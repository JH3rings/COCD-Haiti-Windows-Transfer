"""SAR-only values with optical-conditioned deformable retrieval policies."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"cocd"))
sys.path.insert(0, str(ROOT/"experiments/dlr_unet_study_20261104"))
from unet import PaperUNet
from windows_main.models_da_search import DeformableSearch, SearchContext
from windows_main.models_teacher_target_search_v2 import AnchoredDeformablePolicy


def haar_context(sar):
    pre = sar[:, :6].reshape(len(sar), 3, 2, *sar.shape[-2:]).mean(1)
    delta = sar[:, 6:] - pre
    a,b,c,d = delta[...,::2,::2],delta[...,::2,1::2],delta[...,1::2,::2],delta[...,1::2,1::2]
    bands = torch.cat(((a+b+c+d)/2, (a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2), 1)
    return bands


class OpticalChange(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(6,32,3,2,1),nn.GELU(),
                                     nn.Conv2d(32,64,3,2,1),nn.GELU())
        self.change = nn.Sequential(nn.Conv2d(256,128,1),nn.GELU(),nn.Conv2d(128,128,3,padding=1))

    def forward(self, x):
        before, after = self.encoder(x[:,:6]), self.encoder(x[:,6:])
        return self.change(torch.cat((before,after,after-before,(after-before).abs()),1))


class RetrievalUNet(nn.Module):
    def __init__(self, method="M0", optical=False):
        super().__init__()
        self.method, self.optical = method, optical
        self.backbone = PaperUNet(8)
        self.projections = nn.ModuleList(nn.Conv2d(c,128,1) for c in (256,512,1024))
        self.context = SearchContext(128, dual=False)
        self.policy_head = AnchoredDeformablePolicy(128)
        # Distinct initial directions remove the three-point initialization symmetry.
        with torch.no_grad():
            anchor = torch.tensor([[.5,0.],[0.,.5],[-.5,-.5]])
            self.policy_head.offset.bias.copy_(torch.atanh(anchor/2).repeat(3,1).flatten())
        self.search = DeformableSearch()
        self.inject = nn.Conv2d(128,256,1)
        if method == "M1":
            self.change_projections = nn.ModuleList(nn.Conv2d(8,128,3,padding=1) for _ in range(3))
        if method == "M3":
            self.frequency_query = nn.Conv2d(8,128,1)
        if method == "M4":
            self.channel_head = nn.Conv2d(128,128,1)
            self.eca = nn.Conv1d(1,1,3,padding=1,bias=False)
        if optical:
            self.optical_change = OpticalChange()

    def memory(self, sar):
        assert sar.shape[1] == 8
        maps, x = [], sar
        for i, stage in enumerate(self.backbone.enc):
            x = stage(F.max_pool2d(x,2) if i else x)
            maps.append(x)
        bank = tuple(p(x) for p,x in zip(self.projections,maps[2:]))
        if self.method == "M1":
            pre = sar[:,:6].reshape(len(sar),3,2,*sar.shape[-2:]).mean(1)
            post = sar[:,6:]
            change = torch.cat((pre,post,post-pre,(post-pre).abs()),1)
            bank = tuple(x+p(F.adaptive_avg_pool2d(change,x.shape[-2:]))
                         for p,x in zip(self.change_projections,bank))
        return maps, bank

    def policy(self, bank, sar, optical=None):
        if self.optical != (optical is not None):
            raise ValueError("Optical input is required exclusively for the Teacher")
        context = self.context(bank)
        if self.method == "M3":
            freq = self.frequency_query(haar_context(sar))
            context = context + F.interpolate(freq,size=context.shape[-2:],mode="bilinear",align_corners=False)
        if optical is not None:
            context = context + self.optical_change(optical)
        result = self.policy_head(context)
        if self.method == "M4":
            # Global channel context modulates SAR values, with no additive optical value.
            channel = self.channel_head(context).mean((2,3))
            channel = self.eca(channel[:,None])[:,0,:,None,None]
            result["gain"] = 1 + .5*channel.tanh()
        return result

    def retrieve(self, bank, policy):
        result = self.search(bank,policy)
        if "gain" in policy:
            result = result*policy["gain"]
        return result

    def decode(self, maps, retrieved):
        x = maps[-1]
        for i,(up,dec,skip) in enumerate(zip(self.backbone.up,self.backbone.dec,maps[-2::-1])):
            x = dec(torch.cat((skip,up(x)),1))
            if i == 1:
                x = x + self.inject(retrieved)
        return self.backbone.head(x)

    def details(self, sar, optical=None, forced_policy=None):
        maps,bank = self.memory(sar)
        policy = self.policy(bank,sar,optical) if forced_policy is None else forced_policy
        retrieved = self.retrieve(bank,policy)
        return {"z":self.decode(maps,retrieved),"memory":bank,"maps":maps,
                "policy":policy,"retrieved":retrieved}

    def forward(self, sar, optical=None):
        if self.optical and sar.shape[1] == 20:
            assert optical is None
            sar,optical = sar[:,:8],sar[:,8:]
        return self.details(sar,optical)["z"]


def policy_distribution(policy, memory, preserve_scale=False):
    """Exact bilinear splat in bounded relative coordinates, retaining border clipping."""
    offsets, weights = 2*policy["offsets"].tanh(), policy["weights"]
    b,levels,points,_,h,w = offsets.shape
    radius = int(2*max(h/x.shape[-2] for x in memory))+1
    side = 2*radius+1
    bins = side*side
    dimension = bins*levels if preserve_scale else bins
    yy,xx = torch.meshgrid(torch.arange(h,device=offsets.device),
                           torch.arange(w,device=offsets.device),indexing="ij")
    base_y,base_x = yy.reshape(1,-1,1),xx.reshape(1,-1,1)
    result = weights.new_zeros(b,h*w,dimension)
    for level,mem in enumerate(memory):
        scale = h/mem.shape[-2]
        oy = offsets[:,level,:,0].permute(0,2,3,1).reshape(b,-1,points)
        ox = offsets[:,level,:,1].permute(0,2,3,1).reshape(b,-1,points)
        y,x = (base_y+oy*scale).clamp(0,h-1),(base_x+ox*scale).clamp(0,w-1)
        y0,x0 = y.floor(),x.floor()
        fy,fx = y-y0,x-x0
        weight = weights[:,level].permute(0,2,3,1).reshape(b,-1,points)
        for dy,dx,mass in ((0,0,(1-fy)*(1-fx)),(0,1,(1-fy)*fx),
                           (1,0,fy*(1-fx)),(1,1,fy*fx)):
            yi,xi = (y0+dy).clamp(max=h-1),(x0+dx).clamp(max=w-1)
            index = (yi-base_y+radius).long()*side+(xi-base_x+radius).long()
            if preserve_scale:
                index = index+level*bins
            result = result.scatter_add(2,index,weight*mass)
    return result/result.sum(-1,keepdim=True).clamp_min(1e-8)


def js(p,q):
    middle = (p+q)/2
    return .5*(p*(p.clamp_min(1e-8).log()-middle.clamp_min(1e-8).log())+
               q*(q.clamp_min(1e-8).log()-middle.clamp_min(1e-8).log())).sum(-1)


def kd_losses(student, teacher, output, batch, method):
    with torch.no_grad():
        _, target_bank = teacher.memory(batch["x"])
        target = teacher.policy(target_bank,batch["x"],batch["optical"])
    # M0 follows the original stop-gradient-memory evidence definition.
    bank = tuple(x.detach() for x in output["memory"])
    policy = student.policy(bank,batch["x"])
    mask = F.adaptive_avg_pool2d(batch["valid"].float(),bank[0].shape[-2:]).eq(1)
    keep = mask[:,0].flatten(1)
    ps = policy_distribution(policy,bank,method=="M2")
    pt = policy_distribution(target,bank,method=="M2")
    policy_loss = (js(ps,pt)*keep).sum()/keep.sum().clamp_min(1)
    left,right = student.retrieve(bank,policy),student.retrieve(bank,target).detach()
    evidence = ((1-F.cosine_similarity(left,right,dim=1))*mask[:,0]).sum()/mask.sum().clamp_min(1)
    return policy_loss,evidence
