"""Shared SAR memory/decoder, separate SAR and optical-conditioned policies."""
import copy
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
OLD=Path(__file__).resolve().parents[1]/'dlr_retrieval_five_20261107/model.py'
spec=importlib.util.spec_from_file_location('retrieval_core_shared',OLD)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

class LocalQuery(nn.Module):
    def __init__(self,window=3):
        super().__init__();self.window=window
        self.q=nn.Conv2d(128,32,1);self.k=nn.Conv2d(128,32,1);self.v=nn.Conv2d(128,128,1)
        self.out=nn.Conv2d(128,128,1);self.norm=nn.GroupNorm(8,128)
    def forward(self,sar,opt):
        b,c,h,w=sar.shape;n=self.window**2
        q=F.normalize(self.q(sar),dim=1).flatten(2)
        k=F.normalize(self.k(opt),dim=1)
        k=F.unfold(k,self.window,padding=self.window//2).view(b,32,n,h*w)
        scores=(q[:,:,None]*k).sum(1)*4
        valid=F.unfold(torch.ones_like(opt[:,:1]),self.window,padding=self.window//2).bool()
        weights=scores.masked_fill(~valid,-1e4).softmax(1)
        v=F.unfold(self.v(opt),self.window,padding=self.window//2).view(b,c,n,h*w)
        response=(v*weights[:,None]).sum(2).view(b,c,h,w)
        return sar+self.out(self.norm(response))

class MultiOptical(nn.Module):
    def __init__(self):
        super().__init__();self.enc=nn.ModuleList([nn.Sequential(nn.Conv2d(6,32,3,2,1),nn.GELU()),nn.Sequential(nn.Conv2d(32,64,3,2,1),nn.GELU()),nn.Sequential(nn.Conv2d(64,128,3,2,1),nn.GELU())])
        self.proj=nn.ModuleList(nn.Sequential(nn.Conv2d(c*2,128,1),nn.GroupNorm(8,128)) for c in [32,64,128])
    def forward(self,x):
        pre,post=x.chunk(2,1);out=[]
        for enc,p in zip(self.enc,self.proj):
            pre,post=enc(pre),enc(post);d=post-pre
            out.append(p(torch.cat([d,d.abs()],1)))
        return out

class SharedRetrieval(nn.Module):
    def __init__(self,version='V0',sar_control=False):
        super().__init__();self.version=version;self.sar_control=sar_control
        self.core=base.RetrievalUNet('M0',False)
        self.teacher_context=copy.deepcopy(self.core.context)
        self.teacher_head=copy.deepcopy(self.core.policy_head)
        self.optical=MultiOptical() if version in ('V1','V3') else base.OpticalChange()
        self.cross=LocalQuery(5 if version=='V3' else 3) if version in ('V2','V3','V4') else None
    def optical_input(self,sar,opt):
        if not self.sar_control:return opt
        # Capacity control: same auxiliary network, SAR only; no fabricated observations.
        pre=sar[:,:6].reshape(len(sar),3,2,*sar.shape[-2:]).mean(1);post=sar[:,6:]
        return torch.cat([pre.repeat(1,3,1,1),post.repeat(1,3,1,1)],1)
    def policies(self,bank,sar,opt):
        ps=self.core.policy(bank,sar)
        q=self.teacher_context(bank)
        o=self.optical(self.optical_input(sar,opt))
        if isinstance(o,list):o=sum(F.interpolate(x,size=q.shape[-2:],mode='bilinear',align_corners=False) for x in o)/len(o)
        qt=self.cross(q,o) if self.cross else q+o
        return ps,self.teacher_head(qt)
    def details(self,sar,opt):
        maps,bank=self.core.memory(sar);ps,pt=self.policies(bank,sar,opt)
        es,et=self.core.retrieve(bank,ps),self.core.retrieve(bank,pt)
        return dict(maps=maps,bank=bank,ps=ps,pt=pt,es=es,et=et,zs=self.core.decode(maps,es),zt=self.core.decode(maps,et))
    def forward(self,x):return self.core(x)
    def distill(self,o,sar,opt):
        # Preserve original COCD stop-gradient-value/controller-only KD definition.
        bank=tuple(x.detach() for x in o['bank'])
        ps,pt=self.policies(bank,sar,opt)
        pt={k:v.detach() for k,v in pt.items()}
        p=base.policy_distribution(ps,bank,self.version=='V3');t=base.policy_distribution(pt,bank,self.version=='V3')
        return bank,ps,pt,p,t
