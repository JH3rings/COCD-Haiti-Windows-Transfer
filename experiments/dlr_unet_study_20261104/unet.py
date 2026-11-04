"""Preserved DLR paper-based U-Net body."""
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
 def __init__(self,cin,cout):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.ReLU(),nn.BatchNorm2d(cout),nn.Conv2d(cout,cout,3,padding=1),nn.ReLU(),nn.BatchNorm2d(cout))
 def forward(self,x):return self.net(x)


class PaperUNet(nn.Module):
 """Figure7:64/128/256/512/1024, four pools, four transpose-conv skip decoders."""
 def __init__(self,channels=6):
  super().__init__();widths=[64,128,256,512,1024]
  self.enc=nn.ModuleList([DoubleConv(a,b) for a,b in zip([channels]+widths[:-1],widths)])
  self.up=nn.ModuleList([nn.ConvTranspose2d(a,b,2,2) for a,b in zip(widths[:0:-1],widths[-2::-1])])
  self.dec=nn.ModuleList([DoubleConv(2*b,b) for b in widths[-2::-1]]);self.head=nn.Conv2d(64,1,1)
 def forward(self,x):
  features=[]
  for i,block in enumerate(self.enc):
   if i:x=F.max_pool2d(x,2)
   x=block(x);features.append(x)
  for up,block,skip in zip(self.up,self.dec,features[-2::-1]):x=block(torch.cat([skip,up(x)],1))
  return self.head(x)
