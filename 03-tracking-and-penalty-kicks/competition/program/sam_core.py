"""Small readout and coordinate contracts; no native body-model dependency."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

ARMS=('A_last','B_sequence','C_sequence_axis')
OUT=Path(__file__).resolve().parent
R=OUT.parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
    t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def digest(state):
    h=hashlib.sha256()
    for n,v in sorted(state.items()):
        h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def key(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def crop_inputs(box,K,geometry,flip,size=384):
    scale=size/geometry['side'];origin=np.asarray(geometry['origin'])
    A=np.array([[scale,0,-scale*origin[0]],[0,scale,-scale*origin[1]],[0,0,1.]])
    b=np.asarray(box).reshape(2,2);b=(np.c_[b,np.ones(2)]@A.T)[:,:2].reshape(4)
    kc=A@K
    if flip:
        b=np.array([size-1-b[2],b[1],size-1-b[0],b[3]])
        F=np.array([[-1.,0,size-1],[0,1.,0],[0,0,1.]])
        kc=F@kc@np.diag([-1.,1.,1.])
    return b,kc,A

def elapsed(times,mask):
    """Elapsed seconds since previous VALID observation; never shorten a gap."""
    previous=torch.zeros_like(times[:,0]);seen=torch.zeros_like(mask[:,0]);dt=[]
    for t in range(times.shape[1]):
        dt.append(torch.where(seen,times[:,t]-previous,torch.zeros_like(previous)))
        previous=torch.where(mask[:,t],times[:,t],previous);seen|=mask[:,t]
    return torch.stack(dt,1)

class TokenHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(1024,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.15))
        self.clock=nn.Sequential(nn.Linear(2,64),nn.GELU(),nn.Linear(64,64))
        self.gru=nn.GRUCell(64,64)
        self.classifier=nn.Sequential(nn.Linear(64*3+3,128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,3))
    def forward(self,tokens,mask,times,axis,arm):
        assert arm in ARMS and tokens.shape[1:]==(5,1024)
        if arm=='A_last':tokens=tokens[:,-1:].expand(-1,5,-1)
        t=torch.stack([times/.32,elapsed(times,mask)/.1],-1)
        x=self.project(tokens)+self.clock(t)
        h=torch.zeros(len(x),64,device=x.device);first=h.clone();total=h.clone();seen=torch.zeros(len(x),dtype=torch.bool,device=x.device)
        for i in range(5):
            h=torch.where(mask[:,i,None],self.gru(x[:,i],h),h)
            first=torch.where((mask[:,i]&~seen)[:,None],h,first)
            total=total+torch.where(mask[:,i,None],h,torch.zeros_like(h));seen|=mask[:,i]
        axis=axis if arm=='C_sequence_axis' else torch.zeros_like(axis)
        return self.classifier(torch.cat([total/mask.sum(1).clamp(min=1)[:,None],h,h-first,axis],-1))
