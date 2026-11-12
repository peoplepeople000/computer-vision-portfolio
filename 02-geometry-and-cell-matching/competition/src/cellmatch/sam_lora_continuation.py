"""Local, fixed-encoder decoder-adapter continuation from returned SAM weights."""
from pathlib import Path
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from cellmatch.direct_center_ex import sha
from cellmatch.microsam_lora import insert_adapters

MODEL=Path('models/microsam_vit_b_lm/vit_b.pt')
WARM=Path('outputs/stage467_sam_lora_return_audit/returned/held_b2_adapters.pt')

def load(device='mps',train_decoder=False):
    assert sha(MODEL)=='ffe862fa5b58d78375506041faf0eb3927ced5b5391e8be383e6d7ac07aed924'
    assert sha(WARM)=='106806d53221dd0a00487b8627bcd7f1c0737af977d87b2cf39fc582034c444a'
    state=torch.load(WARM,map_location='cpu',weights_only=True)
    assert state['held_subject']=='subject_b2ba5e' and state['steps']==400
    assert set(state['training_subjects'])=={'subject_5d294c','subject_db6b8b'}
    sam=sam_model_registry['vit_b']();sam.load_state_dict(torch.load(MODEL,map_location='cpu',weights_only=True),strict=True)
    insert_adapters(sam,4);parameters=dict(sam.named_parameters())
    names=[n for n,p in parameters.items() if p.requires_grad]
    assert set(names)==set(state['adapters']) and sum(parameters[n].numel() for n in names)==171008
    with torch.no_grad():
        for n in names:parameters[n].copy_(state['adapters'][n])
    sam.requires_grad_(False)
    if train_decoder:
        for n in names:
            if n.startswith('mask_decoder.'):parameters[n].requires_grad_(True)
        assert sum(p.numel() for p in sam.parameters() if p.requires_grad)==23552
    return sam.to(device).eval(),names,state

def full_adapter_state(sam,names):
    parameters=dict(sam.named_parameters())
    return {n:parameters[n].detach().cpu().clone() for n in names}

def restore(sam,state):
    parameters=dict(sam.named_parameters())
    with torch.no_grad():
        for n,v in state.items():parameters[n].copy_(v.to(parameters[n].device))

def field(sam,gray):
    assert gray.shape==(256,256) and gray.dtype==np.uint8
    rgb=np.repeat(gray[...,None],3,axis=2);image=ResizeLongestSide(1024).apply_image(rgb)
    device=next(sam.parameters()).device
    data=sam.preprocess(torch.as_tensor(image.copy(),device=device).permute(2,0,1))[None]
    with torch.no_grad():value=sam.image_encoder(data).detach()
    assert value.shape==(1,256,64,64) and torch.isfinite(value).all()
    return value

def decode(sam,features,points):
    device=next(sam.parameters()).device
    query=torch.as_tensor(np.asarray(points,dtype=np.float32)*4,device=device)[:,None]
    with torch.no_grad():
        sparse,dense=sam.prompt_encoder(points=(query,torch.ones((len(points),1),dtype=torch.long,device=device)),boxes=None,masks=None)
        pe=sam.prompt_encoder.get_dense_pe()
    logits,score=sam.mask_decoder(image_embeddings=features,image_pe=pe,sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
    return sam.postprocess_masks(logits,(1024,1024),(256,256)),score[:,0]

def update(sam,optimizer,features,points,targets):
    optimizer.zero_grad(set_to_none=True);logits,score=decode(sam,features,points)
    truth=torch.as_tensor(np.asarray(targets,dtype=np.float32),device=logits.device)[:,None]
    binary=torch.nn.functional.binary_cross_entropy_with_logits(logits,truth)
    presence=torch.nn.functional.mse_loss(score,torch.ones_like(score))
    loss=binary+presence;assert torch.isfinite(loss)
    loss.backward();assert all(p.grad is None for p in sam.parameters() if not p.requires_grad)
    torch.nn.utils.clip_grad_norm_([p for p in sam.parameters() if p.requires_grad],5)
    optimizer.step();torch.mps.synchronize()
    return dict(loss=float(loss.detach().cpu()),binary=float(binary.detach().cpu()),presence=float(presence.detach().cpu()))

def metrics(logits,targets):
    pred=np.asarray(logits)[:,0]>0;truth=np.asarray(targets)>0
    ious=(pred&truth).sum((1,2))/(pred|truth).sum((1,2))
    return dict(ious=ious.tolist(),strict_valid=int((ious>.75).sum()),mean_iou=float(ious.mean()))
