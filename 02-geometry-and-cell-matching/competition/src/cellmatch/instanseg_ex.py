"""Fixed image-only InstanSeg EX inference using the official tiled wrapper."""
import warnings
import cv2
import numpy as np
import torch
from instanseg import InstanSeg

MODEL_SHA256='636833d26b5afcc03bde1357b3e669cf015e91dd1edda863aa51c19eee986562'

class InstanSegEX:
    def __init__(self,model_path,device="cpu"):
        import hashlib
        from pathlib import Path
        assert hashlib.sha256(Path(model_path).read_bytes()).hexdigest()==MODEL_SHA256
        self.model=InstanSeg(torch.jit.load(str(model_path),map_location='cpu').eval(),device=device,image_reader='skimage.io',verbosity=0)

    def __call__(self,image,scale):
        if not np.isfinite(scale) or not .5<=scale<=4:raise ValueError('Scale outside fixed resource bounds')
        image=np.asarray(image,np.float32);h,w=image.shape
        resized=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_LINEAR)
        lo,hi=np.percentile(resized,[.1,99.9]);resized=((resized-lo)/(hi-lo+1e-6)).astype(np.float32)
        with warnings.catch_warnings(),torch.inference_mode():
            warnings.filterwarnings('ignore',message='Pixel size not provided')
            labels=self.model.eval_medium_image(resized,pixel_size=None,normalise=False,tile_size=512,batch_size=1,return_image_tensor=False,target='all_outputs')
        cells=labels[0,1].cpu().numpy().astype(np.float32)
        cells=cv2.resize(cells,(w,h),interpolation=cv2.INTER_NEAREST).astype(np.int32)
        assert cells.shape==image.shape and np.min(cells)>=0
        return cells
