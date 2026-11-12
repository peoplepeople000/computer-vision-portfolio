"""Fixed seeded random-walker refinement restricted to a one-pixel boundary band."""
import numpy as np
from scipy.ndimage import binary_erosion,binary_dilation
from skimage.segmentation import random_walker
CROSS=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)
def refine_patch(image,mask):
 image=np.asarray(image,dtype=np.float64);mask=np.asarray(mask,bool);assert image.shape==mask.shape
 core=binary_erosion(mask,structure=CROSS);support=binary_dilation(mask,structure=CROSS)
 if not core.any() or support.all():return mask.copy()
 lo,hi=np.percentile(image,[1,99]);image=np.clip((image-lo)/max(hi-lo,1e-8),0,1)
 if image.std()<1e-8:return mask.copy()
 markers=np.zeros(mask.shape,np.int32);markers[core]=1;markers[~support]=2
 changed=random_walker(image,markers,beta=130,mode='bf')==1
 assert changed[core].all() and not changed[~support].any()
 if not changed.any() or np.count_nonzero(changed&mask)<=.5*np.count_nonzero(changed|mask):return mask.copy()
 return changed
