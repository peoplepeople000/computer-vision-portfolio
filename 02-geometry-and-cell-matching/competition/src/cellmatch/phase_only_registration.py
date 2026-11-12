"""Linear-padded unit-cross-power translation scores on legal valid shifts.

Custom proposal score, not cv.phaseCorrelate's same-size/subpixel procedure.
Partial fields of view and padding boundaries can distort normalized phase.
"""
from collections import OrderedDict
import numpy as np
from scipy.fft import rfftn,irfftn,next_fast_len

def fft_shape(field_shape,template_shape):
 return tuple(next_fast_len(int(a+b-1)) for a,b in zip(field_shape,template_shape))

class PhaseValid:
 """Cache at most8 identical field FFT sizes; caching changes no score."""
 def __init__(self):self.field=None;self.field_original=None;self.cache=OrderedDict()
 def __call__(self,field,template):
  if self.field is None:self.field_original=field;self.field=np.asarray(field,np.float64)
  assert field is self.field_original,'Use one scorer per registration field'
  template=np.asarray(template,np.float64);assert template.ndim==self.field.ndim==2
  valid=tuple(a-b+1 for a,b in zip(self.field.shape,template.shape));assert min(valid)>0
  shape=fft_shape(self.field.shape,template.shape)
  if shape not in self.cache:
   self.cache[shape]=rfftn(self.field,s=shape,workers=1)
   if len(self.cache)>8:self.cache.popitem(last=False)
  self.cache.move_to_end(shape)
  cross=self.cache[shape]*np.conj(rfftn(template,s=shape,workers=1));amplitude=np.abs(cross);epsilon=np.finfo(np.float64).eps*float(amplitude.max())
  normalized=np.divide(cross,amplitude,out=np.zeros_like(cross),where=amplitude>epsilon)
  result=irfftn(normalized,s=shape,workers=1)[:valid[0],:valid[1]].copy();assert np.isfinite(result).all()
  return result

def phase_valid(field,template):return PhaseValid()(field,template)

def raw_fft_valid(field,template):
 """Unnormalized arithmetic control only; never used in production ranking."""
 shape=fft_shape(field.shape,template.shape);cross=rfftn(np.asarray(field,np.float64),s=shape,workers=1)*np.conj(rfftn(np.asarray(template,np.float64),s=shape,workers=1));valid=tuple(a-b+1 for a,b in zip(field.shape,template.shape))
 return irfftn(cross,s=shape,workers=1)[:valid[0],:valid[1]].copy()
