
from PIL import Image
import numpy as np

from uvos import cfg

def imread_indexed(filename, im_size=None):
  """ Load image given filename."""

  im = Image.open(filename)
  if im_size:
    im = im.resize(im_size)
  annotation = np.atleast_3d(im)[...,0]
  if im.getpalette():
    return annotation,np.array(im.getpalette()).reshape((-1,3))
  else:
    return annotation, None

def imwrite_indexed(filename,array,color_palette=cfg.palette):
  """ Save indexed png."""

  if np.atleast_3d(array).shape[2] != 1:
    raise Exception("Saving indexed PNGs requires 2D array.")

  im = Image.fromarray(array)
  im.putpalette(color_palette.ravel())
  im.save(filename, format='PNG')
