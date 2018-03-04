# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# ----------------------------------------------------------------------------

import functools
import os.path as osp

import numpy as np

from PIL import Image
from skimage.io import ImageCollection

from ..misc.config import cfg
from ..misc.io import imread_indexed,imwrite_indexed

#################################
# HELPER FUNCTIONS
#################################

def _load_annotation(filename, obj_id = None):
  """ Load image given filename."""

  annotation,_ = imread_indexed(filename, cfg.IM_SIZE)
  if obj_id:
    assert np.max(annotation) < 100, " %s should be index image" % filename
    annotation = (annotation == int(obj_id)).astype(np.uint8)
  else:
    assert len(np.unique(annotation)) <= 2, "%s annotation should be binary image" % filename
    annotation = (annotation != 0).astype(np.uint8)
  return annotation

def get_sequence_name(name, label_id):
    if label_id is None:
      return name
    else:
      return '%s/%s' % (name, label_id)
#################################
# LOADER CLASSES
#################################

class BaseLoader(ImageCollection):

  """
  Base class to load image sets (inherit from skimage.ImageCollection).

  Arguments:
    path      (string): path to sequence folder.
    regex     (string): regular expression to define image search pattern.
    load_func (func)  : function to load image from disk (see skimage.ImageCollection).

  """

  def __init__(self,name, path,filenames, label_id, load_func=None):

    # Sequence name
    self.name = get_sequence_name(name, label_id)
    self.frames = filenames

    frames = [osp.join(path, fname) for fname in filenames]
    super(BaseLoader, self).__init__(
        frames,load_func=load_func)


  def __str__(self):
    return "< class: '{}' name: '{}', frames: {} >".format(
        type(self).__name__,self.name,len(self))

class Sequence(BaseLoader):

  """
  Load image sequences.

  Arguments:
    name  (string): sequence name.
    regex (string): regular expression to define image search pattern.

  """

  def __init__(self, name, path, frames, label_id, suffix='.jpg'):
    if not frames[0].endswith(suffix):
      frames = [fname.split('.')[0] + suffix for fname in frames]
    super(Sequence, self).__init__(
        name, path, frames, label_id)

class Segmentation(BaseLoader):

  """
  Load image sequences.

  Arguments:
    path          (string): path to sequence folder.
    single_object (bool):   assign same id=1 to each object.
    regex         (string): regular expression to define image search pattern.

  """

  def __init__(self, name, path, frames, label_id=None, suffix=".png"):
    super(Segmentation, self).__init__(name, path, frames, label_id,
       functools.partial(_load_annotation,obj_id=label_id))

    if len(self):
      # Extract color palette from image file
      self.color_palette = Image.open(self.files[0]).getpalette()

      if self.color_palette is not None:
        self.color_palette = np.array(
            self.color_palette).reshape(-1,3)
      else:
        self.color_palette = np.array([[255.0,0,0]])
    else:
      self.color_palette = np.array([])




class Annotation(Segmentation):

  """
  Load ground-truth annotations.

  Arguments:
    name          (string): sequence name.
    single_object (bool):   assign same id=1 to each object.
    regex         (string): regular expression to define image search pattern.

  """

  def __init__(self, name, path, frames,label_id, suffix=".png"):
    super(Annotation, self).__init__(
       name, path, frames, label_id, suffix)
