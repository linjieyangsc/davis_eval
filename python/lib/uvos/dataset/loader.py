# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# ----------------------------------------------------------------------------

__author__ = 'federico perazzi'
__version__ = '2.0.0'

########################################################################
#
# Interface for accessing the DAVIS 2016/2017 dataset.
#
# DAVIS is a video dataset designed for segmentation. The API implemented in
# this file provides functionalities for loading, parsing and visualizing
# images and annotations available in DAVIS. Please visit
# [https://graphics.ethz.ch/~perazzif/davis] for more information on DAVIS,
# including data, paper and supplementary material.
#
########################################################################

from collections import namedtuple

import uvos
import numpy as np

from PIL import Image
from base import Sequence, Annotation, BaseLoader, Segmentation
from base import get_sequence_name
from ..misc.config import cfg,phase,db_read_sequences

from easydict import EasyDict as edict

class UVOSLoader(object):
  """
  Helper class for accessing the DAVIS dataset.

  Arguments:
    year          (string): dataset version (2016,2017).
    phase         (string): dataset set eg. train, val. (See config.phase)
    single_object (bool):   assign same id (==1) to each object.

  Members:
    sequences (list): list of 'Sequence' objects containing RGB frames.
    annotations(list): list of 'Annotation' objects containing ground-truth segmentations.
  """
  def __init__(self,phase,single_object=False):
    super(UVOSLoader, self).__init__()

    self._phase = phase
    self._single_object = single_object


    self._db_sequences = db_read_sequences(self._phase)

    # Load sequences
    self.sequences = []
    self.annotations = []
    for s in self._db_sequences:
      frames_dict = s['frames']
      im_path = s['image_path']
      anno_path = s['anno_path']
      for label_id,frames in frames_dict.iteritems():
        name = s['vid'] #get_sequence_name(s['vid'], label_id)
        self.sequences.append(Sequence(name, im_path, frames, label_id))

        self.annotations.append(Annotation(name, anno_path, frames, label_id))

    self._keys = dict(zip([s.name for s in self.sequences],
      range(len(self.sequences))))



    try:
      self.color_palette = np.array(Image.open(
        self.annotations[0].files[0]).getpalette()).reshape(-1,3)
    except Exception as e:
      self.color_palette = np.array([[0,255,0]])

  def __len__(self):
    """ Number of sequences."""
    return len(self.sequences)

  def __iter__(self):
    """ Iteratator over pairs of (sequence,annotation)."""
    for sequence,annotation in zip(self.sequences,self.annotations):
      yield sequence,annotation

  def __getitem__(self, key):
    """ Get sequences and annotations pairs."""
    if isinstance(key,str) or isinstance(key, unicode):
      sid = self._keys[key]
    elif isinstance(key,int):
      sid = key
    else:
      raise InputError()

    return edict({
      'images'  : self.sequences[sid],
      'annotations': self.annotations[sid]
      })

  def sequence_name_to_id(self,name):
    """ Map sequence name to index."""
    return self._keys[name]

  def sequence_id_to_name(self,sid):
    """ Map index to sequence name."""
    return self.sequences[sid].name

  def iternames(self):
    """ Iterator over sequence names."""
    for s in self.annotations:
      yield s.name, s.frames

  def iteritems(self):
    return self.__iter__()
