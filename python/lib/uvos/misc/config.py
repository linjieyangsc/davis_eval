#!/usr/bin/env python
import yaml

# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from FAST-RCNN (Ross Girshick)
# ----------------------------------------------------------------------------

""" Configuration file."""

import os
import os.path as osp

import sys
from easydict import EasyDict as edict
import json
from enum import Enum

class phase(Enum):
    VAL      = 'all_val'
    VAL_SUBSET = 'val_subset'
    TESTSEEN = 'test-seen'
    TESTUNSEEN = 'test-unseen'

__C = edict()

# Public access to configuration settings
cfg = __C

# Number of CPU cores used to parallelize evaluation.
__C.N_JOBS = 32

# Paths to dataset folders
__C.PATH = edict()

__C.IM_SIZE = (448, 256)
__C.PHASE = phase.VAL

# Multiobject evaluation (Set to False only when evaluating DAVIS 2016)
__C.MULTIOBJECT = True

# Root folder of project
__C.PATH.ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../../..'))

# Data folder
__C.PATH.DATA = osp.abspath('/raid/ljyang/data/LSVOS')

# Path to input images
__C.PATH.SEQUENCES = __C.PATH.DATA

# Path to annotations
__C.PATH.ANNOTATIONS = __C.PATH.DATA

# Color palette
__C.PATH.PALETTE = osp.abspath(osp.join(__C.PATH.ROOT, 'data/palette.txt'))

# Paths to files
__C.FILES = edict()


__C.FILES.DB_INFO = osp.join(__C.PATH.DATA,'{}_seqs.json') 
# Measures and Statistics
__C.EVAL = edict()

# Metrics: J: region similarity, F: contour accuracy, T: temporal stability
__C.EVAL.METRICS = ['J','F']

# Statistics computed for each of the metrics listed above
__C.EVAL.STATISTICS= ['mean','recall','decay']



def db_read_sequences(db_phase=None):
  """ Read list of sequences. """
  print db_phase.value
  info_path = __C.FILES.DB_INFO.format(db_phase.value)
  db_info = json.load(open(info_path))

  sequences = db_info


  return sequences


import numpy as np
__C.palette = np.loadtxt(__C.PATH.PALETTE,dtype=np.uint8).reshape(-1,3)
