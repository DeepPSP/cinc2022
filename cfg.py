"""
"""

import os, pathlib
from copy import deepcopy
from itertools import repeat
from typing import List, NoReturn

import numpy as np
import torch
from easydict import EasyDict as ED

__all__ = [
    "BaseCfg",
    "TrainCfg",
    "ModelCfg",
]


_BASE_DIR = pathlib.Path(__file__).absolute().parent


BaseCfg = ED()
BaseCfg.db_dir = pathlib.Path("/home/wenhao/Jupyter/wenhao/data/CinC2021/")
BaseCfg.project_dir = _BASE_DIR
BaseCfg.log_dir = _BASE_DIR / "log"
BaseCfg.model_dir = _BASE_DIR / "saved_models"
BaseCfg.log_dir.mkdir(exist_ok=True)
BaseCfg.model_dir.mkdir(exist_ok=True)
BaseCfg.fs = 800
BaseCfg.torch_dtype = torch.float32  # "double"

BaseCfg.classes = ["Present", "Absent", "Unknown",]
BaseCfg.states = ["unannotated", "S1", "systolic", "S2", "diastolic",]

# for example, can use scipy.signal.buttord(wp=[15, 250], ws=[5, 400], gpass=1, gstop=40, fs=1000)
BaseCfg.passband = [25, 400]  # Hz, candidates: [20, 500], [15, 250]
BaseCfg.order = 5



# training configurations for machine learning and deep learning
TrainCfg = ED()
TrainCfg.torch_dtype = BaseCfg.torch_dtype

# configs of files
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.project_dir = BaseCfg.project_dir
TrainCfg.log_dir = BaseCfg.log_dir
TrainCfg.model_dir = BaseCfg.model_dir

# TODO: add more fields for TrainCfg

# tasks of training
TrainCfg.tasks = [
    "classification",
    "segmentation",
]

# configs of model selection
# "resnet_leadwise", "multi_scopic_leadwise", "vgg16", "resnet", "vgg16_leadwise", "cpsc", "cpsc_leadwise", etc.

for t in TrainCfg.tasks:
    TrainCfg[t] = ED()

TrainCfg.classification = ED()
TrainCfg.classification.fs = BaseCfg.fs
TrainCfg.classification.passband = BaseCfg.passband
TrainCfg.classification.order = BaseCfg.order
TrainCfg.classification.siglen = 15  # seconds, to adjust

TrainCfg.segmentation = ED()
TrainCfg.segmentation.fs = 500
TrainCfg.segmentation.passband = [15, 250]
TrainCfg.segmentation.order = BaseCfg.order
TrainCfg.segmentation.siglen = 30  # seconds, to adjust



# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
ModelCfg = ED()
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.fs = BaseCfg.fs

# TODO: add more fields for ModelCfg
