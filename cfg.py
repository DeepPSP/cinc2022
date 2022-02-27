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
BaseCfg.fs = 2000
BaseCfg.torch_dtype = torch.float32  # "double"

BaseCfg.passband = [25, 400]



# training configurations for machine learning and deep learning
TrainCfg = ED()
TrainCfg.torch_dtype = BaseCfg.torch_dtype

# configs of files
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.project_dir = BaseCfg.project_dir
TrainCfg.log_dir = BaseCfg.log_dir
TrainCfg.model_dir = BaseCfg.model_dir

# TODO: add more fields for TrainCfg
TrainCfg.fs = BaseCfg.fs
TrainCfg.siglen = 15  # seconds, to adjust



# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
ModelCfg = ED()
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.fs = BaseCfg.fs

# TODO: add more fields for ModelCfg
