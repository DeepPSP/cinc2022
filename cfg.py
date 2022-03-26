"""
"""

import os, pathlib
from copy import deepcopy
from itertools import repeat
from typing import List, NoReturn

import numpy as np
import torch
from torch_ecg.cfg import CFG
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths

from cfg_models import ModelArchCfg


__all__ = [
    "BaseCfg",
    "TrainCfg",
    "ModelCfg",
]


_BASE_DIR = pathlib.Path(__file__).absolute().parent


BaseCfg = CFG()
BaseCfg.db_dir = None
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
TrainCfg = deepcopy(BaseCfg)

TrainCfg.input_type = "raw"  # "raw", "springer"

TrainCfg.train_ratio = 0.8

# configs of signal preprocessing
TrainCfg.normalize = CFG(
    method="z-score",
    mean=0.0,
    std=1.0,
)

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 50
# TODO: automatic adjust batch size according to GPU capacity
# https://stackoverflow.com/questions/45132809/how-to-select-batch-size-automatically-to-fit-gpu
TrainCfg.batch_size = 64
# TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 5e-4  # 1e-3
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = 10

# configs of loss function
# TrainCfg.loss = "BCEWithLogitsLoss"
# TrainCfg.loss = "BCEWithLogitsWithClassWeightLoss"
TrainCfg.loss = "AsymmetricLoss"  # "FocalLoss"
TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")
TrainCfg.flooding_level = 0.0  # flooding performed if positive,

TrainCfg.log_step = 20
TrainCfg.eval_every = 20

# tasks of training
TrainCfg.tasks = [
    "classification",
    "segmentation",
]

# configs of model selection
# "resnet_leadwise", "multi_scopic_leadwise", "vgg16", "resnet", "vgg16_leadwise", "cpsc", "cpsc_leadwise", etc.

for t in TrainCfg.tasks:
    TrainCfg[t] = CFG()

TrainCfg.classification = CFG()
TrainCfg.classification.fs = BaseCfg.fs
TrainCfg.classification.data_format = "channel_first"
TrainCfg.classification.num_channels = 1
TrainCfg.classification.input_len = int(30 * TrainCfg.classification.fs)  # 30 seconds, to adjust
TrainCfg.classification.siglen = TrainCfg.classification.input_len  # alias
TrainCfg.classification.sig_slice_tol = 0.4  # None, do no slicing
TrainCfg.classification.classes = deepcopy(BaseCfg.classes)
TrainCfg.classification.class_map = {c:i for i,c in enumerate(TrainCfg.classification.classes)}
TrainCfg.classification.resample = CFG(fs=TrainCfg.classification.fs)
TrainCfg.classification.bandpass = CFG(
    lowcut=BaseCfg.passband[0],
    highcut=BaseCfg.passband[1],
    filter_type="butter",
    order=BaseCfg.order,
)
TrainCfg.classification.normalize = TrainCfg.normalize
TrainCfg.classification.final_model_name = None
TrainCfg.classification.model_name = "crnn"
TrainCfg.classification.cnn_name = "resnet_nature_comm_bottle_neck_se"
TrainCfg.classification.rnn_name = "none"  # "none", "lstm"
TrainCfg.classification.attn_name = "se"  # "none", "se", "gc", "nl"
TrainCfg.classification.monitor = "challenge_metric"  # accuracy (not recommended)
TrainCfg.classification.loss = "AsymmetricLoss"  # "FocalLoss"
TrainCfg.classification.loss_kw = CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")


TrainCfg.segmentation = CFG()
TrainCfg.segmentation.fs = 500
TrainCfg.segmentation.data_format = "channel_first"
TrainCfg.segmentation.num_channels = 1
TrainCfg.segmentation.input_len = int(30 * TrainCfg.segmentation.fs)  # 30seconds, to adjust
TrainCfg.segmentation.siglen = TrainCfg.segmentation.input_len  # alias
TrainCfg.segmentation.sig_slice_tol = 0.4  # None, do no slicing
TrainCfg.segmentation.classes = BaseCfg.states
TrainCfg.segmentation.class_map = {c:i for i,c in enumerate(TrainCfg.segmentation.classes)}
TrainCfg.segmentation.resample = CFG(fs=TrainCfg.segmentation.fs)
TrainCfg.segmentation.bandpass = CFG(
    lowcut=BaseCfg.passband[0],
    highcut=BaseCfg.passband[1],
    filter_type="butter",
    order=BaseCfg.order,
)
TrainCfg.segmentation.normalize = TrainCfg.normalize
TrainCfg.segmentation.final_model_name = None
TrainCfg.segmentation.model_name = "seq_lab"  # unet
TrainCfg.segmentation.cnn_name = "resnet_nature_comm_bottle_neck_se"
TrainCfg.segmentation.rnn_name = "lstm"  # "none", "lstm"
TrainCfg.segmentation.attn_name = "se"  # "none", "se", "gc", "nl"
TrainCfg.segmentation.monitor = "jaccard"
TrainCfg.segmentation.loss = "AsymmetricLoss"  # "FocalLoss"
TrainCfg.segmentation.loss_kw = CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")



# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted

_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype


ModelCfg = deepcopy(_BASE_MODEL_CONFIG)

for t in TrainCfg.tasks:
    ModelCfg[t] = deepcopy(_BASE_MODEL_CONFIG)
    ModelCfg[t].task = t
    ModelCfg[t].fs = TrainCfg[t].fs


ModelCfg.classification.update(ModelArchCfg.classification)
adjust_cnn_filter_lengths(ModelCfg.classification, ModelCfg.classification.fs)


ModelCfg.classification.input_len = TrainCfg.classification.input_len
ModelCfg.classification.classes = TrainCfg.classification.classes
ModelCfg.classification.model_name = TrainCfg.classification.model_name
ModelCfg.classification.cnn_name = TrainCfg.classification.cnn_name
ModelCfg.classification.rnn_name = TrainCfg.classification.rnn_name
ModelCfg.classification.attn_name = TrainCfg.classification.attn_name




ModelCfg.segmentation.update(ModelArchCfg.segmentation)
adjust_cnn_filter_lengths(ModelCfg.segmentation, ModelCfg.segmentation.fs)

ModelCfg.classification.input_len = TrainCfg.classification.input_len
ModelCfg.classification.classes = TrainCfg.classification.classes
ModelCfg.classification.model_name = TrainCfg.classification.model_name
ModelCfg.classification.cnn_name = TrainCfg.classification.cnn_name
ModelCfg.classification.rnn_name = TrainCfg.classification.rnn_name
ModelCfg.classification.attn_name = TrainCfg.classification.attn_name
