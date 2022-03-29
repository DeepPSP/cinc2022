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
from inputs import InputConfig


__all__ = [
    "BaseCfg",
    "TrainCfg",
    "ModelCfg",
]


_BASE_DIR = pathlib.Path(__file__).absolute().parent


###############################################################################
# Base Configs,
# including path, data type, classes, etc.
###############################################################################

BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.project_dir = _BASE_DIR
BaseCfg.log_dir = _BASE_DIR / "log"
BaseCfg.model_dir = _BASE_DIR / "saved_models"
BaseCfg.log_dir.mkdir(exist_ok=True)
BaseCfg.model_dir.mkdir(exist_ok=True)
BaseCfg.fs = 1000
BaseCfg.torch_dtype = torch.float32  # "double"

BaseCfg.classes = [
    "Present",
    "Absent",
    "Unknown",
]
BaseCfg.states = [
    "unannotated",
    "S1",
    "systolic",
    "S2",
    "diastolic",
]

# for example, can use scipy.signal.buttord(wp=[15, 250], ws=[5, 400], gpass=1, gstop=40, fs=1000)
BaseCfg.passband = [25, 400]  # Hz, candidates: [20, 500], [15, 250]
BaseCfg.order = 5


###############################################################################
# training configurations for machine learning and deep learning
###############################################################################

TrainCfg = deepcopy(BaseCfg)

###########################################
# common configurations for all tasks
###########################################

TrainCfg.checkpoints = _BASE_DIR / "checkpoints"
TrainCfg.checkpoints.mkdir(exist_ok=True)

TrainCfg.train_ratio = 0.8

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 50
# TODO: automatic adjust batch size according to GPU capacity
# https://stackoverflow.com/questions/45132809/how-to-select-batch-size-automatically-to-fit-gpu
TrainCfg.batch_size = 24

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

# configs of callbacks, including early stopping, checkpoint, etc.
TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = 20
TrainCfg.keep_checkpoint_max = 10

# configs of loss function
TrainCfg.loss = "AsymmetricLoss"  # "FocalLoss", "BCEWithLogitsLoss"
TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")
TrainCfg.flooding_level = 0.0  # flooding performed if positive,

# configs of logging
TrainCfg.log_step = 20
# TrainCfg.eval_every = 20

###########################################
# task specific configurations
###########################################

# tasks of training
TrainCfg.tasks = [
    "classification",
    "segmentation",
]

for t in TrainCfg.tasks:
    TrainCfg[t] = CFG()

###########################################
# classification configurations
###########################################

TrainCfg.classification.fs = BaseCfg.fs
TrainCfg.classification.final_model_name = None

# input format configurations
TrainCfg.classification.data_format = "channel_first"
TrainCfg.classification.input_config = InputConfig(
    input_type="waveform",  # "waveform", "spectrogram", "mel", "mfcc", "spectral"
    n_channels=1,
    fs=TrainCfg.classification.fs,
)
TrainCfg.classification.num_channels = TrainCfg.classification.input_config.n_channels
TrainCfg.classification.input_len = int(
    30 * TrainCfg.classification.fs
)  # 30 seconds, to adjust
TrainCfg.classification.siglen = TrainCfg.classification.input_len  # alias
TrainCfg.classification.sig_slice_tol = 0.2  # None, do no slicing
TrainCfg.classification.classes = deepcopy(BaseCfg.classes)
TrainCfg.classification.class_map = {
    c: i for i, c in enumerate(TrainCfg.classification.classes)
}

# preprocess configurations
TrainCfg.classification.resample = CFG(fs=TrainCfg.classification.fs)
TrainCfg.classification.bandpass = CFG(
    lowcut=BaseCfg.passband[0],
    highcut=BaseCfg.passband[1],
    filter_type="butter",
    order=BaseCfg.order,
)

# model choices
TrainCfg.classification.model_name = "crnn"
TrainCfg.classification.cnn_name = "resnet_nature_comm_bottle_neck_se"
TrainCfg.classification.rnn_name = "none"  # "none", "lstm"
TrainCfg.classification.attn_name = "se"  # "none", "se", "gc", "nl"

# loss function choices
TrainCfg.classification.loss = "AsymmetricLoss"  # "FocalLoss"
TrainCfg.classification.loss_kw = CFG(
    gamma_pos=0, gamma_neg=0.2, implementation="deep-psp"
)

# monitor choices
TrainCfg.classification.monitor = "challenge_metric"  # accuracy (not recommended)

###########################################
# classification configurations
###########################################

TrainCfg.segmentation.fs = 1000
TrainCfg.segmentation.final_model_name = None

# input format configurations
TrainCfg.segmentation.data_format = "channel_first"
TrainCfg.segmentation.input_config = InputConfig(
    input_type="waveform",  # "waveform", "spectrogram", "mel", "mfcc", "spectral"
    n_channels=1,
    fs=TrainCfg.segmentation.fs,
)
TrainCfg.segmentation.num_channels = TrainCfg.segmentation.input_config.n_channels
TrainCfg.segmentation.input_len = int(
    30 * TrainCfg.segmentation.fs
)  # 30seconds, to adjust
TrainCfg.segmentation.siglen = TrainCfg.segmentation.input_len  # alias
TrainCfg.segmentation.sig_slice_tol = 0.4  # None, do no slicing
TrainCfg.segmentation.classes = BaseCfg.states
TrainCfg.segmentation.class_map = {
    c: i for i, c in enumerate(TrainCfg.segmentation.classes)
}

# preprocess configurations
TrainCfg.segmentation.resample = CFG(fs=TrainCfg.segmentation.fs)
TrainCfg.segmentation.bandpass = CFG(
    lowcut=BaseCfg.passband[0],
    highcut=BaseCfg.passband[1],
    filter_type="butter",
    order=BaseCfg.order,
)

# model choices
TrainCfg.segmentation.model_name = "seq_lab"  # unet
TrainCfg.segmentation.cnn_name = "resnet_nature_comm_bottle_neck_se"
TrainCfg.segmentation.rnn_name = "lstm"  # "none", "lstm"
TrainCfg.segmentation.attn_name = "se"  # "none", "se", "gc", "nl"

# loss function choices
TrainCfg.segmentation.loss = "AsymmetricLoss"  # "FocalLoss"
TrainCfg.segmentation.loss_kw = CFG(
    gamma_pos=0, gamma_neg=0.2, implementation="deep-psp"
)

# monitor choices
TrainCfg.segmentation.monitor = "jaccard"


###############################################################################
# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
###############################################################################

_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype


ModelCfg = deepcopy(_BASE_MODEL_CONFIG)

for t in TrainCfg.tasks:
    ModelCfg[t] = deepcopy(_BASE_MODEL_CONFIG)
    ModelCfg[t].task = t
    ModelCfg[t].fs = TrainCfg[t].fs

    ModelCfg[t].update(ModelArchCfg[t])

    ModelCfg[t].classes = TrainCfg[t].classes
    ModelCfg[t].num_channels = TrainCfg[t].num_channels
    ModelCfg[t].input_len = TrainCfg[t].input_len
    ModelCfg[t].model_name = TrainCfg[t].model_name
    ModelCfg[t].cnn_name = TrainCfg[t].cnn_name
    ModelCfg[t].rnn_name = TrainCfg[t].rnn_name
    ModelCfg[t].attn_name = TrainCfg[t].attn_name

    # adjust filter length; cnn, rnn, attn choices in model configs
    for mn in [
        "crnn",
        "seq_lab",
        "unet",
    ]:
        if mn not in ModelCfg[t]:
            continue
        adjust_cnn_filter_lengths(ModelCfg[t][mn], ModelCfg[t].fs)
        ModelCfg[t][mn].cnn_name = ModelCfg[t].cnn_name
        ModelCfg[t][mn].rnn_name = ModelCfg[t].rnn_name
        ModelCfg[t][mn].attn_name = ModelCfg[t].attn_name
