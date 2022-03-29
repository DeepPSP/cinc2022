#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

import os, time
from copy import deepcopy

import numpy as np
import torch
from torch.nn.parallel import (
    DistributedDataParallel as DDP,
    DataParallel as DP,
)
from torch.utils.data import DataLoader

from cfg import TrainCfg, ModelCfg
from dataset import CinC2022Dataset
from inputs import (
    InputConfig,
    BaseInput,
    WaveformInput,
    SpectrogramInput,
    MelSpectrogramInput,
    MFCCInput,
    SpectralInput,
)
from model import (
    CRNN_CINC2022,
    SEQ_LAB_NET_CINC2022,
    UNET_CINC2022,
)
from trainer import (
    CINC2022Trainer,
    _MODEL_MAP,
    _set_task,
)

from helper_code import *


CinC2022Dataset.__DEBUG__ = False
CRNN_CINC2022.__DEBUG__ = False
SEQ_LAB_NET_CINC2022.__DEBUG__ = False
UNET_CINC2022.__DEBUG__ = False
CINC2022Trainer.__DEBUG__ = False


TASK = "classification"

# _TrainCfg = deepcopy(TrainCfg[TASK])
# _ModelCfg = deepcopy(ModelCfg[TASK])

_ModelFilename = "final_model.pth.tar"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if _ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print("Finding data files...")

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files == 0:
        raise Exception("No data was provided.")

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    classes = ["Present", "Unknown", "Absent"]
    num_classes = len(classes)

    # TODO: Train the model.
    # general configs and logger
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = data_folder
    train_config.model_dir = model_folder
    train_config.final_model_filename = _ModelFilename
    train_config.debug = False

    train_config.n_epochs = 100
    train_config.batch_size = 24  # 16G (Tesla T4)
    train_config.log_step = 20
    # train_config.max_lr = 1.5e-3
    train_config.early_stopping.patience = 20
    
    # train_config[TASK].cnn_name = "resnet_nature_comm_bottle_neck_se"
    # train_config[TASK].rnn_name = "none"  # "none", "lstm"
    # train_config[TASK].attn_name = "se"  # "none", "se", "gc", "nl"

    _set_task(TASK, train_config)

    model_config = deepcopy(ModelCfg[TASK])

    # adjust model choices if needed
    model_name = model_config.model_name = train_config[TASK].model_name
    model_config[model_name].cnn_name = train_config[TASK].cnn_name
    model_config[model_name].rnn_name = train_config[TASK].rnn_name
    model_config[model_name].attn_name = train_config[TASK].attn_name

    start_time = time.time()

    # ds_train = CinC2022Dataset(train_config, TASK, training=True, lazy=False)
    # ds_val = CinC2022Dataset(train_config, TASK, training=False, lazy=False)

    model_cls = _MODEL_MAP[model_config.model_name]
    model_cls.__DEBUG__ = False

    model = model_cls(config=model_config)
    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model.to(device=DEVICE)

    trainer = CINC2022Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=False,
    )

    best_state_dict = trainer.train()  # including saving model

    del trainer
    del model
    del best_state_dict

    torch.cuda.empty_cache()

    if verbose >= 1:
        print("Done.")


# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments and outputs of this function.
def load_challenge_model(model_folder, verbose):
    """ """
    model_cls = _MODEL_MAP[TrainCfg[TASK].model_name]
    model, train_cfg = model_cls.from_checkpoint(
        path=os.path.join(model_folder, _ModelFilename),
        device=DEVICE,
    )
    model.eval()
    return model


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments and outputs of this function.
def run_challenge_model(model, data, recordings, verbose):
    """ """
    raise NotImplementedError

    return classes, labels, probabilities
