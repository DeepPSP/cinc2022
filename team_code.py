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
from itertools import repeat
from typing import NoReturn, List, Tuple, Dict, Union

import numpy as np
import torch
from torch.nn.parallel import (
    DistributedDataParallel as DDP,
    DataParallel as DP,
)
from torch.utils.data import DataLoader
from torch_ecg.cfg import CFG
from torch_ecg._preprocessors import PreprocManager

from cfg import BaseCfg, TrainCfg, ModelCfg
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
FS = 4000

_ModelFilename = "final_model.pth.tar"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
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
def train_challenge_model(
    data_folder: str, model_folder: str, verbose: int
) -> NoReturn:
    """

    Parameters
    ----------
    data_folder: str,
        path to the folder containing the training data
    model_folder: str,
        path to the folder to save the trained model
    verbose: int,
        verbosity level

    """

    print("\n" + "*" * 100)
    msg = "   CinC2022 challenge training entry starts   ".center(100, "#")
    print(msg)
    print("*" * 100 + "\n")

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
    train_config.debug = False

    if train_config.get("entry_test_flag", False):
        # to test in the file docker_test.py
        train_config.n_epochs = 1
        train_config.batch_size = 32  # 16G (Tesla T4)
        train_config.log_step = 4
        # train_config.max_lr = 1.5e-3
        train_config.early_stopping.patience = 20
    else:
        train_config.n_epochs = 100
        train_config.batch_size = 24  # 16G (Tesla T4)
        train_config.log_step = 20
        # train_config.max_lr = 1.5e-3
        train_config.early_stopping.patience = train_config.n_epochs // 2

    # train_config[TASK].cnn_name = "resnet_nature_comm_bottle_neck_se"
    # train_config[TASK].rnn_name = "none"  # "none", "lstm"
    # train_config[TASK].attn_name = "se"  # "none", "se", "gc", "nl"

    train_config.final_model_name = _ModelFilename
    train_config[TASK].final_model_name = _ModelFilename
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

    print("\n" + "*" * 100)
    msg = "   CinC2022 challenge training entry ends   ".center(100, "#")
    print(msg)
    print("*" * 100 + "\n\n")


# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments and outputs of this function.
def load_challenge_model(
    model_folder: str, verbose: int
) -> Dict[str, Union[CFG, torch.nn.Module]]:
    """

    Parameters
    ----------
    model_folder: str,
        path to the folder containing the trained model
    verbose: int,
        verbosity level

    Returns
    -------
    dict, with items:
        - model: torch.nn.Module,
            the loaded model
        - train_cfg: CFG,
            the training configuration,
            including the list of classes (the ordering is important),
            and the preprocessing configurations

    """
    print("\n" + "*" * 100)
    msg = "   loading CinC2022 challenge model   ".center(100, "#")
    print(msg)
    model_cls = _MODEL_MAP[TrainCfg[TASK].model_name]
    model, train_cfg = model_cls.from_checkpoint(
        path=os.path.join(model_folder, _ModelFilename),
        device=DEVICE,
    )
    model.eval()
    msg = "   CinC2022 challenge model loaded   ".center(100, "#")
    print(msg)
    print("*" * 100 + "\n")
    return dict(model=model, train_cfg=train_cfg)


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments and outputs of this function.
def run_challenge_model(
    model: Dict[str, Union[CFG, torch.nn.Module]],
    data: str,
    recordings: Union[List[np.ndarray], Tuple[List[np.ndarray], List[int]]],
    verbose: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    model: Dict[str, Union[CFG, torch.nn.Module]],
        the trained model (key "model"),
        along with the training configuration (key "train_cfg")
    data: str,
        patient metadata file data, read from a text file
    recordings: List[np.ndarray],
        list of recordings, each recording is a 1D numpy array
    verbose: int,
        verbosity level

    Returns
    -------
    classes: list of str,
        list of class names
    labels: np.ndarray,
        binary prediction
    probabilities: np.ndarray,
        probability prediction

    NOTE
    ----
    the `recordings` are read by `scipy.io.wavfile.read`, with the following possible data types:
    =====================  ===========  ===========  =============
        WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============
    Note that 8-bit PCM is unsigned.

    """

    _model = model["model"]
    _model.to(device=DEVICE)
    train_cfg = model["train_cfg"]
    ppm_config = CFG(random=False)
    ppm_config.update(deepcopy(train_cfg[TASK]))
    ppm = PreprocManager.from_config(ppm_config)

    if not isinstance(recordings[0], np.ndarray):
        recordings, frequencies = recordings
        num_recordings = len(recordings)
    else:
        num_recordings = len(recordings)
        frequencies = list(repeat(FS, num_recordings))

    classes = train_cfg[TASK].classes

    # probabilities, labels, forward_outputs = [], [], []
    features = []
    if BaseCfg.merge_rule.lower() == "avg":
        pooler = torch.nn.AdaptiveAvgPool1d((1,))
    elif BaseCfg.merge_rule.lower() == "max":
        pooler = torch.nn.AdaptiveMaxPool1d((1,))

    for rec, fs in zip(recordings, frequencies):
        rec = _to_dtype(rec, DTYPE)
        rec, _ = ppm(rec, fs)
        # model_output = _model.inference(np.atleast_2d(rec))
        # probabilities.append(model_output.prob)
        # labels.append(model_output.bin_pred)
        # forward_outputs.append(model_output.forward_output)
        for _ in range(3-rec.ndim):
            rec = rec[np.newaxis, :]
        rec = torch.from_numpy(rec.copy().astype(DTYPE)).to(device=DEVICE)
        # rec of shape (1, 1, n_samples)
        features.append(_model.extract_features(rec))  # shape (1, n_features, n_samples)

    features = torch.cat(features, dim=-1)  # shape (1, n_features, n_samples)
    features = pooler(features).squeeze(dim=-1)  # shape (1, n_features)
    forward_output = _model.clf(features)  # shape (1, n_classes)
    probabilities = _model.softmax(forward_output)
    labels = (probabilities == probabilities.max(dim=-1, keepdim=True).values).to(int)
    probabilities = probabilities.squeeze(dim=0).cpu().detach().numpy()
    labels = labels.squeeze(dim=0).cpu().detach().numpy()

    # probabilities = np.concatenate(probabilities, axis=0)
    # labels = np.concatenate(labels, axis=0)
    # forward_outputs = np.concatenate(forward_outputs, axis=0)

    return classes, labels, probabilities


def _to_dtype(data: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """ """
    if data.dtype == dtype:
        return data
    if data.dtype in (np.int8, np.uint8, np.int16, np.int32, np.int64):
        data = data.astype(dtype) / (np.iinfo(data.dtype).max + 1)
    return data
