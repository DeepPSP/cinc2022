"""
"""

# comment out to test in GitHub actions
# import sys

# sys.path.insert(0, "/home/wenhao/Jupyter/wenhao/workspace/torch_ecg/")
# tmp_data_dir = "/home/wenhao/Jupyter/wenhao/data/CinC2022/"

# set test flag
from cfg import set_entry_test_flag

set_entry_test_flag(True)


from copy import deepcopy
from typing import NoReturn

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_ecg.cfg import DEFAULTS
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

from data_reader import CINC2022Reader, CINC2016Reader, EPHNOGRAMReader
from dataset import CinC2022Dataset
from model import CRNN_CINC2022, SEQ_LAB_NET_CINC2022, UNET_CINC2022
from cfg import TrainCfg, ModelCfg, _BASE_DIR
from trainer import CINC2022Trainer


CRNN_CINC2022.__DEBUG__ = False
SEQ_LAB_NET_CINC2022.__DEBUG__ = False
UNET_CINC2022.__DEBUG__ = False
CinC2022Dataset.__DEBUG__ = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


# uncomment to test in GitHub actions
tmp_data_dir = _BASE_DIR / "tmp" / "CINC2022"
tmp_data_dir.mkdir(parents=True, exist_ok=True)
dr = CINC2022Reader(tmp_data_dir)
dr.download(compressed=True)
dr._ls_rec()
del dr

TASK = "classification"


def test_dataset() -> NoReturn:
    """ """
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir

    ds_train = CinC2022Dataset(ds_config, TASK, training=True, lazy=True)
    ds_val = CinC2022Dataset(ds_config, TASK, training=False, lazy=True)

    ds_train._load_all_data()
    ds_val._load_all_data()

    print("dataset test passed")


def test_models() -> NoReturn:
    """ """
    model = CRNN_CINC2022(ModelCfg[TASK])
    model.to(DEVICE)
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir
    ds_val = CinC2022Dataset(ds_config, TASK, training=False, lazy=True)
    ds_val._load_all_data()
    dl = DataLoader(
        dataset=ds_val,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    for idx, (data, labels) in enumerate(dl):
        data.to(DEVICE)
        print(model.inference(data))
        if idx > 10:
            break

    print("models test passed")


def test_trainer() -> NoReturn:
    """ """
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = tmp_data_dir
    # train_config.model_dir = model_folder
    # train_config.final_model_filename = "final_model.pth.tar"
    train_config.debug = True

    train_config.n_epochs = 20
    train_config.batch_size = 24  # 16G (Tesla T4)
    # train_config.log_step = 20
    # # train_config.max_lr = 1.5e-3
    # train_config.early_stopping.patience = 20

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

    best_state_dict = trainer.train()

    print("trainer test passed")


from train_model import train_challenge_model
from run_model import run_model


def test_entry() -> NoReturn:
    """ """

    data_folder = str(tmp_data_dir / "training_data")
    train_challenge_model(data_folder, str(TrainCfg.model_dir), verbose=2)

    output_dir = _BASE_DIR / "tmp" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_model(
        TrainCfg.model_dir,
        data_folder,
        str(output_dir),
        allow_failures=False,
        verbose=2,
    )

    print("entry test passed")


test_team_code = test_entry  # alias


if __name__ == "__main__":
    # test_dataset()  # passed
    # test_models()  # passed
    # test_trainer()  # directly run test_entry
    test_entry()
    set_entry_test_flag(False)
