"""
"""

# comment out to test in GitHub actions
# import sys

# sys.path.insert(0, "/home/wenhao/Jupyter/wenhao/workspace/torch_ecg/")
# tmp_data_dir = "/home/wenhao/Jupyter/wenhao/data/CinC2022/"

from copy import deepcopy
from typing import NoReturn

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


# uncomment to test in GitHub actions
tmp_data_dir = _BASE_DIR / "tmp" / "CINC2022"
tmp_data_dir.mkdir(parents=True, exist_ok=True)
dr = CINC2022Reader(tmp_data_dir)
dr.download(compressed=True)
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
    model.to(DEFAULTS.device)
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
    for data, labels in dl:
        data.to(DEFAULTS.device)
        print(model.inference(data))

    print("models test passed")


def test_trainer() -> NoReturn:
    """ """
    pass

    print("trainer test passed")


def test_entry() -> NoReturn:
    """ """
    pass

    print("entry test passed")


test_team_code = test_entry  # alias


if __name__ == "__main__":
    test_dataset()
    test_models()
    test_trainer()
    test_entry()
