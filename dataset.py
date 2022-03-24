"""
"""

import os, sys, json, time, textwrap
import multiprocessing as mp
from random import shuffle, randint, sample
from copy import deepcopy
from typing import Union, Optional, List, Tuple, Dict, Sequence, Set, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
from easydict import EasyDict as ED
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset

from cfg import BaseCfg, TrainCfg, ModelCfg
from data_reader import CINC2022Reader, CINC2016Reader


__all__ = ["CinC2022Dataset",]


class CinC2022Dataset(Dataset):
    """
    """
    __name__ = "CinC2022Dataset"

    def __init__(self, config:ED, task:str, training:bool=True, lazy:bool=True) -> NoReturn:
        """
        """
        super().__init__()
        self.config = ED(deepcopy(config))
        self.task = task.lower()
        self.training = training
        self.lazy = lazy

        self.reader = CINC2022Reader(self.config.db_dir)

        self.subjects = self._train_test_split()
        df = self.reader.df_stats[self.reader.df_stats["Patient ID"].isin(self.subjects)]
        self.records = [
            f"{row['Patient ID']}_{pos}" \
                for _, row in df.iterrows() for pos in row["Locations"]
        ]
        shuffle(self.records)
        self.siglen = int(self.config[self.task].fs * self.config[self.task].siglen)

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        self._signals = np.array([], dtype=self.dtype)
        self._labels = np.array([], dtype=self.dtype)
        self._masks = np.array([], dtype=self.dtype)
        self.__set_task(task, lazy)

    def __len__(self) -> int:
        """
        """
        return self._signals.shape[0]

    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        return self._signals[index], self._labels[index]

    def __set_task(self, task:str, lazy:bool) -> NoReturn:
        """
        """
        assert task.lower() in self.config.tasks
        raise NotImplementedError

    def _load_all_data(self) -> NoReturn:
        """
        """
        with tqdm(self.records, total=len(self.records)) as pbar:
            for record in pbar:
                raise NotImplementedError

    def _train_test_split(self,
                          train_ratio:float=0.8,
                          force_recompute:bool=False) -> List[str]:
        """
        """
        _train_ratio = int(train_ratio*100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        train_file = self.reader.db_dir / f"train_ratio_{_train_ratio}.json"
        test_file = self.reader.db_dir / f"test_ratio_{_test_ratio}.json"
        aux_train_file = BaseCfg.project_dir / "utils" / f"train_ratio_{_train_ratio}.json"
        aux_test_file = BaseCfg.project_dir / "utils" / f"test_ratio_{_test_ratio}.json"

        if not force_recompute and train_file.exists() and test_file.exists():
            if self.training:
                return json.load(open(train_file, "r"))
            else:
                return json.load(open(test_file, "r"))

        if not force_recompute and aux_train_file.exists() and aux_test_file.exists():
            if self.training:
                return json.load(open(train_file, "r"))
            else:
                return json.load(open(test_file, "r"))

        df_train, df_test = _strafified_train_test_split(
            self.reader.df_stats,
            ["Murmur", "Age", "Sex", "Pregnancy status",],
            test_ratio=1-train_ratio,
        )

        train_set = df_train["Patient ID"].tolist()
        test_set = df_test["Patient ID"].tolist()

        json.dump(train_set, open(train_file, "w"))
        json.dump(test_set, open(test_file, "w"))
        json.dump(train_set, open(aux_train_file, "w"))
        json.dump(test_set, open(aux_test_file, "w"))

        shuffle(train_set)
        shuffle(test_set)

        if self.training:
            return train_set
        else:
            return test_set


def _strafified_train_test_split(df:pd.DataFrame,
                                 strafified_cols:Sequence[str],
                                 test_ratio:float=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    """
    df_inspection = df[strafified_cols].copy()
    for item in strafified_cols:
        all_entities = df_inspection[item].unique().tolist()
        entities_dict = {e: str(i) for i, e in enumerate(all_entities)}
        df_inspection[item] = df_inspection[item].apply(lambda e:entities_dict[e])

    inspection_col_name = "Inspection" * (max([len(c) for c in strafified_cols])//10+1)
    df_inspection[inspection_col_name] = ''
    for idx, row in df_inspection.iterrows():
        cn = "-".join([row[sc] for sc in strafified_cols])
        df_inspection.loc[idx, inspection_col_name] = cn
    item_names = df_inspection[inspection_col_name].unique().tolist()
    item_indices = {
        n: df_inspection.index[df_inspection[inspection_col_name]==n].tolist() for n in item_names
    }
    for n in item_names:
        shuffle(item_indices[n])

    test_indices = []
    for n in item_names:
        item_test_indices = item_indices[n][:round(test_ratio*len(item_indices[n]))]
        test_indices += item_test_indices
    df_test = df.loc[df.index.isin(test_indices)].reset_index(drop=True)
    df_train = df.loc[~df.index.isin(test_indices)].reset_index(drop=True)
    return df_train, df_test
