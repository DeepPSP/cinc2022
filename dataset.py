"""
"""

import json
from random import shuffle, sample
from copy import deepcopy
from typing import Optional, List, Tuple, Sequence, NoReturn

import numpy as np

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import ReprMixin, list_sum
from torch_ecg.utils.utils_data import ensure_siglen, stratified_train_test_split
from torch_ecg._preprocessors import PreprocManager

from cfg import BaseCfg, TrainCfg, ModelCfg  # noqa: F401
from inputs import (  # noqa: F401
    InputConfig,
    WaveformInput,
    SpectrogramInput,
    MelSpectrogramInput,
    MFCCInput,
    SpectralInput,
)  # noqa: F401
from data_reader import PCGDataBase, CINC2022Reader, CINC2016Reader  # noqa: F401
from utils.springer_features import get_springer_features  # noqa: F401


__all__ = [
    "CinC2022Dataset",
]


class CinC2022Dataset(Dataset, ReprMixin):
    """ """

    __name__ = "CinC2022Dataset"

    def __init__(
        self, config: CFG, task: str, training: bool = True, lazy: bool = True
    ) -> NoReturn:
        """ """
        super().__init__()
        self.config = CFG(deepcopy(config))
        # self.task = task.lower()  # task will be set in self.__set_task
        self.training = training
        self.lazy = lazy

        self.reader = CINC2022Reader(
            self.config.db_dir,
            ignore_unannotated=self.config.get("ignore_unannotated", True),
        )

        self.subjects = self._train_test_split()
        df = self.reader.df_stats[
            self.reader.df_stats["Patient ID"].isin(self.subjects)
        ]
        self.records = list_sum(
            [self.reader.subject_records[row["Patient ID"]] for _, row in df.iterrows()]
        )
        if self.config.get("entry_test_flag", False):
            self.records = sample(self.records, int(len(self.records) * 0.2))
        if self.training:
            shuffle(self.records)

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        ppm_config = CFG(random=False)
        ppm_config.update(deepcopy(self.config.classification))
        seg_ppm_config = CFG(random=False)
        seg_ppm_config.update(deepcopy(self.config.segmentation))
        self.ppm = PreprocManager.from_config(ppm_config)
        self.seg_ppm = PreprocManager.from_config(seg_ppm_config)

        self._signals = None
        self._labels = None
        self._masks = None
        self.__set_task(task, lazy)

    def __len__(self) -> int:
        """ """
        if self.lazy:
            return len(self.fdr)
        return self._signals.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        """ """
        if self.lazy:
            return self.fdr[index]
        if self.task in ["multi_task"]:
            return self._signals[index], self._labels[index], self._masks[index]
        return self._signals[index], self._labels[index]

    def __set_task(self, task: str, lazy: bool) -> NoReturn:
        """ """
        assert task.lower() in TrainCfg.tasks, f"illegal task \042{task}\042"
        if (
            hasattr(self, "task")
            and self.task == task.lower()
            and self._signals is not None
            and len(self._signals) > 0
        ):
            return
        self.task = task.lower()
        self.siglen = int(self.config[self.task].fs * self.config[self.task].siglen)
        self.classes = self.config[task].classes
        self.n_classes = len(self.config[task].classes)
        self.lazy = lazy

        if self.task in ["classification", "multi_task"]:
            self.fdr = FastDataReader(
                self.reader, self.records, self.config, self.task, self.ppm
            )
        elif self.task in [
            "segmentation",
        ]:
            self.fdr = FastDataReader(
                self.reader, self.records, self.config, self.task, self.seg_ppm
            )
        else:
            raise ValueError("illegal task")

        if self.lazy:
            return

        self._signals, self._labels, self._masks = [], [], []
        with tqdm(range(len(self.fdr)), desc="Loading data", unit="records") as pbar:
            for idx in pbar:
                values, *labels = self.fdr[idx]
                self._signals.append(values)
                self._labels.append(labels[0])
                if len(labels) == 2:
                    self._masks.append(labels[1])
                else:
                    # raise ValueError("incorrect number of types of labels")
                    assert self.task in ["classification", "segmentation"]

        self._signals = np.concatenate(self._signals, axis=0)
        if self.config[self.task].loss != "CrossEntropyLoss":
            self._labels = np.concatenate(self._labels, axis=0)
        else:
            self._labels = np.array(sum(self._labels)).astype(int)
        if len(self._masks) > 0:
            self._masks = np.concatenate(self._masks, axis=0)
        else:
            self._masks = np.array(self._masks, dtype=self.dtype)

    def _load_all_data(self) -> NoReturn:
        """ """
        self.__set_task(self.task, lazy=False)

    def _train_test_split(
        self, train_ratio: float = 0.8, force_recompute: bool = False
    ) -> List[str]:
        """ """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        train_file = self.reader.db_dir / f"train_ratio_{_train_ratio}.json"
        test_file = self.reader.db_dir / f"test_ratio_{_test_ratio}.json"
        aux_train_file = (
            BaseCfg.project_dir / "utils" / f"train_ratio_{_train_ratio}.json"
        )
        aux_test_file = BaseCfg.project_dir / "utils" / f"test_ratio_{_test_ratio}.json"

        if not force_recompute and train_file.exists() and test_file.exists():
            if self.training:
                return json.loads(train_file.read_text())
            else:
                return json.loads(test_file.read_text())

        if not force_recompute and aux_train_file.exists() and aux_test_file.exists():
            if self.training:
                return json.loads(aux_train_file.read_text())
            else:
                return json.loads(aux_test_file.read_text())

        df_train, df_test = stratified_train_test_split(
            self.reader.df_stats,
            [
                "Murmur",
                "Age",
                "Sex",
                "Pregnancy status",
            ],
            test_ratio=1 - train_ratio,
        )

        train_set = df_train["Patient ID"].tolist()
        test_set = df_test["Patient ID"].tolist()

        train_file.write_text(json.dumps(train_set, ensure_ascii=False))
        aux_train_file.write_text(json.dumps(train_set, ensure_ascii=False))
        test_file.write_text(json.dumps(test_set, ensure_ascii=False))
        aux_test_file.write_text(json.dumps(test_set, ensure_ascii=False))

        shuffle(train_set)
        shuffle(test_set)

        if self.training:
            return train_set
        else:
            return test_set

    @property
    def signals(self) -> np.ndarray:
        return self._signals

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def masks(self) -> np.ndarray:
        return self._masks


class FastDataReader(ReprMixin, Dataset):
    """ """

    def __init__(
        self,
        reader: PCGDataBase,
        records: Sequence[str],
        config: CFG,
        task: str,
        ppm: Optional[PreprocManager] = None,
    ) -> NoReturn:
        """ """
        self.reader = reader
        self.records = records
        self.config = config
        self.task = task
        self.ppm = ppm
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

    def __len__(self) -> int:
        """ """
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        rec = self.records[index]
        values = self.reader.load_data(
            rec,
            data_format=self.config[self.task].data_format,
        )
        if self.ppm:
            values, _ = self.ppm(values, self.reader.fs)
        values = ensure_siglen(
            values,
            siglen=self.config[self.task].input_len,
            fmt=self.config[self.task].data_format,
            tolerance=self.config[self.task].sig_slice_tol,
        ).astype(self.dtype)
        if values.ndim == 2:
            values = values[np.newaxis, ...]

        if self.task in ["classification", "multi_task"]:
            labels = self.reader.load_ann(rec)
            if self.config[self.task].loss != "CrossEntropyLoss":
                labels = (
                    np.isin(self.config[self.task].classes, labels)
                    .astype(self.dtype)[np.newaxis, ...]
                    .repeat(values.shape[0], axis=0)
                )
            else:
                labels = np.array(
                    [
                        self.config[self.task].class_map[labels]
                        for _ in range(values.shape[0])
                    ],
                    dtype=int,
                )

        if self.task in ["segmentation", "multi_task"]:
            masks = self.reader.load_segmentation(rec, seg_format="binary")
            masks = ensure_siglen(
                masks,
                siglen=self.config[self.task].input_len,
                fmt="channel_last",
                tolerance=self.config[self.task].sig_slice_tol,
            ).astype(self.dtype)
            if self.task == "segmentation":
                labels = masks

        if self.task == "multi_task":
            return values, labels, masks

        return values, labels

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
            "ppm",
        ]
