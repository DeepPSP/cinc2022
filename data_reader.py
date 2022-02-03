"""
"""
import os, io, sys, pathlib
import re
import json
import time
import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Tuple, Set, Sequence, NoReturn
from numbers import Real, Number
from collections.abc import Iterable

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import wfdb
import librosa
import torchaudio
import scipy.io.wavfile as sio_wav
from easydict import EasyDict as ED
import IPython
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm

import utils


class PCGDataBase(ABC):
    """
    """
    __name__ = "PCGDataBase"

    def __init__(self,
                 db_name:str,
                 db_dir:str,
                 fs:int=1000,
                 working_dir:Optional[str]=None,
                 verbose:int=2,
                 **kwargs:Any,) -> NoReturn:
        """
        Parameters
        ----------
        db_name: str,
            name of the database
        db_dir: str, optional,
            storage path of the database
        fs: int, default 1000,
            (re-)sampling frequency of the audio
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments
        """
        self.db_name = db_name
        self.db_dir = pathlib.Path(db_dir)
        self._fs = fs
        self.working_dir = pathlib.Path(working_dir or os.getcwd())
        self.working_dir.mkdir(exist_ok=True)
        self.data_ext = None
        self.ann_ext = None
        self.header_ext = "hea"
        self.verbose = verbose
        self._all_records = None

    @abstractmethod
    def _ls_rec(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def load_data(self, rec:str, fs:Optional[int]=None, **kwargs) -> Any:
        """
        load data from the record `rec`
        """
        raise NotImplementedError

    @abstractmethod
    def load_ann(self, rec:str, **kwargs) -> Any:
        """
        load annotations of the record `rec`
        """
        raise NotImplementedError

    @abstractmethod
    def plot(self, rec:str, **kwargs) -> NoReturn:
        """
        """
        raise NotImplementedError

    def play(self, rec:str, **kwargs) -> IPython.display.Audio:
        """
        """
        audio_file = self.db_dir / f"{rec}.{self.data_ext}"
        return IPython.display.Audio(filename=str(audio_file))

    @property
    def all_records(self):
        """
        """
        if self._all_records is None:
            self._ls_rec()
        return self._all_records
    
    @property
    def fs(self):
        """
        """
        return self._fs

    def _reset_fs(fs:int) -> NoReturn:
        """
        """
        self._fs = fs

    @property
    def database_info(self) -> NoReturn:
        """
        """
        info = "\n".join(self.__doc__.split("\n")[1:])
        print(info)


class CINC2022Reader(PCGDataBase):
    """
    """
    __name__ = "CINC2022Reader"

    def __init__(self,
                 db_dir:str,
                 fs:int=1000,
                 working_dir:Optional[str]=None,
                 verbose:int=2,
                 **kwargs:Any) -> NoReturn:
        """
        """
        super().__init__(
            db_name="circor-heart-sound",
            db_dir=db_dir, fs=fs, working_dir=working_dir, verbose=verbose, **kwargs
        )
        self.data_ext = "wav"
        self.ann_ext = "hea"
        self.segmentation_ext = "tsv"
        self.segmentation_map = {
            0: "unannotated",
            1: "S1",
            2: "systolic",
            3: "S2",
            4: "diastolic",
        }
        self.auscultation_locations = {
            "PV", "AV", "MV", "TV", "Phc",
        }

        self._rec_pattern = f"(?P<pid>[\d]+)\_(?P<loc>{'|'.join(self.auscultation_locations)})((?:\_)(?P<num>\d))?"

        self._all_records = None
        self._all_subjects = None
        self._ls_rec()
        
        self._df_stats = None
        self._stats_cols = [
            "Patient ID", "Locations", "Age", "Sex", "Height", "Weight",
            "Pregnancy status", "Murmur", "Murmur locations",
            "Most audible location", "Systolic murmur timing",
            "Systolic murmur shape", "Systolic murmur grading",
            "Systolic murmur pitch", "Systolic murmur quality",
            "Diastolic murmur timing", "Diastolic murmur shape",
            "Diastolic murmur grading", "Diastolic murmur pitch",
            "Diastolic murmur quality", "Campaign", "Additional ID",
        ]
        self._load_stats()

    def _ls_rec(self) -> NoReturn:
        """
        """
        try:
            self._all_records = wfdb.get_record_list(self.db_name)
        except:
            records_file = self.db_dir / "RECORDS"
            if records_file.exists():
                with open(records_file) as f:
                    self._all_records = f.read().splitlines()
            else:
                self._all_records = sorted([
                    f"training_data/{item.name}".replace(".hea", "") \
                        for item in (self.db_dir / "training_data").glob("*.hea")
                ])
                with open(records_file, "w") as f:
                    f.write("\n".join(self._all_records))
        self._all_subjects = sorted(set([
            item.replace("training_data/", "").split("_")[0] for item in self._all_records
        ]))

    def _load_stats(self) -> NoReturn:
        """
        """
        stats_file = self.db_dir / "training_data.csv"
        if stats_file.exists():
            self._df_stats = pd.read_csv(stats_file)
        else:
            self._df_stats = pd.DataFrame()
            with tqdm(self.all_subjects, total=len(self.all_subjects), desc="loading stats") as pbar:
                for s in pbar:
                    f = self.db_dir / "training_data" / f"{s}.txt"
                    with open(f) as f:
                        content = f.read().splitlines()
                    new_row = {"Patient ID": s}
                    localtions = set()
                    for l in content:
                        if not l.startswith("#"):
                            if re. l.split()[0] in self.auscultation_locations:
                                localtions.add(l.split()[0])
                            continue
                        k, v = l.replace("#", "").split(":")
                        k, v = k.strip(), v.strip()
                        if v == "nan":
                            v = ""
                        new_row[k] = v
                    new_row["Locations"] = "+".join(localtions)
                    self._df_stats = self._df_stats.append(
                        new_row, ignore_index=True,
                    )
        self._df_stats = self._df_stats.fillna("")
        self._df_stats.Locations = self._df_stats.Locations.apply(lambda s:s.split("+"))
        self._df_stats["Murmur locations"] = self._df_stats["Murmur locations"].apply(lambda s:s.split("+"))
        self._df_stats["Patient ID"] = self._df_stats["Patient ID"].astype(str)
        self._df_stats = self._df_stats[self._stats_cols]
        for idx, row in self._df_stats.iterrows():
            for c in ["Height", "Weight"]:
                if row[c] == "":
                    self._df_stats.at[idx, c] = np.nan

    def _decompose_rec(self, rec:str) -> Dict[str, str]:
        """
        """
        return list(re.finditer(self._rec_pattern, rec))[0].groupdict()

    def load_data(self, rec:str, fs:Optional[int]=None, fmt:str="channel_first") -> Any:
        """
        load data from the record `rec`
        """
        fs = fs or self.fs
        if fs == -1:
            fs = None
        data_file = self.db_dir / f"{rec}.{self.data_ext}"
        data, _ = librosa.load(data_file, sr=fs, mono=False)
        data = np.atleast_2d(data)
        if fmt.lower() == "channel_last":
            data = data.T
        return data

    def load_ann(self, rec_or_pid:str) -> Any:
        """
        load annotations of the record `rec`
        """
        if rec_or_pid in self.all_subjects:
            return self.df_stats[self.df_stats["Patient ID"] == rec_or_pid].iloc[0]["Murmur"]
        elif rec_or_pid in self.all_records:
            decom = self._decompose_rec(rec_or_pid)
            pid, loc = decom["pid"], decom["loc"]
            row = self.df_stats[self.df_stats["Patient ID"] == pid].iloc[0]
            if row["Murmur"] == "Unknown":
                return "Unknown"
            if loc in row["Murmur locations"]:
                return "Present"
            else:
                return "Absent"
        else:
            raise ValueError(f"{rec_or_pid} is not a valid record or patient ID")

    def load_segmentation(self, rec:str, fmt:str="df", fs:Optional[int]=None) -> Any:
        """
        """
        fs = fs or self.fs
        if fs == -1:
            fs = self.get_fs(rec)
        segmentation_file = self.db_dir / f"{rec}.{self.segmentation_ext}"
        df_seg = pd.read_csv(segmentation_file, sep="\t", header=None)
        df_seg.columns = ["start_t", "end_t", "label"]
        df_seg["wave"] = df_seg["label"].apply(lambda s:self.segmentation_map[s])
        df_seg["start"] = (fs * df_seg["start_t"]).apply(round)
        df_seg["end"] = (fs * df_seg["end_t"]).apply(round)
        if fmt.lower() in ["dataframe", "df",]:
            return df_seg
        elif fmt.lower() in ["dict", "dicts",]:
            # dict of intervals
            raise NotImplementedError
        elif fmt.lower() in ["mask",]:
            raise NotImplementedError

    def get_fs(self, rec:str) -> int:
        """
        """
        return wfdb.rdheader(self.db_dir / rec).fs
    
    @property
    def all_subjects(self) -> List[str]:
        """
        """
        return self._all_subjects

    @property
    def df_stats(self) -> pd.DataFrame:
        """
        """
        return self._df_stats

    def plot(self, rec:str, **kwargs) -> NoReturn:
        """
        """
        raise NotImplementedError


class CINC2016Reader(PCGDataBase):
    """
    """
    __name__ = "CINC2016Reader"



class EPHNOGRAMReader(PCGDataBase):
    """
    """
    __name__ = "EPHNOGRAMReader"
