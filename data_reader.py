"""
"""

import re
from pathlib import Path
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Union, Optional, Any, List, Dict, Tuple, Set, Sequence, NoReturn
from numbers import Real, Number

import numpy as np

np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import wfdb
import librosa
import torchaudio
import scipy.signal as ss
import scipy.io as sio
import scipy.io.wavfile as sio_wav
from easydict import EasyDict as ED
import IPython

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm
from torch_ecg.databases.base import PhysioNetDataBase
from torch_ecg.utils.utils_signal import butter_bandpass_filter

from cfg import BaseCfg
from utils.schmidt_spike_removal import schmidt_spike_removal


__all__ = [
    "PCGDataBase",
    "CINC2022Reader",
    "CINC2016Reader",
    "EPHNOGRAMReader",
]


class PCGDataBase(PhysioNetDataBase):
    """ """

    __name__ = "PCGDataBase"

    def __init__(
        self,
        db_name: str,
        db_dir: str,
        fs: int = 1000,
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
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
        super().__init__(db_name, db_dir, working_dir, verbose, **kwargs)
        self.fs = fs
        self.data_ext = None
        self.ann_ext = None
        self.header_ext = "hea"
        self._all_records = None
        self.dtype = kwargs.get("dtype", np.float32)

    def _auto_infer_units(
        self,
    ) -> NoReturn:
        """
        disable this function implemented in the base class
        """
        print("DO NOT USE THIS FUNCTION for a PCG database!")

    @abstractmethod
    def play(self, rec: str, **kwargs) -> IPython.display.Audio:
        """ """
        raise NotImplementedError

    def _reset_fs(new_fs: int) -> NoReturn:
        """ """
        self.fs = new_fs


class CINC2022Reader(PCGDataBase):
    """ """

    __name__ = "CINC2022Reader"

    def __init__(
        self,
        db_dir: str,
        fs: int = 4000,
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        super().__init__(
            db_name="circor-heart-sound",
            db_dir=db_dir,
            fs=fs,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.data_dir = self.db_dir / "training_data"
        self.data_ext = "wav"
        self.ann_ext = "hea"
        self.segmentation_ext = "tsv"
        self.segmentation_states = [s for s in BaseCfg.states if s != "unannotated"]
        self.segmentation_map = {n: s for n, s in enumerate(BaseCfg.states)}
        self.auscultation_locations = {
            "PV",
            "AV",
            "MV",
            "TV",
            "Phc",
        }

        self._rec_pattern = f"(?P<pid>[\d]+)\_(?P<loc>{'|'.join(self.auscultation_locations)})((?:\_)(?P<num>\d))?"

        self._all_records = None
        self._all_subjects = None
        self._subject_records = None
        self._ls_rec()

        self._df_stats = None
        self._stats_cols = [
            "Patient ID",
            "Locations",
            "Age",
            "Sex",
            "Height",
            "Weight",
            "Pregnancy status",
            "Murmur",
            "Murmur locations",
            "Most audible location",
            "Systolic murmur timing",
            "Systolic murmur shape",
            "Systolic murmur grading",
            "Systolic murmur pitch",
            "Systolic murmur quality",
            "Diastolic murmur timing",
            "Diastolic murmur shape",
            "Diastolic murmur grading",
            "Diastolic murmur pitch",
            "Diastolic murmur quality",
            "Campaign",
            "Additional ID",
        ]
        self._df_stats_records = None
        self._stats_records_cols = [
            "Patient ID",
            "Location",
            "rec",
            "siglen",
            "siglen_sec",
            "Murmur",
        ]
        self._load_stats()

    def _ls_rec(self) -> NoReturn:
        """ """
        try:
            print("Reading the list of records from local file...")
            records_file = self.db_dir / "RECORDS"
            if records_file.exists():
                self._all_records = records_file.read_text().splitlines()
            else:
                self._all_records = sorted(
                    [
                        f"training_data/{item.name}".replace(".hea", "")
                        for item in (self.data_dir).glob("*.hea")
                    ]
                )
                records_file.write_text("\n".join(self._all_records))
        except:
            print("Reading the list of records from PhysioNet...")
            self._all_records = wfdb.get_record_list(self.db_name)
        self._all_records = [
            item.replace("training_data/", "")
            for item in self._all_records
            if (self.db_dir / item).with_suffix(".hea").exists()
        ]
        self._all_subjects = sorted(
            set([item.split("_")[0] for item in self._all_records])
        )
        self._subject_records = defaultdict(list)
        for rec in self._all_records:
            self._subject_records[self.get_subject(rec)].append(rec)
        self._subject_records = dict(self._subject_records)

    def _load_stats(self) -> NoReturn:
        """ """
        print("Reading the statistics from local file...")
        stats_file = self.db_dir / "training_data.csv"
        if stats_file.exists():
            self._df_stats = pd.read_csv(stats_file)
        elif self._all_records is not None and len(self._all_records) > 0:
            print("No cached statistics found, gathering from scratch...")
            self._df_stats = pd.DataFrame()
            with tqdm(
                self.all_subjects, total=len(self.all_subjects), desc="loading stats"
            ) as pbar:
                for s in pbar:
                    f = self.data_dir / f"{s}.txt"
                    content = f.read_text().splitlines()
                    new_row = {"Patient ID": s}
                    localtions = set()
                    for l in content:
                        if not l.startswith("#"):
                            if l.split()[0] in self.auscultation_locations:
                                localtions.add(l.split()[0])
                            continue
                        k, v = l.replace("#", "").split(":")
                        k, v = k.strip(), v.strip()
                        if v == "nan":
                            v = ""
                        new_row[k] = v
                    new_row["Locations"] = "+".join(localtions)
                    self._df_stats = self._df_stats.append(
                        new_row,
                        ignore_index=True,
                    )
            self._df_stats.to_csv(stats_file, index=False)
        else:
            print("No data found locally!")
            return
        self._df_stats = self._df_stats.fillna("")
        self._df_stats.Locations = self._df_stats.Locations.apply(
            lambda s: s.split("+")
        )
        self._df_stats["Murmur locations"] = self._df_stats["Murmur locations"].apply(
            lambda s: s.split("+")
        )
        self._df_stats["Patient ID"] = self._df_stats["Patient ID"].astype(str)
        self._df_stats = self._df_stats[self._stats_cols]
        for idx, row in self._df_stats.iterrows():
            for c in ["Height", "Weight"]:
                if row[c] == "":
                    self._df_stats.at[idx, c] = np.nan

        # load stats of the records
        print("Reading the statistics of the records from local file...")
        stats_file = self.db_dir / "stats_records.csv"
        if stats_file.exists():
            self._df_stats_records = pd.read_csv(stats_file)
        else:
            self._df_stats_records = pd.DataFrame(columns=self._stats_records_cols)
            with tqdm(
                self._df_stats.iterrows(),
                total=len(self._df_stats),
                desc="loading record stats",
            ) as pbar:
                for _, row in pbar:
                    pid = row["Patient ID"]
                    for loc in row["Locations"]:
                        rec = f"{pid}_{loc}"
                        if rec not in self._all_records:
                            continue
                        header = wfdb.rdheader(str(self.data_dir / f"{rec}"))
                        if row["Murmur"] == "Unknown":
                            murmur = "Unknown"
                        if loc in row["Murmur locations"]:
                            murmur = "Present"
                        else:
                            murmur = "Absent"
                        new_row = {
                            "Patient ID": pid,
                            "Location": loc,
                            "rec": rec,
                            "siglen": header.sig_len,
                            "siglen_sec": header.sig_len / header.fs,
                            "Murmur": murmur,
                        }
                        self._df_stats_records = self._df_stats_records.append(
                            new_row,
                            ignore_index=True,
                        )
            self._df_stats_records.to_csv(stats_file, index=False)
        self._df_stats_records = self._df_stats_records.fillna("")

    def _decompose_rec(self, rec: str) -> Dict[str, str]:
        """ """
        return list(re.finditer(self._rec_pattern, rec))[0].groupdict()

    def load_data(
        self, rec: str, fs: Optional[int] = None, data_format: str = "channel_first"
    ) -> np.ndarray:
        """
        load data from the record `rec`
        """
        fs = fs or self.fs
        if fs == -1:
            fs = None
        data_file = self.data_dir / f"{rec}.{self.data_ext}"
        data, _ = librosa.load(data_file, sr=fs, mono=False)
        if data_format.lower() == "flat":
            return data
        data = np.atleast_2d(data)
        if data_format.lower() == "channel_last":
            data = data.T
        return data

    def load_ann(
        self, rec_or_pid: str, class_map: Optional[Dict[str, int]] = None
    ) -> Union[str, int]:
        """
        load annotations of the record `rec`
        """
        _class_map = class_map or {}
        if rec_or_pid in self.all_subjects:
            ann = self.df_stats[self.df_stats["Patient ID"] == rec_or_pid].iloc[0][
                "Murmur"
            ]
        elif rec_or_pid in self.all_records:
            decom = self._decompose_rec(rec_or_pid)
            pid, loc = decom["pid"], decom["loc"]
            row = self.df_stats[self.df_stats["Patient ID"] == pid].iloc[0]
            if row["Murmur"] == "Unknown":
                ann = "Unknown"
            if loc in row["Murmur locations"]:
                ann = "Present"
            else:
                ann = "Absent"
        else:
            raise ValueError(f"{rec_or_pid} is not a valid record or patient ID")
        ann = _class_map.get(ann, ann)
        return ann

    def load_segmentation(
        self, rec: str, seg_format: str = "df", fs: Optional[int] = None
    ) -> Union[pd.DataFrame, np.ndarray, dict]:
        """ """
        fs = fs or self.fs
        if fs == -1:
            fs = self.get_fs(rec)
        segmentation_file = self.data_dir / f"{rec}.{self.segmentation_ext}"
        df_seg = pd.read_csv(segmentation_file, sep="\t", header=None)
        df_seg.columns = ["start_t", "end_t", "label"]
        df_seg["wave"] = df_seg["label"].apply(lambda s: self.segmentation_map[s])
        df_seg["start"] = (fs * df_seg["start_t"]).apply(round)
        df_seg["end"] = (fs * df_seg["end_t"]).apply(round)
        if seg_format.lower() in [
            "dataframe",
            "df",
        ]:
            return df_seg
        elif seg_format.lower() in [
            "dict",
            "dicts",
        ]:
            # dict of intervals
            return {
                k: [
                    [row["start"], row["end"]]
                    for _, row in df_seg[df_seg["wave"] == k].iterrows()
                ]
                for _, k in self.segmentation_map.items()
            }
        elif seg_format.lower() in [
            "mask",
        ]:
            mask = np.zeros(df_seg.end.values[-1], dtype=int)
            for _, row in df_seg.iterrows():
                mask[row["start"] : row["end"]] = int(row["label"])
            return mask
        elif seg_format.lower() in [
            "binary",
        ]:
            bin_mask = np.zeros(
                (len(self.segmentation_states), df_seg.end.values[-1]), dtype=self.dtype
            )
            for _, row in df_seg.iterrows():
                if row["wave"] in self.segmentation_states:
                    bin_mask[
                        self.segmentation_states.index(row["wave"]),
                        row["start"] : row["end"],
                    ] = 1
            return bin_mask
        else:
            raise ValueError(f"{seg_format} is not a valid format")

    def load_meta_data(
        self,
        subject: str,
        keys: Optional[Union[Sequence[str], str]] = None,
    ) -> Union[dict, str, float, int]:
        """ """
        row = self._df_stats[self._df_stats["Patient ID"] == subject].iloc[0]
        meta_data = row.to_dict()
        if keys:
            if isinstance(keys, str):
                for k, v in meta_data.items():
                    if k.lower() == keys.lower():
                        return v
            else:
                _keys = [k.lower() for k in keys]
                return {k: v for k, v in meta_data.items() if k.lower() in _keys}
        return meta_data

    def _load_preprocessed_data(
        self,
        rec: str,
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        passband: Sequence[int] = BaseCfg.passband,
        order: int = BaseCfg.order,
        spike_removal: bool = True,
    ) -> np.ndarray:
        """ """
        fs = fs or self.fs
        data = butter_bandpass_filter(
            self.load_data(rec, fs=fs, data_format="flat"),
            lowcut=passband[0],
            highcut=passband[1],
            fs=fs,
            order=order,
        ).astype(self.dtype)
        if spike_removal:
            data = schmidt_spike_removal(data, fs=fs)
        if data_format.lower() == "flat":
            return data
        data = np.atleast_2d(data)
        if data_format.lower() == "channel_last":
            data = data.T
        return data

    def get_fs(self, rec: str) -> int:
        """ """
        return wfdb.rdheader(self.data_dir / rec).fs

    def get_subject(self, rec: str) -> int:
        """ """
        return self._decompose_rec(rec)["pid"]

    @property
    def all_subjects(self) -> List[str]:
        return self._all_subjects

    @property
    def subject_records(self) -> Dict[str, List[str]]:
        return self._subject_records

    @property
    def df_stats(self) -> pd.DataFrame:
        """ """
        if self._df_stats is None or self._df_stats.empty:
            self._load_stats()
        return self._df_stats

    @property
    def df_stats_records(self) -> pd.DataFrame:
        """ """
        if self._df_stats_records is None or self._df_stats_records.empty:
            self._load_stats()
        return self._df_stats_records

    def play(self, rec: str, **kwargs) -> IPython.display.Audio:
        """ """
        if "data" in kwargs:
            return IPython.display.Audio(
                kwargs["data"], rate=kwargs.get("fs", self.get_fs(rec))
            )
        audio_file = self.data_dir / f"{rec}.{self.data_ext}"
        return IPython.display.Audio(filename=str(audio_file))

    def plot(self, rec: str, **kwargs) -> NoReturn:
        """ """
        raise NotImplementedError


class CINC2016Reader(PCGDataBase):
    """ """

    __name__ = "CINC2016Reader"

    def __init__(
        self,
        db_dir: str,
        fs: int = 2000,
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        super().__init__(
            db_name="challenge-2016",
            db_dir=db_dir,
            fs=fs,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.data_ext = "wav"
        self.ecg_ext = "dat"
        self.ann_ext = "hea"

        self._subsets = [f"training-{s}" for s in "abcde"]

        self._all_records = None
        self._ls_rec()

    def _ls_rec(self) -> NoReturn:
        """ """
        records_file = self.db_dir / "RECORDS"
        if records_file.exists():
            self._all_records = records_file.read_text().splitlines()
        else:
            self._all_records = sorted(self.db_dir.rglob(f"*.{self.header_ext}"))
            self._all_records = [
                str(item)
                .replace(str(self.db_dir), "")
                .replace(f".{self.header_ext}", "")
                .strip(self.db_dir.anchor)
                for item in self._all_records
            ]
            records_file.write_text("\n".join(self._all_records))
        self._all_records = [Path(item).name for item in self._all_records]

    def get_path(self, rec: str, extension: Optional[str] = None) -> Path:
        """ """
        filename = f"{rec}.{extension}" if extension else rec
        return self.db_dir / f"training-{rec[0]}" / filename

    def load_data(
        self, rec: str, fs: Optional[int] = None, data_format: str = "channel_first"
    ) -> Dict[str, np.ndarray]:
        """
        load data from the record `rec`
        """
        data = {
            "PCG": self.load_pcg(rec, fs, data_format),
            "ECG": self.load_ecg(rec, fs, data_format),
        }
        return data

    def load_pcg(
        self, rec: str, fs: Optional[int] = None, data_format: str = "channel_first"
    ) -> np.ndarray:
        """ """
        fs = fs or self.fs
        if fs == -1:
            fs = None
        pcg, _ = librosa.load(self.get_path(rec, self.data_ext), sr=fs, mono=False)
        pcg = np.atleast_2d(pcg)
        if data_format.lower() == "channel_last":
            pcg = pcg.T
        return pcg

    def load_ecg(
        self, rec: str, fs: Optional[int] = None, data_format: str = "channel_first"
    ) -> np.ndarray:
        """ """
        fs = fs or self.fs
        if fs == -1:
            fs = None
        wfdb_rec = wfdb.rdrecord(
            str(self.get_path(rec)), channel_names=["ECG"], physical=True
        )
        ecg = wfdb_rec.p_signal.T
        if fs is not None and fs != wfdb_rec.fs:
            # ecg = ss.resample_poly(ecg, fs, wfdb_rec.fs, axis=-1)
            ecg = librosa.resample(ecg, wfdb_rec.fs, fs, res_type="kaiser_best")
        if data_format.lower() == "channel_last":
            ecg = ecg.T
        return ecg

    def load_ann(self, rec: str) -> str:
        """
        load annotations of the record `rec`
        """
        return wfdb.rdheader(self.get_path(rec)).comments[0]

    def play(self, rec: str, **kwargs) -> IPython.display.Audio:
        """ """
        audio_file = self.get_path(rec, self.data_ext)
        return IPython.display.Audio(filename=str(audio_file))

    def plot(self, rec: str, **kwargs) -> NoReturn:
        """ """
        raise NotImplementedError


class EPHNOGRAMReader(PCGDataBase):
    """ """

    __name__ = "EPHNOGRAMReader"

    def __init__(
        self,
        db_dir: str,
        fs: int = 8000,
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        super().__init__(
            db_name="ephnogram",
            db_dir=db_dir,
            fs=fs,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.data_ext = "mat"
        self.aux_ext = "dat"

        self._df_stats = pd.read_csv(self.db_dir / "ECGPCGSpreadsheet.csv")
        self._df_stats = self._df_stats[
            [
                "Record Name",
                "Subject ID",
                "Record Duration (min)",
                "Age (years)",
                "Gender",
                "Recording Scenario",
                "Num Channels",
                "ECG Notes",
                "PCG Notes",
                "PCG2 Notes",
                "AUX1 Notes",
                "AUX2 Notes",
                "Database Housekeeping",
            ]
        ]
        self._df_stats = self._df_stats[
            ~self._df_stats["Record Name"].isna()
        ].reset_index(drop=True)

        self._ls_rec()
        self.data_dir = self.db_dir / "MAT"
        self.aux_dir = self.db_dir / "WFDB"
        self._channels = [
            "ECG",
            "PCG",
            "PCG2",
            "AUX1",
            "AUX2",
        ]

    def _ls_rec(self) -> NoReturn:
        """ """
        try:
            self._all_records = wfdb.get_record_list(self.db_name)
        except:
            records_file = self.db_dir / "RECORDS"
            if records_file.exists():
                self._all_records = records_file.read_text().splitlines()
            else:
                self._all_records = sorted(self.db_dir.rglob(f"*.{self.header_ext}"))
                self._all_records = [
                    item.replace(str(self.db_dir), "")
                    .replace(f".{self.header_ext}", "")
                    .strip(self.db_dir.anchor)
                    for item in self._all_records
                ]
                records_file.write_text("\n".join(self._all_records))
        self._all_records = [Path(item).name for item in self._all_records]

    def load_data(
        self,
        rec: str,
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        channels: Optional[Union[str, Sequence[str]]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        load data from the record `rec`
        """
        fs = fs or self.fs
        if fs == -1:
            fs = None
        data = sio.loadmat(self.data_dir / f"{rec}.{self.data_ext}")
        data_fs = data["fs"][0][0]
        channels = channels or self._channels
        if isinstance(channels, str):
            channels = [channels]
        data = {
            k: v.astype(np.float32)
            if data_format.lower() == "channel_first"
            else v.astype(np.float32).T
            for k, v in data.items()
            if k in channels
        }
        if fs is not None and fs != data_fs:
            for k in data:
                data[k] = librosa.resample(data[k], data_fs, fs, res_type="kaiser_best")
        return data

    def load_pcg(
        self,
        rec: str,
        fs: Optional[int] = None,
        data_format: str = "channel_first",
    ) -> np.ndarray:
        """ """
        return self.load_data(rec, fs, data_format, "PCG")["PCG"]

    def load_ann(self, rec: str) -> str:
        """
        load annotations of the record `rec`
        """
        raise NotImplementedError("No annotation for this database")

    def play(self, rec: str, channel: str = "PCG") -> IPython.display.Audio:
        """ """
        data = self.load_data(rec, channels=channel)[0]
        return IPython.display.Audio(data=data, rate=8000)

    def plot(self, rec: str, **kwargs) -> NoReturn:
        """ """
        raise NotImplementedError
