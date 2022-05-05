"""
"""

import re
import warnings
import os
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from abc import abstractmethod
from typing import Union, Optional, Any, List, Dict, Tuple, Sequence, NoReturn

import numpy as np
import pandas as pd
import wfdb
import librosa
import torch
import torchaudio
import scipy.signal as ss  # noqa: F401
import scipy.io as sio
import scipy.io.wavfile as sio_wav
import IPython

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm
from torch_ecg.databases.base import PhysioNetDataBase
from torch_ecg.utils.utils_signal import butter_bandpass_filter
from torch_ecg.utils.misc import get_record_list_recursive3

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
        audio_backend: str = "torchaudio",
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
        audio_backend: str, default "torchaudio",
            audio backend to use, can be one of
            "librosa", "torchaudio", "scipy",  "wfdb",
            case insensitive.
            "librosa" or "torchaudio" is recommended.
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(db_name, db_dir, working_dir, verbose, **kwargs)
        self.fs = fs
        self.dtype = kwargs.get("dtype", np.float32)
        self.audio_backend = audio_backend.lower()
        if self.audio_backend == "torchaudio":

            def torchaudio_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
                try:
                    data, new_fs = torchaudio.load(file, normalize=True)
                except Exception:
                    data, new_fs = torchaudio.load(file, normalization=True)
                return data, new_fs

            self._audio_load_func = torchaudio_load
        elif self.audio_backend == "librosa":

            def librosa_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
                data, _ = librosa.load(file, sr=fs, mono=False)
                return torch.from_numpy(data.reshape((-1, data.shape[-1]))), fs

            self._audio_load_func = librosa_load
        elif self.audio_backend == "scipy":

            def scipy_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
                new_fs, data = sio_wav.read(file)
                data = (data / (2**15)).astype(self.dtype)[np.newaxis, :]
                return torch.from_numpy(data), new_fs

            self._audio_load_func = scipy_load
        elif self.audio_backend == "wfdb":
            warnings.warn(
                "loading result using wfdb is inconsistent with other backends"
            )

            def wfdb_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
                record = wfdb.rdrecord(file, physical=True)  # channel last
                sig = record.p_signal.T.astype(self.dtype)
                return torch.from_numpy(sig), record.fs[0]

            self._audio_load_func = wfdb_load
        self.data_ext = None
        self.ann_ext = None
        self.header_ext = "hea"
        self._all_records = None

    def _auto_infer_units(self) -> NoReturn:
        """
        disable this function implemented in the base class
        """
        print("DO NOT USE THIS FUNCTION for a PCG database!")

    @abstractmethod
    def play(self, rec: str, **kwargs) -> IPython.display.Audio:
        """ """
        raise NotImplementedError

    def _reset_fs(self, new_fs: int) -> NoReturn:
        """ """
        self.fs = new_fs


class CINC2022Reader(PCGDataBase):
    """

    The CirCor DigiScope Phonocardiogram Dataset (main resource for CinC2022)

    About CinC2022
    --------------
    1. 5272 heart sound recordings (.wav format) were collected from the main 4 auscultation locations of 1568 subjects, aged between 0 and 21 years (mean ± STD = 6.1 ± 4.3 years), with a duration between 4.8 to 80.4 seconds (mean ± STD = 22.9 ± 7.4 s)
    2. segmentation annotations (.tsv format) regarding the location of fundamental heart sounds (S1 and S2) in the recordings have been obtained using a semi-supervised scheme

    NOTE
    ----
    1. the "Murmur" column (records whether heart murmur can be heard or not) and the "Outcome" column (the expert cardiologist's overall diagnosis using **clinical history, physical examination, analog auscultation, echocardiogram, etc.**) are **NOT RELATED**. All of the 6 combinations (["Present", "Absent", "Unknown"] × ["Abnormal", "Normal"]) occur in the dataset.

    About Heart Murmur
    ------------------
    1. A heart murmur is a blowing, whooshing, or rasping sound heard **during a heartbeat**. The sound is caused by turbulent (rough) blood flow through the heart valves or near the heart. ([source](https://medlineplus.gov/ency/article/003266.htm))

    2. A murmur is a series of vibrations of variable duration, audible with a stethoscope at the chest wall, that emanates from the heart or great vessels. A **systolic murmur** is a murmur that begins during or after the first heart sound and ends before or during the second heart sound. ([source](https://www.ncbi.nlm.nih.gov/books/NBK345/)) A **diastolic murmur** is a sound of some duration occurring during diastole. ([source](https://www.ncbi.nlm.nih.gov/books/NBK346/))

    3. ([Wikipedia](https://en.wikipedia.org/wiki/Heart_murmur)) Heart murmurs may have a distinct pitch, duration and timing. Murmurs have seven main characteristics. These include timing, shape, location, radiation, intensity, pitch and quality
        - Timing refers to whether the murmur is a
            * systolic murmur
            * diastolic murmur. Diastolic murmurs are usually abnormal, and may be early, mid or late diastolic ([source](https://www.utmb.edu/pedi_ed/CoreV2/Cardiology/cardiologyV2/cardiologyV24.html))
            * continuous murmur
        - Shape refers to the intensity over time. Murmurs can be crescendo, decrescendo or crescendo-decrescendo
            * Crescendo murmurs increase in intensity over time
            * Decrescendo murmurs decrease in intensity over time
            * Crescendo-decrescendo murmurs have both shapes over time, resembling a diamond or kite shape
        - Location refers to where the heart murmur is usually heard best. There are **four** places on the anterior chest wall to listen for heart murmurs. Each location roughly corresponds to a specific part of the heart.
            | Region    | Location                                  | Heart Valve Association|
            |-----------|-------------------------------------------|------------------------|
            | Aortic    | 2nd right intercostal space               | Aortic valve           |
            | Pulmonic  | 2nd left intercostal spaces               | Pulmonic valve         |
            | Tricuspid | 4th left intercostal space                | Tricuspid valve        |
            | Mitral    | 5th left mid-clavicular intercostal space | Mitral valve           |
        - Radiation refers to where the sound of the murmur travels.
        - Intensity refers to the loudness of the murmur with grades according to the [Levine scale](https://en.wikipedia.org/wiki/Levine_scale), from 1 to 6
            | Levine scale | Murmur Description                                                                                      |
            |--------------|---------------------------------------------------------------------------------------------------------|
            | 1            | only audible on listening carefully for some time                                                       |
            | 2            | faint but immediately audible on placing the stethoscope on the chest                                   |
            | 3            | loud, readily audible but with no palpable thrill                                                       |
            | 4            | loud with a palpable thrill                                                                             |
            | 5            | loud with a palpable thrill, audible with only the rim of the stethoscope touching the chest            |
            | 6            | loud with a palpable thrill, audible with the stethoscope not touching the chest but lifted just off it |
        - Pitch may be
            * low
            * medium
            * high
        This depends on whether auscultation is best with the bell or diaphragm of a stethoscope.
        - Quality refers to **unusual characteristics** of a murmur. For example
            * blowing
            * harsh
            * rumbling
            * musical

    4. Heart sounds usually has frequency lower than 500 Hz (mostly lower than 300 Hz) (inferred from [source](https://biologicalproceduresonline.biomedcentral.com/articles/10.1186/1480-9222-13-7) Figure 2). frequency of heart sounds is low in range between 20 and 150 Hz.

    5. Instantaneous dominant heart sound frequencies ranged from 130 to 410 Hz (mean ± standard deviation 282 ± 70 Hz). Peak murmur frequencies ranged from 200 to 410 Hz (308 ± 70 Hz) ([source](https://www.ajconline.org/article/0002-9149(89)90491-8/pdf))

    6. innocent murmurs had lower frequencies (below 200 Hz) and a frequency spectrum with a more harmonic structure than pathological cases ([source](https://bmcpediatr.biomedcentral.com/articles/10.1186/1471-2431-7-23)). [Table 4](https://bmcpediatr.biomedcentral.com/articles/10.1186/1471-2431-7-23/tables/4) is very important and copied as follows
        | Group        | Amplitude (%) | Low freq limit (Hz) | High freq limit (Hz) |
        |--------------|---------------|---------------------|----------------------|
        | Vibratory    | 23 ± 9        | 72 ± 15             | 161 ± 22             |
        | Ejection     | 20 ± 9        | 60 ± 9              | 142 ± 51             |
        | Pathological | 30 ± 20       | 52 ± 19             | 299 ± 133            |
        | p-value      | 0.013         | < 0.001             | < 0.001              |

    7. the principal frequencies of heart sounds and murmurs are at the lower end of this range, from 20 to 500 Hz; The murmur containing the highest frequency sound is aortic regurgitation, whose dominant frequencies are approximately 400 Hz. The principal frequencies of other sounds and murmurs are between 100 and 400 Hz ([source1](https://www.sciencedirect.com/science/article/pii/B9780323392761000391), [source2](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/heart-sounds))

    """

    __name__ = "CINC2022Reader"

    def __init__(
        self,
        db_dir: str,
        fs: int = 4000,
        audio_backend: str = "torchaudio",
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        db_dir: str,
            storage path of the database
        fs: int, default 4000,
            (re-)sampling frequency of the audio
        audio_backend: str, default "torchaudio",
            audio backend to use, can be one of
            "librosa", "torchaudio", "scipy",  "wfdb",
            case insensitive.
            "librosa" or "torchaudio" is recommended.
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(
            db_name="circor-heart-sound",
            db_dir=db_dir,
            fs=fs,
            audio_backend=audio_backend,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        if "training_data" in os.listdir(self.db_dir):
            self.data_dir = self.db_dir / "training_data"
        else:
            self.data_dir = self.db_dir
        self.data_ext = "wav"
        self.ann_ext = "hea"
        self.segmentation_ext = "tsv"
        self.segmentation_states = deepcopy(BaseCfg.states)
        self.ignore_unannotated = kwargs.get("ignore_unannotated", True)
        if self.ignore_unannotated:
            self.segmentation_states = [
                s for s in self.segmentation_states if s != "unannotated"
            ]
        self.segmentation_map = {n: s for n, s in enumerate(self.segmentation_states)}
        if self.ignore_unannotated:
            self.segmentation_map[BaseCfg.ignore_index] = "unannotated"
        self.auscultation_locations = {
            "PV",
            "AV",
            "MV",
            "TV",
            "Phc",
        }

        self._rec_pattern = f"(?P<sid>[\\d]+)\\_(?P<loc>{'|'.join(self.auscultation_locations)})((?:\\_)(?P<num>\\d))?"

        self._all_records = None
        self._all_subjects = None
        self._subject_records = None
        self._exceptional_records = ["50782_MV_1"]
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
            "Outcome",  # added in version 1.0.2 in the official phase
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
        """
        list all records in the database
        """
        try:
            print("Reading the list of records from local file...")
            records_file = self.db_dir / "RECORDS"
            if records_file.exists():
                self._all_records = records_file.read_text().splitlines()
            else:
                self._all_records = sorted(
                    [
                        f"{item.name}".replace(".hea", "")
                        for item in (self.data_dir).glob("*.hea")
                    ]
                )
                # records_file.write_text("\n".join(self._all_records))
        except Exception:
            print("Reading the list of records from PhysioNet...")
            try:
                self._all_records = wfdb.get_record_list(self.db_name)
            except Exception:
                self._all_records = []
        if len(self._all_records) == 0:
            self._all_records = get_record_list_recursive3(
                self.data_dir, self._rec_pattern
            )
        self._all_records = [
            item.replace("training_data/", "")
            for item in self._all_records
            if (self.db_dir / item).with_suffix(".hea").exists()
            and item.replace("training_data/", "") not in self._exceptional_records
        ]
        self._all_subjects = sorted(
            set([item.split("_")[0] for item in self._all_records]),
            key=lambda x: int(x),
        )
        self._subject_records = defaultdict(list)
        for rec in self._all_records:
            self._subject_records[self.get_subject(rec)].append(rec)
        self._subject_records = dict(self._subject_records)

    def _load_stats(self) -> NoReturn:
        """
        collect statistics of the database
        """
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
                    locations = set()
                    for line in content:
                        if not line.startswith("#"):
                            if line.split()[0] in self.auscultation_locations:
                                locations.add(line.split()[0])
                            continue
                        k, v = line.replace("#", "").split(":")
                        k, v = k.strip(), v.strip()
                        if v == "nan":
                            v = ""
                        new_row[k] = v
                    new_row["Recording locations:"] = "+".join(locations)
                    self._df_stats = self._df_stats.append(
                        new_row,
                        ignore_index=True,
                    )
            self._df_stats.to_csv(stats_file, index=False)
        else:
            print("No data found locally!")
            return
        self._df_stats = self._df_stats.fillna("")
        try:
            # the column "Locations" is changed to "Recording locations:" in version 1.0.2
            self._df_stats.Locations = self._df_stats.Locations.apply(
                lambda s: s.split("+")
            )
        except AttributeError:
            self._df_stats["Locations"] = self._df_stats["Recording locations:"].apply(
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
                    sid = row["Patient ID"]
                    for loc in row["Locations"]:
                        rec = f"{sid}_{loc}"
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
                            "Patient ID": sid,
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

    def _decompose_rec(self, rec: Union[str, int]) -> Dict[str, str]:
        """

        decompose a record name into its components (subject, location, and number)

        Parameters
        ----------
        rec: str or int,
            the record name or the index of the record in `self.all_records`

        Returns
        -------
        dict,
            the components (subject, location, and number) of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        return list(re.finditer(self._rec_pattern, rec))[0].groupdict()

    def load_data(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        data_type: str = "np",
    ) -> np.ndarray:
        """

        load data from the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        fs : int, optional,
            the sampling frequency of the record, defaults to `self.fs`
        data_format : str, optional,
            the format of the returned data, defaults to `channel_first`
            can be `channel_last`, `channel_first`, `flat`,
            case insensitive
        data_type : str, default "np",
            the type of the returned data, can be one of "pt", "np",
            case insensitive

        Returns
        -------
        data : np.ndarray,
            the data of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        fs = fs or self.fs
        if fs == -1:
            fs = None
        data_file = self.data_dir / f"{rec}.{self.data_ext}"
        data, data_fs = self._audio_load_func(data_file, fs)
        # data of shape (n_channels, n_samples), of type torch.Tensor
        if fs is not None and data_fs != fs:
            data = torchaudio.transforms.Resample(data_fs, fs)(data)
        if data_format.lower() == "channel_last":
            data = data.T
        elif data_format.lower() == "flat":
            data = data.reshape(-1)
        if data_type.lower() == "np":
            data = data.numpy()
        elif data_type.lower() != "pt":
            raise ValueError(f"Unsupported data type: {data_type}")
        return data

    def load_ann(
        self, rec_or_sid: Union[str, int], class_map: Optional[Dict[str, int]] = None
    ) -> Union[str, int]:
        """

        load classification annotation of the record `rec` or the subject `sid`

        Parameters
        ----------
        rec_or_sid : str or int,
            the record name or the index of the record in `self.all_records`
            or the subject id
        class_map : dict, optional,
            the mapping of the annotation classes

        Returns
        -------
        ann : str or int,
            the class of the record,
            or the number of the class if `class_map` is provided

        """
        if isinstance(rec_or_sid, int):
            rec_or_sid = self[rec_or_sid]
        _class_map = class_map or {}
        if rec_or_sid in self.all_subjects:
            ann = self.df_stats[self.df_stats["Patient ID"] == rec_or_sid].iloc[0][
                "Murmur"
            ]
        elif rec_or_sid in self.all_records:
            decom = self._decompose_rec(rec_or_sid)
            sid, loc = decom["sid"], decom["loc"]
            row = self.df_stats[self.df_stats["Patient ID"] == sid].iloc[0]
            if row["Murmur"] == "Unknown":
                ann = "Unknown"
            if loc in row["Murmur locations"]:
                ann = "Present"
            else:
                ann = "Absent"
        else:
            raise ValueError(f"{rec_or_sid} is not a valid record or patient ID")
        ann = _class_map.get(ann, ann)
        return ann

    def load_segmentation(
        self, rec: Union[str, int], seg_format: str = "df", fs: Optional[int] = None
    ) -> Union[pd.DataFrame, np.ndarray, dict]:
        """

        load the segmentation of the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        seg_format : str, default `df`,
            the format of the returned segmentation,
            can be `df`, `dict`, `mask`, `binary`,
            case insensitive

        Returns
        -------
        pd.DataFrame or np.ndarray or dict,
            the segmentation of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        fs = fs or self.fs
        if fs == -1:
            fs = self.get_fs(rec)
        segmentation_file = self.data_dir / f"{rec}.{self.segmentation_ext}"
        df_seg = pd.read_csv(segmentation_file, sep="\t", header=None)
        df_seg.columns = ["start_t", "end_t", "label"]
        if self.ignore_unannotated:
            df_seg["label"] = df_seg["label"].apply(
                lambda x: x - 1 if x > 0 else BaseCfg.ignore_index
            )
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
            # mask = np.zeros(df_seg.end.values[-1], dtype=int)
            mask = np.full(df_seg.end.values[-1], BaseCfg.ignore_index, dtype=int)
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
        """

        load meta data of the subject `subject`

        Parameters
        ----------
        subject : str,
            the subject id
        keys : str or sequence of str, optional,
            the keys of the meta data to be returned,
            if None, return all meta data

        Returns
        -------
        meta_data : dict or str or float or int,
            the meta data of the subject

        """
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

    def load_outcome(self, subject: str) -> str:
        """

        load the expert cardiologist's overall diagnosis of  of the subject `subject`

        Parameters
        ----------
        subject : str,
            the subject id

        Returns
        -------
        str,
            the expert cardiologist's overall diagnosis,
            one of `Normal`, `Abnormal`

        """
        row = self._df_stats[self._df_stats["Patient ID"] == subject].iloc[0]
        return row.Outcome

    def _load_preprocessed_data(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        passband: Sequence[int] = BaseCfg.passband,
        order: int = BaseCfg.filter_order,
        spike_removal: bool = True,
    ) -> np.ndarray:
        """

        load preprocessed data of the record `rec`,
        with preprocessing procedure:
            - resample to `fs` (if `fs` is not None)
            - bandpass filter
            - spike removal

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        fs : int, optional,
            the sampling frequency of the returned data
        data_format : str, default `channel_first`,
            the format of the returned data,
            can be `channel_first`, `channel_last` or `flat`,
            case insensitive
        passband : sequence of int, default `BaseCfg.passband`,
            the passband of the bandpass filter
        order : int, default `BaseCfg.filter_order`,
            the order of the bandpass filter
        spike_removal : bool, default True,
            whether to remove spikes using the Schmmidt algorithm

        Returns
        -------
        data : np.ndarray,
            the preprocessed data of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
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

    def get_fs(self, rec: Union[str, int]) -> int:
        """

        get the original sampling frequency of the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`

        Returns
        -------
        int,
            the original sampling frequency of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        return wfdb.rdheader(self.data_dir / rec).fs

    def get_subject(self, rec: Union[str, int]) -> str:
        """

        get the subject id (Patient ID) of the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`

        Returns
        -------
        str,
            the subject id (Patient ID) of the record

        """
        return self._decompose_rec(rec)["sid"]

    @property
    def all_subjects(self) -> List[str]:
        return self._all_subjects

    @property
    def subject_records(self) -> Dict[str, List[str]]:
        return self._subject_records

    @property
    def df_stats(self) -> pd.DataFrame:
        if self._df_stats is None or self._df_stats.empty:
            self._load_stats()
        return self._df_stats

    @property
    def df_stats_records(self) -> pd.DataFrame:
        if self._df_stats_records is None or self._df_stats_records.empty:
            self._load_stats()
        return self._df_stats_records

    def play(self, rec: Union[str, int], **kwargs) -> IPython.display.Audio:
        """

        play the record `rec` in a Juptyer Notebook

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        kwargs : dict,
            optional keyword arguments including `data`, `fs`,
            if specified, the data will be played instead of the record

        Returns
        -------
        IPython.display.Audio,
            the audio object of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
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
        audio_backend: str = "torchaudio",
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        super().__init__(
            db_name="challenge-2016",
            db_dir=db_dir,
            fs=fs,
            audio_backend=audio_backend,
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

    def get_path(self, rec: Union[str, int], extension: Optional[str] = None) -> Path:
        """ """
        if isinstance(rec, int):
            rec = self[rec]
        filename = f"{rec}.{extension}" if extension else rec
        return self.db_dir / f"training-{rec[0]}" / filename

    def load_data(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
    ) -> Dict[str, np.ndarray]:
        """
        load data from the record `rec`
        """
        if isinstance(rec, int):
            rec = self[rec]
        data = {
            "PCG": self.load_pcg(rec, fs, data_format),
            "ECG": self.load_ecg(rec, fs, data_format),
        }
        return data

    def load_pcg(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
    ) -> np.ndarray:
        """ """
        if isinstance(rec, int):
            rec = self[rec]
        fs = fs or self.fs
        if fs == -1:
            fs = None
        pcg, _ = librosa.load(self.get_path(rec, self.data_ext), sr=fs, mono=False)
        pcg = np.atleast_2d(pcg)
        if data_format.lower() == "channel_last":
            pcg = pcg.T
        return pcg

    def load_ecg(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
    ) -> np.ndarray:
        """ """
        if isinstance(rec, int):
            rec = self[rec]
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

    def load_ann(self, rec: Union[str, int]) -> str:
        """
        load annotations of the record `rec`
        """
        return wfdb.rdheader(self.get_path(rec)).comments[0]

    def play(self, rec: Union[str, int], **kwargs) -> IPython.display.Audio:
        """ """
        audio_file = self.get_path(rec, self.data_ext)
        return IPython.display.Audio(filename=str(audio_file))

    def plot(self, rec: Union[str, int], **kwargs) -> NoReturn:
        """ """
        raise NotImplementedError


class EPHNOGRAMReader(PCGDataBase):
    """ """

    __name__ = "EPHNOGRAMReader"

    def __init__(
        self,
        db_dir: str,
        fs: int = 8000,
        audio_backend: str = "torchaudio",
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        super().__init__(
            db_name="ephnogram",
            db_dir=db_dir,
            fs=fs,
            audio_backend=audio_backend,
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
        except Exception:
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
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        channels: Optional[Union[str, Sequence[str]]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        load data from the record `rec`
        """
        if isinstance(rec, int):
            rec = self[rec]
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
        rec: Union[str, int],
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

    def play(self, rec: Union[str, int], channel: str = "PCG") -> IPython.display.Audio:
        """ """
        data = self.load_data(rec, channels=channel)[0]
        return IPython.display.Audio(data=data, rate=8000)

    def plot(self, rec: Union[str, int], **kwargs) -> NoReturn:
        """ """
        raise NotImplementedError
