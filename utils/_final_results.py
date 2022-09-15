"""
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Union, NoReturn, Dict

import pandas as pd
import requests
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfg import BaseCfg


_URLS = {
    "summary": "https://moody-challenge.physionet.org/2022/results/summary.tsv",
    "official_murmur_scores": "https://moody-challenge.physionet.org/2022/results/official_murmur_scores.tsv",
    "official_outcome_scores": "https://moody-challenge.physionet.org/2022/results/official_outcome_scores.tsv",
    "unofficial_murmur_scores": "https://moody-challenge.physionet.org/2022/results/unofficial_murmur_scores.tsv",
    "unofficial_outcome_scores": "https://moody-challenge.physionet.org/2022/results/unofficial_outcome_scores.tsv",
}


def _fetch_final_results() -> Dict[str, pd.DataFrame]:
    df = {}
    for name, url in _URLS.items():
        _http_get(url, BaseCfg.log_dir / f"{name}.tsv")
        df[name] = pd.read_csv(BaseCfg.log_dir / f"{name}.tsv", sep="\t")
    df["official_murmur_scores"].insert(
        3,
        "Ranking Metric (Weighted Accuracy on Test Set)",
        df["official_murmur_scores"]["Weighted Accuracy on Test Set"].values,
    )
    df["official_outcome_scores"].insert(
        3,
        "Ranking Metric (Cost on Test Set)",
        df["official_outcome_scores"]["Cost on Test Set"].values,
    )
    df["unofficial_murmur_scores"].insert(
        2,
        "Ranking Metric (Weighted Accuracy on Test Set)",
        df["unofficial_murmur_scores"]["Weighted Accuracy on Test Set"].values,
    )
    df["unofficial_outcome_scores"].insert(
        2,
        "Ranking Metric (Cost on Test Set)",
        df["unofficial_outcome_scores"]["Cost on Test Set"].values,
    )
    for name in _URLS:
        os.remove(BaseCfg.log_dir / f"{name}.tsv")
    return df


def _http_get(url: str, fname: Union[str, Path]) -> NoReturn:
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=Path(fname).name,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def main() -> NoReturn:
    df = _fetch_final_results()
    updated = False
    save_path = BaseCfg.log_dir.parent / "results" / "final_scores.xlsx"
    df_old = (
        pd.read_excel(save_path, sheet_name=None, engine="openpyxl")
        if save_path.exists()
        else None
    )
    if df_old is None:
        updated = True
    else:
        for name in _URLS:
            if not df_old[name].equals(df[name]):
                updated = True
                break
    if updated:
        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            for name, df_ in df.items():
                df_.to_excel(writer, sheet_name=name, index=False)
        print(f"final results saved to {str(save_path)}")
    else:
        mtime = datetime.fromtimestamp(save_path.stat().st_mtime)
        mtime = datetime.strftime(mtime, "%Y-%m-%d %H:%M:%S")
        print(f"final results is up-to-date, last updated at {mtime}")


if __name__ == "__main__":
    main()
