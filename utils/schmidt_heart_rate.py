"""
"""

from typing import Optional, Tuple

import numpy as np
from easydict import EasyDict as ED
from torch_ecg.utils.utils_signal import butter_bandpass_filter

from .schmidt_spike_removal import schmidt_spike_removal
from .springer_features import homomorphic_envelope_with_hilbert


__all__ = [
    "get_schmidt_heart_rate",
]


def get_schmidt_heart_rate(signal:np.ndarray, fs:int, config:Optional[dict]=None) -> Tuple[float, float]:
    """
    """
    cfg = ED(
        order=2,
        lowcut=25,
        highcut=400,
        lpf_freq=8,
    )
    cfg.update(config or {})
    filtered_signal = butter_bandpass_filter(
        signal, fs=fs,
        lowcut=cfg.lowcut, highcut=cfg.highcut,
        order=cfg.order,
    )
    filtered_signal = schmidt_spike_removal(filtered_signal, fs)

    homomorphic_envelope = homomorphic_envelope_with_hilbert(filtered_signal, fs, cfg.lpf_freq)
    y = homomorphic_envelope - np.mean(homomorphic_envelope)
    auto_corr = np.correlate(y, y, mode="full")[len(homomorphic_envelope):]

    min_index = round(0.3 * fs)  # hr 200
    max_index = round(2 * fs)  # hr 30

    index = np.argmax(auto_corr[min_index:max_index]) + min_index
    mean_hr = 60 / (index / fs)

    max_sys_duration = round(index / 2)
    min_sys_duration = min(round(0.2 * fs), max_sys_duration)
    systolic_time_interval = \
        (np.argmax(auto_corr[min_sys_duration:max_sys_duration+1]) + min_sys_duration) / fs

    return mean_hr, systolic_time_interval
