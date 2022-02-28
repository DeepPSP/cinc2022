"""
"""

from decimal import localcontext, Decimal, ROUND_HALF_UP
from typing import Optional, Sequence

import numpy as np
import scipy.signal as SS
from easydict import EasyDict as ED

from .utils_signal import butter_bandpass_filter, normalize
from .schmidt_spike_removal import schmidt_spike_removal
from .springer_dwt import get_dwt_features


__all__ = [
    "get_springer_features",
]


def get_springer_features(signal:np.ndarray, fs:int, feature_fs:int, config:Optional[dict]=None) -> np.ndarray:
    """
    """
    cfg = ED(
        order=2,
        lowcut=25,
        highcut=400,
        lpf_freq=8,
        seg_tol=0.1,
        psd_freq_lim=(40,60),
        wavelet_level=3,
        wavelet_name="db7",
    )
    cfg.update(config or {})
    filtered_signal = butter_bandpass_filter(
        signal, fs=fs,
        lowcut=cfg.lowcut, highcut=cfg.highcut,
        order=cfg.order,
    )
    filtered_signal = schmidt_spike_removal(filtered_signal, fs)

    homomorphic_envelope = \
        homomorphic_envelope_with_hilbert(filtered_signal, fs, lpf_freq=cfg.lpf_freq)
    downsampled_homomorphic_envelope = \
        SS.resample_poly(homomorphic_envelope, feature_fs, fs)
    downsampled_homomorphic_envelope = \
        normalize(downsampled_homomorphic_envelope, method="z-score", mean=0.0, std=1.0)

    amplitude_envelope = hilbert_envelope(filtered_signal, fs)
    downsampled_hilbert_envelope = \
        SS.resample_poly(amplitude_envelope, feature_fs, fs)
    downsampled_hilbert_envelope = \
        normalize(downsampled_hilbert_envelope, method="z-score", mean=0.0, std=1.0)

    psd = get_PSD_feature(filtered_signal, fs, freq_lim=cfg.psd_freq_lim)
    psd = SS.resample_poly(psd, len(downsampled_homomorphic_envelope), len(psd))
    psd = normalize(psd, method="z-score", mean=0.0, std=1.0)

    wavelet_feature = np.abs(get_dwt_features(filtered_signal, fs, feature_fs, config=cfg))
    wavelet_feature = wavelet_feature[:len(homomorphic_envelope)]
    wavelet_feature = SS.resample_poly(wavelet_feature, feature_fs, fs)
    wavelet_feature = normalize(wavelet_feature, method="z-score", mean=0.0, std=1.0)

    springer_features = np.concatenate([
        downsampled_homomorphic_envelope,
        downsampled_hilbert_envelope,
        psd,
        wavelet_feature,
    ])
    return springer_features


def hilbert_envelope(signal:np.ndarray, fs:int) -> np.ndarray:
    """
    """
    return np.abs(SS.hilbert(signal))


def homomorphic_envelope_with_hilbert(signal:np.ndarray, fs:int, lpf_freq:int=8, order:int=1) -> np.ndarray:
    """
    """
    amplitude_envelope = hilbert_envelope(signal, fs)
    homomorphic_envelope = np.exp(
        butter_bandpass_filter(np.log(amplitude_envelope), 0, lpf_freq, fs, order=order)
    )
    homomorphic_envelope[0] = homomorphic_envelope[1]
    return homomorphic_envelope


def get_PSD_feature(signal:np.ndarray,
                    fs:int,
                    freq_lim:Sequence[int]=(40,60),
                    window_size:float=1/40,
                    overlap_size:float=1/80,) -> np.ndarray:
    """
    """
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP
        nperseg = int(Decimal(fs*window_size).to_integral_value())
        noverlap = int(Decimal(fs*overlap_size).to_integral_value())
    f, t, Sxx = SS.spectrogram(
        signal, fs,
        nperseg=nperseg, noverlap=noverlap, nfft=fs,
        return_onesided=True, scaling="density", mode="psd",
    )
    inds = np.where((f >= freq_lim[0]) & (f <= freq_lim[1]))[0]
    psd = np.mean(Sxx[inds, :], axis=0)
    return psd
