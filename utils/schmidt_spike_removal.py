"""
"""

import numpy as np


__all__ = ["schmidt_spike_removal",]


def schmidt_spike_removal(original_signal:np.ndarray,
                          fs:int,
                          window_size:float=0.5,
                          threshold:float=3.0,
                          eps:float=1e-4) -> np.ndarray:
    """
    """
    window_size = round(fs * window_size)
    nframes = original_signal.shape[0] // window_size
    sig_start = (original_signal.shape[0] - window_size * nframes) // 2
    frames = original_signal[sig_start: sig_start + window_size * nframes].reshape((nframes, window_size))
    MAAs = np.abs(frames).max(axis=1)  # of shape (nframes,)

    while len(np.where(MAAs > threshold * np.median(MAAs))[0]) > 0:
        frame_num = np.where(MAAs==MAAs.max())[0][0]
        spike_position = np.argmax(np.abs(frames[frame_num]))
        zero_crossings = np.where(np.diff(np.sign(frames[frame_num])))[0]
        spike_start = np.where(zero_crossings <= spike_position)[0]
        spike_start = zero_crossings[spike_start[-1]] if len(spike_start) > 0 else 0
        spike_end = np.where(zero_crossings >= spike_position)[0]
        spike_end = zero_crossings[spike_end[0]] + 1 if len(spike_end) > 0 else window_size
        # print(f"frame_num = {frame_num}, spike_position = {spike_position}, spike_start = {spike_start}, spike_end = {spike_end}")
        frames[frame_num, spike_start:spike_end] = eps
        MAAs = np.abs(frames).max(axis=1)

    despiked_signal = original_signal.copy()
    despiked_signal[sig_start: sig_start + window_size * nframes] = frames.reshape((-1,))

    return despiked_signal
