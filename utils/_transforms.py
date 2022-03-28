"""
Transforms that does not exist in old versions of torchaudio.

Added transforms:
- ``PitchShift`` (added in ``torchaudio`` 0.9.x)

NOTE:
- ``PitchShift`` seems to raise the following error:
``"arange_cuda" not implemented for 'ComplexFloat'`` or
``"arange_cpu" not implemented for 'ComplexFloat'``
"""

import math
from typing import Optional, NoReturn, Callable

import torch
from torch import Tensor

# from torchaudio.functional import phase_vocoder


__all__ = [
    "PitchShift",
    "pitch_shift",
]


class PitchShift(torch.nn.Module):
    r"""Shift the pitch of a waveform by ``n_steps`` steps.
    Args:
        waveform (Tensor): The input waveform of shape `(..., time)`.
        sample_rate (int): Sample rate of `waveform`.
        n_steps (int): The (fractional) steps to shift `waveform`.
        bins_per_octave (int, optional): The number of steps per octave (Default : ``12``).
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins (Default: ``512``).
        win_length (int or None, optional): Window size. If None, then ``n_fft`` is used. (Default: ``None``).
        hop_length (int or None, optional): Length of hop between STFT windows. If None, then ``win_length // 4``
            is used (Default: ``None``).
        window (Tensor or None, optional): Window tensor that is applied/multiplied to each frame/window.
            If None, then ``torch.hann_window(win_length)`` is used (Default: ``None``).
    Example
        >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
        >>> transform = transforms.PitchShift(sample_rate, 4)
        >>> waveform_shift = transform(waveform)  # (channel, time)

    """
    __constants__ = [
        "sample_rate",
        "n_steps",
        "bins_per_octave",
        "n_fft",
        "win_length",
        "hop_length",
    ]

    def __init__(
        self,
        sample_rate: int,
        n_steps: int,
        bins_per_octave: int = 12,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        wkwargs: Optional[dict] = None,
    ) -> NoReturn:
        super().__init__()
        self.n_steps = n_steps
        self.bins_per_octave = bins_per_octave
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        window = (
            window_fn(self.win_length)
            if wkwargs is None
            else window_fn(self.win_length, **wkwargs)
        )
        self.register_buffer("window", window)

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension `(..., time)`.
        Returns:
            Tensor: The pitch-shifted audio of shape `(..., time)`.

        """

        return pitch_shift(
            waveform,
            self.sample_rate,
            self.n_steps,
            self.bins_per_octave,
            self.n_fft,
            self.win_length,
            self.hop_length,
            self.window,
        )


def pitch_shift(
    waveform: Tensor,
    sample_rate: int,
    n_steps: int,
    bins_per_octave: int = 12,
    n_fft: int = 512,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    window: Optional[Tensor] = None,
) -> Tensor:
    """
    Shift the pitch of a waveform by ``n_steps`` steps.
    Args:
        waveform (Tensor): The input waveform of shape `(..., time)`.
        sample_rate (int): Sample rate of `waveform`.
        n_steps (int): The (fractional) steps to shift `waveform`.
        bins_per_octave (int, optional): The number of steps per octave (Default: ``12``).
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins (Default: ``512``).
        win_length (int or None, optional): Window size. If None, then ``n_fft`` is used. (Default: ``None``).
        hop_length (int or None, optional): Length of hop between STFT windows. If None, then
            ``win_length // 4`` is used (Default: ``None``).
        window (Tensor or None, optional): Window tensor that is applied/multiplied to each frame/window.
            If None, then ``torch.hann_window(win_length)`` is used (Default: ``None``).
    Returns:
        Tensor: The pitch-shifted audio waveform of shape `(..., time)`.

    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(window_length=win_length, device=waveform.device)

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    ori_len = shape[-1]
    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    phase_advance = torch.linspace(
        0, math.pi * hop_length, spec_f.shape[-2], device=spec_f.device
    )[..., None]
    spec_stretch = phase_vocoder(spec_f, rate, phase_advance)
    len_stretch = int(round(ori_len / rate))
    waveform_stretch = torch.istft(
        spec_stretch,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=len_stretch,
    )
    waveform_shift = resample(waveform_stretch, int(sample_rate / rate), sample_rate)
    shift_len = waveform_shift.size()[-1]
    if shift_len > ori_len:
        waveform_shift = waveform_shift[..., :ori_len]
    else:
        waveform_shift = torch.nn.functional.pad(
            waveform_shift, [0, ori_len - shift_len]
        )

    # unpack batch
    waveform_shift = waveform_shift.view(shape[:-1] + waveform_shift.shape[-1:])
    return waveform_shift


def phase_vocoder(
    complex_specgrams: Tensor, rate: float, phase_advance: Tensor
) -> Tensor:
    r"""Given a STFT tensor, speed up in time without modifying pitch by a
    factor of ``rate``.
    Args:
        complex_specgrams (Tensor): Dimension of `(..., freq, time, complex=2)`
        rate (float): Speed-up factor
        phase_advance (Tensor): Expected phase advance in each bin. Dimension of (freq, 1)
    Returns:
        Tensor: Complex Specgrams Stretch with dimension of `(..., freq, ceil(time/rate), complex=2)`
    Example
        >>> freq, hop_length = 1025, 512
        >>> # (channel, freq, time, complex=2)
        >>> complex_specgrams = torch.randn(2, freq, 300, 2)
        >>> rate = 1.3 # Speed up by 30%
        >>> phase_advance = torch.linspace(
        >>>    0, math.pi * hop_length, freq)[..., None]
        >>> x = phase_vocoder(complex_specgrams, rate, phase_advance)
        >>> x.shape # with 231 == ceil(300 / 1.3)
        torch.Size([2, 1025, 231, 2])

    """

    # pack batch
    shape = complex_specgrams.size()
    complex_specgrams = complex_specgrams.reshape([-1] + list(shape[-3:]))

    time_steps = torch.arange(
        0,
        complex_specgrams.size(-2),
        rate,
        device=complex_specgrams.device,
        dtype=complex_specgrams.dtype,
    )

    alphas = time_steps % 1.0
    phase_0 = angle(complex_specgrams[..., :1, :])

    # Time Padding
    complex_specgrams = torch.nn.functional.pad(complex_specgrams, [0, 0, 0, 2])

    # (new_bins, freq, 2)
    complex_specgrams_0 = complex_specgrams.index_select(-2, time_steps.long())
    complex_specgrams_1 = complex_specgrams.index_select(-2, (time_steps + 1).long())

    angle_0 = angle(complex_specgrams_0)
    angle_1 = angle(complex_specgrams_1)

    norm_0 = torch.norm(complex_specgrams_0, p=2, dim=-1)
    norm_1 = torch.norm(complex_specgrams_1, p=2, dim=-1)

    phase = angle_1 - angle_0 - phase_advance
    phase = phase - 2 * math.pi * torch.round(phase / (2 * math.pi))

    # Compute Phase Accum
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[..., :-1]], dim=-1)
    phase_acc = torch.cumsum(phase, -1)

    mag = alphas * norm_1 + (1 - alphas) * norm_0

    real_stretch = mag * torch.cos(phase_acc)
    imag_stretch = mag * torch.sin(phase_acc)

    complex_specgrams_stretch = torch.stack([real_stretch, imag_stretch], dim=-1)

    # unpack batch
    complex_specgrams_stretch = complex_specgrams_stretch.reshape(
        shape[:-3] + complex_specgrams_stretch.shape[1:]
    )

    return complex_specgrams_stretch
