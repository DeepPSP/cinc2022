"""
"""

import inspect
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import NoReturn, Union, List, Sequence

import numpy as np
import torch
from torchaudio import transforms as TT
from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.utils.misc import ReprMixin


__all__ = [
    "InputConfig",
]


class InputConfig(CFG):
    """
    """
    __name__ = "InputConfig"

    def __init__(self, *args: Union[CFG, dict], input_type:str, n_channels:int, n_samples:int=-1, **kwargs: dict) -> NoReturn:
        """
        """
        super().__init__(*args, input_type=input_type, n_channels=n_channels, n_samples=n_samples, **kwargs)
        assert "n_channels" in self and self.n_channels > 0
        assert "n_samples" in self and (self.n_samples > 0 or self.n_samples == -1)
        assert "input_type" in self and self.input_type.lower() in ["waveform", "spectrogram", "mel_spectrogram", "melspectrogram", "mel", "mfcc", "spectral"]
        self.input_type = self.input_type.lower()
        if self.input_type in ["spectrogram", "mel_spectrogram", "melspectrogram", "mel", "mfcc", "spectral"]:
            assert "n_bins" in self
        self.pop("compute_input_shape", None)

    def compute_input_shape(self, waveform_shape:Union[Sequence[int], torch.Size]) -> torch.Size:
        """
        """
        if self.input_type == "waveform":
            return torch.Size(waveform_shape)
        return torch.Size([self.n_channels, self.n_bins, self.n_samples])


class BaseInput(ReprMixin, ABC):
    """
    """
    __name__ = "BaseInput"

    def __init__(self, config: InputConfig) -> NoReturn:
        """
        """
        assert isinstance(config, InputConfig)
        self._config = deepcopy(config)
        self._values = None
        self._dtype = self._config.get("dtype", DEFAULTS.torch_dtype)
        self._device = self._config.get("device", DEFAULTS.device)
        self._post_init()

    def __call__(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        """
        return self.from_waveform(waveform)

    @abstractmethod
    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def _post_init(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    @property
    def values(self) -> torch.Tensor:
        return self._values

    @property
    def n_channels(self) -> int:
        return self._config.n_channels

    @property
    def n_samples(self) -> int:
        if self.values is not None:
            return self.values.shape[-1]
        return self._config.n_samples

    @property
    def input_type(self) -> str:
        return self._config.input_type

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return ["input_type", "n_channels", "n_samples", "dtype", "device"]


class WaveformInput(BaseInput):
    """
    """
    __name__ = "WaveformInput"

    def _post_init(self) -> NoReturn:
        """
        """
        assert self.input_type == "waveform"

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        """
        self._values = torch.as_tensor(waveform).to(self.device, self.dtype)
        return self._values


class _SpectralInput(BaseInput):
    """
    """
    __name__ = "_SpectralInput"

    @property
    def n_bins(self) -> int:
        return self._config.n_bins

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return super().extra_repr_keys() + ["n_bins"]


class SpectrogramInput(_SpectralInput):
    """
    """
    __name__ = "SpectrogramInput"

    def _post_init(self) -> NoReturn:
        """
        """
        assert self.input_type in ["spectrogram"]
        assert "n_bins" in self._config
        assert self.n_channels == 1
        args = inspect.getfullargspec(TT.Spectrogram.__init__).args
        for k in ["self", "n_fft"]:
            args.remove(k)
        kwargs = {k: self._config[k] for k in args if k in self._config}
        kwargs["n_fft"] = (self.n_bins - 1) * 2
        self._transform = TT.Spectrogram(**kwargs).to(self.device, self.dtype)

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        """
        self._values = self._transform(torch.as_tensor(waveform).to(self.device, self.dtype))
        return self._values


class MelSpectrogramInput(_SpectralInput):
    """
    """
    __name__ = "MelSpectrogramInput"

    def _post_init(self) -> NoReturn:
        """
        """
        assert self.input_type in ["mel_spectrogram", "mel", "melspectrogram",]
        assert "n_bins" in self._config
        assert self.n_channels == 1
        self.fs = self._config.get("fs", self._config.get("sample_rate", None))
        assert self.fs is not None
        args = inspect.getfullargspec(TT.MelSpectrogram.__init__).args
        for k in ["self", "sample_rate", "n_fft", "n_mels"]:
            args.remove(k)
        kwargs = {k: self._config[k] for k in args if k in self._config}
        kwargs["n_fft"] = (self.n_bins - 1) * 2
        kwargs["sample_rate"] = self.fs
        kwargs["n_mels"] = self.n_bins
        self._transform = TT.MelSpectrogram(**kwargs).to(self.device, self.dtype)

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        """
        self._values = self._transform(torch.as_tensor(waveform).to(self.device, self.dtype))
        return self._values


class MFCCInput(_SpectralInput):
    """
    """
    __name__ = "MFCCInput"

    def _post_init(self) -> NoReturn:
        """
        """
        assert self.input_type in ["mfcc",]
        assert "n_bins" in self._config
        assert self.n_channels == 1
        self.fs = self._config.get("fs", self._config.get("sample_rate", None))
        assert self.fs is not None
        args = inspect.getfullargspec(TT.MFCC.__init__).args
        for k in ["self", "sample_rate", "n_mfcc"]:
            args.remove(k)
        kwargs = {k: self._config[k] for k in args if k in self._config}
        kwargs["n_mfcc"] = self.n_bins
        kwargs["sample_rate"] = self.fs
        kwargs["melkwargs"] = kwargs.get("melkwargs", {})
        kwargs["melkwargs"].update(
            dict(
                n_fft=(self.n_bins - 1) * 2,
                n_mels=self.n_bins,
            )
        )
        self._transform = TT.MFCC(**kwargs).to(self.device, self.dtype)

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        """
        self._values = self._transform(torch.as_tensor(waveform).to(self.device, self.dtype))
        return self._values


class SpectralInput(_SpectralInput):
    """

    Concatenation of 3 different types of spectrograms:
    - Spectrogram
    - MelSpectrogram
    - MFCC

    Example
    -------
    >>> input_config = InputConfig(
    ...     input_type="spectral",
    ...     n_bins=224,
    ...     n_channels=3,
    ...     fs=1000,
    ...     normalized=True
    ... )
    >>> inputer = SpectralInput(input_config)
    >>> inputer
    SpectralInput(
        n_bins     = 224,
        n_channels = 3,
        n_samples  = -1,
        input_type = 'spectral',
        dtype      = torch.float32,
        device     = device(type='cuda')
    )
    >>> inputer(torch.rand(1,1000*30).to(inputer.device)).shape
    torch.Size([3, 224, 135])
    """
    __name__ = "SpectralInput"
    
    def _post_init(self) -> NoReturn:
        """
        """
        assert self.input_type in ["spectral",]
        assert "n_bins" in self._config
        assert self.n_channels == 3
        self.fs = self._config.get("fs", self._config.get("sample_rate", None))
        assert self.fs is not None
        self._transforms = []
        # spectrogram
        args = inspect.getfullargspec(TT.Spectrogram.__init__).args
        for k in ["self", "n_fft"]:
            args.remove(k)
        spectro_kwargs = {k: self._config[k] for k in args if k in self._config}
        spectro_kwargs["n_fft"] = (self.n_bins - 1) * 2
        self._transforms.append(TT.Spectrogram(**spectro_kwargs).to(self.device, self.dtype))
        # mel spectrogram
        args = inspect.getfullargspec(TT.MelSpectrogram.__init__).args
        for k in ["self", "sample_rate", "n_fft", "n_mels"]:
            args.remove(k)
        mel_kwargs = {k: self._config[k] for k in args if k in self._config}
        mel_kwargs["n_fft"] = (self.n_bins - 1) * 2
        mel_kwargs["sample_rate"] = self.fs
        mel_kwargs["n_mels"] = self.n_bins
        self._transforms.append(TT.MelSpectrogram(**mel_kwargs).to(self.device, self.dtype))
        # MFCC
        args = inspect.getfullargspec(TT.MFCC.__init__).args
        for k in ["self", "sample_rate", "n_mfcc"]:
            args.remove(k)
        mfcc_kwargs = {k: self._config[k] for k in args if k in self._config}
        mfcc_kwargs["n_mfcc"] = self.n_bins
        mfcc_kwargs["sample_rate"] = self.fs
        mfcc_kwargs["melkwargs"] = mfcc_kwargs.get("melkwargs", {})
        mfcc_kwargs["melkwargs"].update(deepcopy(mel_kwargs))
        mfcc_kwargs["melkwargs"].pop("sample_rate")
        self._transforms.append(TT.MFCC(**mfcc_kwargs).to(self.device, self.dtype))

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        """
        self._values = torch.as_tensor(waveform).to(self.device, self.dtype)
        cat_dim = 0 if self.values.ndim == 2 else 1
        self._values = torch.cat(
            [transform(self._values.clone()) for transform in self._transforms],
            dim=cat_dim,
        )
        return self._values
