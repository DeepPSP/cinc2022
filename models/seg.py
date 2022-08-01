"""
"""

from copy import deepcopy
from typing import Union, Optional, NoReturn, Any

import numpy as np
import torch
from torch import Tensor
from torch_ecg.cfg import CFG
from torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET
from torch_ecg.models.unets.ecg_unet import ECG_UNET
from torch_ecg.components.outputs import (
    SequenceLabellingOutput,
)
from torch_ecg.utils import add_docstring

from cfg import ModelCfg


__all__ = [
    "SEQ_LAB_NET_CINC2022",
    "UNET_CINC2022",
]


class SEQ_LAB_NET_CINC2022(ECG_SEQ_LAB_NET):
    """ """

    __DEBUG__ = True
    __name__ = "SEQ_LAB_NET_CINC2022"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> NoReturn:
        """

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        Usage
        -----
        ```python
        from cfg import ModelCfg
        task = "segmentation"
        model_cfg = deepcopy(ModelCfg[task])
        model = ECG_SEQ_LAB_NET_CINC2022(model_cfg)
        ````
        """
        _config = CFG(deepcopy(ModelCfg.segmentation))
        _config = _config[_config.model_name]
        _config.update(deepcopy(config) or {})
        if _config[_config.model_name].reduction == 1:
            _config[_config.model_name].recover_length = True
        super().__init__(
            _config.classes,
            _config.num_channels,
            _config[_config.model_name],
        )

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
        bin_pred_threshold: float = 0.5,
    ) -> SequenceLabellingOutput:
        """

        auxiliary function to `forward`, for CINC2022,

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        bin_pred_threshold: float, default 0.5,
            threshold for binary predictions,
            works only if "unannotated" not in `self.classes`

        Returns
        -------
        output: SequenceLabellingOutput, with items:
            - classes: list of str,
                list of the class names
            - prob: ndarray or DataFrame,
                scalar (probability) predictions,
                (and binary predictions if `class_names` is True)
            - pred: ndarray,
                the array of binary predictions
        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
        if "unannotated" in self.classes:
            prob = self.softmax(self.forward(_input))
            pred = torch.argmax(prob, dim=-1)
        else:
            prob = self.sigmoid(self.forward(_input))
            pred = (prob > bin_pred_threshold).int() * (
                prob == prob.max(dim=-1, keepdim=True).values
            ).int()
        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        return SequenceLabellingOutput(
            classes=self.classes,
            prob=prob,
            pred=pred,
        )

    @add_docstring(inference.__doc__)
    def inference_CINC2022(
        self,
        input: Union[np.ndarray, Tensor],
    ) -> SequenceLabellingOutput:
        """
        alias for `self.inference`
        """
        return self.inference(input)


class UNET_CINC2022(ECG_UNET):
    """ """

    __DEBUG__ = True
    __name__ = "UNET_CINC2022"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> NoReturn:
        """

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        Usage
        -----
        ```python
        from cfg import ModelCfg
        task = "segmentation"
        model_cfg = deepcopy(ModelCfg[task])
        model = ECG_SEQ_LAB_NET_CINC2022(model_cfg)
        ````
        """
        _config = CFG(deepcopy(ModelCfg.segmentation))
        _config = _config[_config.model_name]
        _config.update(deepcopy(config) or {})
        super().__init__(
            _config.classes,
            _config.num_channels,
            _config[_config.model_name],
        )

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
    ) -> SequenceLabellingOutput:
        """

        auxiliary function to `forward`, for CINC2022,

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)

        Returns
        -------
        output: SequenceLabellingOutput, with items:
            - classes: list of str,
                list of the class names
            - prob: ndarray or DataFrame,
                scalar (probability) predictions,
                (and binary predictions if `class_names` is True)
            - pred: ndarray,
                the array of binary predictions
        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
        prob = self.softmax(self.forward(_input))
        pred = torch.argmax(prob, dim=-1)
        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        return SequenceLabellingOutput(
            classes=self.classes,
            prob=prob,
            pred=pred,
        )

    @add_docstring(inference.__doc__)
    def inference_CINC2022(
        self,
        input: Union[np.ndarray, Tensor],
    ) -> SequenceLabellingOutput:
        """
        alias for `self.inference`
        """
        return self.inference(input)
