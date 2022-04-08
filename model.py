"""
Models, including:
    - 1D models accepcting raw audio signal input
    - 2D models accepcting spectrogram input
"""

from copy import deepcopy
from typing import Union, Optional, NoReturn, Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_ecg.cfg import CFG
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET
from torch_ecg.models.unets.ecg_unet import ECG_UNET
from torch_ecg.components.outputs import (
    ClassificationOutput,
    SequenceLabelingOutput,
)

from cfg import ModelCfg


__all__ = [
    "CRNN_CINC2022",
    "SEQ_LAB_NET_CINC2022",
    "UNET_CINC2022",
]


class CRNN_CINC2022(ECG_CRNN):
    """ """

    __DEBUG__ = True
    __name__ = "CRNN_CINC2022"

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
        task = "classification"
        model_cfg = deepcopy(ModelCfg[task])
        model = ECG_CRNN_CINC2022(model_cfg)
        ````
        """
        config = CFG(deepcopy(ModelCfg.classification))
        config.update(deepcopy(config) or {})
        super().__init__(
            config.classes,
            config.num_channels,
            config[config.model_name],
        )

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
    ) -> ClassificationOutput:
        """finished, checked,

        auxiliary function to `forward`, for CINC2022,

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        class_names: bool, default False,
            if True, the returned scalar predictions will be a `DataFrame`,
            with class names for each scalar prediction

        Returns
        -------
        output: ClassificationOutput, with items:
            - classes: list of str,
                list of the class names
            - prob: ndarray or DataFrame,
                scalar (probability) predictions,
                (and binary predictions if `class_names` is True)
            - pred: ndarray,
                the array of class number predictions
            - bin_pred: ndarray,
                the array of binary predictions
            - forward_output: ndarray,
                the array of output of the model's forward function,
                useful for producing challenge result using
                multiple recordings
        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
        forward_output = self.forward(_input)
        prob = self.softmax(forward_output)
        pred = torch.argmax(prob, dim=-1)
        bin_pred = (prob == prob.max(dim=-1, keepdim=True).values).to(int)
        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        bin_pred = bin_pred.cpu().detach().numpy()
        forward_output = forward_output.cpu().detach().numpy()
        if class_names:
            prob = pd.DataFrame(prob, columns=self.classes)
            prob["pred"] = ""
            for row_idx in range(len(prob)):
                prob.at[row_idx, "pred"] = pred[row_idx]
            prob["pred"] = prob["pred"].apply(lambda x: self.classes[x])
            pred = prob["pred"].values

        return ClassificationOutput(
            classes=self.classes,
            prob=prob,
            pred=pred,
            bin_pred=bin_pred,
            forward_output=forward_output,
        )

    @torch.no_grad()
    def inference_CINC2022(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
    ) -> ClassificationOutput:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names)


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
        config = CFG(deepcopy(ModelCfg.segmentation))
        config.update(deepcopy(config) or {})
        if config[config.model_name].reduction == 1:
            config[config.model_name].recover_length = True
        super().__init__(
            config.classes,
            config.num_channels,
            config[config.model_name],
        )

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
    ) -> SequenceLabelingOutput:
        """finished, checked,

        auxiliary function to `forward`, for CINC2022,

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)

        Returns
        -------
        output: SequenceLabelingOutput, with items:
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

        return SequenceLabelingOutput(
            classes=self.classes,
            prob=prob,
            pred=pred,
        )

    @torch.no_grad()
    def inference_CINC2022(
        self,
        input: Union[np.ndarray, Tensor],
    ) -> SequenceLabelingOutput:
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
        config = CFG(deepcopy(ModelCfg.segmentation))
        config.update(deepcopy(config) or {})
        super().__init__(
            config.classes,
            config.num_channels,
            config[config.model_name],
        )

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
    ) -> SequenceLabelingOutput:
        """finished, checked,

        auxiliary function to `forward`, for CINC2022,

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)

        Returns
        -------
        output: SequenceLabelingOutput, with items:
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

        return SequenceLabelingOutput(
            classes=self.classes,
            prob=prob,
            pred=pred,
        )

    @torch.no_grad()
    def inference_CINC2022(
        self,
        input: Union[np.ndarray, Tensor],
    ) -> SequenceLabelingOutput:
        """
        alias for `self.inference`
        """
        return self.inference(input)
