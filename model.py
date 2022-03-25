"""
"""

from copy import deepcopy
from typing import Union, Optional, Sequence, Tuple, NoReturn, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import Tensor
from torch_ecg.cfg import CFG
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET

from cfg import ModelCfg


class ECG_CRNN_CINC2022(ECG_CRNN):
    """
    """
    __name__ = "ECG_CRNN_CINC2022"

    def __init__(self, classes:Sequence[str], n_channels:int, config:Optional[CFG]=None) -> NoReturn:
        """

        Parameters
        ----------
        classes: list,
            list of the classes for classification
        n_channels: int,
            number of input channels
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        model_config = CFG(deepcopy(ModelCfg))
        model_config.update(deepcopy(config) or {})
        super().__init__(classes, n_channels, model_config)

    @torch.no_grad()
    def inference(self,
                  input:Union[np.ndarray,Tensor],
                  class_names:bool=False,) -> Tuple[Union[np.ndarray,pd.DataFrame],np.ndarray]:
        """ finished, checked,

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
        prob: ndarray or DataFrame,
            scalar (probability) predictions, (and binary predictions if `class_names` is True)
        pred: ndarray,
            the array of class number predictions
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
        if class_names:
            prob = pd.DataFrame(prob, columns=self.classes)
            prob["pred"] = ""
            for row_idx in range(len(prob)):
                prob.at[row_idx, "pred"] = pred[row_idx]
            prob["pred"] = prob["pred"].apply(lambda x: self.classes[x])
            pred = prob["pred"].values
        return prob, pred

    @torch.no_grad()
    def inference_CINC2021(self,
                           input:Union[np.ndarray,Tensor],
                           class_names:bool=False,
                           bin_pred_thr:float=0.5) -> Tuple[Union[np.ndarray,pd.DataFrame],np.ndarray]:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)
