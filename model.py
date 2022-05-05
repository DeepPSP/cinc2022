"""
Models, including:
    - 1D models accepcting raw audio signal input
    - 2D models accepcting spectrogram input
"""

from copy import deepcopy
from typing import Union, Optional, NoReturn, Any, Tuple

import numpy as np
import pandas as pd
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from torch_ecg.cfg import CFG
from torch_ecg.models.cnn.vgg import VGG16
from torch_ecg.models.cnn.resnet import ResNet
from torch_ecg.models.cnn.multi_scopic import MultiScopicCNN
from torch_ecg.models.cnn.densenet import DenseNet
from torch_ecg.models._nets import MLP
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET
from torch_ecg.models.unets.ecg_unet import ECG_UNET
from torch_ecg.components.outputs import (
    ClassificationOutput,
    SequenceLabelingOutput,
)
from torch_ecg.utils import add_docstring

from cfg import ModelCfg
from wav2vec2 import Wav2Vec2Model, components as w2v2_components


__all__ = [
    "Wav2Vec2_CINC2022",
    "CRNN_CINC2022",
    "SEQ_LAB_NET_CINC2022",
    "UNET_CINC2022",
]


class Wav2Vec2_CINC2022(Wav2Vec2Model):
    """ """

    __DEBUG__ = True
    __name__ = "Wav2Vec2_CINC2022"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> NoReturn:
        """

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        _config = CFG(deepcopy(ModelCfg.classification))
        _config.update(deepcopy(config) or {})
        self.config = _config[_config.model_name]
        assert "encoder" in self.config, "encoder is a required key in config"
        if "rnn" in self.config:
            self.config.pop("rnn", None)
        if "attn" in self.config:
            self.config.pop("attn", None)
        self.classes = deepcopy(_config.classes)
        self.n_classes = len(_config.classes)

        self.squeeze = None

        cnn_choice = self.config.cnn.name.lower()
        cnn_config = self.config.cnn[self.config.cnn.name]
        encoder_in_features = None
        if "wav2vec2" in cnn_choice:
            cnn = w2v2_components._get_feature_extractor(
                norm_mode=cnn_config.norm_mode,
                shapes=cnn_config.ch_ks_st,
                bias=cnn_config.bias,
            )
            encoder_in_features = cnn_config.ch_ks_st[-1][0]
        elif "vgg16" in cnn_choice:
            cnn = VGG16(_config.num_channels, **cnn_config)
        elif "resnet" in cnn_choice:
            cnn = ResNet(_config.num_channels, **cnn_config)
        elif "multi_scopic" in cnn_choice:
            cnn = MultiScopicCNN(_config.num_channels, **cnn_config)
        elif "densenet" in cnn_choice or "dense_net" in cnn_choice:
            cnn = DenseNet(_config.num_channels, **cnn_config)
        else:
            raise NotImplementedError(
                f"the CNN \042{cnn_choice}\042 not implemented yet"
            )
        if encoder_in_features is None:
            _, encoder_in_features, _ = cnn.compute_output_shape()
            cnn = torch.nn.Sequential(
                cnn, Rearrange("batch chan seqlen -> batch seqlen chan")
            )

        encoder_config = self.config.encoder[self.config.encoder.name]
        encoder = w2v2_components._get_encoder(
            in_features=encoder_in_features,
            embed_dim=encoder_config.embed_dim,
            dropout_input=encoder_config.projection_dropout,
            pos_conv_kernel=encoder_config.pos_conv_kernel,
            pos_conv_groups=encoder_config.pos_conv_groups,
            num_layers=encoder_config.num_layers,
            num_heads=encoder_config.num_heads,
            attention_dropout=encoder_config.attention_dropout,
            ff_interm_features=encoder_config.ff_interm_features,
            ff_interm_dropout=encoder_config.ff_interm_dropout,
            dropout=encoder_config.dropout,
            layer_norm_first=encoder_config.layer_norm_first,
            layer_drop=encoder_config.layer_drop,
        )
        # encoder output shape: (batch, seq_len, embed_dim)

        if self.config.global_pool.lower() == "max":
            pool = torch.nn.AdaptiveMaxPool1d((1,), return_indices=False)
        elif self.config.global_pool.lower() == "avg":
            pool = torch.nn.AdaptiveAvgPool1d((1,))
        elif self.config.global_pool.lower() == "attn":
            raise NotImplementedError("Attentive pooling not implemented yet!")

        clf = MLP(
            in_channels=encoder_config.embed_dim,
            out_channels=self.config.clf.out_channels + [self.n_classes],
            activation=self.config.clf.activation,
            bias=self.config.clf.bias,
            dropouts=self.config.clf.dropouts,
            skip_last_activation=True,
        )

        aux = torch.nn.Sequential(
            Rearrange("batch seqlen chan -> batch chan seqlen"),
            pool,
            Rearrange("batch chan seqlen -> batch (chan seqlen)"),
            clf,
        )

        super().__init__(cnn, encoder, aux)

        if _config.get("outcomes", None) is not None:
            self.outcome_head = torch.nn.Sequential(
                Rearrange("batch seqlen chan -> batch chan seqlen"),
                pool,
                Rearrange("batch chan seqlen -> batch (chan seqlen)"),
                MLP(
                    in_channels=self.clf.in_channels,
                    skip_last_activation=True,
                    **_config.outcome_head,
                ),
            )
        else:
            self.outcome_head = None
        if _config.get("states", None) is not None:
            self.states = _config.get("states")
            _config.segmentation_head.out_channels.append(len(self.states))
            self.segmentation_head = MLP(
                in_channels=self.clf.in_channels,
                skip_last_activation=True,
                **_config.segmentation_head,
            )
        else:
            self.segmentation_head = None
            self.states = None

        if "wav2vec2" in cnn_choice:
            self.squeeze = Rearrange("batch 1 seqlen -> batch seqlen")

        # for inference
        # classification: if single-label, use softmax; otherwise (multi-label) use sigmoid
        # sequence tagging: if "unannotated" counted in `classes`, use softmax; otherwise use sigmoid
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, waveforms: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """

        Parameters
        ----------
        waveforms: Tensor,
            shape: (batch, num_channels, seq_len)

        Returns
        -------
        pred: Tensor,
            of shape (batch_size, n_classes)
        outcome: Tensor, optional,
            of shape (batch_size, n_outcomes)
        segmentation: Tensor, optional,
            of shape (batch_size, seqlen, n_states)

        """
        batch_size, channels, seq_len = waveforms.shape

        if self.squeeze is not None:
            waveforms = self.squeeze(waveforms)
            return super().forward(waveforms)[0]
        features = self.feature_extractor(waveforms)
        features = self.encoder(features)
        if self.aux is not None:
            pred = self.aux(features)

        if self.outcome_head is not None:
            outcome = self.outcome_head(features)
            return pred, outcome

        if self.segmentation_head is not None:
            segmentation = self.segmentation_head(features)
            if self.config.segmentation_head.get("recover_length", True):
                segmentation = F.interpolate(
                    segmentation.permute(0, 2, 1),
                    size=seq_len,
                    mode="linear",
                    align_corners=True,
                ).permute(0, 2, 1)
            return pred, segmentation

        return pred

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
        seg_thr: float = 0.5,
    ) -> Union[
        ClassificationOutput,
        Tuple[ClassificationOutput, ClassificationOutput],
        Tuple[ClassificationOutput, SequenceLabelingOutput],
    ]:
        """

        auxiliary function to `forward`, for CINC2022,

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        class_names: bool, default False,
            if True, the returned scalar predictions will be a `DataFrame`,
            with class names for each scalar prediction
        seg_thr: float, default 0.5,
            threshold for making binary predictions for
            the optional segmentaion head

        Returns
        -------
        clf_output: ClassificationOutput, with items:
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
        outcome_output: ClassificationOutput, optional, with items:
            - classes: list of str,
                list of the outcome class names
            - prob: ndarray,
                scalar (probability) predictions,
            - pred: ndarray,
                the array of outcome class number predictions
        seg_output: SequenceLabelingOutput, optional, with items:
            - classes: list of str,
                list of the state class names
            - prob: ndarray,
                scalar (probability) predictions,
            - pred: ndarray,
                the array of binarized prediction

        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
        outcome, states = None, None
        if self.outcome_head is not None:
            forward_output, outcome = self.forward(_input)
        elif self.segmentation_head is not None:
            forward_output, states = self.forward(_input)
        else:
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

        clf_output = ClassificationOutput(
            classes=self.classes,
            prob=prob,
            pred=pred,
            bin_pred=bin_pred,
            forward_output=forward_output,
        )

        if outcome is not None:
            prob = self.softmax(outcome)
            pred = torch.argmax(prob, dim=-1)
            bin_pred = (prob == prob.max(dim=-1, keepdim=True).values).to(int)
            prob = prob.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            bin_pred = bin_pred.cpu().detach().numpy()
            outcome = outcome.cpu().detach().numpy()
            outcome_output = ClassificationOutput(
                classes=self.config.outcomes,
                prob=prob,
                pred=pred,
                bin_pred=bin_pred,
                outcome=outcome,
            )
            return clf_output, outcome_output

        if states is not None:
            # if "unannotated" in self.config.states, use softmax
            # else use sigmoid
            if "unannotated" in self.states:
                prob = self.softmax(states)
                pred = torch.argmax(prob, dim=-1)
            else:
                prob = self.sigmoid(states)
                pred = (prob > seg_thr).int() * (
                    prob == prob.max(dim=-1, keepdim=True).values
                ).int()
            prob = prob.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            seg_output = SequenceLabelingOutput(
                classes=self.states,
                prob=prob,
                pred=pred,
            )
            return clf_output, seg_output

        return clf_output

    @add_docstring(inference.__doc__)
    def inference_CINC2022(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
        seg_thr: float = 0.5,
    ) -> Union[
        ClassificationOutput,
        Tuple[ClassificationOutput, ClassificationOutput],
        Tuple[ClassificationOutput, SequenceLabelingOutput],
    ]:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names)


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
        _config = CFG(deepcopy(ModelCfg.multi_task))
        _config.update(deepcopy(config) or {})
        super().__init__(
            _config.classes,
            _config.num_channels,
            _config[_config.model_name],
        )
        if _config.get("outcomes", None) is not None:
            self.outcome_head = MLP(
                in_channels=self.clf.in_channels,
                skip_last_activation=True,
                **_config.outcome_head,
            )
        else:
            self.outcome_head = None
        if _config.get("states", None) is not None:
            self.states = _config.get("states")
            _config.segmentation_head.out_channels.append(len(self.states))
            self.segmentation_head = MLP(
                in_channels=self.clf.in_channels,
                skip_last_activation=True,
                **_config.segmentation_head,
            )
        else:
            self.segmentation_head = None
            self.states = None

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, channels, seq_len)

        Returns
        -------
        pred: Tensor,
            of shape (batch_size, n_classes)
        outcome: Tensor, optional,
            of shape (batch_size, n_outcomes)
        segmentation: Tensor, optional,
            of shape (batch_size, seqlen, n_states)

        """
        batch_size, channels, seq_len = input.shape

        features = self.extract_features(input)

        if self.pool:
            pooled_features = self.pool(features)  # (batch_size, channels, pool_size)
            # features = features.squeeze(dim=-1)
            pooled_features = rearrange(
                pooled_features,
                "batch_size channels pool_size -> batch_size (channels pool_size)",
            )
        else:
            # pooled_features of shape (batch_size, channels) or (batch_size, seq_len, channels)
            pooled_features = features

        # print(f"clf in shape = {x.shape}")
        pred = self.clf(pooled_features)  # batch_size, n_classes

        if self.outcome_head is not None:
            outcome = self.outcome_head(pooled_features)
            return pred, outcome

        if self.segmentation_head is not None:
            features = features.permute(0, 2, 1)  # (batch_size, seq_len, channels)
            segmentation = self.segmentation_head(features)
            if self.config.segmentation_head.get("recover_length", True):
                segmentation = F.interpolate(
                    segmentation.permute(0, 2, 1),
                    size=seq_len,
                    mode="linear",
                    align_corners=True,
                ).permute(0, 2, 1)
            return pred, segmentation

        return pred

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
        seg_thr: float = 0.5,
    ) -> Union[
        ClassificationOutput,
        Tuple[ClassificationOutput, ClassificationOutput],
        Tuple[ClassificationOutput, SequenceLabelingOutput],
    ]:
        """

        auxiliary function to `forward`, for CINC2022,

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        class_names: bool, default False,
            if True, the returned scalar predictions will be a `DataFrame`,
            with class names for each scalar prediction
        seg_thr: float, default 0.5,
            threshold for making binary predictions for
            the optional segmentaion head

        Returns
        -------
        clf_output: ClassificationOutput, with items:
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
        outcome_output: ClassificationOutput, optional, with items:
            - classes: list of str,
                list of the outcome class names
            - prob: ndarray,
                scalar (probability) predictions,
            - pred: ndarray,
                the array of outcome class number predictions
        seg_output: SequenceLabelingOutput, optional, with items:
            - classes: list of str,
                list of the state class names
            - prob: ndarray,
                scalar (probability) predictions,
            - pred: ndarray,
                the array of binarized prediction

        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
        outcome, states = None, None
        if self.outcome_head is not None:
            forward_output, outcome = self.forward(_input)
        elif self.segmentation_head is not None:
            forward_output, states = self.forward(_input)
        else:
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

        clf_output = ClassificationOutput(
            classes=self.classes,
            prob=prob,
            pred=pred,
            bin_pred=bin_pred,
            forward_output=forward_output,
        )

        if outcome is not None:
            prob = self.softmax(outcome)
            pred = torch.argmax(prob, dim=-1)
            bin_pred = (prob == prob.max(dim=-1, keepdim=True).values).to(int)
            prob = prob.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            bin_pred = bin_pred.cpu().detach().numpy()
            outcome = outcome.cpu().detach().numpy()
            outcome_output = ClassificationOutput(
                classes=self.config.outcomes,
                prob=prob,
                pred=pred,
                bin_pred=bin_pred,
                outcome=outcome,
            )
            return clf_output, outcome_output

        if states is not None:
            # if "unannotated" in self.config.states, use softmax
            # else use sigmoid
            if "unannotated" in self.states:
                prob = self.softmax(states)
                pred = torch.argmax(prob, dim=-1)
            else:
                prob = self.sigmoid(states)
                pred = (prob > seg_thr).int() * (
                    prob == prob.max(dim=-1, keepdim=True).values
                ).int()
            prob = prob.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            seg_output = SequenceLabelingOutput(
                classes=self.states,
                prob=prob,
                pred=pred,
            )
            return clf_output, seg_output

        return clf_output

    @add_docstring(inference.__doc__)
    def inference_CINC2022(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
        seg_thr: float = 0.5,
    ) -> Union[
        ClassificationOutput,
        Tuple[ClassificationOutput, ClassificationOutput],
        Tuple[ClassificationOutput, SequenceLabelingOutput],
    ]:
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
    ) -> SequenceLabelingOutput:
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
        if "unannotated" in self.classes:
            prob = self.softmax(self.forward(_input))
            pred = torch.argmax(prob, dim=-1)
        else:
            prob = self.sigmoid(self.forward(_input))
            pred = (prob > bin_pred_threshold).int() * (
                prob == prob.max(dim=-1, keepdim=True)
            ).int()
        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        return SequenceLabelingOutput(
            classes=self.classes,
            prob=prob,
            pred=pred,
        )

    @add_docstring(inference.__doc__)
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
    ) -> SequenceLabelingOutput:
        """

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

    @add_docstring(inference.__doc__)
    def inference_CINC2022(
        self,
        input: Union[np.ndarray, Tensor],
    ) -> SequenceLabelingOutput:
        """
        alias for `self.inference`
        """
        return self.inference(input)
