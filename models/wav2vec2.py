"""
"""

from copy import deepcopy
from typing import Union, Optional, NoReturn, Any, Tuple, Dict

import numpy as np
from einops.layers.torch import Rearrange
import torch
from torch import Tensor
from torch_ecg.cfg import CFG
from torch_ecg.models.cnn.vgg import VGG16
from torch_ecg.models.cnn.resnet import ResNet
from torch_ecg.models.cnn.multi_scopic import MultiScopicCNN
from torch_ecg.models.cnn.densenet import DenseNet
from torch_ecg.models._nets import MLP
from torch_ecg.components.outputs import (
    ClassificationOutput,
    SequenceLabellingOutput,
)
from torch_ecg.utils import add_docstring

from cfg import ModelCfg
from wav2vec2_ta import Wav2Vec2Model, components as w2v2_components
from .heads import MultiTaskHead
from .outputs import CINC2022Outputs


__all__ = ["Wav2Vec2_CINC2022"]


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
        if config is None:
            _config = deepcopy(ModelCfg.classification)
        else:
            _config = deepcopy(config)
        self.config = _config[_config.model_name]
        assert "encoder" in self.config, "encoder is a required key in config"
        if "rnn" in self.config:
            self.config.pop("rnn", None)
        if "attn" in self.config:
            self.config.pop("attn", None)
        self.classes = deepcopy(_config.classes)
        self.n_classes = len(_config.classes)

        self.outcomes = _config.outcomes
        self.states = _config.states

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

        super().__init__(cnn, encoder)

        if self.config.global_pool.lower() == "max":
            pool = torch.nn.AdaptiveMaxPool1d((1,), return_indices=False)
        elif self.config.global_pool.lower() == "avg":
            pool = torch.nn.AdaptiveAvgPool1d((1,))
        elif self.config.global_pool.lower() == "attn":
            raise NotImplementedError("Attentive pooling not implemented yet!")

        self.pool = torch.nn.Sequential(
            Rearrange("batch seqlen chan -> batch chan seqlen"),
            pool,
            Rearrange("batch chan seqlen -> batch (chan seqlen)"),
        )

        self.clf = MLP(
            in_channels=encoder_config.embed_dim,
            out_channels=self.config.clf.out_channels + [self.n_classes],
            activation=self.config.clf.activation,
            bias=self.config.clf.bias,
            dropouts=self.config.clf.dropouts,
            skip_last_activation=True,
        )

        self.extra_heads = MultiTaskHead(
            in_channels=self.clf.in_channels,
            config=_config,
        )

        if "wav2vec2" in cnn_choice:
            self.squeeze = Rearrange("batch 1 seqlen -> batch seqlen")

        # for inference
        # classification: if single-label, use softmax; otherwise (multi-label) use sigmoid
        # sequence tagging: if "unannotated" counted in `classes`, use softmax; otherwise use sigmoid
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(-1)

    def forward(
        self,
        waveforms: Tensor,
        labels: Optional[Dict[str, Tensor]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """

        Parameters
        ----------
        waveforms: Tensor,
            shape: (batch, num_channels, seq_len)
        labels: dict of Tensor, optional,
            the labels of the input data, including:
            - "outcome": the outcome labels, of shape (batch_size, n_outcomes)
            - "segmentation": the segmentation labels, of shape (batch_size, seq_len, n_states)

        Returns
        -------
        dict of Tensor, with items:
            - "murmur": the murmur predictions, of shape (batch_size, n_classes)
            - "outcome": the outcome predictions, of shape (batch_size, n_outcomes)
            - "segmentation": the segmentation predictions, of shape (batch_size, seq_len, n_states)
            - "outcome_loss": loss of the outcome predictions
            - "segmentation_loss": loss of the segmentation predictions

        """
        batch_size, channels, seq_len = waveforms.shape

        if self.squeeze is not None:
            waveforms = self.squeeze(waveforms)
        features, _ = self.feature_extractor(waveforms, seq_len)
        features = self.encoder(features)
        pooled_features = self.pool(features)

        pred = self.clf(pooled_features)

        out = self.extra_heads(
            features.permute(0, 2, 1), pooled_features, seq_len, labels
        )
        out["murmur"] = pred

        return out

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
        seg_thr: float = 0.5,
    ) -> CINC2022Outputs:
        """
        auxiliary function to `forward`, for CINC2022,

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        seg_thr: float, default 0.5,
            threshold for making binary predictions for
            the optional segmentaion head

        Returns
        -------
         Returns
        -------
        CINC2022Outputs, with attributes:
            - murmur_output: ClassificationOutput, with items:
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
            segmentation_output: SequenceLabellingOutput, optional, with items:
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
        forward_output = self.forward(_input)

        prob = self.softmax(forward_output["murmur"])
        pred = torch.argmax(prob, dim=-1)
        bin_pred = (prob == prob.max(dim=-1, keepdim=True).values).to(int)
        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        bin_pred = bin_pred.cpu().detach().numpy()

        murmur_output = ClassificationOutput(
            classes=self.classes,
            prob=prob,
            pred=pred,
            bin_pred=bin_pred,
            forward_output=forward_output["murmur"].cpu().detach().numpy(),
        )

        if forward_output.get("outcome", None) is not None:
            prob = self.softmax(forward_output["outcome"])
            pred = torch.argmax(prob, dim=-1)
            bin_pred = (prob == prob.max(dim=-1, keepdim=True).values).to(int)
            prob = prob.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            bin_pred = bin_pred.cpu().detach().numpy()
            outcome_output = ClassificationOutput(
                classes=self.outcomes,
                prob=prob,
                pred=pred,
                bin_pred=bin_pred,
                forward_output=forward_output["outcome"],
            )
        else:
            outcome_output = None

        if forward_output.get("segmentation", None) is not None:
            # if "unannotated" in self.states, use softmax
            # else use sigmoid
            if "unannotated" in self.states:
                prob = self.softmax(forward_output["segmentation"])
                pred = torch.argmax(prob, dim=-1)
            else:
                prob = self.sigmoid(forward_output["segmentation"])
                pred = (prob > seg_thr).int() * (
                    prob == prob.max(dim=-1, keepdim=True).values
                ).int()
            prob = prob.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            segmentation_output = SequenceLabellingOutput(
                classes=self.states,
                prob=prob,
                pred=pred,
                forward_output=forward_output["segmentation"],
            )
        else:
            segmentation_output = None

        return CINC2022Outputs(murmur_output, outcome_output, segmentation_output)

    @add_docstring(inference.__doc__)
    def inference_CINC2022(
        self,
        input: Union[np.ndarray, Tensor],
        seg_thr: float = 0.5,
    ) -> CINC2022Outputs:
        """
        alias for `self.inference`
        """
        return self.inference(input, seg_thr)
