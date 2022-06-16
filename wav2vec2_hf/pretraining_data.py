"""
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Config
from transformers.pytorch_utils import torch_int_div
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)
from torch_ecg.cfg import CFG

from .pretraining_cfg import PreTrainModelCfg


__all__ = [
    "DataCollatorForWav2Vec2Pretraining",
    "get_pretraining_datacollator",
]


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.
    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    config: Wav2Vec2Config
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        reformat list to dict and set to pytorch format

        Parameters
        ----------
        features : List[Dict[str, Union[List[int], torch.Tensor]]]
            List of features to be processed.
            features are dicts with the following keys:
                - "input_values": List (Tensor) of input values.
                - "attention_mask": Optional (List or Tensor) of attention mask.

        """
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self._get_feat_extract_output_lengths(
            batch["input_values"].shape[-1]
        )
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.config.mask_time_prob,
            self.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(
            mask_time_indices, dtype=torch.long, device=device
        )
        batch["sampled_negative_indices"] = torch.tensor(
            sampled_negative_indices, dtype=torch.long, device=device
        )

        return batch

    def _get_feature_vector_attention_mask(
        self,
        feature_vector_length: int,
        attention_mask: torch.LongTensor,
        add_adapter: Optional[bool] = None,
    ) -> torch.LongTensor:
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(
            non_padded_lengths, add_adapter=add_adapter
        )
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[
            (
                torch.arange(attention_mask.shape[0], device=attention_mask.device),
                output_lengths - 1,
            )
        ] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def _get_feat_extract_output_lengths(
        self,
        input_lengths: Union[torch.LongTensor, int],
        add_adapter: Optional[bool] = None,
    ) -> int:
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length: int, kernel_size: int, stride: int) -> int:
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch_int_div(input_length - kernel_size, stride) + 1

        for kernel_size, stride in zip(
            self.config.conv_kernel, self.config.conv_stride
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(
                    input_lengths, 1, self.config.adapter_stride
                )

        return input_lengths


def get_pretraining_datacollator(
    cfg: Optional[CFG] = None,
) -> DataCollatorForWav2Vec2Pretraining:
    """ """
    if cfg is None:
        cfg = PreTrainModelCfg
    assert hasattr(cfg, "get_Wav2Vec2Config"), "cfg must have get_Wav2Vec2Config method"
    assert hasattr(
        cfg, "get_Wav2Vec2FeatureExtractor"
    ), "cfg must have get_Wav2Vec2FeatureExtractor method"

    extra_options = {
        "padding": cfg.get("padding", None),
        "max_length": cfg.get("max_length", None),
        "pad_to_multiple_of": cfg.get("pad_to_multiple_of", None),
    }
    extra_options = {k: v for k, v in extra_options.items() if v is not None}
    return DataCollatorForWav2Vec2Pretraining(
        cfg.get_Wav2Vec2Config(),
        cfg.get_Wav2Vec2FeatureExtractor(),
        **extra_options,
    )
