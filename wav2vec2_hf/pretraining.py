"""
pretrain a Wav2Vec model from scratch
modified from:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any, NoReturn

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator  # noqa: F401
from transformers import (  # noqa: F401
    AdamW,
    Trainer,
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    get_scheduler,
    is_wandb_available,
    set_seed,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (  # noqa: F401
    _compute_mask_indices,
    _sample_negative_indices,
)
from torch_ecg.components.trainer import BaseTrainer

from .pretraining_cfg import PreTrainCfg, PreTrainModelCfg  # noqa: F401


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

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(
            batch["input_values"].shape[-1]
        )
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(
            mask_time_indices, dtype=torch.long, device=device
        )
        batch["sampled_negative_indices"] = torch.tensor(
            sampled_negative_indices, dtype=torch.long, device=device
        )

        return batch


def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)


def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


class Wav2Vec2PreTrainingTrainer(BaseTrainer):
    """ """

    __name__ = "Wav2Vec2PreTrainingTrainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        raise NotImplementedError

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> NoReturn:
        """

        setup the dataloaders for training and validation

        Parameters
        ----------
        train_dataset: Dataset, optional,
            the training dataset
        val_dataset: Dataset, optional,
            the validation dataset

        """
        raise NotImplementedError

    def run_one_step(
        self, *data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        """

        Parameters
        ----------
        data: tuple of Tensors,
            the data to be processed for training one step (batch),
            should be of the following order:
            signals, labels, *extra_tensors

        Returns
        -------
        preds: Tensor,
            the predictions of the model for the given data
        labels: Tensor,
            the labels of the given data
        masks: Tensor, optional,
            the state masks of the given data, if available

        """
        raise NotImplementedError

        # compute num of losses
        # num_losses = batch["mask_time_indices"].sum()
        # sub_attention_mask = batch.pop("sub_attention_mask", None)
        # sub_attention_mask = (
        #     sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
        # )
        # percent_masked = num_losses / sub_attention_mask.sum()

        # # forward
        # outputs = model(**batch)

        # # divide loss by gradient accumulation steps since gradients
        # # are accumulated for multiple backward passes in PyTorch
        # loss = outputs.loss / args.gradient_accumulation_steps
        # accelerator.backward(loss)

        # # make sure that `num_losses` is summed for distributed training
        # # and average gradients over losses of all devices
        # if accelerator.state.num_processes > 1:
        #     num_losses = accelerator.gather(num_losses).sum()
        #     gradient_multiplier = accelerator.state.num_processes / num_losses
        #     multiply_grads(model.module.parameters(), gradient_multiplier)
        # else:
        #     multiply_grads(model.parameters(), 1 / num_losses)

        # # update step
        # if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

        #     # compute grad norm for monitoring
        #     scale = (
        #         accelerator.scaler._scale.item()
        #         if hasattr(accelerator, "scaler") and accelerator.scaler is not None
        #         else 1
        #     )
        #     if accelerator.state.num_processes > 1:
        #         grad_norm = get_grad_norm(model.module.parameters(), scale)
        #     else:
        #         grad_norm = get_grad_norm(model.parameters(), scale)

        #     # update parameters
        #     optimizer.step()
        #     optimizer.zero_grad()

        #     if not accelerator.optimizer_step_was_skipped:
        #         lr_scheduler.step()
        #     elif accelerator.is_local_main_process:
        #         progress_bar.write(
        #             f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
        #         )

        #     # update gumbel temperature
        #     gumbel_temperature = max(
        #         args.max_gumbel_temperature * args.gumbel_temperature_decay**completed_steps,
        #         args.min_gumbel_temperature,
        #     )
        #     if hasattr(model, "module"):
        #         model.module.set_gumbel_temperature(gumbel_temperature)
        #     else:
        #         model.set_gumbel_temperature(gumbel_temperature)

        #     progress_bar.update(1)
        #     completed_steps += 1

        # # 6. Log all results
        # if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0:
        #     loss.detach()
        #     outputs.contrastive_loss.detach()
        #     outputs.diversity_loss.detach()

        #     if accelerator.state.num_processes > 1:
        #         loss = accelerator.gather(loss).sum()
        #         outputs.contrastive_loss = accelerator.gather(outputs.contrastive_loss).sum()
        #         outputs.diversity_loss = accelerator.gather(outputs.diversity_loss).sum()
        #         percent_masked = accelerator.gather(percent_masked).sum()

        #     train_logs = {
        #         "loss": (loss * args.gradient_accumulation_steps) / num_losses,
        #         "constrast_loss": outputs.contrastive_loss / num_losses,
        #         "div_loss": outputs.diversity_loss / num_losses,
        #         "%_mask_idx": percent_masked / accelerator.num_processes,
        #         "ppl": outputs.codevector_perplexity,
        #         "lr": torch.tensor(optimizer.param_groups[0]["lr"]),
        #         "temp": torch.tensor(gumbel_temperature),
        #         "grad_norm": torch.tensor(grad_norm),
        #     }
        #     log_str = ""
        #     for k, v in train_logs.items():
        #         log_str += "| {}: {:.3e}".format(k, v.item())

        #     if accelerator.is_local_main_process:
        #         progress_bar.write(log_str)
        #         if is_wandb_available():
        #             wandb.log(train_logs)

        # # save model every `args.saving_steps` steps
        # if (step + 1) % (args.gradient_accumulation_steps * args.saving_steps) == 0:
        #     if (args.push_to_hub and epoch < args.num_train_epochs - 1) or args.output_dir is not None:
        #         accelerator.wait_for_everyone()
        #         unwrapped_model = accelerator.unwrap_model(model)
        #         unwrapped_model.save_pretrained(
        #             args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        #         )

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
        raise NotImplementedError
        # init logs
        # val_logs = {
        #     "val_loss": 0,
        #     "val_contrastive_loss": 0,
        #     "val_diversity_loss": 0,
        #     "val_num_losses": 0,
        # }
        # for step, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         batch.pop("sub_attention_mask", None)
        #         outputs = model(**batch)

        #     val_logs["val_loss"] += outputs.loss
        #     val_logs["val_contrastive_loss"] += outputs.contrastive_loss
        #     val_logs["val_diversity_loss"] += outputs.diversity_loss
        #     val_logs["val_num_losses"] += batch["mask_time_indices"].sum()

        # # sum over devices in multi-processing
        # if accelerator.num_processes > 1:
        #     val_logs = {k: accelerator.gather(v).sum() for k, v in val_logs.items()}

        # val_logs = {k: v / val_logs["val_num_losses"] for k, v in val_logs.items()}

        # log_str = ""
        # for k, v in val_logs.items():
        #     log_str += "| {}: {:.3e}".format(k, v.item())

        # if accelerator.is_local_main_process:
        #     progress_bar.write(log_str)
        #     if is_wandb_available():
        #         wandb.log(val_logs)

        # if args.output_dir is not None:
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.save_pretrained(
        #         args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        #     )

    def _setup_optimizer(self) -> NoReturn:
        """ """
        raise NotImplementedError

    def _setup_criterion(self) -> NoReturn:
        """ """
        raise NotImplementedError

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, usually 0,
        but can be 1 for some models, e.g. RR_LSTM
        """
        raise NotImplementedError

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        """ """
        raise NotImplementedError

    @property
    def save_prefix(self) -> str:
        raise NotImplementedError

    def extra_log_suffix(self) -> str:
        raise NotImplementedError
