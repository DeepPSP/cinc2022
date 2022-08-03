"""
"""

import os
import sys
import argparse
import textwrap
from copy import deepcopy
from typing import Any, Optional, Tuple, NoReturn, Dict, List, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import (  # noqa: F401
    DistributedDataParallel as DDP,
    DataParallel as DP,
)  # noqa: F401

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm

from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.misc import str2bool
from torch_ecg.utils.utils_nn import default_collate_fn
from torch_ecg.utils.utils_data import mask_to_intervals  # noqa: F401

from models import (  # noqa: F401
    Wav2Vec2_CINC2022,
    CRNN_CINC2022,
    SEQ_LAB_NET_CINC2022,
    UNET_CINC2022,
)
from cfg import BaseCfg, TrainCfg, ModelCfg
from dataset import CinC2022Dataset
from utils.scoring_metrics import compute_metrics  # noqa: F401
from utils.augmentations import AugmenterManager

if BaseCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "CINC2022Trainer",
]


class CINC2022Trainer(BaseTrainer):
    """ """

    __DEBUG__ = True
    __name__ = "CINC2022Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        model: Module,
            the model to be trained
        model_config: dict,
            the configuration of the model,
            used to keep a record in the checkpoints
        train_config: dict,
            the configuration of the training,
            including configurations for the data loader, for the optimization, etc.
            will also be recorded in the checkpoints.
            `train_config` should at least contain the following keys:
                "monitor": str,
                "loss": str,
                "n_epochs": int,
                "batch_size": int,
                "learning_rate": float,
                "lr_scheduler": str,
                    "lr_step_size": int, optional, depending on the scheduler
                    "lr_gamma": float, optional, depending on the scheduler
                    "max_lr": float, optional, depending on the scheduler
                "optimizer": str,
                    "decay": float, optional, depending on the optimizer
                    "momentum": float, optional, depending on the optimizer
        device: torch.device, optional,
            the device to be used for training
        lazy: bool, default True,
            whether to initialize the data loader lazily

        """
        super().__init__(
            model, CinC2022Dataset, model_config, train_config, device, lazy
        )

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
        if train_dataset is None:
            train_dataset = self.dataset_cls(
                config=self.train_config,
                task=self.train_config.task,
                training=True,
                lazy=False,
            )

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None
        if val_dataset is None:
            val_dataset = self.dataset_cls(
                config=self.train_config,
                task=self.train_config.task,
                training=False,
                lazy=False,
            )

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
        num_workers = 4

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        if self.train_config.debug:
            self.val_train_loader = DataLoader(
                dataset=val_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            self.val_train_loader = None
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def _setup_augmenter_manager(self) -> NoReturn:
        """ """
        self.augmenter_manager = AugmenterManager.from_config(config=self.train_config)

    def train_one_epoch(self, pbar: tqdm) -> NoReturn:
        """
        train one epoch, and update the progress bar

        Parameters
        ----------
        pbar: tqdm,
            the progress bar for training

        """
        for epoch_step, input_tensors in enumerate(self.train_loader):
            self.global_step += 1
            # input_tensors is assumed to be a dict of tensors, with the following items:
            # "waveforms" (required): the input waveforms
            # "murmur" (optional): the murmur labels, for classification task and multi task
            # "outcome" (optional): the outcome labels, for classification task and multi task
            # "segmentation" (optional): the segmentation labels, for segmentation task and multi task
            input_tensors["waveforms"] = self.augmenter_manager(
                input_tensors["waveforms"]
            )

            # out_tensors is a dict of tensors, with the following items:
            # - "murmur": the murmur predictions, of shape (batch_size, n_classes)
            # - "outcome": the outcome predictions, of shape (batch_size, n_outcomes)
            # - "segmentation": the segmentation predictions, of shape (batch_size, seq_len, n_states)
            # - "outcome_loss": loss of the outcome predictions
            # - "segmentation_loss": loss of the segmentation predictions
            # - "total_extra_loss": total loss of the extra heads
            out_tensors = self.run_one_step(input_tensors)

            loss = self.criterion(
                out_tensors[self._criterion_key], input_tensors[self._criterion_key]
            ).to(self.dtype) + out_tensors.get(
                "total_extra_loss", torch.tensor(0.0)
            ).to(
                self.dtype
            )
            if self.train_config.flooding_level > 0:
                flood = (
                    loss - self.train_config.flooding_level
                ).abs() + self.train_config.flooding_level
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                flood.backward()
            else:
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
            self.optimizer.step()
            self._update_lr()

            if self.global_step % self.train_config.log_step == 0:
                train_step_metrics = {"loss": loss.item()}
                if self.scheduler:
                    train_step_metrics.update({"lr": self.scheduler.get_last_lr()[0]})
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                            "lr": self.scheduler.get_last_lr()[0],
                        }
                    )
                else:
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                        }
                    )
                if self.train_config.flooding_level > 0:
                    train_step_metrics.update({"flood": flood.item()})
                self.log_manager.log_metrics(
                    metrics=train_step_metrics,
                    step=self.global_step,
                    epoch=self.epoch,
                    part="train",
                )
            pbar.update(input_tensors["waveforms"].shape[self.batch_dim])

    def run_one_step(
        self, input_tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """

        Parameters
        ----------
        input_tensors: dict of Tensors,
            the tensors to be processed for training one step (batch), with the following items:
                - "waveforms" (required): the input waveforms
                - "murmur" (optional): the murmur labels, for classification task and multi task
                - "outcome" (optional): the outcome labels, for classification task and multi task
                - "segmentation" (optional): the segmentation labels, for segmentation task and multi task

        Returns
        -------
        out_tensors: dict of Tensors, with the following items:
            - "murmur": the murmur predictions, of shape (batch_size, n_classes)
            - "outcome": the outcome predictions, of shape (batch_size, n_outcomes)
            - "segmentation": the segmentation predictions, of shape (batch_size, seq_len, n_states)
            - "outcome_loss": loss of the outcome predictions
            - "segmentation_loss": loss of the segmentation predictions
            - "total_extra_loss": total loss of the extra heads

        """
        input_tensors = {k: v.to(self.device) for k, v in input_tensors.items()}
        waveforms = input_tensors.pop("waveforms")
        out_tensors = self.model(waveforms, input_tensors)
        return out_tensors

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
        if self.train_config.task == "multi_task":
            return self.evaluate_multi_task(data_loader)

        self.model.eval()

        all_outputs = []
        all_labels = []

        for input_tensors in data_loader:
            # input_tensors is assumed to be a dict of tensors, with the following items:
            # "waveforms" (required): the input waveforms
            # "murmur" (optional): the murmur labels, for classification task and multi task
            # "outcome" (optional): the outcome labels, for classification task and multi task
            # "segmentation" (optional): the segmentation labels, for segmentation task and multi task
            waveforms = input_tensors.pop("waveforms")
            waveforms = waveforms.to(device=self.device, dtype=self.dtype)
            labels = {k: v.numpy() for k, v in input_tensors.items() if v is not None}

            all_labels.append(labels)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            all_outputs.append(self._model.inference(waveforms))

        if self.val_train_loader is not None and self.train_config.task not in [
            "segmentation"
        ]:
            head_num = 5
            head_scalar_preds = [
                all_outputs[idx].murmur_outputs.prob for idx in range(head_num)
            ]
            head_bin_preds = [
                all_outputs[idx].murmur_outputs.bin_pred for idx in range(head_num)
            ]
            head_preds_classes = [
                np.array(all_outputs[0].murmur_outputs.classes)[np.where(row)]
                for row in head_bin_preds
            ]
            head_labels = [all_labels[idx]["murmur"] for idx in range(head_num)]
            head_labels_classes = [
                np.array(all_outputs[0].murmur_outputs.classes)[np.where(row)]
                for row in head_labels
            ]
            for n in range(head_num):
                msg = textwrap.dedent(
                    f"""
                ----------------------------------------------
                murmur scalar prediction:    {[round(item, 3) for item in head_scalar_preds[n].tolist()]}
                murmur binary prediction:    {head_bin_preds[n].tolist()}
                murmur labels:               {head_labels[n].astype(int).tolist()}
                murmur predicted classes:    {head_preds_classes[n].tolist()}
                murmur label classes:        {head_labels_classes[n].tolist()}
                ----------------------------------------------
                """
                )
                self.log_manager.log_message(msg)
            if "outcome" in input_tensors:
                head_scalar_preds = [
                    all_outputs[idx].outcome_outputs.prob for idx in range(head_num)
                ]
                head_bin_preds = [
                    all_outputs[idx].outcome_outputs.bin_pred for idx in range(head_num)
                ]
                head_preds_classes = [
                    np.array(all_outputs[0].outcome_outputs.classes)[np.where(row)]
                    for row in head_bin_preds
                ]
                head_labels = [all_labels[idx]["outcome"] for idx in range(head_num)]
                head_labels_classes = [
                    np.array(all_outputs[0].outcome_outputs.classes)[np.where(row)]
                    for row in head_labels
                ]
                for n in range(head_num):
                    msg = textwrap.dedent(
                        f"""
                    ----------------------------------------------
                    outcome scalar prediction:    {[round(item, 3) for item in head_scalar_preds[n].tolist()]}
                    outcome binary prediction:    {head_bin_preds[n].tolist()}
                    outcome labels:               {head_labels[n].astype(int).tolist()}
                    outcome predicted classes:    {head_preds_classes[n].tolist()}
                    outcome label classes:        {head_labels_classes[n].tolist()}
                    ----------------------------------------------
                    """
                    )
                    self.log_manager.log_message(msg)

        # TODO: correct the following code
        raise NotImplementedError
        # metrics = compute_metrics(
        #     classes=classes,
        #     labels=all_labels,
        #     scalar_outputs=all_scalar_preds,
        #     binary_outputs=all_bin_preds,
        # )
        # eval_res = dict(
        #     auroc=metrics["auroc"],
        #     auprc=metrics["auprc"],
        #     f_measure=metrics["f_measure"],
        #     accuracy=metrics["accuracy"],
        #     weighted_accuracy=metrics["weighted_accuracy"],
        #     challenge_cost=metrics["challenge_cost"],
        #     task_score=metrics["task_score"],
        #     neg_challenge_metric=-metrics["task_score"],
        # )

        # # in case possible memeory leakage?
        # del all_scalar_preds, all_bin_preds, all_labels

        # self.model.train()

        # return eval_res

    def evaluate_multi_task(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
        self.model.eval()

        all_scalar_preds = []
        all_bin_preds = []
        all_aux_preds = []
        all_labels = []
        all_masks = []

        for signals, labels, masks in data_loader:
            signals = signals.to(device=self.device, dtype=self.dtype)
            labels = labels.numpy()
            masks = masks.numpy()
            all_labels.append(labels)
            all_masks.append(masks)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_output, aux_output = self._model.inference(signals)
            all_scalar_preds.append(model_output.prob)
            all_bin_preds.append(model_output.bin_pred)

        raise NotImplementedError

    def _setup_optimizer(self) -> NoReturn:
        """ """
        # TODO: adjust for multi-task
        if self.train_config.task == "multi_task":
            raise NotImplementedError
        super()._setup_optimizer()

    def _setup_criterion(self) -> NoReturn:
        """ """
        # TODO: adjust for multi-task
        if self.train_config.task == "multi_task":
            raise NotImplementedError
        super()._setup_criterion()

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, usually 0,
        but can be 1 for some models, e.g. RR_LSTM
        """
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        """ """
        return [
            "task",
        ]

    @property
    def save_prefix(self) -> str:
        return f"task-{self.train_config.task}_{self._model.__name__}_{self.model_config.cnn_name}_epoch"

    def extra_log_suffix(self) -> str:
        return f"task-{self.train_config.task}_{super().extra_log_suffix()}_{self.model_config.cnn_name}"

    @property
    def _criterion_key(self) -> str:
        return {
            "multi_task": "murmur",
            "classification": "murmur",
            "segmentation": "segmentation",
        }[self.train_config.task]


def collate_fn(
    batch: Sequence[Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray]]]
) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    """ """
    if isinstance(batch[0], dict):
        keys = batch[0].keys()
        collated = default_collate_fn([tuple(b[k] for k in keys) for b in batch])
        return {k: collated[i] for i, k in enumerate(keys)}
    else:
        return default_collate_fn(batch)


def get_args(**kwargs: Any):
    """NOT checked,"""
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train the Model on CINC2022 database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="the batch size for training",
        dest="batch_size",
    )
    # parser.add_argument(
    #     "-c", "--cnn-name",
    #     type=str, default="multi_scopic_leadwise",
    #     help="choice of cnn feature extractor",
    #     dest="cnn_name")
    # parser.add_argument(
    #     "-r", "--rnn-name",
    #     type=str, default="none",
    #     help="choice of rnn structures",
    #     dest="rnn_name")
    # parser.add_argument(
    #     "-a", "--attn-name",
    #     type=str, default="se",
    #     help="choice of attention structures",
    #     dest="attn_name")
    parser.add_argument(
        "--keep-checkpoint-max",
        type=int,
        default=10,
        help="maximum number of checkpoints to keep. If set 0, all checkpoints will be kept",
        dest="keep_checkpoint_max",
    )
    # parser.add_argument(
    #     "--optimizer", type=str, default="adam",
    #     help="training optimizer",
    #     dest="train_optimizer")
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="train with more debugging information",
        dest="debug",
    )

    args = vars(parser.parse_args())

    cfg.update(args)

    return CFG(cfg)


_MODEL_MAP = {
    "wav2vec2": Wav2Vec2_CINC2022,
    "crnn": CRNN_CINC2022,
    "seq_lab": SEQ_LAB_NET_CINC2022,
    "unet": UNET_CINC2022,
}


def _set_task(task: str, config: CFG) -> NoReturn:
    """"""
    assert task in config.tasks
    config.task = task
    for item in [
        "classes",
        "monitor",
        "final_model_name",
        "loss",
    ]:
        config[item] = config[task][item]


if __name__ == "__main__":
    # WARNING: most training were done in notebook,
    # NOT in cli
    train_config = get_args(**TrainCfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: adjust for CINC2022
    for task in train_config.tasks:
        _set_task(task, train_config)
        model_config = deepcopy(ModelCfg[task])
        model_config = deepcopy(ModelCfg[task])

        # adjust model choices if needed
        model_name = model_config.model_name = train_config[task].model_name
        if "cnn" in model_config[model_name]:
            model_config[model_name].cnn.name = train_config[task].cnn_name
        if "rnn" in model_config[model_name]:
            model_config[model_name].rnn.name = train_config[task].rnn_name
        if "attn" in model_config[model_name]:
            model_config[model_name].attn.name = train_config[task].attn_name

        model_cls = _MODEL_MAP[train_config[task].model_name]
        model_cls.__DEBUG__ = False
        model = model_cls(config=model_config)

        if torch.cuda.device_count() > 1:
            model = DP(model)
            # model = DDP(model)
        model.to(device=device)

        trainer = CINC2022Trainer(
            model=model,
            model_config=model_config,
            train_config=train_config,
            device=device,
            lazy=False,
        )

        try:
            best_model_state_dict = trainer.train()
        except KeyboardInterrupt:
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
