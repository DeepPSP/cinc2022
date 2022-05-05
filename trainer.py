"""
"""

import os
import sys
import argparse
import logging
import textwrap
from copy import deepcopy
from typing import Any, Optional, Tuple, NoReturn, Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import (  # noqa: F401
    DistributedDataParallel as DDP,
    DataParallel as DP,
)  # noqa: F401

from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.misc import str2bool
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from torch_ecg.utils.utils_data import mask_to_intervals  # noqa: F401

from model import (  # noqa: F401
    Wav2Vec2_CINC2022,
    CRNN_CINC2022,
    SEQ_LAB_NET_CINC2022,
    UNET_CINC2022,
)
from cfg import BaseCfg, TrainCfg, ModelCfg
from dataset import CinC2022Dataset
from utils.scoring_metrics import compute_metrics

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
        signals, *labels = data
        signals = signals.to(device=self.device, dtype=self.dtype)
        labels = (lb.to(device=self.device, dtype=self.dtype) for lb in labels)
        # print(f"signals: {signals.shape}")
        # print(f"labels: {labels.shape}")
        preds = self.model(signals)
        return (preds, *labels)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
        if self.train_config.task == "multi_task":
            return self.evaluate_multi_task(data_loader)

        self.model.eval()

        all_scalar_preds = []
        all_bin_preds = []
        all_labels = []

        for signals, labels in data_loader:
            signals = signals.to(device=self.device, dtype=self.dtype)
            labels = labels.numpy()
            all_labels.append(labels)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_output = self._model.inference(signals)
            all_scalar_preds.append(model_output.prob)
            all_bin_preds.append(model_output.bin_pred)

        all_scalar_preds = np.concatenate(all_scalar_preds, axis=0)
        all_bin_preds = np.concatenate(all_bin_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        classes = data_loader.dataset.classes

        if self.val_train_loader is not None:
            msg = f"all_scalar_preds.shape = {all_scalar_preds.shape}, all_labels.shape = {all_labels.shape}"
            self.log_manager.log_message(msg, level=logging.DEBUG)
            head_num = 5
            head_scalar_preds = all_scalar_preds[:head_num, ...]
            head_bin_preds = all_bin_preds[:head_num, ...]
            head_preds_classes = [
                np.array(classes)[np.where(row)] for row in head_bin_preds
            ]
            head_labels = all_labels[:head_num, ...]
            head_labels_classes = [
                np.array(classes)[np.where(row)] for row in head_labels
            ]
            for n in range(head_num):
                msg = textwrap.dedent(
                    f"""
                ----------------------------------------------
                scalar prediction:    {[round(n, 3) for n in head_scalar_preds[n].tolist()]}
                binary prediction:    {head_bin_preds[n].tolist()}
                labels:               {head_labels[n].astype(int).tolist()}
                predicted classes:    {head_preds_classes[n].tolist()}
                label classes:        {head_labels_classes[n].tolist()}
                ----------------------------------------------
                """
                )
                self.log_manager.log_message(msg)

        (auroc, auprc, accuracy, f_measure, challenge_metric,) = compute_metrics(
            classes=classes,
            labels=all_labels,
            scalar_outputs=all_scalar_preds,
            binary_outputs=all_bin_preds,
        )
        eval_res = dict(
            auroc=auroc,
            auprc=auprc,
            accuracy=accuracy,
            f_measure=f_measure,
            challenge_metric=challenge_metric,
            neg_challenge_metric=-challenge_metric,
        )

        # in case possible memeory leakage?
        del all_scalar_preds, all_bin_preds, all_labels

        self.model.train()

        return eval_res

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
        raise NotImplementedError

    def _setup_criterion(self) -> NoReturn:
        """ """
        # TODO: adjust for multi-task
        raise NotImplementedError

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
