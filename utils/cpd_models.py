"""Methods and modules for experiments with seq2seq modeld ('indid', 'bce' and 'combided')"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np

from sklearn.base import BaseEstimator


class ClassicBaseline(nn.Module):
    """Class for classic (from ruptures) Baseline models."""

    def __init__(
        self,
        model: BaseEstimator,
        pen: float = None,
        n_pred: int = None,
        device: str = "cpu",
    ) -> None:
        """Initialize ClassicBaseline model.

        :param model: core model (from ruptures)
        :param pen: penalty parameter (for ruptures models)
        :param n_pred: maximum number of predicted CPs (for ruptures models)
        :param device: 'cpu' or 'cuda' (if available)
        """
        super().__init__()
        self.device = device
        self.model = model
        self.pen = pen
        self.n_pred = n_pred

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get predictions of a ClassicBaseline models.

        :param inputs: input signal
        :return: tensor with predicted change point labels
        """
        all_predictions = []
        for i, seq in enumerate(inputs):
            # (n_samples, n_dims)
            try:
                signal = seq.flatten(1, 2).detach().cpu().numpy()
            except:
                signal = seq.detach().cpu().numpy()
            algo = self.model.fit(signal)
            cp_pred = []
            if self.pen is not None:
                cp_pred = self.model.predict(pen=self.pen)
            elif self.n_pred is not None:
                cp_pred = self.model.predict(self.n_pred)
            else:
                cp_pred = self.model.predict()
            cp_pred = cp_pred[0]
            baselines_pred = np.zeros(inputs.shape[1])
            baselines_pred[cp_pred:] = np.ones(inputs.shape[1] - cp_pred)
            all_predictions.append(baselines_pred)
        out = torch.from_numpy(np.array(all_predictions))
        return out


class CPDModel(pl.LightningModule):
    """Pytorch Lightning wrapper for change point detection models."""

    def __init__(
        self,
        loss_type: str,
        args: dict,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ) -> None:
        """Initialize CPD model.

        :param experiment_type: type of data used for training
        :param loss_type: type of loss function for training special CPD or common BCE loss
        :param args: dict with supplementary argumemnts
        :param model: base model
        :param train_dataset: train data
        :param test_dataset: test data
        """
        super().__init__()

        self.experiments_name = args["experiments_name"]
        self.model = model

        self.learning_rate = args["learning"]["lr"]
        self.batch_size = args["learning"]["batch_size"]
        self.num_workers = args["num_workers"]

        self.loss = nn.BCELoss()

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward step for CPD model.

        :param inputs: batch of data
        :return: predictions
        """
        return self.model(inputs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Train CPD model.

        :param batch: data for training
        :param batch_idx: index of batch (special for pytorch-lightning)
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())

        train_loss = self.loss(pred.squeeze(), labels.float().squeeze())
        train_accuracy = (
            ((pred.squeeze() > 0.5).long() == labels.squeeze()).float().mean()
        )

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", train_accuracy, prog_bar=True)

        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test CPD model.

        :param batch: data for validation
        :param batch_idx: index of batch (special for pytorch-lightning)
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())

        val_loss = self.loss(pred.squeeze(), labels.float().squeeze())
        val_accuracy = (
            ((pred.squeeze() > 0.5).long() == labels.squeeze()).float().mean()
        )

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_accuracy, prog_bar=True)

        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer.

        :return: optimizer for training CPD model
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return opt

    def train_dataloader(self) -> DataLoader:
        """Initialize dataloader for training.

        :return: dataloader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Initialize dataloader for validation.

        :return: dataloader for validation
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
