import math
import random
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn

from framework.FaultSegV3 import FaultSegV3
from seisDataset.gpu_aug import FaultSegGPUAugmentor

DEFAULT_WEIGHTS = Path(__file__).with_name("faultsegv3.ckpt")


class FaultSeg(pl.LightningModule):
    def __init__(
            self,
            restore_path=DEFAULT_WEIGHTS,
            reg_factor=2.0,
            tv_factor=0.0,
            start_lr=1.5e-4,
            end_lr=5e-6,
            decay_steps=100000,
            cosine=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = FaultSegV3(c=8, drop_path_rate=0.1)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.decay_steps = decay_steps
        self.cosine = cosine
        self.reg_factor = reg_factor
        self.tv_factor = tv_factor
        self.gpu_aug = FaultSegGPUAugmentor()

        if restore_path:
            self._load_weights(restore_path)

    def _load_weights(self, restore_path):
        checkpoint = torch.load(restore_path, map_location="cpu", weights_only=True)
        source_weights = checkpoint.get("model", checkpoint)
        model_weights = self.model.state_dict()
        compatible = {
            name: value
            for name, value in source_weights.items()
            if name in model_weights and value.shape == model_weights[name].shape
        }
        if not compatible:
            raise ValueError(f"no compatible model weights found in {restore_path}")
        skipped = sorted(set(source_weights).difference(compatible))
        model_weights.update(compatible)
        self.model.load_state_dict(model_weights)
        print(f"loaded {len(compatible)} tensors from {restore_path}; skipped: {skipped}")

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if isinstance(batch, dict) and "samples" in batch:
            return self.gpu_aug(batch)
        return batch

    def forward(self, seismic, rank=None, chunk_size=64):
        """Use raw forward in training and Rank4 by default in evaluation."""
        return self.model(seismic, rank=rank, chunk_size=chunk_size)

    def training_step(self, batch, batch_idx):
        seis, target = batch
        if random.randint(0, 1):
            seis = seis.permute(0, 1, 2, 4, 3)
            target = target.permute(0, 1, 2, 4, 3)

        self.last_batch = {"seis": seis.detach(), "target": target.detach()}
        prediction = self.model.forward_raw(seis)
        loss = MaskDice()(prediction, target)
        self.log("seg_loss", loss, prog_bar=True)
        self.last_pred_target = prediction
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.start_lr)

        def lr_multiplier(step):
            if step >= self.decay_steps:
                return self.end_lr / self.start_lr
            cosine = 0.5 * (1 + math.cos(math.pi * step / self.decay_steps))
            learning_rate = self.end_lr + (self.start_lr - self.end_lr) * cosine
            return learning_rate / self.start_lr

        if not self.cosine:
            return optimizer

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class MaskDice(nn.Module):
    def __init__(self, smooth=1, alpha=0.5, ignore=-1.0):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.ignore = ignore

    def forward(self, prediction, target):
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("prediction and target batch sizes differ")

        prediction = prediction.contiguous().view(prediction.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        valid = target != self.ignore
        true_positive = torch.where(valid, prediction * target, 0).sum(dim=1)
        false_negative = torch.where(valid, target * (1 - prediction), 0).sum(dim=1)
        false_positive = torch.where(valid, (1 - target) * prediction, 0).sum(dim=1)
        dice = (true_positive + self.smooth) / (
                true_positive
                + self.alpha * false_negative
                + (1 - self.alpha) * false_positive
                + self.smooth
        )
        return (1 - dice).mean()
