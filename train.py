import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from framework.custom_callback import CustomCallback, EMACallback
from framework.framework import FaultSeg
from seisDataset.dataset import (
    DATASET_CHOICES,
    FaultSegDataset,
    MultiScaleBatchSampler,
    gpu_collate,
    prepare_faultseg_datasets,
)


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "framework" / "faultsegv3.ckpt"


class SizeAction(argparse.Action):
    """Replace the default scales on the first explicit --size option."""

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest)
        if current is self.default:
            current = []
        setattr(namespace, self.dest, [*current, tuple(values)])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train FaultSegV3")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="all",
        help="dataset to prepare and train on (default: merged V1 + V2)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing dataset ZIP/extracted caches",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="download/extract the selected dataset without starting training",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=4,
        action=SizeAction,
        default=[
            (32, 128, 128, 128),
            (8, 256, 176, 176),
            (64, 64, 64, 64),
        ],
        metavar=("BATCH", "D", "H", "W"),
        help="one scale and its batch size; repeat for multiple scales",
    )
    parser.add_argument(
        "--workers", "--num-workers", "--work", type=int, default=16,
        help="DataLoader worker processes (default: 16)",
    )
    parser.add_argument(
        "--resume", nargs="?", type=Path, const=Path("model_weights/last.ckpt"),
        help="resume full training state from a Lightning checkpoint; default path: model_weights/last.ckpt",
    )
    parser.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS,
        help="initial model weights for a new run (default: framework/faultsegv3.ckpt)",
    )
    parser.add_argument(
        "--no-weights", action="store_true", help="start a new run with random weights",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
    parser.add_argument("--end-lr", type=float, default=5e-6, help="final cosine learning rate")
    parser.add_argument(
        "--cosine", action=argparse.BooleanOptionalAction, default=True,
        help="enable or disable cosine learning rate decay (default: enabled)",
    )
    parser.add_argument(
        "--cosine-steps", "--decay-steps", dest="decay_steps", type=int,
        default=100000, help="steps until cosine learning rate reaches --end-lr",
    )
    parser.add_argument(
        "--ema", action=argparse.BooleanOptionalAction, default=True,
        help="enable or disable EMA (default: enabled)",
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.9996,
        help="EMA decay factor (default: 0.9996)",
    )
    parser.add_argument(
        "--save-every-steps", type=int, default=2500,
        help="interval for weights and resumable checkpoints (default: 2500)",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if not args.prepare_only and args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
    faultseg_dirs = prepare_faultseg_datasets(args.dataset, args.data_root)
    if args.prepare_only:
        return

    num_workers = args.workers

    train_dataset = FaultSegDataset(
        faultseg_dirs=faultseg_dirs,
        target_shape=[scale[1:] for scale in args.size],
    )
    batch_sampler = MultiScaleBatchSampler(len(train_dataset), args.size)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=gpu_collate,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model = FaultSeg(
        restore_path=None if args.resume or args.no_weights else args.weights,
        start_lr=args.lr,
        end_lr=args.end_lr,
        decay_steps=args.decay_steps,
        cosine=args.cosine,
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs("model_weights", exist_ok=True)
    callbacks = [
        CustomCallback(
            print_every_n_steps=10,
            save_weights_every_n_steps=args.save_every_steps,
            viz_every_n_steps=500,
            viz_samples=4,
            save_dir="model_weights",
            results_dir="results",
        ),
    ]
    if args.ema:
        callbacks.append(EMACallback(decay=args.ema_decay, ema_scope="model", update_bn=True))
    callbacks.append(
        ModelCheckpoint(
            dirpath="model_weights",
            filename="step-{step}",
            every_n_train_steps=args.save_every_steps,
            save_top_k=1,
            save_last=True,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed",
        logger=TensorBoardLogger("logs", name="faultSeg"),
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=False,
        use_distributed_sampler=False,
        sync_batchnorm=True,
    )
    trainer.fit(model, train_loader, ckpt_path=str(args.resume) if args.resume else None)


if __name__ == "__main__":
    main()
