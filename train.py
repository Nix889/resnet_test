"""
Train a ResNet to regress the needle angle(s) (in degrees) of a gauge image,
plus its scale's ref_angle_min/ref_angle_max reference marks if the CSV has
them (annotate.py always labels both -- see DOCUMENTATION.md).

Needle count and whether reference marks are present are both auto-detected
from --train-csv's columns -- there's no separate flag for either here, so
training a one-needle model and then a two-needle model, or a needle-only
model and then one that also predicts reference marks, is just a matter of
pointing --train-csv/--val-csv at CSVs built from a different labels file.

Expects images/train/dataset/{train,val}.csv (built by split_dataset.py) and
images/train/ as the image root (see DOCUMENTATION.md for the full layout).

Output directory defaults to runs/<arch>/<timestamp> and will contain:
  best_model.pt   -- checkpoint of the best epoch
  params.json     -- training hyperparameters and final metrics
  history.json    -- per-epoch train/val loss and MAE
  metrics.png     -- loss and MAE vs epoch plot

Example:
    python train.py --arch resnet50 --epochs 30 --batch-size 32
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (DATASET_DIR, TRAIN_DIR, GaugeAngleDataset, build_model,
                     build_transforms, get_criterion, plot_training_history,
                     run_epoch, set_seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", default=str(DATASET_DIR / "train.csv"),
                   help="CSV with filename,angle for training")
    p.add_argument("--val-csv", default=str(DATASET_DIR / "val.csv"),
                   help="CSV with filename,angle for validation")
    p.add_argument("--img-dir", default=str(TRAIN_DIR),
                   help="Root the CSV filenames are resolved against")
    p.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    p.add_argument("--pretrained", dest="pretrained", action="store_true", default=True,
                   help="Start from ImageNet-pretrained weights (default)")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                   help="Train the backbone from scratch")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--loss", default="smoothl1", choices=["mse", "l1", "smoothl1"])
    p.add_argument("--augment", action="store_true",
                   help="Enable in-memory photometric augmentation each epoch, on top of "
                        "whatever augment.py already added to train.csv (see DOCUMENTATION.md)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: runs/<arch>/<timestamp>)")
    p.add_argument("--patience", type=int, default=10, help="Early-stopping patience in epochs (0 disables)")
    p.add_argument("--device", default=None, help="cuda / cpu / mps (default: auto-detect)")
    p.add_argument("--amp", dest="amp", action="store_true", default=True,
                   help="Mixed-precision training on CUDA (default: on; no effect on CPU/MPS)")
    p.add_argument("--no-amp", dest="amp", action="store_false",
                   help="Disable mixed precision (fp32 throughout) for debugging/max reproducibility")
    return p.parse_args()


def get_device(requested=None):
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested but torch.cuda.is_available() is False. "
                "Either PyTorch's CPU-only build is installed (reinstall the CUDA wheel "
                "per requirements.txt), or no compatible GPU/driver was found."
            )
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(args, log_prefix=""):
    """Runs a full training loop and returns (history, best_val_mae, checkpoint_path).

    Factored out of main() so compare_pretrained.py can call it twice
    (pretrained=True/False) with the exact same data/hyperparameters.
    """
    set_seed(args.seed)
    device = get_device(args.device)
    if device.type == "cuda":
        # Input size is fixed (Resize to --img-size every batch), so letting
        # cuDNN autotune/cache the fastest conv algorithm for that shape is a
        # safe win. No effect on CPU/MPS.
        torch.backends.cudnn.benchmark = True
        print(f"{log_prefix}Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(f"{log_prefix}No GPU in use -- training on {device}. If you have a "
              f"CUDA GPU, see requirements.txt for the CUDA-enabled torch wheel.")
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("runs") / args.arch / datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.img_size, augment=args.augment)
    train_ds = GaugeAngleDataset(args.train_csv, args.img_dir, transform=train_tf)
    val_ds = GaugeAngleDataset(args.val_csv, args.img_dir, transform=eval_tf)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"--train-csv has {len(train_ds)} rows and --val-csv has {len(val_ds)} -- "
            f"both must be non-empty. If these came from split_dataset.py, try "
            f"--no-stratify or a smaller --strat-bins."
        )
    if train_ds.num_needles != val_ds.num_needles:
        raise ValueError(
            f"--train-csv has {train_ds.num_needles} needle(s) ({train_ds.angle_cols}) but "
            f"--val-csv has {val_ds.num_needles} ({val_ds.angle_cols}) -- both must match."
        )
    if train_ds.num_reference != val_ds.num_reference:
        raise ValueError(
            f"--train-csv has reference columns {train_ds.ref_cols} but --val-csv has "
            f"{val_ds.ref_cols} -- both must match."
        )
    num_needles = train_ds.num_needles
    num_reference = train_ds.num_reference

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.arch, pretrained=args.pretrained, num_needles=num_needles,
                         num_reference=num_reference).to(device)
    criterion = get_criterion(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    # Mixed precision only ever engages on CUDA; a disabled GradScaler is a
    # no-op, so this is inert on CPU/MPS regardless of --amp.
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    history = []
    ckpt_path = output_dir / "best_model.pt"

    print(f"{log_prefix}Training {args.arch} (pretrained={args.pretrained}, needles={num_needles}, "
          f"reference_marks={bool(num_reference)}) on {device} (amp={amp_enabled}) | "
          f"train={len(train_ds)} val={len(val_ds)} samples")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_mae = run_epoch(model, train_loader, device, criterion, optimizer,
                                           desc=f"{log_prefix}epoch {epoch} train", scaler=scaler)
        val_loss, val_mae = run_epoch(model, val_loader, device, criterion, optimizer=None,
                                       desc=f"{log_prefix}epoch {epoch} val")
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_mae": train_mae,
                         "val_loss": val_loss, "val_mae": val_mae})
        print(f"{log_prefix}Epoch {epoch:03d}/{args.epochs} "
              f"train_loss={train_loss:.4f} train_mae={train_mae:.2f}deg "
              f"val_loss={val_loss:.4f} val_mae={val_mae:.2f}deg "
              f"({time.time() - t0:.1f}s)")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "pretrained": args.pretrained,
                "img_size": args.img_size,
                "num_needles": num_needles,
                "num_reference": num_reference,
                "val_mae": val_mae,
                "epoch": epoch,
            }, ckpt_path)
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                print(f"{log_prefix}Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    params = {
        "arch": args.arch,
        "pretrained": args.pretrained,
        "img_size": args.img_size,
        "num_needles": num_needles,
        "num_reference": num_reference,
        "epochs_trained": len(history),
        "best_val_mae": best_val_mae,
        "checkpoint": str(ckpt_path),
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "augment": args.augment,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "amp": amp_enabled,
    }
    with open(output_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    plot_training_history(history, output_dir, prefix=log_prefix)
    print(f"{log_prefix}Best val MAE: {best_val_mae:.2f} degrees. Checkpoint: {ckpt_path}")
    return history, best_val_mae, ckpt_path


def main():
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
