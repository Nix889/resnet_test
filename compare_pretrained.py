"""
Train the same architecture twice -- once from ImageNet-pretrained weights,
once from scratch -- with identical data, hyperparameters and seed, then
report which converges faster / generalizes better on this dataset.

Expects images/train/dataset/{train,val}.csv (built by split_dataset.py) and
images/train/ as the image root (see DOCUMENTATION.md for the full layout).

Example:
    python compare_pretrained.py --arch resnet50 --epochs 30
"""
import argparse
import json
from pathlib import Path

from common import DATASET_DIR, TRAIN_DIR
from train import train_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", default=str(DATASET_DIR / "train.csv"))
    p.add_argument("--val-csv", default=str(DATASET_DIR / "val.csv"))
    p.add_argument("--img-dir", default=str(TRAIN_DIR))
    p.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--loss", default="smoothl1", choices=["mse", "l1", "smoothl1"])
    p.add_argument("--augment", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="runs/compare")
    return p.parse_args()


def build_train_args(cmp_args, pretrained: bool, output_dir: Path):
    """Builds the argparse.Namespace that train_model() expects, matching
    train.py's --arch/--epochs/... flags but with `pretrained` fixed by the
    caller instead of read from argv."""
    ns = argparse.Namespace(
        train_csv=cmp_args.train_csv,
        val_csv=cmp_args.val_csv,
        img_dir=cmp_args.img_dir,
        arch=cmp_args.arch,
        pretrained=pretrained,
        epochs=cmp_args.epochs,
        batch_size=cmp_args.batch_size,
        lr=cmp_args.lr,
        weight_decay=cmp_args.weight_decay,
        img_size=cmp_args.img_size,
        loss=cmp_args.loss,
        augment=cmp_args.augment,
        num_workers=cmp_args.num_workers,
        seed=cmp_args.seed,
        output_dir=str(output_dir),
        patience=cmp_args.patience,
        device=cmp_args.device,
    )
    return ns


def main():
    cmp_args = parse_args()
    root = Path(cmp_args.output_dir)

    results = {}
    for pretrained in (True, False):
        tag = "pretrained" if pretrained else "scratch"
        run_dir = root / tag
        train_args = build_train_args(cmp_args, pretrained, run_dir)
        history, best_val_mae, ckpt_path = train_model(train_args, log_prefix=f"[{tag}] ")
        results[tag] = {"history": history, "best_val_mae": best_val_mae, "checkpoint": str(ckpt_path)}

    root.mkdir(parents=True, exist_ok=True)
    with open(root / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Comparison summary ===")
    for tag, res in results.items():
        print(f"{tag:>10}: best val MAE = {res['best_val_mae']:.2f} deg "
              f"({len(res['history'])} epochs run) -> {res['checkpoint']}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for tag, res in results.items():
            epochs = [h["epoch"] for h in res["history"]]
            val_mae = [h["val_mae"] for h in res["history"]]
            ax.plot(epochs, val_mae, label=tag)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MAE (deg)")
        ax.set_title(f"Pretrained vs. scratch -- {cmp_args.arch}")
        ax.legend()
        plot_path = root / "comparison.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to {plot_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
