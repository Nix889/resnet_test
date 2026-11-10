"""
Evaluate a trained checkpoint on a held-out, labeled test CSV.

Reports per-needle MAE/RMSE/max-error (using circular angular distance, so
e.g. 359deg vs. 1deg correctly counts as 2deg of error, not 358), plus
per-reference-mark stats if the checkpoint was trained with ref_angle_min/
ref_angle_max. Needle count and reference-mark presence are both read from
the checkpoint (see train.py), not from a flag.

Pass --value-min/--value-max (the physical values at the scale's min/max
reference marks, e.g. 0 and 160) to also compute the actual physical
reading via compute_reading() -- this only works for a checkpoint trained
with reference marks. See DOCUMENTATION.md, "Reference marks and computing
a physical reading."

Pass --vis-dir to save per-sample visual overlays showing both true and
predicted angle lines drawn on each image.

Expects images/train/dataset/test.csv (built by split_dataset.py) and
images/train/ as the image root (see DOCUMENTATION.md for the full layout).

--vis-dir arrows originate from each row's stored center_x/center_y (see
annotate.py) when test.csv has them, falling back to the image center for
any row that doesn't (e.g. labels created before that column existed).

Example:
    python test.py --checkpoint runs/resnet50/20260703_143022/best_model.pt \
        --value-min 0 --value-max 160 --unit PSI \
        --output-csv runs/resnet50/20260703_143022/test_predictions.csv --plot

    python test.py --checkpoint runs/resnet50/20260703_143022/best_model.pt --output-csv runs/resnet50/20260703_143022/test_predictions.csv --vis-dir runs/resnet50/20260703_143022/visualizations
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (DATASET_DIR, TRAIN_DIR, GaugeAngleDataset, build_model,
                     build_transforms, circular_abs_error_deg, compute_reading,
                     decode_angles, draw_angle_overlay, get_center_columns)

REF_LABELS = ["ref_min", "ref_max"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-csv", default=str(DATASET_DIR / "test.csv"))
    p.add_argument("--img-dir", default=str(TRAIN_DIR))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-csv", default=None, help="Where to write per-sample predictions")
    p.add_argument("--plot", action="store_true", help="Save predicted-vs-actual scatter plots")
    p.add_argument("--value-min", type=float, default=None,
                   help="Physical value at ref_angle_min (requires a checkpoint with reference marks)")
    p.add_argument("--value-max", type=float, default=None,
                   help="Physical value at ref_angle_max")
    p.add_argument("--unit", default="", help="Unit label for printed readings, e.g. PSI")
    p.add_argument("--vis-dir", default=None,
                   help="Directory to save visual overlay images (default: <output_csv parent>/visualizations)")
    p.add_argument("--device", default=None)
    return p.parse_args()


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_angle_stats(label, err_col):
    mae = err_col.mean().item()
    rmse = torch.sqrt((err_col ** 2).mean()).item()
    max_err = err_col.max().item()
    within_5 = (err_col <= 5).float().mean().item() * 100
    within_10 = (err_col <= 10).float().mean().item() * 100
    print(f"[{label}] MAE={mae:.2f}deg RMSE={rmse:.2f}deg max={max_err:.2f}deg "
          f"within5deg={within_5:.1f}% within10deg={within_10:.1f}%")
    return dict(mae=mae, rmse=rmse, max_err=max_err, within_5=within_5, within_10=within_10)


def main():
    args = parse_args()
    device = get_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    num_needles = ckpt["num_needles"]
    num_reference = ckpt.get("num_reference", 0)
    model = build_model(ckpt["arch"], pretrained=False, num_needles=num_needles,
                         num_reference=num_reference)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    _, eval_tf = build_transforms(ckpt["img_size"], augment=False)
    dataset = GaugeAngleDataset(args.test_csv, args.img_dir, transform=eval_tf)
    if dataset.num_needles != num_needles or dataset.num_reference != num_reference:
        raise ValueError(
            f"Checkpoint was trained with {num_needles} needle(s) and "
            f"{num_reference} reference column(s), but --test-csv has "
            f"{dataset.num_needles} needle(s) ({dataset.angle_cols}) and "
            f"{dataset.num_reference} reference column(s) ({dataset.ref_cols})."
        )
    if (args.value_min is None) != (args.value_max is None):
        raise ValueError("--value-min and --value-max must be given together.")
    if args.value_min is not None and num_reference != 2:
        raise ValueError(
            "--value-min/--value-max require a checkpoint trained with reference marks "
            "(ref_angle_min/ref_angle_max) -- this one has none. Re-label with "
            "annotate.py (it now always labels both reference marks) and retrain."
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)

    filenames = dataset.df[dataset.filename_col].tolist()
    all_preds_raw, all_targets_raw = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            preds = model(images).cpu()
            all_preds_raw.append(preds)
            all_targets_raw.append(targets)

    preds_raw = torch.cat(all_preds_raw)
    targets_raw = torch.cat(all_targets_raw)
    pred_deg = decode_angles(preds_raw)     # (N, num_needles + num_reference)
    true_deg = decode_angles(targets_raw)
    abs_err = circular_abs_error_deg(pred_deg, true_deg)  # squaring drops the sign, so this also feeds RMSE

    pred_needle, true_needle, err_needle = (t[:, :num_needles] for t in (pred_deg, true_deg, abs_err))
    if num_reference:
        pred_ref = pred_deg[:, num_needles:]
        true_ref = true_deg[:, num_needles:]
        err_ref = abs_err[:, num_needles:]

    print(f"Test samples: {len(dataset)} | needles: {num_needles} | reference marks: {bool(num_reference)}")
    per_needle_stats = []
    for i in range(num_needles):
        label = f"needle_{i + 1}" if num_needles > 1 else "needle"
        per_needle_stats.append(print_angle_stats(label, err_needle[:, i]))
    if num_needles > 1:
        print(f"[combined] MAE={err_needle.mean().item():.2f}deg (averaged across all needles)")

    per_ref_stats = []
    if num_reference:
        for i, label in enumerate(REF_LABELS):
            per_ref_stats.append(print_angle_stats(label, err_ref[:, i]))

    reading_specs = []  # (label, pred_reading, true_reading) per needle, if requested
    if args.value_min is not None:
        unit_suffix = f" {args.unit}" if args.unit else ""
        for i in range(num_needles):
            label = f"needle_{i + 1}" if num_needles > 1 else "needle"
            pred_reading = compute_reading(pred_needle[:, i], pred_ref[:, 0], pred_ref[:, 1],
                                            args.value_min, args.value_max)
            true_reading = compute_reading(true_needle[:, i], true_ref[:, 0], true_ref[:, 1],
                                            args.value_min, args.value_max)
            reading_specs.append((label, pred_reading, true_reading))

            err = (pred_reading - true_reading).abs()
            print(f"[{label} reading] MAE={err.mean().item():.3f}{unit_suffix} "
                  f"max={err.max().item():.3f}{unit_suffix} "
                  f"(fully predicted: needle + reference-mark angles both from the model)")

            # Diagnostic: use ground-truth reference angles to isolate how much of the
            # reading error is coming from the needle prediction alone, vs. from the
            # reference-mark predictions -- see DOCUMENTATION.md.
            oracle_ref_reading = compute_reading(pred_needle[:, i], true_ref[:, 0], true_ref[:, 1],
                                                  args.value_min, args.value_max)
            oracle_err = (oracle_ref_reading - true_reading).abs()
            print(f"[{label} reading, oracle ref] MAE={oracle_err.mean().item():.3f}{unit_suffix} "
                  f"(needle-only error, using ground-truth reference angles)")

    if args.output_csv:
        out_data = {"filename": filenames}
        for i in range(num_needles):
            suffix = f"_{i + 1}" if num_needles > 1 else ""
            out_data[f"true_angle{suffix}"] = true_needle[:, i].numpy()
            out_data[f"predicted_angle{suffix}"] = pred_needle[:, i].numpy()
            out_data[f"abs_error{suffix}"] = err_needle[:, i].numpy()
        for i, label in enumerate(REF_LABELS[:num_reference]):
            out_data[f"true_{label}"] = true_ref[:, i].numpy()
            out_data[f"predicted_{label}"] = pred_ref[:, i].numpy()
            out_data[f"abs_error_{label}"] = err_ref[:, i].numpy()
        for label, pred_reading, true_reading in reading_specs:
            out_data[f"true_reading_{label}"] = true_reading.numpy()
            out_data[f"predicted_reading_{label}"] = pred_reading.numpy()
            out_data[f"reading_abs_error_{label}"] = (pred_reading - true_reading).abs().numpy()
        out_df = pd.DataFrame(out_data)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output_csv, index=False)
        print(f"Per-sample predictions written to {args.output_csv}")

    if args.plot:
        import matplotlib.pyplot as plt
        plot_specs = []  # (title, true_np, pred_np, axis_range)
        for i in range(num_needles):
            label = f"needle {i + 1}" if num_needles > 1 else "needle"
            plot_specs.append((f"{label} angle (MAE={per_needle_stats[i]['mae']:.2f} deg)",
                                true_needle[:, i].numpy(), pred_needle[:, i].numpy(), (0, 360)))
        for i, label in enumerate(REF_LABELS[:num_reference]):
            plot_specs.append((f"{label} angle (MAE={per_ref_stats[i]['mae']:.2f} deg)",
                                true_ref[:, i].numpy(), pred_ref[:, i].numpy(), (0, 360)))
        for label, pred_reading, true_reading in reading_specs:
            lo, hi = sorted((args.value_min, args.value_max))
            plot_specs.append((f"{label} reading", true_reading.numpy(), pred_reading.numpy(), (lo, hi)))

        fig, axes = plt.subplots(1, len(plot_specs), figsize=(6 * len(plot_specs), 6), squeeze=False)
        for ax, (title, t, p, (lo, hi)) in zip(axes[0], plot_specs):
            ax.scatter(t, p, alpha=0.5, s=15)
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="ideal")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.set_title(title)
            ax.legend()
        fig.tight_layout()
        plot_path = Path(args.output_csv).with_suffix(".png") if args.output_csv else Path("test_predictions.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")

    if args.vis_dir or args.output_csv:
        vis_dir = Path(args.vis_dir) if args.vis_dir else (
            Path(args.output_csv).parent / "visualizations" if args.output_csv else Path("test_visualizations"))
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visual overlays to {vis_dir} ...")
        center_cols = get_center_columns(dataset.df)
        for idx, fname in enumerate(filenames):
            img_path = dataset.img_dir / fname
            if not img_path.exists():
                continue
            angles = {}
            for i in range(num_needles):
                suffix = f"_{i + 1}" if num_needles > 1 else ""
                label = f"needle{i + 1}" if num_needles > 1 else "needle"
                angles[f"true_{label}"] = float(true_needle[idx, i])
                angles[f"pred_{label}"] = float(pred_needle[idx, i])
            if num_reference:
                for i, rl in enumerate(REF_LABELS[:num_reference]):
                    angles[f"true_{rl}"] = float(true_ref[idx, i])
                    angles[f"pred_{rl}"] = float(pred_ref[idx, i])

            center = None
            if center_cols:
                cx, cy = dataset.df.iloc[idx][center_cols]
                if pd.notna(cx) and pd.notna(cy):
                    center = (float(cx), float(cy))

            out_name = Path(fname).stem + "_vis.png"
            draw_angle_overlay(img_path, angles, vis_dir / out_name, center=center)
        print(f"Visual overlays saved to {vis_dir}")


if __name__ == "__main__":
    main()
