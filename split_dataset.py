"""
Splits images/train/original/labels.csv into train/val/test CSVs, written to
the sibling images/train/dataset folder.

The split is decided on ORIGINAL images only. If images/train/augmented
(built by augment.py) exists, augmented copies are added to the TRAIN split
alone, keyed off their source_filename -- augmented copies of a val/test
image are never included anywhere, so validation/test stay clean and there's
no leakage between an original image and its own augmented copies ending up
on both sides of the split.

By default the split is STRATIFIED by angle: rows are grouped into equal-
width angle bins (per needle, jointly for two-needle gauges) and each group
is split proportionally, so train/val/test all see a similar angle
distribution instead of e.g. all near-0deg samples landing in test by chance
on a small dataset. Use --no-stratify for a plain random split instead.

Angle columns (angle_1, and angle_2 for two-needle gauges) are auto-detected
from the labels CSV header, so this script works unchanged for either mode.
ref_angle_min/ref_angle_max, and center_x/center_y, if present, are carried
through into every split too, but do not affect stratification (only needle
angles do). center_x/center_y are original-image-only: augmented rows get
NaN for them, since augment.py's translate/scale moves the pixel center and
augmented images are train-only and never overlaid.

Output CSVs use a subfolder-prefixed filename column ("original/x.jpg" or
"augmented/x.jpg") so a single --img-dir (images/train) resolves both.

Example:
    python split_dataset.py --train-frac 0.7 --val-frac 0.15
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from common import (AUGMENTED_DIR, DATASET_DIR, ORIGINAL_DIR, get_angle_columns,
                     get_center_columns, get_reference_columns)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-labels", default=str(ORIGINAL_DIR / "labels.csv"))
    p.add_argument("--augmented-labels", default=str(AUGMENTED_DIR / "labels.csv"),
                   help="Optional; skipped with a warning if it doesn't exist")
    p.add_argument("--output-dir", default=str(DATASET_DIR))
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify", dest="stratify", action="store_true", default=True,
                   help="Stratify the split by binned angle so train/val/test each keep "
                        "a similar angle distribution (default: on)")
    p.add_argument("--no-stratify", dest="stratify", action="store_false",
                   help="Fall back to a plain random split")
    p.add_argument("--strat-bins", type=int, default=12,
                   help="Equal-width angle bins per needle used for stratification "
                        "(default: 12, i.e. 30deg bins)")
    return p.parse_args()


def stratified_indices(df, angle_cols, bins, train_frac, val_frac, seed):
    """Groups rows by binned angle (jointly across all needle columns) and
    splits each group proportionally, so every split sees a similar angle
    distribution instead of e.g. all near-0deg samples landing in test."""
    rng = random.Random(seed)
    bin_width = 360.0 / bins

    def bin_of(value):
        return min(int(value // bin_width), bins - 1)

    groups = {}
    for idx, row in df.iterrows():
        key = tuple(bin_of(float(row[c])) for c in angle_cols)
        groups.setdefault(key, []).append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for idxs in groups.values():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = round(n * train_frac)
        n_val = min(round(n * val_frac), n - n_train)
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])
    return train_idx, val_idx, test_idx


def plain_indices(df, train_frac, val_frac, seed):
    shuffled = df.sample(frac=1.0, random_state=seed).index.tolist()
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def main():
    args = parse_args()
    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0 (remainder becomes test)")

    df = pd.read_csv(args.original_labels).reset_index(drop=True)
    angle_cols = get_angle_columns(df)
    ref_cols = get_reference_columns(df)
    center_cols = get_center_columns(df)
    if not angle_cols:
        raise ValueError(f"No angle_N columns found in {args.original_labels}")

    if args.stratify:
        train_i, val_i, test_i = stratified_indices(
            df, angle_cols, args.strat_bins, args.train_frac, args.val_frac, args.seed)
    else:
        train_i, val_i, test_i = plain_indices(df, args.train_frac, args.val_frac, args.seed)

    for split_name, idxs in (("train", train_i), ("val", val_i), ("test", test_i)):
        if not idxs:
            raise ValueError(
                f"The {split_name} split ended up with 0 images -- this can happen with a "
                f"small dataset combined with stratification (each angle bin is split "
                f"independently, and rounding can zero out a small bin's share). Try "
                f"--strat-bins with a smaller value, --no-stratify, or label more images."
            )

    train_df = df.loc[train_i].copy()
    val_df = df.loc[val_i].copy()
    test_df = df.loc[test_i].copy()
    train_original_filenames = set(train_df["filename"])

    out_cols = ["filename"] + angle_cols + ref_cols + center_cols
    train_df["filename"] = "original/" + train_df["filename"].astype(str)
    val_df["filename"] = "original/" + val_df["filename"].astype(str)
    test_df["filename"] = "original/" + test_df["filename"].astype(str)
    train_df, val_df, test_df = train_df[out_cols], val_df[out_cols], test_df[out_cols]

    aug_path = Path(args.augmented_labels)
    if aug_path.exists():
        aug_df = pd.read_csv(aug_path)
        aug_angle_cols = get_angle_columns(aug_df)
        aug_ref_cols = get_reference_columns(aug_df)
        if aug_angle_cols != angle_cols or aug_ref_cols != ref_cols:
            raise ValueError(
                f"Columns differ between original (angles={angle_cols}, ref={ref_cols}) and "
                f"augmented (angles={aug_angle_cols}, ref={aug_ref_cols}) labels -- re-run "
                f"augment.py after changing needle count or reference-mark columns."
            )
        aug_train = aug_df[aug_df["source_filename"].isin(train_original_filenames)].copy()
        aug_train["filename"] = "augmented/" + aug_train["filename"].astype(str)
        for col in center_cols:
            if col not in aug_train.columns:
                aug_train[col] = float("nan")
        train_df = pd.concat([train_df, aug_train[out_cols]], ignore_index=True)
        print(f"Added {len(aug_train)} augmented images to the train split "
              f"(val/test remain unaugmented).")
    else:
        print(f"No augmented labels found at {aug_path} -- run augment.py first "
              f"if you want them included. Continuing with originals only.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    mode = f"stratified ({args.strat_bins} bins/needle)" if args.stratify else "plain random"
    print(f"Split mode: {mode} | angle columns: {angle_cols} | reference columns: {ref_cols} | "
          f"center columns: {center_cols or 'none'}")
    print(f"Original images: {len(df)} | train: {len(train_df)} (incl. augmented) | "
          f"val: {len(val_df)} | test: {len(test_df)}")
    print(f"Written to {output_dir}/train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
