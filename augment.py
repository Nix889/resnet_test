"""
Creates augmented copies of every image in images/train/original, and writes
them to the sibling images/train/augmented folder together with a labels.csv
that carries over the angle_N column(s) for every needle, plus
ref_angle_min/ref_angle_max if the source labels have them.

Available controls, combined per generated copy:
  - photometric + angle-preserving geometric (default on): brightness and
    contrast jitter, slight blur, small translate/scale. Translate and
    uniform scale are geometric, but neither rotates nor mirrors the image,
    so the pivot->tip *direction* -- and therefore the angle label -- is
    unchanged; only rotation/flip would require relabeling, and neither is
    applied here.
  - --brightness: strength of brightness jitter (default 0.25)
  - --contrast: strength of contrast jitter (default 0.25)

Every augmented copy is resized to 200x200 pixels before saving -- no label
relabeling is ever needed, since nothing applied here rotates or mirrors the
image.

Run this before split_dataset.py; split_dataset.py splits on the original
images only and adds these augmented copies to the train split alone, so
validation/test stay unaugmented (no leakage).

Example:
    python augment.py --num-augments 3 --brightness 0.3 --contrast 0.2
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision import transforms

from common import (AUGMENTED_DIR, ORIGINAL_DIR, get_angle_columns,
                     get_reference_columns, set_seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", default=str(ORIGINAL_DIR))
    p.add_argument("--labels", default=str(ORIGINAL_DIR / "labels.csv"))
    p.add_argument("--output-dir", default=str(AUGMENTED_DIR))
    p.add_argument("--output-labels", default=None,
                   help="Default: <output-dir>/labels.csv")
    p.add_argument("--num-augments", type=int, default=3,
                   help="Augmented copies to generate per original image")
    p.add_argument("--photometric", dest="photometric", action="store_true", default=True,
                   help="Apply brightness/contrast jitter/blur/translate/scale (default: on)")
    p.add_argument("--no-photometric", dest="photometric", action="store_false")
    p.add_argument("--brightness", type=float, default=0.25,
                   help="Strength of brightness jitter (default: 0.25)")
    p.add_argument("--contrast", type=float, default=0.25,
                   help="Strength of contrast jitter (default: 0.25)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_photometric_augmenter(brightness=0.25, contrast=0.25):
    """Photometric plus small angle-preserving geometric transforms (pure
    translate/scale, no rotation/flip): never rotates or mirrors the image,
    so the pivot->tip angle -- and therefore the label -- never changes."""
    return transforms.Compose([
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    ])


def main():
    args = parse_args()
    set_seed(args.seed)

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_labels_path = (Path(args.output_labels) if args.output_labels
                           else output_dir / "labels.csv")

    labels_df = pd.read_csv(args.labels)
    angle_cols = get_angle_columns(labels_df) + get_reference_columns(labels_df)
    if not angle_cols:
        raise ValueError(f"No angle_N columns found in {args.labels}")

    photometric_augmenter = (build_photometric_augmenter(args.brightness, args.contrast)
                             if args.photometric else None)

    resize_tf = transforms.Resize((200, 200))

    rows = []
    for _, row in labels_df.iterrows():
        src_path = image_dir / str(row["filename"])
        if not src_path.exists():
            print(f"Skipping {src_path} (not found)")
            continue
        image = Image.open(src_path).convert("RGB")
        stem, suffix = Path(row["filename"]).stem, Path(row["filename"]).suffix

        for i in range(args.num_augments):
            out_image = image
            angles = {c: float(row[c]) for c in angle_cols}

            if photometric_augmenter is not None:
                out_image = photometric_augmenter(out_image)

            out_image = resize_tf(out_image)

            out_name = f"{stem}_aug{i}{suffix}"
            out_image.save(output_dir / out_name)
            rows.append({
                "filename": out_name,
                **angles,
                "source_filename": row["filename"],
            })

    out_df = pd.DataFrame(rows, columns=["filename", *angle_cols, "source_filename"])
    out_df.to_csv(output_labels_path, index=False)
    print(f"Generated {len(out_df)} augmented images ({args.num_augments} per original) "
          f"in {output_dir}")
    print(f"Modes: photometric={args.photometric} brightness={args.brightness} "
          f"contrast={args.contrast} resize=200x200")
    print(f"Labels written to {output_labels_path}")


if __name__ == "__main__":
    main()
