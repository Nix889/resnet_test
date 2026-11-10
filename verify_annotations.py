"""
Interactive viewer to visually spot-check an existing labels CSV: steps
through each labeled image drawing its annotated needle angle(s) and
reference-mark angles (if present) as arrows from the shared pivot, using
the same corrected angle->vector math as common.py::draw_angle_overlay().

Unlike annotate.py, this tool never writes anything -- it's read-only, for
confirming that stored labels actually match what's in the photo (e.g.
right after annotating, or after a change to the angle/overlay math).

Each arrow's origin is the row's stored center_x/center_y (see annotate.py)
when present, otherwise the image's pixel center -- the same fallback rule
test.py --vis-dir uses. Rows without reference-mark columns just draw the
needle angle(s).

Keys while the viewer is open:
    right / n / d        next image
    left / p / a          previous image
    q                     quit

Example:
    python verify_annotations.py
    python verify_annotations.py --labels images/train/original/labels_2needle.csv
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import (ORIGINAL_DIR, angle_to_vector, get_angle_columns,
                     get_center_columns, get_reference_columns)

REF_LABELS = ["ref_min", "ref_max"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=str(ORIGINAL_DIR / "labels.csv"))
    p.add_argument("--image-dir", default=str(ORIGINAL_DIR))
    p.add_argument("--radius-ratio", type=float, default=0.4,
                   help="Arrow length as a fraction of the smaller image dimension (default: 0.4)")
    return p.parse_args()


def build_palette(labels):
    palette = ["red", "lime", "cyan", "magenta", "yellow", "orange"]
    return {label: palette[i % len(palette)] for i, label in enumerate(labels)}


def draw_row(ax, image_dir, row, angle_cols, ref_cols, center_cols, radius_ratio):
    img_path = Path(image_dir) / str(row["filename"])
    img = plt.imread(str(img_path))
    h, w = img.shape[:2]

    if center_cols and pd.notna(row[center_cols[0]]) and pd.notna(row[center_cols[1]]):
        cx, cy = float(row[center_cols[0]]), float(row[center_cols[1]])
        origin_note = "stored center"
    else:
        cx, cy = w / 2, h / 2
        origin_note = "image-center fallback"
    radius = min(h, w) * radius_ratio

    needle_labels = [f"needle_{i + 1}" if len(angle_cols) > 1 else "needle"
                     for i in range(len(angle_cols))]
    labels = needle_labels + REF_LABELS[:len(ref_cols)]
    colors = build_palette(labels)

    ax.clear()
    ax.imshow(img)
    ax.axis("off")
    for label, col in zip(labels, angle_cols + ref_cols):
        dx, dy = angle_to_vector(float(row[col]), radius)
        ax.arrow(cx, cy, dx, dy, head_width=radius * 0.04, head_length=radius * 0.06,
                  fc=colors[label], ec=colors[label], linewidth=3, label=label)
    ax.plot(cx, cy, "wo", markeredgecolor="black", markersize=6)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"{row['filename']}  ({origin_note})\n"
                 f"←/a=prev  →/d=next  q=quit")


def main():
    args = parse_args()
    df = pd.read_csv(args.labels)
    angle_cols = get_angle_columns(df)
    if not angle_cols:
        raise ValueError(f"No angle_N columns found in {args.labels}")
    ref_cols = get_reference_columns(df)
    center_cols = get_center_columns(df)

    print(f"{len(df)} labeled images | needle columns: {angle_cols} | "
          f"reference columns: {ref_cols or 'none'} | "
          f"center columns: {center_cols or 'none (image-center fallback for every row)'}")

    state = {"idx": 0}
    fig, ax = plt.subplots(figsize=(8, 8))

    def show():
        row = df.iloc[state["idx"]]
        draw_row(ax, args.image_dir, row, angle_cols, ref_cols, center_cols, args.radius_ratio)
        fig.canvas.draw()

    def on_key(event):
        if event.key in ("right", "n", "d"):
            state["idx"] = min(state["idx"] + 1, len(df) - 1)
            show()
        elif event.key in ("left", "p", "a"):
            state["idx"] = max(state["idx"] - 1, 0)
            show()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()


if __name__ == "__main__":
    main()
