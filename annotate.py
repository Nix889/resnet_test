"""
Interactive click-to-label tool to build (and edit) labels.csv for the
images in images/train/original.

Steps through every image in --image-dir, in order -- not just unlabeled
ones -- so a previously-saved label can be revisited and corrected within
the same run (labels sometimes turn out displaced after the fact: a
mis-click, a re-cropped photo, etc.). For each image:
  - if it's already labeled, its current needle/reference-mark angles are
    drawn as an overlay (same corrected math as
    common.py::draw_angle_overlay() / verify_annotations.py), so you can see
    at a glance whether it's still right before deciding to redo it;
  - press 'r' to (re)label it: click the gauge's CENTER once -- the shared
    pivot for every point in the image -- then one TIP click per point:
    needle 1's tip (and, in --needles 2 mode, needle 2's tip right after),
    then the scale's min reference mark, then its max reference mark.
    Finishing the click sequence overwrites that image's row (or adds a new
    one) and saves immediately.

Each point's angle is computed as the counter-clockwise angle from the
positive x-axis to the center->tip vector (standard math convention,
0-360 deg -- no range restriction, so full-360-degree dials are labeled
exactly like bounded-arc gauges). Adjust angle_from_points() if your project
needs a different zero-reference.

Keys:
    n / right          next image
    p / left / b       previous image (freely revisit any image, labeled or
                        not -- this is how you go back and fix one)
    r                  (re)label the current image: start collecting clicks
                        (overwrites the existing label once the full click
                        sequence below is finished)
    z                  while collecting clicks: restart this image's clicks
                        from scratch
    escape / c         while collecting clicks: cancel and go back to just
                        viewing (nothing is saved)
    s                  skip forward without labeling (alias for 'n')
    q                  quit and save

Progress is saved to the CSV after every completed edit, so it's safe to
quit and resume later -- and safe to go back and fix a mislabeled image in
the same or a later run, since nothing here is append-only.

To test a one-needle model and a two-needle model independently on the same
photos, label into two separate files and point split_dataset.py/train.py at
whichever one you're testing -- e.g.:
    python annotate.py --needles 1 --output images/train/original/labels_1needle.csv
    python annotate.py --needles 2 --output images/train/original/labels_2needle.csv
See DOCUMENTATION.md for the full one-vs-two-needle workflow, and for how
the reference marks feed into computing an actual physical reading.

Example:
    python annotate.py                # 1 needle + reference marks
    python annotate.py --needles 2   # 2 needles + reference marks
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import CENTER_COLS, REFERENCE_COLS, ORIGINAL_DIR, angle_to_vector

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
OVERLAY_PALETTE = ["red", "lime", "cyan", "magenta", "yellow", "orange"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", default=str(ORIGINAL_DIR))
    p.add_argument("--output", default=str(ORIGINAL_DIR / "labels.csv"))
    p.add_argument("--needles", type=int, default=1, choices=[1, 2],
                   help="Number of needles to click per image (default: 1)")
    return p.parse_args()


def angle_from_points(pivot, tip):
    dx = tip[0] - pivot[0]
    dy = -(tip[1] - pivot[1])  # flip y: image coords grow downward
    angle = math.degrees(math.atan2(dy, dx))
    return angle % 360


def build_point_plan(needles):
    """Returns [(display_label, column_name), ...] in click order. Every
    point shares the image's single gauge-center click as its pivot and gets
    one tip click of its own; the scale's min/max reference marks are always
    included."""
    plan = [(f"needle {i + 1}", f"angle_{i + 1}") for i in range(needles)]
    plan.append(("min reference mark", REFERENCE_COLS[0]))
    plan.append(("max reference mark", REFERENCE_COLS[1]))
    return plan


def load_labels(output_path, expected_cols):
    """Loads an existing labels CSV, accepting both centerless (legacy) and
    centered files -- center_x/center_y are backfilled as NaN on rows that
    predate them. Errors if the file's angle/reference columns don't match
    the requested --needles mode."""
    if output_path.exists():
        df = pd.read_csv(output_path)
        existing_cols = [c for c in df.columns if c != "filename" and c not in CENTER_COLS]
        if sorted(existing_cols) != sorted(expected_cols):
            raise ValueError(
                f"{output_path} already has columns {existing_cols}, which doesn't "
                f"match the requested mode ({expected_cols}). Use a different "
                f"--output file for a different mode."
            )
        for col in CENTER_COLS:
            if col not in df.columns:
                df[col] = float("nan")
    else:
        df = pd.DataFrame(columns=["filename"] + expected_cols + CENTER_COLS)
    return df


def draw_existing_annotation(ax, img, row, point_plan):
    """Draws a row's currently-saved angles as arrows from its stored (or
    image-center-fallback) pivot -- the same convention test.py --vis-dir and
    verify_annotations.py use -- so you can see what's saved before deciding
    whether to redo it."""
    h, w = img.shape[:2]
    cx, cy = row[CENTER_COLS[0]], row[CENTER_COLS[1]]
    if pd.isna(cx) or pd.isna(cy):
        cx, cy = w / 2, h / 2
    radius = min(h, w) * 0.4
    for i, (label, col) in enumerate(point_plan):
        dx, dy = angle_to_vector(float(row[col]), radius)
        color = OVERLAY_PALETTE[i % len(OVERLAY_PALETTE)]
        ax.arrow(cx, cy, dx, dy, head_width=radius * 0.04, head_length=radius * 0.06,
                  fc=color, ec=color, linewidth=2, label=label)
    ax.plot(cx, cy, "wo", markeredgecolor="black", markersize=6)
    ax.legend(loc="upper right", fontsize=8)


class AnnotationSession:
    """Drives one persistent matplotlib window across every image in
    `all_images`, so next/prev navigation and re-editing a previously-saved
    label don't require closing/reopening a window per image."""

    def __init__(self, all_images, df, point_plan, output_path):
        self.all_images = all_images
        self.df = df
        self.point_plan = point_plan
        self.output_path = output_path
        self.needed = 1 + len(point_plan)
        labeled = set(df["filename"])
        self.idx = next((i for i, p in enumerate(all_images) if p.name not in labeled), 0)
        self.editing = False
        self.clicks = []
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def run(self):
        self._render()
        plt.show()

    def _row_index(self, filename):
        matches = self.df.index[self.df["filename"] == filename]
        return matches[0] if len(matches) else None

    def _current_path(self):
        return self.all_images[self.idx]

    def _instructions(self, is_labeled):
        name = self._current_path().name
        pos = f"{self.idx + 1}/{len(self.all_images)}"
        if self.editing:
            if not self.clicks:
                prompt = "click gauge CENTER"
            else:
                point_idx = min(len(self.clicks) - 1, len(self.point_plan) - 1)
                prompt = f"{self.point_plan[point_idx][0]}: click TIP"
            return f"{name}  ({pos})  EDITING -- {prompt}\nz=restart  esc/c=cancel  q=quit"
        status = "labeled" if is_labeled else "not labeled"
        return (f"{name}  ({pos})  [{status}]\n"
                f"r=relabel  n/->=next  p/<-=prev  q=quit")

    def _render(self):
        path = self._current_path()
        img = plt.imread(path)
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.axis("off")
        row_idx = self._row_index(path.name)
        if self.editing:
            for i, (x, y) in enumerate(self.clicks):
                self.ax.plot(x, y, "bo" if i == 0 else "rx")
        elif row_idx is not None:
            draw_existing_annotation(self.ax, img, self.df.loc[row_idx], self.point_plan)
        self.ax.set_title(self._instructions(row_idx is not None))
        self.fig.canvas.draw()

    def _start_edit(self):
        self.editing = True
        self.clicks = []
        self._render()

    def _cancel_edit(self):
        self.editing = False
        self.clicks = []
        self._render()

    def _goto(self, delta):
        if self.editing:
            return
        self.idx = max(0, min(len(self.all_images) - 1, self.idx + delta))
        self._render()

    def _save_current(self, angles, center):
        path = self._current_path()
        data = {"filename": path.name, **angles,
                CENTER_COLS[0]: center[0], CENTER_COLS[1]: center[1]}
        row_idx = self._row_index(path.name)
        if row_idx is None:
            self.df = pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)
        else:
            for col, val in data.items():
                self.df.loc[row_idx, col] = val
        self.df.to_csv(self.output_path, index=False)
        angle_str = ", ".join(f"{k}={v:.2f}" for k, v in angles.items())
        print(f"{path.name}: {angle_str}, center=({center[0]:.1f}, {center[1]:.1f}) (saved)")

    def _on_click(self, event):
        if not self.editing or event.xdata is None or event.ydata is None:
            return
        self.clicks.append((event.xdata, event.ydata))
        if len(self.clicks) == self.needed:
            center = self.clicks[0]
            angles = {col: angle_from_points(center, self.clicks[i + 1])
                      for i, (_, col) in enumerate(self.point_plan)}
            self._save_current(angles, center)
            self.editing = False
            self.clicks = []
            if self.idx < len(self.all_images) - 1:
                self.idx += 1
        self._render()

    def _on_key(self, event):
        if event.key == "q":
            plt.close(self.fig)
        elif self.editing:
            if event.key == "z":
                self.clicks = []
                self._render()
            elif event.key in ("escape", "c"):
                self._cancel_edit()
        else:
            if event.key in ("n", "right", "s"):
                self._goto(1)
            elif event.key in ("p", "left", "b"):
                self._goto(-1)
            elif event.key == "r":
                self._start_edit()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_path = Path(args.output)
    point_plan = build_point_plan(args.needles)
    expected_cols = [col for _, col in point_plan]

    all_images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    if not all_images:
        print(f"No images found in {image_dir}")
        return

    df = load_labels(output_path, expected_cols)
    labeled = set(df["filename"].tolist())
    remaining = [p for p in all_images if p.name not in labeled]
    print(f"{len(labeled)} already labeled, {len(remaining)} remaining "
          f"({args.needles}-needle + min/max reference marks).")
    print("n/->=next  p/<-/b=prev  r=(re)label current image  q=quit&save")

    session = AnnotationSession(all_images, df, point_plan, output_path)
    session.run()

    print(f"Done. {len(session.df)} total labels in {output_path}")


if __name__ == "__main__":
    main()
