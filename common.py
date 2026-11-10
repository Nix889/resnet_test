"""
Shared building blocks for the gauge-needle-angle regression project:
dataset loading, model construction, transforms, train/eval loops,
training-history plotting, and angle-overlay drawing.

Imported by train.py, test.py, predict.py and compare_pretrained.py so the
data pipeline and model definition are identical everywhere they're used.
"""
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Fixed on-disk layout this project expects (see DOCUMENTATION.md):
#   images/train/original   - raw labeled images + labels.csv (source of truth)
#   images/train/augmented  - offline-augmented images + labels.csv, built by augment.py
#   images/train/dataset    - train.csv/val.csv/test.csv, built by split_dataset.py
#   images/predict          - unlabeled images, default target for predict.py
# All scripts accept path overrides, but default to this layout.
IMAGES_DIR = Path("images")
TRAIN_DIR = IMAGES_DIR / "train"
ORIGINAL_DIR = TRAIN_DIR / "original"
AUGMENTED_DIR = TRAIN_DIR / "augmented"
DATASET_DIR = TRAIN_DIR / "dataset"
PREDICT_DIR = IMAGES_DIR / "predict"

# Maps a CLI-friendly arch name to its torchvision constructor and its
# pretrained-weights enum, so build_model() can support new depths (101, 152,
# ...) just by adding a line here.
ARCH_TABLE = {
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    "resnet34": (models.resnet34, models.ResNet34_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
    "resnet101": (models.resnet101, models.ResNet101_Weights),
    "resnet152": (models.resnet152, models.ResNet152_Weights),
}

# Angle columns follow the "angle_<needle index>" convention (angle_1,
# angle_2, ...) so any script can auto-detect how many needles a given
# labels/dataset CSV describes just from its header. This is what lets you
# switch between one- and two-needle gauges by pointing scripts at a
# different CSV (see annotate.py --needles), without a separate mode flag
# needed everywhere else.
ANGLE_COL_RE = re.compile(r"^angle_(\d+)$")

# Reference-mark columns: the angular position of the gauge scale's min and
# max ticks, measured the same way as a needle angle (from the gauge center
# -- the shared pivot clicked once per image in annotate.py -- to the mark's
# tip). annotate.py always labels both. Kept as fixed, distinct column names
# rather than folded into the angle_N numbering, so needle-count
# auto-detection is never affected by whether reference marks are also
# present (older CSVs without them still load). See DOCUMENTATION.md,
# "Reference marks and computing a physical reading."
REFERENCE_COLS = ["ref_angle_min", "ref_angle_max"]

# Pixel coordinates of the gauge center click (the shared pivot every angle
# in a row is measured from) -- written by annotate.py going forward. Kept
# optional: rows labeled before this column existed simply lack it, and
# every consumer (test.py, verify_annotations.py) falls back to the image
# center for those rows rather than requiring re-annotation.
CENTER_COLS = ["center_x", "center_y"]


def get_angle_columns(df) -> list:
    """Returns the angle_N (needle) columns present in `df`, sorted by
    needle index."""
    matches = []
    for c in df.columns:
        m = ANGLE_COL_RE.match(c)
        if m:
            matches.append((int(m.group(1)), c))
    matches.sort(key=lambda t: t[0])
    return [c for _, c in matches]


def get_reference_columns(df) -> list:
    """Returns [ref_angle_min, ref_angle_max] if both are present in `df`,
    [] if neither is. Errors if only one is, since a lone reference mark
    can't define a scale."""
    present = [c for c in REFERENCE_COLS if c in df.columns]
    if len(present) == 1:
        raise ValueError(
            f"Found only '{present[0]}' -- reference marks must be labeled as a "
            f"pair ({REFERENCE_COLS}), not one alone."
        )
    return present


def get_center_columns(df) -> list:
    """Returns [center_x, center_y] if both are present in `df`, [] if
    neither is (older labels files, or any row predating this column).
    Errors if only one is, since a lone coordinate can't define a pivot."""
    present = [c for c in CENTER_COLS if c in df.columns]
    if len(present) == 1:
        raise ValueError(
            f"Found only '{present[0]}' -- the gauge center must have both "
            f"{CENTER_COLS}, not one alone."
        )
    return present


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch: str, pretrained: bool = True, num_needles: int = 1,
                 num_reference: int = 0) -> nn.Module:
    """Build a ResNet with its classification head replaced by a
    2*(num_needles + num_reference)-wide linear regression output: (sin, cos)
    of each needle's angle, followed by (sin, cos) of each reference-mark
    angle (ref_angle_min, ref_angle_max), if any. Predicting (sin, cos)
    rather than a raw degree value is what makes the output correctly
    circular -- 359deg and 1deg end up close together in (sin, cos) space,
    as they should be, instead of being penalized as if they were nearly
    opposite. See DOCUMENTATION.md, "Angle encoding: circular (sin/cos),
    full 360 degrees"."""
    if arch not in ARCH_TABLE:
        raise ValueError(
            f"Unknown arch '{arch}'. Supported: {sorted(ARCH_TABLE)}"
        )
    ctor, weights_cls = ARCH_TABLE[arch]
    weights = weights_cls.DEFAULT if pretrained else None
    model = ctor(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2 * (num_needles + num_reference))
    return model


def encode_angles(angles_deg) -> torch.Tensor:
    """[a1, a2, ...] degrees -> flat tensor [sin1, cos1, sin2, cos2, ...]."""
    angles_rad = torch.deg2rad(torch.as_tensor(angles_deg, dtype=torch.float32))
    return torch.stack([torch.sin(angles_rad), torch.cos(angles_rad)], dim=-1).flatten()


def decode_angles(vec: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_angles, batched: (..., 2*N) -> (..., N) degrees in
    [0, 360). Works whether or not the (sin, cos) pair is unit-normalized,
    since atan2 only depends on their ratio/signs."""
    n_needles = vec.shape[-1] // 2
    vec = vec.reshape(*vec.shape[:-1], n_needles, 2)
    deg = torch.rad2deg(torch.atan2(vec[..., 0], vec[..., 1]))
    return deg % 360


def circular_abs_error_deg(pred_deg: torch.Tensor, true_deg: torch.Tensor) -> torch.Tensor:
    """Smallest absolute difference between two angles in degrees, correctly
    treating e.g. 359 vs 1 as 2 degrees apart rather than 358."""
    diff = (pred_deg - true_deg + 180) % 360 - 180
    return diff.abs()


def compute_reading(needle_deg, ref_min_deg, ref_max_deg, value_min: float, value_max: float):
    """Converts a needle angle into a physical reading, using this image's
    own predicted (or ground-truth) ref_angle_min/ref_angle_max.

    Between ref_min_deg and ref_max_deg there are always two candidate arcs
    (going one way around the circle vs. the other); only one of them is the
    gauge's actual scale, the other is the dead zone the needle never
    enters. Rather than assuming a fixed sweep direction (which would break
    for any gauge that doesn't match the assumption), this picks whichever
    arc actually contains the needle -- which it always does, by physical
    construction. See DOCUMENTATION.md, "Which arc is the scale."

    All of needle_deg/ref_min_deg/ref_max_deg may be Python floats or
    (broadcastable) torch tensors; the result matches.
    """
    needle_deg = torch.as_tensor(needle_deg, dtype=torch.float32)
    ref_min_deg = torch.as_tensor(ref_min_deg, dtype=torch.float32)
    ref_max_deg = torch.as_tensor(ref_max_deg, dtype=torch.float32)

    span_cw = (ref_min_deg - ref_max_deg) % 360
    progress_cw = (ref_min_deg - needle_deg) % 360
    fraction_cw = progress_cw / span_cw.clamp(min=1e-6)
    fraction_ccw = (360 - progress_cw) / (360 - span_cw).clamp(min=1e-6)
    fraction = torch.where(progress_cw <= span_cw, fraction_cw, fraction_ccw)
    fraction = fraction.clamp(0.0, 1.0)
    return value_min + fraction * (value_max - value_min)


def build_transforms(img_size: int = 224, augment: bool = False):
    """
    Train/eval transforms.

    This in-memory pipeline is photometric-only (color jitter, blur, random
    erasing) -- it never rotates or flips, because these torchvision
    transforms only see the image, not its angle label(s), so they have no
    way to relabel a rotated/flipped sample. For additional on-disk
    augmentation (photometric plus small angle-preserving translate/scale;
    still no rotation/flip), see `augment.py`.
    """
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if not augment:
        return eval_tf, eval_tf

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])
    return train_tf, eval_tf


class GaugeAngleDataset(Dataset):
    """Reads a CSV with a `filename` column plus one or more `angle_N`
    columns (angle_1 for a single-needle gauge, angle_1 + angle_2 for a
    two-needle gauge -- auto-detected from the header), optionally followed
    by `ref_angle_min`/`ref_angle_max` (the gauge scale's reference marks,
    also auto-detected), and the images they refer to, resolved as img_dir /
    filename. `filename` may include a subfolder prefix (e.g.
    "original/gauge_0001.jpg" or "augmented/gauge_0001_aug0.jpg") so a
    single img_dir (TRAIN_DIR) can serve rows pointing into either the
    original/ or augmented/ sibling folder -- this is exactly what
    split_dataset.py's output CSVs do. See DOCUMENTATION.md for the full
    expected layout.

    The target tensor is [needle angles..., reference angles...] encoded
    together via encode_angles(); `.num_needles`/`.num_reference` record how
    to split model output back into the two groups (see test.py/predict.py)."""

    def __init__(self, csv_file, img_dir, transform=None, filename_col: str = "filename"):
        self.df = pd.read_csv(csv_file)
        if filename_col not in self.df.columns:
            raise ValueError(
                f"CSV must contain a '{filename_col}' column, found {list(self.df.columns)}"
            )
        self.angle_cols = get_angle_columns(self.df)
        if not self.angle_cols:
            raise ValueError(
                f"No angle_N columns found in {csv_file} (expected e.g. 'angle_1', "
                f"plus 'angle_2' for two-needle gauges); found {list(self.df.columns)}"
            )
        self.ref_cols = get_reference_columns(self.df)
        self.num_needles = len(self.angle_cols)
        self.num_reference = len(self.ref_cols)
        self.target_cols = self.angle_cols + self.ref_cols
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.filename_col = filename_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / str(row[self.filename_col])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = encode_angles([float(row[c]) for c in self.target_cols])
        return image, target


def run_epoch(model, loader, device, criterion, optimizer=None, desc=None, scaler=None):
    """One pass over `loader`. Trains if optimizer is given, otherwise just
    evaluates. Returns (avg_loss, mae_degrees), where mae_degrees is the
    circular mean absolute error averaged over every needle and sample.

    `scaler` (a torch.amp.GradScaler) enables CUDA mixed-precision training:
    forward+loss run under autocast, backward/step go through the scaler.
    On CPU (or if scaler is None/disabled), this is fully equivalent to plain
    fp32 training -- autocast(enabled=False) and a disabled GradScaler are
    no-ops, so behavior there is unchanged."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    use_amp = device.type == "cuda" and scaler is not None and scaler.is_enabled()

    total_loss, total_abs_err, n_samples, n_errs = 0.0, 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds = model(images)
                loss = criterion(preds, targets)

            if is_train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            # Decode in fp32 regardless of autocast, so the MAE metric is
            # never computed at reduced precision.
            pred_deg = decode_angles(preds.detach().float())
            true_deg = decode_angles(targets.float())
            abs_err = circular_abs_error_deg(pred_deg, true_deg)
            total_abs_err += abs_err.sum().item()
            n_errs += abs_err.numel()

    return total_loss / n_samples, total_abs_err / n_errs


def get_criterion(name: str):
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "smoothl1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss '{name}'. Use mse, l1 or smoothl1.")


def plot_training_history(history, output_dir, prefix=""):
    """Plot training/validation loss and MAE vs epoch and save as metrics.png."""
    import matplotlib.pyplot as plt
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_mae = [h["train_mae"] for h in history]
    val_mae = [h["val_mae"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_loss, label="train_loss")
    ax1.plot(epochs, val_loss, label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{prefix}Loss vs Epoch")
    ax1.legend()

    ax2.plot(epochs, train_mae, label="train_mae")
    ax2.plot(epochs, val_mae, label="val_mae")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (deg)")
    ax2.set_title(f"{prefix}MAE vs Epoch")
    ax2.legend()

    fig.tight_layout()
    plot_path = Path(output_dir) / "metrics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{prefix}Metrics plot saved to {plot_path}")


def angle_to_vector(angle_deg, radius=1.0):
    """The exact inverse of annotate.py::angle_from_points(): given the
    counter-clockwise-from-+x-axis, y-up angle convention used for every
    stored label, returns the (dx, dy) pixel offset from a pivot to a point
    `radius` away at that angle. `dy` is negated relative to the math
    convention because pixel/display y grows downward while the labeling
    convention is y-up -- so `angle_from_points(pivot, (pivot[0] + dx,
    pivot[1] + dy))` recovers `angle_deg`. This is the single point of truth
    for turning a stored angle back into a drawable direction; anything that
    renders an angle (draw_angle_overlay, verify_annotations.py) should go
    through this rather than re-deriving the trig."""
    angle_rad = math.radians(angle_deg)
    dx = radius * math.cos(angle_rad)
    dy = -radius * math.sin(angle_rad)
    return dx, dy


def draw_angle_overlay(image_path, angles, output_path, center=None,
                       radius_ratio=0.4, colors=None, linewidth=3):
    """Draw angle line(s) on an image and save the result.

    image_path: path to the source image
    angles: dict mapping label -> angle_in_degrees
    output_path: where to save the annotated image
    center: (x, y) pivot in pixels; defaults to image center
    radius_ratio: line length as fraction of smaller image dimension
    colors: optional dict mapping label -> matplotlib color string
    """
    import matplotlib.pyplot as plt
    img = plt.imread(str(image_path))
    h, w = img.shape[:2]
    cx, cy = (w / 2, h / 2) if center is None else center
    radius = min(h, w) * radius_ratio

    if colors is None:
        palette = ["red", "lime", "cyan", "magenta", "yellow", "orange"]
        colors = {label: palette[i % len(palette)] for i, label in enumerate(angles)}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.axis("off")

    for label, angle_deg in angles.items():
        dx, dy = angle_to_vector(angle_deg, radius)
        ax.arrow(cx, cy, dx, dy, head_width=radius * 0.04,
                 head_length=radius * 0.06, fc=colors[label],
                 ec=colors[label], linewidth=linewidth, label=label)

    ax.legend(loc="upper right", fontsize=10)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
