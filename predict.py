"""
Run inference with a trained checkpoint on new, unlabeled gauge images
(no CSV of ground-truth angles required -- use test.py for that).

The number of needles the checkpoint was trained for (1 or 2), and whether
it also predicts ref_angle_min/ref_angle_max reference marks, are both read
from the checkpoint itself. Pass --value-min/--value-max (the physical
values at those reference marks, e.g. 0 and 160) to print the actual
computed reading instead of just raw angles -- only works for a checkpoint
trained with reference marks. See DOCUMENTATION.md, "Reference marks and
computing a physical reading."

Pass --vis-dir to save per-image visual overlays with predicted angle
lines drawn on each image. Unlike test.py, these images have no stored
gauge center (they're unlabeled), so overlay arrows always originate from
the image center.

Defaults to images/predict/ -- the folder reserved exclusively for
prediction-time images (see DOCUMENTATION.md).

Examples:
    python predict.py --checkpoint runs/resnet50/20260703_143022/best_model.pt
    python predict.py --checkpoint runs/resnet50/20260703_143022/best_model.pt --image gauge_0001.jpg
    python predict.py --checkpoint runs/resnet50/20260703_143022/best_model.pt --value-min 0 --value-max 160 --unit PSI
    python predict.py --checkpoint runs/resnet50/20260703_143022/best_model.pt --vis-dir runs/resnet50/20260703_143022/vis
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from common import (PREDICT_DIR, build_model, build_transforms, compute_reading,
                     decode_angles, draw_angle_overlay)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--image", help="Path to a single image")
    group.add_argument("--image-dir", default=None,
                        help=f"Directory of images to predict on (default: {PREDICT_DIR})")
    p.add_argument("--value-min", type=float, default=None,
                   help="Physical value at ref_angle_min (requires a checkpoint with reference marks)")
    p.add_argument("--value-max", type=float, default=None,
                   help="Physical value at ref_angle_max")
    p.add_argument("--unit", default="", help="Unit label for printed readings, e.g. PSI")
    p.add_argument("--vis-dir", default="images/predict/results",
                   help="Directory to save visual overlay images (default: alongside each input image)")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if not args.image and not args.image_dir:
        args.image_dir = str(PREDICT_DIR)
    if (args.value_min is None) != (args.value_max is None):
        p.error("--value-min and --value-max must be given together.")
    return args


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_needles = ckpt["num_needles"]
    num_reference = ckpt.get("num_reference", 0)
    model = build_model(ckpt["arch"], pretrained=False, num_needles=num_needles,
                         num_reference=num_reference)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    _, eval_tf = build_transforms(ckpt["img_size"], augment=False)
    return model, eval_tf, num_needles, num_reference


def predict_one(model, transform, image_path, device):
    """Returns the full decoded angle vector in degrees: needle angle(s)
    followed by reference-mark angles (ref_min, ref_max), if any."""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model(tensor).cpu()
    return decode_angles(raw)[0]


def format_prediction(angles, num_needles, value_min=None, value_max=None, unit=""):
    needle_angles = angles[:num_needles]
    ref_angles = angles[num_needles:]
    unit_suffix = f" {unit}" if unit else ""

    if value_min is not None:
        readings = [compute_reading(a, ref_angles[0], ref_angles[1], value_min, value_max).item()
                    for a in needle_angles]
        if num_needles == 1:
            return f"{readings[0]:.2f}{unit_suffix}"
        return ", ".join(f"needle_{i + 1}={r:.2f}{unit_suffix}" for i, r in enumerate(readings))

    if num_needles == 1:
        parts = [f"{needle_angles[0].item():.2f} deg"]
    else:
        parts = [f"needle_{i + 1}={a.item():.2f} deg" for i, a in enumerate(needle_angles)]
    if len(ref_angles):
        parts.append(f"ref_min={ref_angles[0].item():.2f} deg")
        parts.append(f"ref_max={ref_angles[1].item():.2f} deg")
    return ", ".join(parts)


def main():
    args = parse_args()
    device = get_device(args.device)
    model, transform, num_needles, num_reference = load_model(args.checkpoint, device)

    if args.value_min is not None and num_reference != 2:
        raise ValueError(
            "--value-min/--value-max require a checkpoint trained with reference marks "
            "(ref_angle_min/ref_angle_max) -- this one has none. Re-label with "
            "annotate.py (it now always labels both reference marks) and retrain, "
            "or omit --value-min/--value-max to see raw angles."
        )

    vis_dir = Path(args.vis_dir) if args.vis_dir else None

    def report(path, display_name):
        angles = predict_one(model, transform, path, device)
        text = format_prediction(angles, num_needles, args.value_min, args.value_max, args.unit)
        print(f"{display_name}: {text}")

        if vis_dir:
            overlay_angles = {}
            for i in range(num_needles):
                suffix = f"_{i + 1}" if num_needles > 1 else ""
                label = "needle" if num_needles == 1 else f"needle_{i + 1}"
                overlay_angles[f"pred_{label}"] = float(angles[i])
            if num_reference:
                overlay_angles["pred_ref_min"] = float(angles[num_needles])
                overlay_angles["pred_ref_max"] = float(angles[num_needles + 1])
            out_name = Path(display_name).stem + "_vis.png"
            draw_angle_overlay(path, overlay_angles, vis_dir / out_name)

    if args.image:
        report(args.image, args.image)
    else:
        image_dir = Path(args.image_dir)
        paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
        if not paths:
            print(f"No images found in {image_dir}")
            return
        for path in paths:
            report(path, path.name)


if __name__ == "__main__":
    main()
