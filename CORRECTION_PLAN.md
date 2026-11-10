# Correction Plan — overlay angle bug, doc gaps, annotation verification tool

## Context

The 2026-07-03 session recorded at `DOCUMENTATION.md:682` ("Metrics plotting,
organized model saving, visual overlays, and augment.py overhaul") added:
metrics.png plotting, per-run output dirs, `draw_angle_overlay()` for test/predict
visual overlays, and an augment.py rewrite to photometric-only + resize.

Reviewing that work: the **angle labels and augmentation are correct**, but the
**overlay rendering is wrong** — which is exactly the "visualization of
predicted/annotated angles is not right" symptom. There are also several stale
doc references left over from the same session, and we want a tool to visually
verify annotations.

## Findings

### 1. BUG (correctness) — `draw_angle_overlay` uses the wrong angle→vector math
`annotate.py::angle_from_points` (annotate.py:61) defines the label as the CCW
angle from the +x axis in a **y-up** frame:
`dx = tip.x-pivot.x`, `dy = -(tip.y-pivot.y)`, `angle = atan2(dy,dx) % 360`.

`common.py::draw_angle_overlay` (common.py:373-375) instead draws with
`angle_rad = radians(angle_deg - 90)`, `dx = cos`, `dy = +sin` in **y-down** pixel
space. That injects a spurious −90° rotation **and** a y-flip, so every arrow is
drawn ~90° off and mirrored.

Evidence:
- Numerically: an annotated 90° (needle straight up) renders pointing **right**.
- Rendered `runs/resnet18/20260703_155742/D1B1A9_vis.png`: the `true_needle`
  arrow points down-right (~400 bar) while the real needle in the photo points
  up-left (~105 bar). Ref-min/ref-max are likewise rotated wrong.
- The stored labels are correct: `angle_1=149.34°`, `ref_min=218.45°`,
  `ref_max=316.91°` all match the photo under the annotation convention.

**Fix (single point of truth):** replace the buggy math with the exact inverse of
`angle_from_points`:
```
angle_rad = math.radians(angle_deg)
dx = radius * math.cos(angle_rad)
dy = -radius * math.sin(angle_rad)   # negate: display/pixel y grows downward
```
This corrects `test.py --vis-dir` and `predict.py --vis-dir` at once (both call
this function).

### 2. Annotation & augmentation ARE held properly (but mis-described)
- Annotation math is consistent and the labels verified correct above.
- `augment.py` preserves angles: `RandomAffine(degrees=0, translate, scale)` —
  pure translation and uniform scale keep the pivot→tip **direction**, so the
  copied labels stay valid; there is no rotation/flip. Correct as-is.
- **Doc gap:** DOCUMENTATION.md and augment.py call this "photometric-only" with
  "no geometric transforms," but translate+scale *are* geometric (angle-
  preserving) transforms. Reword for accuracy (no code change).

### 3. Stale documentation from the same session
- `common.py::build_transforms` docstring (common.py:184) still tells users to run
  `augment.py --rotate --hflip --vflip` — those flags were **removed** this same
  session (DOCUMENTATION.md:705-707).
- `test.py` (test.py:23-28) and `predict.py` (predict.py:20-23) usage examples
  still use the old `runs/exp/...` path; the new default is `runs/<arch>/<timestamp>`.
- DOCUMENTATION.md §"Visual overlays" (698-703) doesn't mention overlays originate
  from the image center, not the true gauge-center pivot.

### 4. New — annotation verification tool (interactive viewer)
New `verify_annotations.py`: an **interactive matplotlib viewer** (like
`annotate.py::label_one`) that steps through the labeled images with next/prev/quit
keys, drawing each image's annotated needle angle(s) + ref-mark angles with the
**corrected** angle→vector math. Reuses `get_angle_columns`/`get_reference_columns`
and the shared vector helper (below) from common.py.

## Decision: store the gauge center (chosen)
So overlay arrows originate exactly at the clicked pivot, `annotate.py` will save
the first (center) click as `center_x`/`center_y` columns, used as the overlay
origin wherever available. This is **backward-compatible with graceful fallback**:
- The existing 170-row `labels.csv` has **no** center columns → tools fall back to
  image center (current behavior) until images are re-annotated. No forced
  re-annotation.
- `predict.py` runs on unlabeled images (no stored center) → image-center fallback.
- `augment.py`'s translate/scale moves the pixel center, so augmented copies get
  **no** center (they're train-only and never overlaid — a non-issue).

## Files to change
1. **common.py** — (a) core fix: add a shared `angle_to_vector(angle_deg, radius)`
   helper implementing the correct inverse of `angle_from_points`
   (`dx=r·cos θ`, `dy=-r·sin θ`), and use it in `draw_angle_overlay` (replacing the
   buggy `-90` / `+sin` math); (b) drop the removed-flag reference in
   `build_transforms` docstring.
2. **annotate.py** — write `center_x`/`center_y` (the first click) into the labels
   CSV; make the column-schema check accept both centered and legacy centerless
   files (append centers going forward; old rows stay NaN).
3. **split_dataset.py** — carry `center_x`/`center_y` through for original rows when
   present (augmented rows get NaN); skip cleanly when the columns are absent.
   Centers are not used for stratification.
4. **test.py** — pass the per-image stored center to `draw_angle_overlay` when the
   columns are present in `test.csv`; else image center. `predict.py` — image
   center (documented).
5. **augment.py** + **DOCUMENTATION.md** §augmentation — reword "photometric-only /
   no geometric" to "photometric + angle-preserving geometric (small translate/
   scale; no rotation/flip)".
6. **test.py**, **predict.py** — update `runs/exp` usage examples to
   `runs/<arch>/<timestamp>`.
7. **DOCUMENTATION.md** — document the overlay fix, the new `center_x`/`center_y`
   columns + fallback, `verify_annotations.py` in the tooling list, and add a dated
   record entry.
8. **verify_annotations.py** (new).

## Verification
- Numeric unit check on `angle_to_vector`: 0°→right `(r,0)`, 90°→up `(0,-r)`,
  180°→left `(-r,0)`, 270°→down `(0,r)`; and round-trip
  `angle_from_points(pivot, pivot+angle_to_vector(θ))` ≈ θ.
- Re-render D1B1A9 via `predict.py --vis-dir` (or the viewer) and confirm the needle
  arrow now lies **along** the needle (up-left ~105 bar), ref-min lower-left,
  ref-max lower-right.
- Run `verify_annotations.py` over the current `labels.csv` (centerless → image-
  center origin) and spot-check a handful; then re-annotate one image to confirm the
  stored-center origin path works end-to-end (annotate → verify).
