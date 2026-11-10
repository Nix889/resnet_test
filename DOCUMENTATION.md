# Gauge Needle Angle Regression — Analysis & Development Record

## 1. Problem framing

The task is: given a photo of an analog gauge, predict the angle of its
needle, as a real number in degrees. That is a **regression** problem, not
classification, even though the model architecture (ResNet) is normally
associated with classification.

### Why "just change the output layer" works

A torchvision ResNet ends in `model.fc`, a single `nn.Linear(in_features,
num_classes)` layer sitting on top of the convolutional feature extractor.
For classification, `num_classes` is the number of categories and the layer
is trained with cross-entropy against integer labels. For this task the
change is:

```python
model.fc = nn.Linear(in_features, 2 * num_needles)
```

`2 * num_needles` outputs, trained with a regression loss (`MSELoss`,
`L1Loss`, or `SmoothL1Loss`) against `(sin, cos)` pairs instead of a class
index (see "Angle encoding" below for why `sin`/`cos` rather than a raw
degree value). Everything below `fc` (the convolutional backbone) is
unchanged — it's still doing the same job it always does: turning an image
into a feature vector. Only the head and the loss target change. This is
implemented in `common.py::build_model()`.

### Loss function choice

Three options are wired up via `--loss`, all operating elementwise on the
`(sin, cos)` output vector:

| Loss | Behavior | When it's the better pick |
|---|---|---|
| `mse` | Quadratic penalty | Clean data, few outliers/mislabels |
| `l1` | Linear penalty | Robust to outliers, but noisier gradients near 0 |
| `smoothl1` (Huber) | Quadratic near 0, linear for large errors | **Default.** Good middle ground — labeling a needle angle by hand (see §4) is itself imprecise, so a loss that doesn't let a few noisy labels dominate the gradient is a safer default. |

### Angle encoding: circular (sin/cos), full 360 degrees

Earlier versions of this project assumed gauges never wrap around (e.g. a
0-270° pressure gauge) and regressed the raw degree value directly through a
single output neuron. Feedback from the user overturned that assumption:
some dials genuinely sweep the full 360° (e.g. compass-style dials), and
annotation shouldn't be range-limited at all. Regressing a raw degree value
directly is a bad fit for that case, because 1° and 359° are visually almost
identical but numerically far apart (358° of naive error), which distorts
both the loss and any bounded-arc assumption in the code.

The fix, now used for every gauge regardless of how wide its actual sweep
is: each needle's angle is encoded as `(sin θ, cos θ)` --

- `common.py::encode_angles()` — degrees to `(sin, cos)` pairs, used when
  building the dataset's target tensor (`GaugeAngleDataset.__getitem__`).
- `common.py::decode_angles()` — the inverse, `atan2(sin, cos) mod 360`,
  used to turn model output (and targets) back into degrees for logging,
  `test.py`, and `predict.py`. Works even if the predicted `(sin, cos)` pair
  isn't unit-normalized, since `atan2` only depends on their ratio/signs.
- `common.py::circular_abs_error_deg()` — `((pred - true + 180) mod 360) -
  180`, then `abs()`. This is the metric reported everywhere ("MAE
  degrees"), and it correctly scores 359° vs. 1° as 2° apart, not 358°.

This one representation now covers both a bounded-arc gauge (whose labels
simply never populate most of the circle) and a full-360° gauge, so there's
no separate "bounded" vs. "circular" mode to choose between — annotation,
training and evaluation are unrestricted to any range by construction.

### Evaluation metric

Mean Absolute Error (MAE) in degrees is used throughout (`common.py::run_epoch`,
`test.py`), computed circularly as described above — it's directly
interpretable ("the model is off by 4.2° on average") in a way that raw loss
values are not. For a multi-needle model, `run_epoch`'s MAE is averaged over
every needle and sample; `test.py` additionally breaks it down per needle,
plus RMSE, max error, and the fraction of predictions within 5° and 10°,
useful thresholds for judging "is this accurate enough to read a real
gauge."

### Data augmentation: photometric (always safe)

`build_transforms()`'s in-memory `--augment` pipeline (used during training,
regenerated every epoch) is **photometric-only**: brightness/contrast jitter
and slight blur. `augment.py`'s on-disk pipeline additionally applies small
translate/scale, which *are* geometric transforms -- but pure translation and
uniform scale don't rotate or mirror the image, so the pivot->tip *direction*
of every needle/reference-mark angle is unchanged, and the angle labels can
still be copied unchanged. Only rotation/flip would require relabeling, and
neither is applied by either pipeline, so no label relabeling is ever
required.

Every image produced by `augment.py` is resized to a standard 120×120 pixels,
and the adjustable `--brightness`/`--contrast` flags control the strength of
the ColorJitter transform.

### Pretrained vs. from-scratch

Per user request, both are implemented and directly comparable:

- `train.py --pretrained` (default) starts from ImageNet weights
  (`torchvision.models.ResNetXX_Weights.DEFAULT`) — usually faster to
  converge and more label-efficient, since low-level filters (edges,
  gradients, circles — all relevant to reading a needle) transfer well from
  ImageNet.
- `train.py --no-pretrained` trains the same architecture from random
  initialization.
- `compare_pretrained.py` runs both back-to-back with identical data,
  hyperparameters, and random seed, then writes `comparison.json` and a
  `comparison.png` plot of validation MAE per epoch for both runs, so the
  choice can be justified empirically on your actual dataset rather than
  assumed.

### Supported architectures

`common.py::ARCH_TABLE` maps `resnet18/34/50/101/152` to their torchvision
constructors and weights enums. All scripts accept `--arch <name>`; adding a
deeper/newer variant (e.g. a Wide ResNet) is one line in that table.

### One or two needles, switchable independently

Per user feedback, a gauge may have up to two needles at once, and the
project needs to let a one-needle model be tested before committing to a
two-needle one. Rather than a `--needles` flag threaded through every
script, needle count is **inferred from the labels CSV itself**:

- Angle columns follow the convention `angle_1` (single needle) or
  `angle_1` + `angle_2` (two needles). `common.py::get_angle_columns()`
  auto-detects however many of these a given CSV has via the regex
  `^angle_(\d+)$`.
- `GaugeAngleDataset` exposes `.num_needles`/`.angle_cols` from whatever CSV
  it's given; `build_model(..., num_needles=...)` sizes the output layer
  (`2 * num_needles`) to match; `train.py` reads `num_needles` off the
  dataset and stores it in the checkpoint, so `test.py`/`predict.py` never
  need it passed in either.
- The only script that needs an explicit mode is `annotate.py --needles
  {1,2}`, since it's the one place needle count can't be inferred (nothing
  to infer it *from* yet) — it determines how many needle tip clicks to
  collect per image (after the single shared gauge-center click) and which
  `angle_N` columns to write.

This means testing one-needle vs. two-needle is just a matter of which
labels file you annotate into and which resulting `dataset/*.csv` you point
`train.py` at — see §2/§3 for the concrete two-CSV workflow. There is no
assumption anywhere that needle 1 and needle 2 are interchangeable: whichever
needle you click first when labeling becomes (and must stay) `angle_1` for
every image, since the model has no other way to learn a consistent mapping
between "first clicked" and "which physical needle."

### Reference marks and computing a physical reading

The original design (before this project existed) used two separate
classical CV algorithms: one to find the gauge's center via circle
detection, another to find the scale's min/max tick marks, so a needle angle
could be converted into an actual physical reading (e.g. "47 PSI") by
interpolating between them. The question that led to this section: could
the CNN just predict the reference marks too, and skip the separate
classical algorithms entirely?

**Yes, and it reuses the exact same machinery already built for needles,**
rather than adding a new kind of output:

- The reference marks are regressed as **angles from a pivot**, exactly like
  a needle, not as raw `(x, y)` pixel coordinates. A CNN never needs to
  explicitly output "the center is at pixel (120, 85)" to answer "what
  angle is this" -- it already doesn't, for the needle -- so there's no
  reason the reference marks would need that either. Raw pixel-coordinate
  regression from a globally-pooled feature is also just a weaker fit for
  this than the network directly answering an angle, and it would carry
  irrelevant information (absolute framing/zoom) that the actual downstream
  question (fraction of the way from min to max) doesn't care about.
- Concretely, this means two new columns, `ref_angle_min`/`ref_angle_max`
  (`common.py::REFERENCE_COLS`), auto-detected via
  `get_reference_columns()` exactly like `angle_N` needle columns are via
  `get_angle_columns()` -- and treated identically by every downstream
  script (same encode/decode, same rotate/flip relabeling in `augment.py`,
  same carry-through in `split_dataset.py`). `build_model()`'s output layer
  becomes `2 * (num_needles + num_reference)`; `GaugeAngleDataset` builds one
  combined target `[needle angles..., ref_angle_min, ref_angle_max]` and
  exposes `.num_needles`/`.num_reference` so `test.py`/`predict.py` know how
  to split the decoded output back into its two halves.
- `annotate.py` always labels both reference marks (there is no opt-out
  flag): after the needle tip(s), you click the min mark, then the max
  mark. Every point in an image -- each needle, the min mark, the max mark
  -- measures its angle from one **shared pivot: the gauge's center**,
  clicked exactly once per image as the first click. For the single- and
  two-needle gauges currently in scope, everything physically shares the
  dial center, so a per-point pivot click was pure annotation overhead;
  sharing the center click removes it. (An earlier iteration gave every
  point its own independent pivot+tip pair to allow for multi-dial gauges
  with separate centers -- see the development record -- and that design
  could be restored if such gauges enter scope.)

**Which physical values `value_min`/`value_max` correspond to is *not*
predicted by the model.** Reading the printed "0" and "160" off a dial face
is a text/OCR problem, a fundamentally different task from angle regression,
and per the user's confirmation those values are fixed and already known per
gauge type -- so they're supplied as plain `--value-min`/`--value-max`
arguments to `test.py`/`predict.py` at evaluation/inference time, not
trained on.

**Which arc is the scale.** Between `ref_angle_min` and `ref_angle_max`
there are always two candidate arcs around the circle, and only one of them
is the gauge's actual scale -- the other is the dead zone the needle
mechanically never enters. Rather than assume a fixed sweep direction (which
would break for any gauge that doesn't match the assumption, and camera
framing here is known to vary shot-to-shot, so there's no reliable fixed
per-type convention to lean on), `common.py::compute_reading()` uses the
needle's own predicted angle to pick the arc automatically, every time,
since the needle is guaranteed by physical construction to be inside the
valid one:

```python
span_cw = (ref_min_deg - ref_max_deg) % 360        # size of the clockwise-from-min arc
progress_cw = (ref_min_deg - needle_deg) % 360      # how far the needle is along it, going that way
if progress_cw <= span_cw:
    fraction = progress_cw / span_cw                # needle is on the CW arc
else:
    fraction = (360 - progress_cw) / (360 - span_cw)  # needle is on the CCW arc instead
fraction = clip(fraction, 0, 1)
reading = value_min + fraction * (value_max - value_min)
```

Verified against hand-derived expected values during development (needle at
`ref_angle_min` → `value_min`; at `ref_angle_max` → `value_max`; exactly
halfway around a 270° arc → the midpoint value; a reversed min/max
convention still resolves correctly) -- see the development record.

One acknowledged limitation: if a gauge's true sweep is very close to
exactly 180°, the two candidate arcs are nearly the same size, and a noisy
needle prediction near the middle becomes genuinely ambiguous between them.
That's an identifiability limit of using only these three angles, not a bug
-- it doesn't affect the vast majority of analog gauges, which sweep
roughly 90-270°, but would need a 4th reference point (or a fixed sweep
convention as a tie-breaker) to resolve for a gauge that really does use a
~180° arc.

**`test.py`/`predict.py --value-min`/`--value-max`/`--unit`** convert
predicted angles into an actual reading. `test.py` reports it two ways: the
fully-predicted reading (both needle and reference angles from the model --
the real deployment number) and an "oracle ref" diagnostic (predicted
needle angle combined with *ground-truth* reference angles), which isolates
how much of the reading error is coming from the needle prediction alone vs.
the reference-mark predictions -- useful for knowing which part of the
model needs more training data if reading accuracy isn't good enough yet.

---

## 2. Data format

The fixed on-disk layout (`common.py` exposes each path as a constant):

```
images/
  train/
    original/          # your raw labeled images -- source of truth
      gauge_0001.jpg
      gauge_0002.jpg
      ...
      labels.csv        # filename,angle_1[,angle_2][,ref_angle_min,ref_angle_max]  (built by annotate.py, or your own)
    augmented/          # sibling folder, generated by augment.py
      gauge_0001_aug0.jpg
      ...
      labels.csv        # filename,angle_1[,angle_2][,ref_angle_min,ref_angle_max],source_filename
    dataset/            # sibling folder, generated by split_dataset.py
      train.csv         # same columns -- filenames prefixed "original/" or "augmented/"
      val.csv           # same columns -- always "original/" only, never augmented
      test.csv          # same columns -- always "original/" only, never augmented
  predict/              # unlabeled images, exclusively for predict.py
```

`original/labels.csv` and `augmented/labels.csv` schema:

| column | type | meaning |
|---|---|---|
| `filename` | string | image filename (bare, no subfolder) |
| `angle_1` | float | needle 1's angle in degrees (0-360, unrestricted) |
| `angle_2` | float | needle 2's angle in degrees -- present only for two-needle gauges |
| `ref_angle_min` | float | the scale's min-tick angle in degrees -- always written by `annotate.py` (older labels files without it still load) |
| `ref_angle_max` | float | the scale's max-tick angle in degrees -- always written by `annotate.py` (older labels files without it still load) |
| `center_x` / `center_y` | float | pixel coordinates of the gauge-center click -- the shared pivot every angle in the row was measured from. Written by `annotate.py` for every newly-labeled row; absent (NaN) on rows labeled before this column existed. `test.py --vis-dir` and `verify_annotations.py` use it as the overlay arrow origin when present, falling back to the image center otherwise; `predict.py` runs on unlabeled images with no stored center, so it always uses the image-center fallback. |
| `source_filename` | string | (`augmented/labels.csv` only) which original image this was generated from |

`dataset/{train,val,test}.csv` schema is the same columns, but `filename` is
prefixed with its subfolder (`original/...` or `augmented/...`) so a single
`--img-dir images/train` resolves every row — see `GaugeAngleDataset` in
`common.py`. Needle count (1 or 2) and whether reference marks are present
are both read off however many `angle_N`/`ref_angle_*` columns exist; see
§1, "One or two needles" and "Reference marks."

### Building labels for unlabeled images: `annotate.py`

Since the images on hand aren't labeled yet, `annotate.py` provides a
minimal click-based workflow: it opens each image in `images/train/original`,
you click the gauge's center once (the shared pivot for everything in that
image), then needle 1's tip -- and, in `--needles 2` mode, needle 2's tip
right after -- then the scale's min reference mark and its max reference
mark (always, for every image). Each angle is computed as the
counter-clockwise angle from the image's positive x-axis to the center→tip
vector (0-360°, unrestricted -- full-360° dials are labeled exactly like
bounded-arc gauges). Progress is saved to
`images/train/original/labels.csv` after every completed edit, so the
process can be stopped and resumed. The tool steps through *every* image,
not just unlabeled ones (it starts at the first unlabeled one, but `n`/`p`
navigate freely across the whole set): an already-labeled image shows its
current annotation as an overlay so you can spot a bad label at a glance,
and `r` re-enters click mode to overwrite it -- this is how a mislabeled or
displaced annotation gets fixed without hand-editing the CSV or starting a
new run from scratch.

```
python annotate.py                # 1 needle + scale reference marks (default)
python annotate.py --needles 2   # 2 needles + scale reference marks
```

Every point (each needle, the min mark, the max mark) shares the single
gauge-center click as its pivot and gets one tip click of its own -- see
§1, "Reference marks," for the history of this choice. The center click
itself is also saved (`center_x`/`center_y`), so overlay tools (`test.py
--vis-dir`, `verify_annotations.py`) can draw arrows from the true pivot
instead of the image center; labels files written before this column
existed simply have it blank for those rows, and those tools fall back to
the image center for them -- no forced re-annotation.

This defines the "relative to image frame" labeling convention referenced
in the augmentation section above. If your gauges are always photographed in
a consistent, fixed orientation, this is equivalent to "relative to the
dial's zero mark" for practical purposes.

**Testing one needle, then two, on the same photos:** label into two
separate files and keep them (and their downstream datasets) side by side
rather than overwriting one with the other:

```
python annotate.py --needles 1 --output images/train/original/labels_1needle.csv
python annotate.py --needles 2 --output images/train/original/labels_2needle.csv

python split_dataset.py --original-labels images/train/original/labels_1needle.csv \
    --output-dir images/train/dataset_1needle
python split_dataset.py --original-labels images/train/original/labels_2needle.csv \
    --output-dir images/train/dataset_2needle

python train.py --train-csv images/train/dataset_1needle/train.csv \
    --val-csv images/train/dataset_1needle/val.csv --output-dir runs/1needle
python train.py --train-csv images/train/dataset_2needle/train.csv \
    --val-csv images/train/dataset_2needle/val.csv --output-dir runs/2needle
```

Everything downstream (`augment.py`, `train.py`, `test.py`, `predict.py`)
just follows however many `angle_N` columns the CSV you point it at has, so
no other flags change between the two runs.

### Expanding the training set: `augment.py`

Generates `--num-augments` copies of each image in `images/train/original`,
writing them to the sibling `images/train/augmented` folder with their own
`labels.csv` that carries over the `angle_N` label(s) and reference-mark
columns unchanged. Available controls, combined per copy:

- **photometric** (on by default): brightness and contrast jitter, slight
  blur, small translate/scale. Never moves the needle, so labels are copied
  unchanged.
- **`--brightness`** (default 0.25): strength of brightness jitter.
- **`--contrast`** (default 0.25): strength of contrast jitter.

Every augmented copy is resized to **120×120 pixels** before saving. Since
all operations are photometric (no rotation, flip, or geometric warp), no
label relabeling is needed -- labels are always correct as-is.

```
python augment.py --num-augments 3 --brightness 0.3 --contrast 0.2
```

This is optional and separate from `train.py --augment`, which instead
applies photometric jitter in-memory each epoch (see §1); the two can be
combined.

### Splitting: `split_dataset.py`

```
python split_dataset.py --train-frac 0.7 --val-frac 0.15
```

The split membership is decided on `images/train/original/labels.csv`
**only**. If `images/train/augmented/labels.csv` exists, augmented rows are
appended to `dataset/train.csv` alone, matched to the train split via their
`source_filename` — augmented copies of a validation/test image are never
included anywhere. This is the reason for the original/augmented/dataset
split in the first place: if augmentation happened before splitting, a
near-duplicate of a validation image could end up in the training set
(data leakage), silently inflating validation/test scores. Running
`split_dataset.py` before or after `augment.py` gives the same, correct
result either way, since the split always keys off the original filenames.

`center_x`/`center_y`, if present, are carried through for original rows into
whichever split they land in; augmented rows never get one (see above), and
if the source labels have no center columns at all, they're skipped
entirely -- no forced backfill.

By default the split is **stratified**: rows are grouped into `--strat-bins`
equal-width angle bins (default 12, i.e. 30° bins; jointly across both
`angle_N` columns for a two-needle gauge) and each group is split
proportionally, so train/val/test all see a similar angle distribution
instead of a plain random split risking e.g. every near-0° sample landing in
test by chance on a small dataset. If a bin is too small to give every split
a share after rounding, `split_dataset.py` raises a clear error rather than
silently producing an empty split (which would otherwise crash later,
opaquely, inside the training loop) -- lower `--strat-bins`, pass
`--no-stratify`, or label more images. Verified during development against a
synthetic dataset small enough to trigger this exact edge case.

---

## 3. Scripts reference

| Script | Purpose |
|---|---|
| `annotate.py` | Click-to-label tool to build `images/train/original/labels.csv` (`--needles 1\|2`; gauge center clicked once as the shared pivot and saved as `center_x`/`center_y`, min/max reference marks always labeled) |
| `verify_annotations.py` | Read-only interactive viewer (next/prev/quit) that steps through a labels CSV drawing each image's annotated needle/reference angles from the stored (or image-center-fallback) pivot, to visually spot-check annotations without needing a trained model |
| `augment.py` | Generates augmented copies into `images/train/augmented` (brightness/contrast jitter, blur, translate/scale, resized to 120×120) |
| `split_dataset.py` | Leakage-safe, stratified split into `images/train/dataset/{train,val,test}.csv` |
| `train.py` | Trains one ResNet, saves `best_model.pt` + `params.json` + `history.json` + `metrics.png` (arch auto-detected, needle count + reference marks auto-detected) |
| `compare_pretrained.py` | Trains pretrained vs. scratch back-to-back and compares |
| `test.py` | Evaluates a checkpoint on `dataset/test.csv` (per-needle/per-reference-mark MAE/RMSE/plot/`--vis-dir` overlays, optional `--value-min`/`--value-max` reading) |
| `predict.py` | Runs inference on `images/predict/` (or any file/folder given, optional `--vis-dir` overlays, `--value-min`/`--value-max` reading) |
| `common.py` | Shared dataset class, model builder, angle encode/decode, `compute_reading()`, transforms, train/eval loop, `plot_training_history()`, `angle_to_vector()`/`draw_angle_overlay()`, path constants |

### Typical end-to-end workflow

```bash
# 1. Label images/train/original (skip if you already have labels.csv there)
python annotate.py

# 2. Optional: expand the training pool with augmented copies (120x120, brightness/contrast jitter)
python augment.py --num-augments 3 --brightness 0.3 --contrast 0.2

# 3. Split into train/val/test (augmented copies only ever join train)
python split_dataset.py

# 4. Train (output defaults to runs/<arch>/<timestamp>/ -- includes params.json, history.json, metrics.png)
python train.py --arch resnet50 --epochs 30

# 5. Evaluate on the held-out test set
python test.py --checkpoint runs/resnet50/20260703_143022/best_model.pt \
    --output-csv runs/resnet50/20260703_143022/test_predictions.csv --plot

# 5b. With visual overlays (predicted vs true angle lines drawn on each image)
python test.py --checkpoint runs/resnet50/20260703_143022/best_model.pt \
    --output-csv runs/resnet50/20260703_143022/test_predictions.csv \
    --vis-dir runs/resnet50/20260703_143022/visualizations

# 5c. If the labels include reference marks (annotate.py always writes them),
#     also get an actual physical reading (value_min/value_max are the printed
#     scale endpoints, e.g. 0/160 PSI)
python test.py --checkpoint runs/resnet50/20260703_143022/best_model.pt \
    --value-min 0 --value-max 160 --unit PSI --plot

# 6. Run on brand-new, unlabeled images placed in images/predict/
python predict.py --checkpoint runs/resnet50/20260703_143022/best_model.pt
python predict.py --checkpoint runs/resnet50/20260703_143022/best_model.pt --value-min 0 --value-max 160 --unit PSI
python predict.py --checkpoint runs/resnet50/20260703_143022/best_model.pt --vis-dir runs/resnet50/20260703_143022/predict_vis

# Optional: is pretrained actually helping on this dataset?
python compare_pretrained.py --arch resnet50 --epochs 30 --output-dir runs/compare
```

Checkpoints (`best_model.pt`) store the model weights plus `arch`,
`pretrained`, `img_size`, `num_needles`, and `num_reference`, so
`test.py` / `predict.py` reconstruct the exact right model/preprocessing
automatically — no need to re-specify `--arch` or a needle/reference-mark
flag at test time. Each training run also saves `params.json` (training
hyperparameters and final metrics), `history.json` (per-epoch metrics), and
`metrics.png` (loss and MAE vs epoch plot).

---

## 4. Development record

**2026-07-02** — Initial version.

Requirements gathered from the user via clarifying questions before writing
any code:

1. *Dataset format*: no labeled dataset exists yet (images only) → designed
   the `filename,angle` CSV + image-folder format from scratch, and built
   `annotate.py` as a lightweight way to actually produce that CSV from the
   user's existing unlabeled images, plus `split_dataset.py` to partition it.
2. *Angle range*: confirmed bounded, non-wrapping arc (typical analog gauge)
   rather than full 360° wraparound → single-output linear regression head
   with raw-degree targets; circular sin/cos encoding documented as a future
   extension rather than built now, to avoid unused complexity.
3. *Pretrained weights*: user wants both alternatives available and
   compared, not just one → `train.py` supports `--pretrained/--no-pretrained`
   and `compare_pretrained.py` was added specifically to run both under
   identical conditions and report which wins on the user's own data.

Architecture decisions made independently, recorded here for future
reference:

- Loss defaults to Huber/SmoothL1 rather than MSE, anticipating that
  hand-labeled angles (via `annotate.py`) will have some annotation noise.
- Rotation/flip augmentations were intentionally left out of `--augment`
  (see §1, "Data augmentation caveat") rather than implemented with
  automatic label correction, since the correct label transform depends on
  a labeling convention the user hadn't fixed yet — shipping a guess here
  risked silently corrupting training data. Photometric-only augmentation
  was implemented instead as a safe default.
- Checkpoints self-describe (`arch`, `pretrained`, `img_size`) so
  `test.py`/`predict.py` don't require re-passing architecture flags and
  can't silently mismatch preprocessing between train and test.

**2026-07-02** — Adapted to a fixed folder structure.

The user specified the on-disk layout directly rather than leaving it
open: `images/train/original` (all original images), a sibling
`images/train/augmented` for augmentation output, a sibling
`images/train/dataset` created by the split step, and a separate
`images/predict` reserved exclusively for prediction-time images. Changes
made:

- Added `IMAGES_DIR`/`TRAIN_DIR`/`ORIGINAL_DIR`/`AUGMENTED_DIR`/
  `DATASET_DIR`/`PREDICT_DIR` constants to `common.py` as the single source
  of truth for these paths; every script now defaults to them instead of
  requiring `--train-csv`/`--img-dir`/etc. to be passed by hand.
- Added `augment.py`: previously augmentation only happened in-memory during
  training (`train.py --augment`). Since the user wants an actual
  `augmented/` folder on disk, this new script materializes label-safe
  (photometric + safe-geometric, never rotation/flip — same constraint as
  §1) copies as real image files with their own `labels.csv`.
- Rewrote `split_dataset.py` to decide the split on **original** images only
  and append augmented copies to the train split alone, keyed by
  `source_filename`. This wasn't explicitly requested, but was necessary to
  implement correctly: augmenting before splitting (or splitting over the
  combined original+augmented pool) risks a near-duplicate of a validation
  or test image ending up in training, which would leak information and
  quietly inflate validation/test scores. Splitting on originals first and
  only ever adding augmented rows to `train.csv` avoids that regardless of
  what order `augment.py` and `split_dataset.py` are run in.
- `dataset/{train,val,test}.csv` now store subfolder-prefixed filenames
  (`original/x.jpg`, `augmented/x.jpg`) so `GaugeAngleDataset` can resolve
  both sibling folders through one shared `--img-dir images/train`.
- `predict.py`'s `--image`/`--image-dir` are no longer mutually-required;
  with neither given it now defaults to `images/predict`.

**2026-07-02** — Implemented all previously-documented future work, per
explicit user feedback on five points:

1. *Full 360° annotation, no range limit*: `annotate.py` was already
   computing `angle % 360` with no range restriction, but the model itself
   assumed a bounded, non-wrapping arc (single raw-degree output). Replaced
   that with the circular `(sin, cos)` encoding described in §1 --
   `build_model()` now outputs `2 * num_needles` values, `encode_angles()`/
   `decode_angles()`/`circular_abs_error_deg()` were added to `common.py`,
   and `run_epoch()`/`test.py`/`predict.py` were updated to decode through
   them. This supersedes the "bounded arc" decision recorded above --
   circular encoding is now used for every gauge, bounded-sweep or full-360,
   since it strictly generalizes the bounded case.
2. *Rotation/flip in `augment.py`*: added `--rotate`/`--max-rotation`,
   `--hflip`, `--vflip`, each relabeling every `angle_N` column via the
   formulas in §1. This was explicitly blocked in the previous entry
   ("shipping a guess here risked silently corrupting training data") until
   two things were resolved: a fixed labeling convention (§2, "relative to
   image frame") and circular angle encoding (point 1 above, which removes
   the "falls outside the valid range" failure mode for rotation). Verified
   correct via a synthetic-image round-trip test (see §1) rather than
   trusting the derivation alone.
3. *Two-needle gauges*: added `angle_1`/`angle_2` column auto-detection
   (`get_angle_columns()`), `annotate.py --needles {1,2}`, and per-needle
   metrics in `test.py`/`predict.py` (see §1, "One or two needles").
   Needle count is inferred from the CSV everywhere except `annotate.py`
   itself, rather than added as a flag on every script, specifically so a
   one-needle labels file and a two-needle labels file can coexist and be
   tested independently, per the user's request to "test if the models can
   handle one needle first."
4. *Export format*: user confirmed `.pt` (the existing checkpoint format)
   is sufficient for now; no ONNX/TorchScript export was added.
5. *Stratification*: `split_dataset.py` now stratifies by binned angle by
   default (`--strat-bins`, `--no-stratify` to opt out), as described in
   §2/§3.

All five changes were exercised end-to-end (not just syntax-checked) against
a synthetic dataset with known ground-truth angles before being considered
done: `augment.py --rotate --hflip --vflip` output was verified by
re-detecting the needle's actual pixel position in each generated image and
comparing it to the label `augment.py` recorded; `split_dataset.py`,
`train.py`, `test.py`, and `predict.py` were each run for both a 1-needle
and a 2-needle synthetic dataset. This is also how the stratification empty-
split edge case (next paragraph) was found.

One issue surfaced by that testing, not by inspection: on a small dataset,
stratified splitting can round a bin's share down to zero for one of
train/val/test, which previously reached `run_epoch()` as an empty
`DataLoader` and crashed with an opaque `ZeroDivisionError` several frames
deep. Fixed by validating all three splits are non-empty in
`split_dataset.py` right after computing them (clear error naming which
split and suggesting `--strat-bins`/`--no-stratify`), plus a second check in
`train.py::train_model()` in case a hand-built CSV (not from
`split_dataset.py`) is empty.

**2026-07-02** — Added scale reference marks (`ref_angle_min`/`ref_angle_max`)
and physical-reading computation, replacing the two classical circle-
detection/scale-detection algorithms from the original design with a single
model that also predicts the reference marks as angles.

This followed directly from a design discussion (not a one-shot request):
the user first asked whether predicting reference-point *coordinates* could
replace the classical algorithms; the recommendation was to predict
reference *angles* from a shared pivot instead (reusing 100% of the
already-built/tested `(sin, cos)` machinery, rather than adding weaker,
less-relevant pixel-coordinate regression) -- see §1, "Reference marks," for
the full reasoning. Two clarifying questions shaped the implementation:

1. *Camera framing*: user confirmed it's "mixed / not consistent enough to
   rely on" across photos → ruled out a per-gauge-type calibration shortcut
   (classify gauge type, look up fixed reference angles); per-image
   reference-angle regression, as proposed, is the right call here.
2. *Min/max physical values*: user confirmed these are fixed and already
   known per gauge type → kept entirely out of the model (no OCR/digit
   recognition needed); supplied as plain `--value-min`/`--value-max`
   arguments at evaluation/inference time instead.

The user then specifically asked whether min+max alone would be a
sufficient angular reference, and separately flagged the "which of the two
candidate arcs is the scale" ambiguity as their real concern. Both are
addressed in `common.py::compute_reading()`: min+max is mathematically
sufficient for a linear gauge scale (2-point calibration, standard
practice), and the arc ambiguity is resolved by using the needle's own
predicted angle to pick whichever arc it actually falls on, rather than
assuming a fixed sweep direction -- see §1 for the full derivation, caveats
(non-linear scales, near-180° sweeps), and the annotation-cost tradeoff of
giving every point (needle, min mark, max mark) its own independent pivot
click rather than assuming a shared gauge center.

Implementation touched every script: `common.py` gained
`REFERENCE_COLS`/`get_reference_columns()`, `build_model(..., num_reference=)`,
and `compute_reading()`; `GaugeAngleDataset` builds one combined
`[needle angles..., reference angles...]` target and exposes
`.num_reference`/`.ref_cols`; `annotate.py` gained `--ref-marks {0,2}` with a
generalized "point plan" click flow (needles and reference marks are now
just two kinds of entries in the same list, each getting its own pivot+tip
pair); `augment.py`/`split_dataset.py` treat reference columns identically
to needle columns for rotate/flip relabeling and split carry-through
(reference marks rotate/flip exactly like a needle does); `train.py` stores
`num_reference` in the checkpoint; `test.py`/`predict.py` split decoded
output back into needle vs. reference angles and, given
`--value-min`/`--value-max`, compute the actual reading via
`compute_reading()`.

Verified end-to-end on synthetic data before considering it done, the same
way as the previous round: generated gauge images with a needle plus
colored min/max tick marks at known angles, ran them through
`augment.py --rotate --hflip --vflip`, and re-detected all three markers'
actual pixel positions to confirm the recorded labels were still correct
(max ~2.4° discrepancy across all three, consistent with the earlier
needle-only test). Also independently verified `compute_reading()` against
hand-derived expected values (needle at `ref_angle_min` → `value_min`;
at `ref_angle_max` → `value_max`; exactly halfway along the arc → the
midpoint value; a reversed min/max convention still resolves correctly).
Then ran `split_dataset.py` → `train.py` → `test.py` → `predict.py`
end-to-end on the synthetic reference-mark dataset, confirming the 6-output
model (1 needle + 2 reference marks) trains, evaluates with per-mark stats
plus both reading-error variants (fully-predicted and the "oracle ref"
diagnostic), and that requesting a reading from a checkpoint trained
without reference marks fails with a clear, specific error rather than a
shape mismatch deep in the model.

**2026-07-03** — Simplified annotation: one shared gauge-center pivot per
image, and reference marks are always labeled.

Two changes to `annotate.py`, both per explicit user request:

1. *Shared center pivot*: the previous click flow collected an independent
   pivot+tip pair for every point (each needle, min mark, max mark) to stay
   correct for hypothetical multi-dial gauges with separate centers. The
   user clarified that for the gauges actually in scope (single- and
   two-needle), the gauge's center is the pivot for the needle(s) *and*
   both scale limits -- so the per-point pivot clicks were pure annotation
   overhead. The flow is now: click the gauge CENTER once, then one TIP
   click per point (needle 1[, needle 2], min mark, max mark) -- e.g. 5
   clicks instead of 8 for a 1-needle image with reference marks. The
   angle convention (counter-clockwise from the positive x-axis to the
   pivot→tip vector, `angle_from_points()`) is untouched, so nothing
   downstream changes: the CSV schema, circular encoding, `augment.py`'s
   rotate/flip relabeling formulas (rigid transforms rotate the pivot→tip
   vector identically wherever the pivot sits), and `compute_reading()` all
   behave exactly as before. The old multi-dial rationale is preserved in
   §1, "Reference marks," in case independent per-point pivots need to
   come back.
2. *Reference marks always labeled*: the `--ref-marks {0,2}` flag was
   removed; every annotated image now gets `ref_angle_min`/`ref_angle_max`
   alongside its `angle_N` column(s). Downstream auto-detection
   (`get_reference_columns()` etc.) is unchanged, so labels files created
   before reference marks existed still load, and old checkpoints still
   self-describe via `num_reference`.

Because only which clicks feed which `angle_from_points()` call changed --
not the angle math, columns, or any training/eval code -- verification was
a syntax compile of all scripts plus a headless exercise of the new click
flow (simulated center+tip click sequences through `label_one`'s handler,
checking the written CSV columns and that the computed angles match
hand-derived expected values for known click coordinates).

**2026-07-03** — Metrics plotting, organized model saving, visual overlays,
and augment.py overhaul.

Changes across this session:

1. *Metrics vs epoch plot*: Added `common.py::plot_training_history()` which
   generates a 2-panel `metrics.png` (loss and MAE vs epoch) saved to every
   training output directory, so performance over time is always available
   alongside the checkpoint.

2. *Organized model saving*: `train.py`'s default output directory is now
   `runs/<arch>/<timestamp>` (e.g. `runs/resnet50/20260703_143022`) instead
   of the fixed `runs/exp`, preventing accidental overwrites between runs.
   A `params.json` file records all training hyperparameters (learning rate,
   batch size, loss, weight decay, etc.) plus the best val MAE for reference.

3. *Visual overlays for test and predict*: Added
   `common.py::draw_angle_overlay()` which draws labeled angle arrow lines
   (needle angles and/or reference marks) on the original image and saves
   the annotated copy. `test.py --vis-dir` produces overlays showing both
   true and predicted angle lines on each test image. `predict.py --vis-dir`
   produces overlays showing predicted angle lines on inference images.

4. *Augment.py simplified to photometric-only + resize*: Removed the
   `--rotate`/`--hflip`/`--vflip` flags and all geometric relabeling logic.
   Added explicit `--brightness` and `--contrast` flags to control the
   ColorJitter range. Every augmented copy is now resized to 120×120 pixels
   before saving. Since all operations are photometric (no geometric
   transform that would move where the needle points), labels are copied
   unchanged -- no relabeling formulas needed.

**2026-07-04** — Fixed overlay angle math, stored the gauge center for
overlays, corrected augmentation wording, added an annotation-verification
viewer.

A review of the 2026-07-03 session found the underlying angle labels and
augmentation were correct, but overlay *rendering* was wrong -- exactly the
"visualization of predicted/annotated angles is not right" symptom:

1. *Overlay math bug*: `draw_angle_overlay()` computed its arrow direction as
   `angle_rad = radians(angle_deg - 90)`, `dx = cos`, `dy = +sin` in pixel
   (y-down) space -- not the inverse of `annotate.py::angle_from_points()`'s
   convention (`dy = -(tip.y - pivot.y)`, i.e. y-up). The true inverse is
   `dx = r*cos(theta)`, `dy = -r*sin(theta)`, no `-90` offset. The bug
   injected a spurious 90-degree rotation plus a y-flip, so every rendered
   arrow pointed roughly 90 degrees off and mirrored from the actual
   annotated direction, even though the stored angle value itself was always
   correct. Fixed by adding `common.py::angle_to_vector(angle_deg, radius)`
   as the single, exact inverse of `angle_from_points()`, and routing
   `draw_angle_overlay()` through it -- this corrects both `test.py
   --vis-dir` and `predict.py --vis-dir` at once, since both call the same
   function.
2. *Stored gauge center for overlays*: overlay arrows previously always
   originated at the image's pixel center rather than the actual clicked
   gauge center, since `annotate.py` never saved it. `annotate.py` now writes
   `center_x`/`center_y` (the first, shared-pivot click) alongside the angle
   columns; `test.py --vis-dir` and the new `verify_annotations.py` use it as
   the overlay origin when present, falling back to the image center
   otherwise. Backward-compatible: the existing 170-row `labels.csv` has no
   center columns, so every row in it (and any row labeled before this
   change) keeps using the image-center fallback until re-annotated -- no
   forced re-labeling. `predict.py` runs on unlabeled images and has no
   stored center available, so it always uses the image-center fallback
   (documented behavior, not a bug). `split_dataset.py` carries
   `center_x`/`center_y` through for original rows when present; augmented
   rows get NaN, since `augment.py`'s translate/scale moves the pixel center
   and augmented images are train-only and never overlaid.
3. *Doc-gap fix, no code change*: `augment.py` and this document described
   the on-disk augmentation as "photometric-only" / "no geometric
   transforms," but the translate/scale it applies (via
   `RandomAffine(degrees=0, ...)`) *is* geometric -- it just preserves the
   pivot->tip angle, since pure translation and uniform scale don't rotate or
   mirror the image, which is why labels can still be copied unchanged.
   Reworded throughout to "photometric plus small angle-preserving geometric
   transforms" instead of implying no geometric transform is used at all.
4. *New tool, `verify_annotations.py`*: a read-only, interactive matplotlib
   viewer (next/prev/quit, modeled on `annotate.py::label_one`) that steps
   through an existing labels CSV and draws each image's annotated
   needle/reference angles using the corrected `angle_to_vector()` math and
   the stored-center-or-image-center-fallback origin, for spot-checking
   annotations without needing a trained model.
5. Also updated stale leftovers found during this review, from the same
   2026-07-03 session: `build_transforms()`'s docstring no longer tells users
   to run `augment.py --rotate --hflip --vflip` (those flags were removed
   that same session); `test.py`/`predict.py` usage examples now show the
   actual default output path (`runs/<arch>/<timestamp>/...`) instead of the
   old fixed `runs/exp`.

Verified numerically (`angle_to_vector`: 0 degrees -> `(r, 0)` [right], 90
degrees -> `(0, -r)` [up], 180 degrees -> `(-r, 0)` [left], 270 degrees ->
`(0, r)` [down]; and that `angle_from_points(pivot, pivot + angle_to_vector(
theta))` recovers `theta`) and visually, by rendering an existing labeled
image's overlay with both the old and new formulas and confirming the arrow
only lies along the true annotated direction (matching the photo) after the
fix.

**2026-07-04** — `annotate.py` reworked to support navigation and in-place
editing of already-saved labels.

Per user report: some previously-correct labels had ended up displaced, with
no way to go back and fix them short of hand-editing the CSV -- the old flow
opened one modal window per *unlabeled* image only (`label_one()`), moved
forward automatically, and never revisited a filename already in the CSV.
Replaced that with a single persistent window (`AnnotationSession`) that
steps through *every* image, labeled or not:

- `n`/`right`/`s` and `p`/`left`/`b` navigate forward/back across the full
  image list, not just the remaining unlabeled ones.
- An already-labeled image shows its current annotation as an overlay
  (`draw_existing_annotation()`, using the same corrected `angle_to_vector()`
  math and stored-center-or-fallback origin as `draw_angle_overlay()` /
  `verify_annotations.py`), so a bad label is visible immediately on
  revisiting it instead of only showing up later in `test.py --vis-dir`.
- `r` enters click mode for the current image and, once the full click
  sequence is finished, overwrites its row (or adds a new one) and saves
  immediately; `z` restarts the in-progress clicks, `escape`/`c` cancels
  back to just viewing without changing anything.

This replaces the previous one-image-per-window, forward-only flow
entirely -- `label_one()` no longer exists. The click math itself
(`angle_from_points()`), the CSV schema, and the center-storage/fallback
behavior (see the entry above) are unchanged; only how you navigate between
images and get back into an already-saved one changed. Per explicit user
instruction, this was verified with the syntax-compile check only (no
generated test images) -- see CLAUDE.md.

### Possible future work

- ONNX/TorchScript export for deployment outside Python (explicitly
  deferred per user, not needed right now).
- A "which needle is which" consistency check across a labels CSV (e.g.
  flagging images where `angle_1`/`angle_2` look swapped relative to a
  reference), if two-needle labeling in practice turns out error-prone.
- Non-linear scale support (piecewise/higher-order fit instead of the
  current 2-point linear interpolation in `compute_reading()`), if any
  gauge type turns out not to have a linear angle-to-value mapping.
- A tie-breaking mechanism (extra reference point, or a per-gauge-type fixed
  sweep direction) for gauges whose true sweep is close to 180°, where the
  two candidate arcs in `compute_reading()` become hard to disambiguate from
  a noisy needle prediction alone -- see §1.
- Independent per-needle reference marks (e.g. `ref_angle_min_1`/`_max_1`
  and `ref_angle_min_2`/`_max_2`), if a two-needle gauge turns out to have
  two genuinely independent scales rather than sharing one. That would also
  be the moment to restore per-point pivot clicks in `annotate.py`
  (replaced 2026-07-03 by the single shared gauge-center click), since
  independent scales usually mean independent dial centers.
- A heatmap/keypoint-style auxiliary prediction head (as used in pose
  estimation) instead of plain FC regression, if measured angular precision
  on real data turns out insufficient -- would need a real architecture
  change (predicting spatial heatmaps from intermediate feature maps rather
  than a single pooled vector), so only worth it if the simpler approach's
  accuracy doesn't hold up.

#Use l2 for loss function and test different learning rate schedulers