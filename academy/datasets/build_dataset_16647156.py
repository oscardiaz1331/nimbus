"""
build_dataset_cloud_only.py — CloudVision
==========================================
Converts the Kontas-2017 sky/cloud dataset into a YOLO11-seg compatible structure
with a single class: cloud.

Sky (original class 1) is treated as background — no annotation needed, since the
model only needs to learn where clouds are, not where the sky is. This simplifies
the problem and tends to produce better cloud masks on small datasets.

Source layout expected:
    datasets/16647156/
        kontas_2017/
            images/          ← original RGB images
            seg_masks/       ← semantic segmentation masks (grayscale PNG)
                                    0 → camera mask  (ignored)
                                    1 → sky          (background — not annotated)
                                    2 → low-layer clouds   ┐
                                    3 → mid-layer clouds   ├─ all → class 0 (cloud)
                                    4 → high-layer clouds  ┘
            validation.csv   ← filenames designated for the val split

Output layout produced:
    datasets/yolo_ready_cloud_only/
        dataset.yaml         ← nc: 1, names: {0: cloud}
        images/
            train/
            val/
        labels/
            train/           ← YOLO polygon .txt files
            val/
"""

import os
import csv
import shutil
import yaml
import cv2
import numpy as np

# =====================================================================
# SOURCE PATHS  (read-only — never modified)
# =====================================================================
SRC_DIR = "datasets/16647156"
SRC_ROOT_DIR = os.path.join(SRC_DIR, "kontas_2017")

VALIDATION_CSV = os.path.join(SRC_ROOT_DIR, "validation.csv")
SRC_IMAGES_DIR = os.path.join(SRC_ROOT_DIR, "images")
SRC_MASKS_DIR = os.path.join(SRC_ROOT_DIR, "seg_masks")

# =====================================================================
# DESTINATION PATHS  (created fresh on each run)
# =====================================================================
DST_DIR = "datasets/16647156_yolo"
TRAIN_IMG_DIR = os.path.join(DST_DIR, "train", "images")
VAL_IMG_DIR = os.path.join(DST_DIR, "val", "images")
TRAIN_LABEL_DIR = os.path.join(DST_DIR, "train", "labels")
VAL_LABEL_DIR = os.path.join(DST_DIR, "val", "labels")
OUTPUT_YAML = os.path.join(DST_DIR, "data.yaml")

# =====================================================================
# CLASS DEFINITION
# =====================================================================
CLOUD_ID = 0
CLASSES = {CLOUD_ID: "cloud"}  # single-class: nc=1

# Contour simplification tolerance.
# Lower = more polygon points (more precise, heavier labels).
# Higher = fewer points (smoother, may miss fine cloud edges).
# Recommended range: 0.001 – 0.005
CONTOUR_EPSILON = 0.001


# =====================================================================
# MASK CONVERSION
# =====================================================================
def mask_to_yolo_labels(mask_path: str) -> list[str]:
    """
    Convert a single semantic mask PNG into YOLO segmentation label lines
    for the single-class (cloud) configuration.

    Only pixels with original values 2, 3 or 4 are annotated as cloud.
    Sky (1) and camera mask (0) are implicitly treated as background.

    YOLO segmentation format (one instance per line):
        <class_id>  <x1> <y1>  <x2> <y2>  ...  <xn> <yn>
        All coordinates normalised to [0, 1].

    Args:
        mask_path: Path to the grayscale mask PNG.

    Returns:
        List of label strings, one per detected cloud polygon instance.
        Empty list if no cloud pixels are found or the mask cannot be read.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  WARNING: Could not read mask: {mask_path}")
        return []

    h, w = mask.shape

    # Merge all three cloud layers into a single binary mask.
    # Sky (1) and camera mask (0) are not included → treated as background.
    cloud_binary = ((mask == 2) | (mask == 3) | (mask == 4)).astype(np.uint8) * 255

    if cloud_binary.max() == 0:
        return []  # image contains no cloud pixels

    contours, _ = cv2.findContours(
        cloud_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for contour in contours:
        if len(contour) < 3:
            continue  # degenerate contour — skip

        # Douglas-Peucker simplification to reduce polygon complexity
        epsilon = CONTOUR_EPSILON * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        if len(simplified) < 3:
            continue  # simplification collapsed the polygon — skip

        # Normalise to [0, 1] and clamp for floating-point safety
        points = simplified.reshape(-1, 2).astype(float)
        points[:, 0] /= w
        points[:, 1] /= h
        points = np.clip(points, 0.0, 1.0)

        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
        lines.append(f"{CLOUD_ID} {coords}")

    return lines


# =====================================================================
# MAIN
# =====================================================================
def main():
    # --- Sanity checks ---------------------------------------------------
    if not os.path.exists(SRC_IMAGES_DIR):
        raise FileNotFoundError(f"Source images directory not found: {SRC_IMAGES_DIR}")
    if not os.path.exists(SRC_MASKS_DIR):
        raise FileNotFoundError(f"Source masks directory not found: {SRC_MASKS_DIR}")

    # --- Create destination structure ------------------------------------
    print(f"Creating YOLO dataset structure at: '{DST_DIR}'")
    for d in (TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LABEL_DIR, VAL_LABEL_DIR):
        os.makedirs(d, exist_ok=True)

    # --- Load validation split filenames ---------------------------------
    val_filenames: set[str] = set()
    if os.path.exists(VALIDATION_CSV):
        with open(VALIDATION_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["fileNames"].strip().strip(",")
                if name:
                    val_filenames.add(name)
        print(f"  Validation split: {len(val_filenames)} files")
    else:
        print(f"  WARNING: {VALIDATION_CSV} not found — all data routed to train.")

    # --- Process images and convert masks --------------------------------
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    all_images = sorted(os.listdir(SRC_IMAGES_DIR))
    train_count = val_count = skipped = empty = 0

    print(f"Processing {len(all_images)} source images...")

    for filename in all_images:
        if not filename.lower().endswith(valid_extensions):
            continue

        base_name = os.path.splitext(filename)[0]
        src_img_path = os.path.join(SRC_IMAGES_DIR, filename)

        # Find corresponding mask (any supported extension)
        mask_filename = None
        for ext in valid_extensions:
            candidate = base_name + ext
            if os.path.exists(os.path.join(SRC_MASKS_DIR, candidate)):
                mask_filename = candidate
                break

        if mask_filename is None:
            print(f"  WARNING: No mask found for '{filename}' — skipping.")
            skipped += 1
            continue

        src_mask_path = os.path.join(SRC_MASKS_DIR, mask_filename)

        # Route to train or val based on CSV
        is_val = base_name in val_filenames
        dest_img_dir = VAL_IMG_DIR if is_val else TRAIN_IMG_DIR
        dest_lbl_dir = VAL_LABEL_DIR if is_val else TRAIN_LABEL_DIR

        # Copy image (source never modified)
        shutil.copy2(src_img_path, os.path.join(dest_img_dir, filename))

        # Convert mask → YOLO polygon .txt
        label_lines = mask_to_yolo_labels(src_mask_path)
        label_path = os.path.join(dest_lbl_dir, base_name + ".txt")

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        if not label_lines:
            # Cloud-free image: write empty .txt so YOLO treats it as background.
            # These are useful — they teach the model not to predict clouds on clear sky.
            empty += 1

        if is_val:
            val_count += 1
        else:
            train_count += 1

    annotated = (train_count + val_count) - empty
    print(f"\nDone.")
    print(f"  Train : {train_count}  |  Val : {val_count}  |  Skipped : {skipped}")
    print(f"  With clouds : {annotated}  |  Cloud-free (background) : {empty}")

    # --- Write dataset.yaml ----------------------------------------------
    dataset_config = {
        "path": DST_DIR.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASSES),
        "names": CLASSES,
    }

    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    print(f"\n  dataset.yaml → {OUTPUT_YAML}")
    print(f"  Classes ({len(CLASSES)}): {CLASSES}")
    print("\nDataset ready. Update 'data=' in your train script to point to this yaml.")


if __name__ == "__main__":
    main()
