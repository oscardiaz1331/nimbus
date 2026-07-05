import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "swimseg"
SRC_DATASET_DIR = Path(f"datasets/{DATASET}")

IMAGES_DIR = SRC_DATASET_DIR / "images"
MASKS_DIR = SRC_DATASET_DIR / "GTmaps"

OUTPUT_DIR = Path("datasets/swimnseg_yolo")

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6
RANDOM_SEED = 42

CLASS_ID = 0
CLASS_NAME = "cloud"

CONTOUR_EPSILON = 0.001
MIN_CONTOUR_AREA = 10

# ============================================================
# OUTPUT PATHS
# ============================================================

TRAIN_IMAGES_DIR = OUTPUT_DIR / "train" / "images"
VAL_IMAGES_DIR = OUTPUT_DIR / "valid" / "images"

TRAIN_LABELS_DIR = OUTPUT_DIR / "train" / "labels"
VAL_LABELS_DIR = OUTPUT_DIR / "valid" / "labels"

TEST_IMAGES_DIR = OUTPUT_DIR / "test" / "images"
TEST_LABELS_DIR = OUTPUT_DIR / "test" / "labels"

# ============================================================
# HELPERS
# ============================================================


def create_output_structure() -> None:

    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    VAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    VAL_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    TEST_LABELS_DIR.mkdir(parents=True, exist_ok=True)


def find_image_mask_pairs():

    pairs = []

    for ext in ["*.png", "*.jpg"]:
        for image_path in sorted(IMAGES_DIR.glob(ext)):

            stem = image_path.stem

            candidate_masks = [
                MASKS_DIR / f"{stem}_GT.png",
                MASKS_DIR / f"{stem}_GT.jpg",
            ]

            mask_path = None

            for candidate in candidate_masks:
                if candidate.exists():
                    mask_path = candidate
                    break

            if mask_path is None:
                print(f"[WARNING] Mask not found for " f"{image_path.name}")
                continue

            pairs.append((image_path, mask_path))

    return pairs


# ============================================================
# MASK -> YOLO SEGMENTATION
# ============================================================


def mask_to_yolo_segmentation(mask_path: Path) -> list[str]:

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"[WARNING] Cannot read " f"{mask_path}")
        return []

    h, w = mask.shape

    #
    # Cloud = black pixels
    # Sky   = white pixels
    #
    binary_mask = (mask == 255).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    labels = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA:
            continue

        epsilon = CONTOUR_EPSILON * cv2.arcLength(contour, True)

        contour = cv2.approxPolyDP(contour, epsilon, True)

        if len(contour) < 3:
            continue

        points = contour.reshape(-1, 2).astype(np.float32)

        points[:, 0] /= w
        points[:, 1] /= h

        points = np.clip(points, 0.0, 1.0)

        coordinates = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)

        labels.append(f"{CLASS_ID} {coordinates}")

    return labels


# ============================================================
# PROCESS SAMPLE
# ============================================================


def process_sample(
    image_path: Path, mask_path: Path, dst_image_dir: Path, dst_label_dir: Path
):

    shutil.copy2(image_path, dst_image_dir / f"{DATASET}_{image_path.name}")

    label_lines = mask_to_yolo_segmentation(mask_path)

    label_file = dst_label_dir / f"{DATASET}_{image_path.stem}.txt"

    with open(label_file, "w", encoding="utf-8") as f:

        f.write("\n".join(label_lines))


# ============================================================
# DATASET YAML
# ============================================================


def create_dataset_yaml():

    yaml_dict = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": str(TRAIN_IMAGES_DIR.relative_to(OUTPUT_DIR)),
        "val": str(VAL_IMAGES_DIR.relative_to(OUTPUT_DIR)),
        "nc": 1,
        "names": {CLASS_ID: CLASS_NAME},
    }

    yaml_path = OUTPUT_DIR / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:

        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"data.yaml created at " f"{yaml_path}")


# ============================================================
# MAIN
# ============================================================


def main():

    random.seed(RANDOM_SEED)

    create_output_structure()

    pairs = find_image_mask_pairs()

    if len(pairs) == 0:
        raise RuntimeError("No image-mask pairs found.")

    random.shuffle(pairs)

    n = len(pairs)

    train_end = int(n * TRAIN_RATIO)

    val_end = train_end + int(n * VAL_RATIO)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    print(f"Total samples : {len(pairs)}")

    print(f"Train samples : {len(train_pairs)}")

    print(f"Val samples   : {len(val_pairs)}")

    print(f"Test samples  : {len(test_pairs)}")

    print("\nProcessing train split...")

    for image_path, mask_path in train_pairs:

        process_sample(image_path, mask_path, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)

    print("Processing validation split...")

    for image_path, mask_path in val_pairs:

        process_sample(image_path, mask_path, VAL_IMAGES_DIR, VAL_LABELS_DIR)

    print("Processing test split...")

    for image_path, mask_path in test_pairs:

        process_sample(image_path, mask_path, TEST_IMAGES_DIR, TEST_LABELS_DIR)
        create_dataset_yaml()

    print("\nDataset conversion completed.")


if __name__ == "__main__":
    main()
