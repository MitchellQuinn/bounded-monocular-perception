"""Shared test helpers for ROI-FCN preprocessing v0.1."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
REPO_ROOT = PROJECT_ROOT.parents[1]
SYNTHETIC_ROOT = REPO_ROOT / "02_synthetic-data-processing-v4.0"
for path in (SRC_ROOT, SYNTHETIC_ROOT):
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def ensure_preprocessing_root(root: Path) -> Path:
    (root / "input").mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(parents=True, exist_ok=True)
    return root


def write_rectangle_image(
    image_path: Path,
    *,
    width: int,
    height: int,
    box_xyxy: tuple[int, int, int, int] | None = None,
) -> None:
    canvas = np.full((height, width), 255, dtype=np.uint8)
    x1, y1, x2, y2 = box_xyxy or (width // 4, height // 4, (3 * width) // 4, (3 * height) // 4)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color=0, thickness=-1)
    ok = cv2.imwrite(str(image_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write fixture image: {image_path}")


def build_input_split(
    preprocessing_root: Path,
    dataset_reference: str,
    split_name: str,
    row_specs: Iterable[dict[str, object]],
) -> pd.DataFrame:
    split_root = ensure_preprocessing_root(preprocessing_root) / "input" / dataset_reference / split_name
    images_dir = split_root / "images"
    manifests_dir = split_root / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, spec in enumerate(row_specs):
        width = int(spec.get("width", 64))
        height = int(spec.get("height", 64))
        filename = str(spec.get("filename", f"{split_name}_{index:03d}.png"))
        capture_success = bool(spec.get("capture_success", True))
        write_image = bool(spec.get("write_image", True))
        image_path = images_dir / filename
        if write_image:
            if bool(spec.get("corrupt_image", False)):
                image_path.write_bytes(b"not-a-real-image")
            else:
                write_rectangle_image(
                    image_path,
                    width=width,
                    height=height,
                    box_xyxy=spec.get("box_xyxy"),
                )

        rows.append(
            {
                "run_id": str(spec.get("run_id", f"{dataset_reference}_{split_name}")),
                "sample_id": str(spec.get("sample_id", f"{dataset_reference}_{split_name}_{index:03d}")),
                "frame_index": int(spec.get("frame_index", index)),
                "image_filename": filename,
                "distance_m": float(spec.get("distance_m", 1.0 + index)),
                "image_width_px": width,
                "image_height_px": height,
                "capture_success": capture_success,
            }
        )

    samples_df = pd.DataFrame(rows)
    samples_df.to_csv(manifests_dir / "samples.csv", index=False)
    (manifests_dir / "run.json").write_text(
        json.dumps({"RunId": f"{dataset_reference}_{split_name}"}, indent=4) + "\n",
        encoding="utf-8",
    )
    return samples_df


def build_dataset(
    preprocessing_root: Path,
    dataset_reference: str,
    *,
    train_rows: Iterable[dict[str, object]],
    validate_rows: Iterable[dict[str, object]],
) -> None:
    build_input_split(preprocessing_root, dataset_reference, "train", train_rows)
    build_input_split(preprocessing_root, dataset_reference, "validate", validate_rows)
