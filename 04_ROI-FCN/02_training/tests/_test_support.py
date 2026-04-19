"""Shared test helpers for ROI-FCN training v0.1."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from roi_fcn_training_v0_1.contracts import EXPECTED_GEOMETRY_SCHEMA


def ensure_training_root(root: Path) -> Path:
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "notebooks").mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _gaussian_image(canvas_width: int, canvas_height: int, center_x: float, center_y: float) -> np.ndarray:
    ys = np.arange(canvas_height, dtype=np.float32)
    xs = np.arange(canvas_width, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    dist_sq = (grid_x - float(center_x)) ** 2 + (grid_y - float(center_y)) ** 2
    image = np.exp(-dist_sq / (2.0 * 4.0 * 4.0)).astype(np.float32)
    return image[None, :, :]


def build_dataset_reference(
    training_root: Path,
    dataset_reference: str,
    *,
    train_centers: Iterable[tuple[float, float]],
    validate_centers: Iterable[tuple[float, float]],
    canvas_width: int = 96,
    canvas_height: int = 64,
    roi_width: int = 20,
    roi_height: int = 20,
) -> None:
    root = ensure_training_root(training_root)
    for split_name, centers in (("train", list(train_centers)), ("validate", list(validate_centers))):
        split_root = root / "datasets" / dataset_reference / split_name
        arrays_dir = split_root / "arrays"
        manifests_dir = split_root / "manifests"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        images = []
        target_original = []
        target_canvas = []
        source_wh = []
        resized_wh = []
        padding = []
        resize_scale = []
        sample_ids = []
        image_filenames = []
        npz_row_index = []
        bbox_xyxy = []
        confidence = []

        for index, (center_x, center_y) in enumerate(centers):
            sample_id = f"{dataset_reference}_{split_name}_{index:03d}"
            image_filename = f"{sample_id}.png"
            bbox = [center_x - 6.0, center_y - 5.0, center_x + 6.0, center_y + 5.0]
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_filename": image_filename,
                    "image_width_px": canvas_width,
                    "image_height_px": canvas_height,
                    "pack_roi_fcn_stage_status": "success",
                    "npz_filename": f"{dataset_reference}__{split_name}__shard_0000.npz",
                    "npz_row_index": index,
                    "locator_canvas_width_px": canvas_width,
                    "locator_canvas_height_px": canvas_height,
                    "locator_resize_scale": 1.0,
                    "locator_resized_width_px": canvas_width,
                    "locator_resized_height_px": canvas_height,
                    "locator_pad_left_px": 0,
                    "locator_pad_right_px": 0,
                    "locator_pad_top_px": 0,
                    "locator_pad_bottom_px": 0,
                    "locator_center_x_px": center_x,
                    "locator_center_y_px": center_y,
                    "bootstrap_center_x_px": center_x,
                    "bootstrap_center_y_px": center_y,
                    "bootstrap_bbox_x1": bbox[0],
                    "bootstrap_bbox_y1": bbox[1],
                    "bootstrap_bbox_x2": bbox[2],
                    "bootstrap_bbox_y2": bbox[3],
                }
            )
            images.append(_gaussian_image(canvas_width, canvas_height, center_x, center_y))
            target_original.append([center_x, center_y])
            target_canvas.append([center_x, center_y])
            source_wh.append([canvas_width, canvas_height])
            resized_wh.append([canvas_width, canvas_height])
            padding.append([0, 0, 0, 0])
            resize_scale.append(1.0)
            sample_ids.append(sample_id)
            image_filenames.append(image_filename)
            npz_row_index.append(index)
            bbox_xyxy.append(bbox)
            confidence.append(1.0)

        shard_path = arrays_dir / f"{dataset_reference}__{split_name}__shard_0000.npz"
        np.savez(
            shard_path,
            locator_input_image=np.stack(images, axis=0).astype(np.float32),
            target_center_xy_original_px=np.asarray(target_original, dtype=np.float32),
            target_center_xy_canvas_px=np.asarray(target_canvas, dtype=np.float32),
            source_image_wh_px=np.asarray(source_wh, dtype=np.int32),
            resized_image_wh_px=np.asarray(resized_wh, dtype=np.int32),
            padding_ltrb_px=np.asarray(padding, dtype=np.int32),
            resize_scale=np.asarray(resize_scale, dtype=np.float32),
            sample_id=np.asarray(sample_ids),
            image_filename=np.asarray(image_filenames),
            npz_row_index=np.asarray(npz_row_index, dtype=np.int64),
            bootstrap_bbox_xyxy_px=np.asarray(bbox_xyxy, dtype=np.float32),
            bootstrap_confidence=np.asarray(confidence, dtype=np.float32),
            locator_geometry_schema=np.asarray(EXPECTED_GEOMETRY_SCHEMA),
        )

        pd.DataFrame(rows).to_csv(manifests_dir / "samples.csv", index=False)
        (manifests_dir / "run.json").write_text(
            json.dumps(
                {
                    "RunId": f"{dataset_reference}_{split_name}",
                    "PreprocessingContract": {
                        "ContractVersion": "rb-preprocess-roi-fcn-v0_1",
                        "CurrentRepresentation": {
                            "Kind": "roi_fcn_locator_npz",
                            "StorageFormat": "npz",
                            "ArrayKeys": [
                                "locator_input_image",
                                "target_center_xy_original_px",
                                "target_center_xy_canvas_px",
                                "source_image_wh_px",
                                "resized_image_wh_px",
                                "padding_ltrb_px",
                                "resize_scale",
                                "sample_id",
                                "image_filename",
                                "npz_row_index",
                                "bootstrap_bbox_xyxy_px",
                                "bootstrap_confidence",
                                "locator_geometry_schema",
                            ],
                            "ImageLayout": "N,C,H,W",
                            "Channels": 1,
                            "CanvasWidth": canvas_width,
                            "CanvasHeight": canvas_height,
                            "ImageKind": "full_frame_locator_canvas",
                            "ImageColorMode": "grayscale",
                            "NormalizationRange": [0.0, 1.0],
                            "AspectRatioPolicy": "preserve_with_padding",
                            "PadValue": 0,
                            "TargetType": "crop_center_point",
                            "TargetGeneration": "training_loader_gaussian_from_canvas_center",
                            "TargetSource": "edge_roi_v1_bootstrap",
                            "FixedROICropWidthPx": roi_width,
                            "FixedROICropHeightPx": roi_height,
                        },
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
