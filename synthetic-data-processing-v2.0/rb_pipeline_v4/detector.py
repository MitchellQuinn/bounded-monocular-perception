"""YOLO detector adapters used by v4 detect stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts import Detection, ObjectDetector
from .image_io import to_grayscale_uint8


class UltralyticsYoloDetector(ObjectDetector):
    """Ultralytics YOLO adapter implementing the ObjectDetector protocol."""

    def __init__(
        self,
        *,
        model_path: str,
        conf_threshold: float,
        iou_threshold: float,
        imgsz: int,
        max_det: int,
        device: str | None,
    ) -> None:
        model_path_text = str(model_path).strip()
        if not model_path_text:
            raise ValueError("model_path is required when using UltralyticsYoloDetector")

        self.model_path = model_path_text
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.device = device
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics YOLO import failed. Ensure package is installed in the active environment."
            ) from exc

        model_ref = str(self.model_path).strip()
        path_obj = Path(model_ref)

        if path_obj.is_file():
            self._model = YOLO(str(path_obj))
            return self._model

        # Allow model names like 'yolov8n.pt' to auto-resolve/download via Ultralytics.
        looks_like_explicit_path = (
            path_obj.is_absolute()
            or "/" in model_ref
            or "\\" in model_ref
            or model_ref.startswith(".")
        )
        if looks_like_explicit_path and not path_obj.exists():
            raise FileNotFoundError(f"YOLO model weights not found: {path_obj}")

        self._model = YOLO(model_ref)
        return self._model

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        model = self._load_model()
        results = model.predict(
            source=image_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4), dtype=np.float32)
        cls_ids = boxes.cls.cpu().numpy().astype(np.int64) if boxes.cls is not None else np.empty((0,), dtype=np.int64)
        conf = boxes.conf.cpu().numpy().astype(np.float32) if boxes.conf is not None else np.zeros((len(xyxy),), dtype=np.float32)

        names_obj = getattr(result, "names", None)
        if names_obj is None:
            names_obj = getattr(model, "names", {})

        out: list[Detection] = []
        for idx in range(int(xyxy.shape[0])):
            class_id = int(cls_ids[idx]) if idx < cls_ids.shape[0] else -1
            confidence = float(conf[idx]) if idx < conf.shape[0] else 0.0

            class_name = str(class_id)
            if isinstance(names_obj, dict):
                if class_id in names_obj:
                    class_name = str(names_obj[class_id])
            elif isinstance(names_obj, list) and 0 <= class_id < len(names_obj):
                class_name = str(names_obj[class_id])

            x1, y1, x2, y2 = [float(value) for value in xyxy[idx].tolist()]
            out.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return out


class EdgeRoiDetector(ObjectDetector):
    """v1-style edge detector that returns one ROI bbox centered on edge centroid."""

    def __init__(
        self,
        *,
        blur_kernel_size: int,
        canny_low_threshold: int,
        canny_high_threshold: int,
        foreground_threshold: int,
        padding_px: int,
        min_foreground_px: int,
        close_kernel_size: int,
        class_id: int = 0,
        class_name: str = "defender",
    ) -> None:
        self.blur_kernel_size = self._normalized_blur_kernel_size(blur_kernel_size)
        self.canny_low_threshold = max(0, min(255, int(canny_low_threshold)))
        self.canny_high_threshold = max(0, min(255, int(canny_high_threshold)))
        self.foreground_threshold = max(0, min(255, int(foreground_threshold)))
        self.padding_px = max(0, int(padding_px))
        self.min_foreground_px = max(1, int(min_foreground_px))
        self.close_kernel_size = max(1, int(close_kernel_size))
        self.class_id = int(class_id)
        self.class_name = str(class_name).strip() or "defender"

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        gray = to_grayscale_uint8(image_bgr)
        processed = gray
        if self.blur_kernel_size > 1:
            processed = cv2.GaussianBlur(processed, (self.blur_kernel_size, self.blur_kernel_size), 0)

        edges = cv2.Canny(processed, int(self.canny_low_threshold), int(self.canny_high_threshold))
        if self.close_kernel_size > 1:
            close_kernel = np.ones((self.close_kernel_size, self.close_kernel_size), dtype=np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

        edge_black_on_white = np.full(gray.shape, 255, dtype=np.uint8)
        edge_black_on_white[edges > 0] = 0
        foreground_mask = edge_black_on_white < int(self.foreground_threshold)
        if not bool(np.any(foreground_mask)):
            return []

        ys, xs = np.where(foreground_mask)
        if xs.size < int(self.min_foreground_px):
            return []

        raw_x1 = max(0, int(xs.min()) - self.padding_px)
        raw_y1 = max(0, int(ys.min()) - self.padding_px)
        raw_x2 = min(gray.shape[1] - 1, int(xs.max()) + self.padding_px)
        raw_y2 = min(gray.shape[0] - 1, int(ys.max()) + self.padding_px)

        roi_width = max(1, int(raw_x2 - raw_x1 + 1))
        roi_height = max(1, int(raw_y2 - raw_y1 + 1))

        center_x = float(xs.mean())
        center_y = float(ys.mean())

        centered_x1 = int(round(center_x - (roi_width / 2.0)))
        centered_y1 = int(round(center_y - (roi_height / 2.0)))
        x1, x2_exclusive = _clamp_interval_by_size(
            centered_x1,
            roi_width,
            limit=int(gray.shape[1]),
        )
        y1, y2_exclusive = _clamp_interval_by_size(
            centered_y1,
            roi_height,
            limit=int(gray.shape[0]),
        )

        confidence = min(1.0, float(xs.size) / float(max(1, roi_width * roi_height)))
        detection = Detection(
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=float(confidence),
            x1=float(x1),
            y1=float(y1),
            x2=float(x2_exclusive),
            y2=float(y2_exclusive),
            center_x_px=float(center_x),
            center_y_px=float(center_y),
        )
        return [detection]

    @staticmethod
    def _normalized_blur_kernel_size(value: int) -> int:
        kernel = max(1, int(value))
        if kernel % 2 == 0:
            kernel += 1
        return kernel


def _clamp_interval_by_size(start: int, size: int, *, limit: int) -> tuple[int, int]:
    total = max(1, int(limit))
    extent = max(1, int(size))

    if extent >= total:
        return 0, total

    s = int(start)
    e = s + extent
    if s < 0:
        s = 0
        e = extent
    if e > total:
        e = total
        s = max(0, total - extent)
    return s, e
