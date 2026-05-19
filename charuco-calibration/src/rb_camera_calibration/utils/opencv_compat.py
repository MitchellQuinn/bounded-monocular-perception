"""Compatibility helpers for OpenCV ArUco/ChArUco APIs.

OpenCV's Python ArUco bindings changed across 4.x releases.  This module keeps
those differences local so detection and calibration code can operate on simple
Python data and contract objects.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from rb_camera_calibration.contracts import (
    CharucoBoardConfig,
    PLACEHOLDER_ARUCO_DICTIONARY,
)


PREFERRED_DICTIONARY_NAMES: tuple[str, ...] = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
)


def import_cv2() -> Any:
    """Import OpenCV with a helpful error when unavailable."""
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import environment dependent
        raise RuntimeError(
            "OpenCV is required for camera calibration. Install opencv-python "
            "with ArUco/ChArUco support in the repository virtual environment."
        ) from exc
    return cv2


def require_aruco() -> Any:
    """Return ``cv2.aruco`` or raise a clear compatibility error."""
    cv2 = import_cv2()
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError(
            "The installed OpenCV build does not expose cv2.aruco. "
            "Install an OpenCV package with contrib/objdetect ArUco support."
        )
    return aruco


def available_dictionary_names() -> tuple[str, ...]:
    """Return available OpenCV predefined dictionary constant names."""
    aruco = require_aruco()
    names = []
    for name in dir(aruco):
        if not name.startswith("DICT_"):
            continue
        value = getattr(aruco, name)
        if isinstance(value, int):
            names.append(name)
    return tuple(sorted(set(names)))


def common_dictionary_names() -> tuple[str, ...]:
    """Return common predefined dictionaries present in this OpenCV build."""
    available = set(available_dictionary_names())
    preferred = [name for name in PREFERRED_DICTIONARY_NAMES if name in available]
    remaining = [name for name in sorted(available) if name not in preferred]
    return tuple(preferred + remaining)


def normalize_dictionary_name(name: str) -> str:
    """Normalize user-provided dictionary names to OpenCV constant names."""
    cleaned = name.strip()
    if cleaned == PLACEHOLDER_ARUCO_DICTIONARY:
        raise ValueError(
            "aruco_dictionary is still the placeholder <ARUCO_DICTIONARY>. "
            "Run the dictionary probe and set an explicit OpenCV dictionary "
            "constant such as DICT_5X5_100 before calibration."
        )
    if cleaned.startswith("cv2.aruco."):
        cleaned = cleaned.removeprefix("cv2.aruco.")
    if cleaned.startswith("aruco."):
        cleaned = cleaned.removeprefix("aruco.")
    if not cleaned.startswith("DICT_"):
        cleaned = f"DICT_{cleaned}"
    return cleaned


def resolve_aruco_dictionary(name: str) -> Any:
    """Resolve a predefined OpenCV dictionary by name."""
    aruco = require_aruco()
    normalized = normalize_dictionary_name(name)
    if not hasattr(aruco, normalized):
        choices = ", ".join(common_dictionary_names())
        raise ValueError(
            f"Unknown OpenCV ArUco dictionary {name!r}. Available dictionaries: {choices}."
        )
    return aruco.getPredefinedDictionary(getattr(aruco, normalized))


def create_detector_parameters() -> Any:
    """Create detector parameters across OpenCV API versions."""
    aruco = require_aruco()
    if hasattr(aruco, "DetectorParameters"):
        return aruco.DetectorParameters()
    if hasattr(aruco, "DetectorParameters_create"):
        return aruco.DetectorParameters_create()
    raise RuntimeError("OpenCV ArUco detector parameters API is unavailable.")


def create_charuco_board(config: CharucoBoardConfig, *, reverse_dimensions: bool = False) -> Any:
    """Create an OpenCV ChArUco board from a dependency-light config."""
    aruco = require_aruco()
    cv2 = import_cv2()
    dictionary = resolve_aruco_dictionary(config.aruco_dictionary)
    squares_x = config.squares_y if reverse_dimensions else config.squares_x
    squares_y = config.squares_x if reverse_dimensions else config.squares_y
    if hasattr(aruco, "CharucoBoard"):
        try:
            return aruco.CharucoBoard(
                (int(squares_x), int(squares_y)),
                float(config.square_length_m),
                float(config.marker_length_m),
                dictionary,
            )
        except TypeError:
            return aruco.CharucoBoard(
                cv2.Size(int(squares_x), int(squares_y)),
                float(config.square_length_m),
                float(config.marker_length_m),
                dictionary,
            )
    if hasattr(aruco, "CharucoBoard_create"):
        return aruco.CharucoBoard_create(
            int(squares_x),
            int(squares_y),
            float(config.square_length_m),
            float(config.marker_length_m),
            dictionary,
        )
    raise RuntimeError("OpenCV ChArUco board creation API is unavailable.")


def config_with_reversed_dimensions(config: CharucoBoardConfig) -> CharucoBoardConfig:
    """Return a board config with ``squares_x`` and ``squares_y`` swapped."""
    return replace(config, squares_x=config.squares_y, squares_y=config.squares_x)


def decode_image_bytes(image_bytes: bytes) -> Any:
    """Decode encoded image bytes to a BGR/gray OpenCV image."""
    cv2 = import_cv2()
    import numpy as np

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Could not decode CameraFrame.image_bytes as an image.")
    return image


def encode_png(image: Any) -> bytes:
    """Encode an OpenCV image as PNG bytes."""
    cv2 = import_cv2()
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("OpenCV failed to encode frame as PNG.")
    return bytes(encoded)


def to_gray(image: Any) -> Any:
    """Convert a BGR/BGRA image to grayscale when needed."""
    cv2 = import_cv2()
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_markers(image: Any, dictionary_name: str) -> tuple[Any, Any, Any]:
    """Detect ArUco markers using the best API available."""
    aruco = require_aruco()
    dictionary = resolve_aruco_dictionary(dictionary_name)
    parameters = create_detector_parameters()
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers(image)
    return aruco.detectMarkers(image, dictionary, parameters=parameters)


def detect_charuco_board(
    image: Any,
    config: CharucoBoardConfig,
    *,
    reverse_dimensions: bool = False,
) -> dict[str, Any]:
    """Detect a ChArUco board and return raw OpenCV arrays in a dict."""
    aruco = require_aruco()
    board = create_charuco_board(config, reverse_dimensions=reverse_dimensions)
    parameters = create_detector_parameters()

    if hasattr(aruco, "CharucoDetector"):
        try:
            detector = aruco.CharucoDetector(board, detectorParams=parameters)
        except TypeError:
            detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)
        return {
            "board": board,
            "charuco_corners": charuco_corners,
            "charuco_ids": charuco_ids,
            "marker_corners": marker_corners,
            "marker_ids": marker_ids,
        }

    marker_corners, marker_ids, rejected = detect_markers(image, config.aruco_dictionary)
    if marker_ids is None or len(marker_ids) == 0:
        return {
            "board": board,
            "charuco_corners": None,
            "charuco_ids": None,
            "marker_corners": marker_corners,
            "marker_ids": marker_ids,
            "rejected": rejected,
        }
    if not hasattr(aruco, "interpolateCornersCharuco"):
        raise RuntimeError("OpenCV ChArUco corner interpolation API is unavailable.")
    _count, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        image,
        board,
    )
    return {
        "board": board,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
    }


def ids_to_tuple(ids: Any) -> tuple[int, ...]:
    """Convert OpenCV id arrays to a tuple of ints."""
    if ids is None:
        return ()
    import numpy as np

    return tuple(int(item) for item in np.asarray(ids).reshape(-1).tolist())


def charuco_corners_to_list(corners: Any) -> list[list[float]]:
    """Convert ChArUco corner arrays to ``[[x, y], ...]``."""
    if corners is None:
        return []
    import numpy as np

    points = np.asarray(corners, dtype=float).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in points]


def marker_corners_to_list(marker_corners: Any) -> list[list[list[float]]]:
    """Convert marker corner arrays to ``[[[x, y] * 4], ...]``."""
    if marker_corners is None:
        return []
    import numpy as np

    serialised: list[list[list[float]]] = []
    for corners in marker_corners:
        points = np.asarray(corners, dtype=float).reshape(-1, 2)
        serialised.append([[float(x), float(y)] for x, y in points])
    return serialised


def corners_from_list(points: Iterable[Iterable[float]]) -> Any:
    """Convert serialised ChArUco corners back to OpenCV shape."""
    import numpy as np

    return np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)


def ids_from_list(ids: Iterable[int]) -> Any:
    """Convert serialised ids back to OpenCV shape."""
    import numpy as np

    return np.asarray(list(ids), dtype=np.int32).reshape(-1, 1)


def marker_corners_from_list(markers: Iterable[Iterable[Iterable[float]]]) -> list[Any]:
    """Convert serialised marker corners back to OpenCV marker-corner arrays."""
    import numpy as np

    return [np.asarray(marker, dtype=np.float32).reshape(1, 4, 2) for marker in markers]


def board_point_metrics(marker_corners: Any, image_shape: tuple[int, ...]) -> dict[str, Any]:
    """Compute bounds, center, area, edge margin, and simple pose proxies."""
    cv2 = import_cv2()
    import numpy as np

    if marker_corners is None or len(marker_corners) == 0:
        return {
            "center_xy_px": None,
            "bounds_xyxy_px": None,
            "area_fraction": 0.0,
            "edge_margin_px": None,
            "quad_xy_px": None,
            "roll_like_angle_deg": 0.0,
            "perspective_skew_score": 0.0,
        }

    points = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1, 2) for c in marker_corners])
    height, width = int(image_shape[0]), int(image_shape[1])
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    area_fraction = float(cv2.contourArea(hull) / max(float(width * height), 1.0))
    edge_margin = float(min(min_xy[0], min_xy[1], width - max_xy[0], height - max_xy[1]))
    rect = cv2.minAreaRect(points.astype("float32"))
    box = cv2.boxPoints(rect)
    side_lengths = [
        float(np.linalg.norm(box[(i + 1) % 4] - box[i]))
        for i in range(4)
    ]
    longest = max(side_lengths) if side_lengths else 0.0
    shortest = min(side_lengths) if side_lengths else 0.0
    skew = 0.0 if longest <= 0 else float(1.0 - shortest / longest)
    angle = float(rect[2])
    if angle < -45.0:
        angle += 90.0
    return {
        "center_xy_px": (float(center[0]), float(center[1])),
        "bounds_xyxy_px": (
            float(min_xy[0]),
            float(min_xy[1]),
            float(max_xy[0]),
            float(max_xy[1]),
        ),
        "area_fraction": area_fraction,
        "edge_margin_px": edge_margin,
        "quad_xy_px": [[float(x), float(y)] for x, y in box],
        "roll_like_angle_deg": angle,
        "perspective_skew_score": skew,
    }


def draw_detection_overlay(image: Any, detection_extras: Mapping[str, Any]) -> Any:
    """Draw marker and ChArUco overlays from serialised detection extras."""
    cv2 = import_cv2()
    aruco = require_aruco()
    import numpy as np

    overlay = image.copy()
    marker_corners_raw = detection_extras.get("marker_corners_xy", ())
    marker_ids_raw = detection_extras.get("marker_ids", ())
    charuco_corners_raw = detection_extras.get("charuco_corners_xy", ())
    charuco_ids_raw = detection_extras.get("charuco_ids", ())
    board_quad_raw = detection_extras.get("board_quad_xy_px")

    if marker_corners_raw and marker_ids_raw:
        marker_corners = marker_corners_from_list(marker_corners_raw)
        marker_ids = ids_from_list(marker_ids_raw)
        aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
    if charuco_corners_raw and charuco_ids_raw and hasattr(aruco, "drawDetectedCornersCharuco"):
        charuco_corners = corners_from_list(charuco_corners_raw)
        charuco_ids = ids_from_list(charuco_ids_raw)
        aruco.drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids, (255, 0, 0))
    if board_quad_raw:
        pts = np.asarray(board_quad_raw, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    return overlay
