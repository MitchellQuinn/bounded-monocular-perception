from __future__ import annotations

import pytest

from rb_camera_calibration.contracts import CharucoBoardConfig
from rb_camera_calibration.detection.dictionary_probe import probe_image
from rb_camera_calibration.utils import opencv_compat as cvx


def test_dictionary_resolver_rejects_placeholder() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        cvx.resolve_aruco_dictionary("<ARUCO_DICTIONARY>")


def test_dictionary_resolver_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown OpenCV ArUco dictionary"):
        cvx.resolve_aruco_dictionary("DICT_DOES_NOT_EXIST")


def test_dictionary_probe_ranks_matching_synthetic_board() -> None:
    cv2 = cvx.import_cv2()
    import numpy as np

    config = CharucoBoardConfig(
        squares_x=5,
        squares_y=7,
        square_length_m=0.04,
        marker_length_m=0.02,
        aruco_dictionary="DICT_5X5_100",
    )
    board = cvx.create_charuco_board(config)
    image = np.zeros((600, 400), dtype=np.uint8)
    board.generateImage((400, 600), image, 10, 1)

    report = probe_image(image, dictionary_names=("DICT_4X4_50", "DICT_5X5_100"))

    assert report.best_candidate is not None
    assert report.best_candidate.dictionary_name == "DICT_5X5_100"
    assert report.best_candidate.marker_count > 0


def test_opencv_compat_fails_gracefully_without_aruco(monkeypatch) -> None:
    class NoAruco:
        pass

    monkeypatch.setattr(cvx, "import_cv2", lambda: NoAruco())

    with pytest.raises(RuntimeError, match="does not expose cv2.aruco"):
        cvx.require_aruco()
