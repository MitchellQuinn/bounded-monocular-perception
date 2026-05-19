"""Diagnostic utility for finding the likely ArUco dictionary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from rb_camera_calibration.contracts import (
    DictionaryProbeCandidate,
    DictionaryProbeConfidence,
    DictionaryProbeReport,
)
from rb_camera_calibration.utils import opencv_compat as cvx


def confidence_for_marker_count(marker_count: int) -> DictionaryProbeConfidence:
    """Map marker count to a simple diagnostic confidence label."""
    if marker_count >= 8:
        return DictionaryProbeConfidence.HIGH
    if marker_count >= 4:
        return DictionaryProbeConfidence.MEDIUM
    if marker_count > 0:
        return DictionaryProbeConfidence.LOW
    return DictionaryProbeConfidence.NONE


def probe_image(
    image: object,
    *,
    frame_id: str | None = None,
    dictionary_names: Iterable[str] | None = None,
) -> DictionaryProbeReport:
    """Try common predefined dictionaries against one decoded image."""
    names = tuple(dictionary_names) if dictionary_names is not None else cvx.common_dictionary_names()
    candidates: list[DictionaryProbeCandidate] = []
    for name in names:
        try:
            _corners, ids, _rejected = cvx.detect_markers(image, name)
        except Exception as exc:
            candidates.append(
                DictionaryProbeCandidate(
                    dictionary_name=name,
                    marker_count=0,
                    marker_ids=(),
                    confidence=DictionaryProbeConfidence.NONE,
                    usefulness_score=0.0,
                    extras={"error": str(exc)},
                )
            )
            continue
        marker_ids = cvx.ids_to_tuple(ids)
        marker_count = len(marker_ids)
        unique_count = len(set(marker_ids))
        usefulness = float(marker_count + unique_count * 0.25)
        candidates.append(
            DictionaryProbeCandidate(
                dictionary_name=name,
                marker_count=marker_count,
                marker_ids=marker_ids,
                confidence=confidence_for_marker_count(marker_count),
                usefulness_score=usefulness,
            )
        )
    ranked = tuple(
        sorted(
            candidates,
            key=lambda item: (item.marker_count, len(set(item.marker_ids)), item.usefulness_score),
            reverse=True,
        )
    )
    best = ranked[0] if ranked and ranked[0].marker_count > 0 else None
    image_size = None
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        image_size = (int(shape[1]), int(shape[0]))
    warnings = ()
    if best is None:
        warnings = (
            "No markers were detected with common predefined dictionaries. "
            "Check focus, exposure, and whether the board uses a custom dictionary.",
        )
    return DictionaryProbeReport(
        frame_id=frame_id,
        candidates=ranked,
        best_candidate=best,
        image_size_wh_px=image_size,
        warning_messages=warnings,
    )


def probe_image_path(path: str | Path) -> DictionaryProbeReport:
    """Load an image from disk and probe common dictionaries."""
    cv2 = cvx.import_cv2()
    resolved = Path(path)
    image = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image for dictionary probe: {resolved}")
    return probe_image(image, frame_id=resolved.name)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for dictionary probing."""
    parser = argparse.ArgumentParser(description="Probe common OpenCV ArUco dictionaries.")
    parser.add_argument("--image", required=True, help="Path to a frame image.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full probe report as JSON.",
    )
    args = parser.parse_args(argv)

    report = probe_image_path(args.image)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0

    if report.best_candidate is None:
        print("No likely dictionary found.")
    else:
        best = report.best_candidate
        print(
            "Best candidate: "
            f"{best.dictionary_name} markers={best.marker_count} "
            f"ids={list(best.marker_ids)} confidence={best.confidence.value}"
        )
    print("Top candidates:")
    for candidate in report.candidates[:10]:
        print(
            f"  {candidate.dictionary_name:24s} "
            f"markers={candidate.marker_count:3d} "
            f"ids={list(candidate.marker_ids)} "
            f"confidence={candidate.confidence.value}"
        )
    for warning in report.warning_messages:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
