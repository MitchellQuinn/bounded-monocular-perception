"""Merge accepted frames from existing calibration sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

from rb_camera_calibration.capture.session_store import (
    CalibrationSessionStore,
    load_accepted_frames_from_manifest,
)
from rb_camera_calibration.io.config_loader import build_session_config, default_calibration_runs_root


def discover_merge_session_roots(current_session_root: Path) -> tuple[Path, ...]:
    """Find session directories worth offering for a one-click merge.

    The scan includes the package-local run directory, the current session's
    parent, and the legacy repository-root ``calibration_runs`` directory that
    early versions of this tool used.
    """
    current = current_session_root.resolve()
    default_root = default_calibration_runs_root()
    candidate_roots = [
        default_root,
        current_session_root.parent,
        default_root.parent.parent / "calibration_runs",
    ]
    discovered: list[Path] = []
    seen: set[Path] = set()
    for root in candidate_roots:
        if not root.exists() or not root.is_dir():
            continue
        for manifest in sorted(root.glob("*/session_manifest.json")):
            session_root = manifest.parent.resolve()
            if session_root == current or session_root in seen:
                continue
            discovered.append(session_root)
            seen.add(session_root)
    return tuple(discovered)


def merge_session_roots_into_store(
    store: CalibrationSessionStore,
    source_session_roots: tuple[Path, ...],
) -> tuple[int, Path]:
    """Copy accepted frames from source sessions into an existing store."""
    store.initialise(load_existing=True)
    output_root = store.session_root.resolve()
    existing_keys = {
        (frame.frame_hash.value, frame.charuco_corner_count, frame.marker_count)
        for frame in store.accepted_frames
    }
    merged_count = 0
    for source_root in source_session_roots:
        if source_root.resolve() == output_root:
            continue
        manifest_path = source_root / "session_manifest.json"
        frames = load_accepted_frames_from_manifest(manifest_path)
        if not frames:
            continue
        for frame in frames:
            key = (frame.frame_hash.value, frame.charuco_corner_count, frame.marker_count)
            if key in existing_keys:
                continue
            store.copy_accepted_frame_from(frame, source_session_root=source_root)
            existing_keys.add(key)
            merged_count += 1
    store.write_session_manifest()
    return merged_count, store.manifest_path


def merge_sessions(
    *,
    output_session_root: Path,
    source_session_roots: tuple[Path, ...],
    board_config_path: Path,
    camera_config_path: Path,
    capture_policy_path: Path | None = None,
) -> tuple[int, Path]:
    """Copy accepted frames from source sessions into one output session."""
    session_config = build_session_config(
        board_config_path=board_config_path,
        camera_config_path=camera_config_path,
        capture_policy_path=capture_policy_path,
        session_root=output_session_root,
    )
    store = CalibrationSessionStore(session_config)
    return merge_session_roots_into_store(store, source_session_roots)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge accepted ChArUco calibration frames from existing sessions.",
    )
    parser.add_argument(
        "source_session_roots",
        nargs="+",
        type=Path,
        help="Existing run directories containing session_manifest.json.",
    )
    parser.add_argument(
        "--output-session-root",
        required=True,
        type=Path,
        help="Target run directory to create or append to.",
    )
    parser.add_argument("--board-config", required=True, type=Path)
    parser.add_argument("--camera-config", required=True, type=Path)
    parser.add_argument("--capture-policy", default=None, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    merged_count, manifest_path = merge_sessions(
        output_session_root=args.output_session_root,
        source_session_roots=tuple(args.source_session_roots),
        board_config_path=args.board_config,
        camera_config_path=args.camera_config,
        capture_policy_path=args.capture_policy,
    )
    print(f"Merged {merged_count} accepted frames.")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
