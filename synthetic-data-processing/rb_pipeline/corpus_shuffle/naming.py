"""Filename and sample-id regeneration helpers for corpus shuffle."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from .exceptions import CorpusShuffleValidationError

_FRAME_TOKEN_PATTERN = re.compile(r"f\d{6}")


def ensure_frame_token(value: str, field_name: str) -> None:
    """Validate that a value contains a parseable frame token."""

    if not isinstance(value, str) or not value.strip():
        raise CorpusShuffleValidationError(
            f"{field_name} is required and must be a non-empty string."
        )

    if _FRAME_TOKEN_PATTERN.search(value) is None:
        raise CorpusShuffleValidationError(
            f"{field_name}={value!r} does not contain a parseable f###### token."
        )


def _replace_frame_token(value: str, new_frame_index: int, field_name: str) -> str:
    ensure_frame_token(value, field_name)
    replacement = f"f{new_frame_index:06d}"
    replaced, count = _FRAME_TOKEN_PATTERN.subn(replacement, value, count=1)
    if count != 1:
        raise CorpusShuffleValidationError(
            f"Could not replace frame token in {field_name}={value!r}."
        )
    return replaced


def build_output_name(source_image_filename: str, new_frame_index: int) -> str:
    """Build output image filename with updated frame token."""

    normalized = str(source_image_filename).replace("\\", "/")
    source_path = PurePosixPath(normalized)
    replaced_name = _replace_frame_token(source_path.name, new_frame_index, "image_filename")
    return str(source_path.with_name(replaced_name))


def build_output_sample_id(source_sample_id: str, new_frame_index: int) -> str:
    """Build output sample_id with updated frame token."""

    return _replace_frame_token(source_sample_id, new_frame_index, "sample_id")

