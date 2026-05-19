"""Frame hashing utilities."""

from __future__ import annotations

import hashlib

from rb_camera_calibration.contracts import (
    DEFAULT_FRAME_HASH_ALGORITHM,
    DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES,
    FrameHash,
)


def hash_bytes(
    data: bytes,
    *,
    algorithm: str = DEFAULT_FRAME_HASH_ALGORITHM,
    digest_size_bytes: int = DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES,
) -> FrameHash:
    """Hash bytes using the contract's default BLAKE2b-128 representation."""
    if algorithm != DEFAULT_FRAME_HASH_ALGORITHM:
        raise ValueError(
            f"Unsupported frame hash algorithm {algorithm!r}; "
            f"expected {DEFAULT_FRAME_HASH_ALGORITHM!r}."
        )
    digest = hashlib.blake2b(data, digest_size=digest_size_bytes).hexdigest()
    return FrameHash(
        value=digest,
        algorithm=algorithm,
        digest_size_bytes=digest_size_bytes,
    )
