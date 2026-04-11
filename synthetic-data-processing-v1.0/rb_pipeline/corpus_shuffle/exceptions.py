"""Exceptions for input corpus shuffle operations."""

from __future__ import annotations


class CorpusShuffleError(RuntimeError):
    """Base exception for corpus shuffle failures."""


class CorpusShuffleValidationError(CorpusShuffleError):
    """Raised when inputs fail validation."""

