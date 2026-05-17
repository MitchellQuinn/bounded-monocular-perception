"""Compatibility shim for the v0.3 live inference contract surface.

The canonical contract module now lives at
``live_inference.interfaces.contracts``.  The top-level ``interfaces`` package is
retained so copied worker/engine modules keep importing the same boundary
objects while v0.3 gradually removes legacy naming.
"""

from live_inference.interfaces.contracts import *  # noqa: F403
from live_inference.interfaces.contracts import __all__
