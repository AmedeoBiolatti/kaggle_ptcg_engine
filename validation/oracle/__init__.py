"""Oracle: differential testing of the custom C++ engine against cabt.

See README.md for the record-and-replay ("tape") design.
"""
from .canonical import (
    canonical_options,
    canonical_state,
    diff,
    option_descriptor,
)

__all__ = ["canonical_state", "canonical_options", "option_descriptor", "diff"]
