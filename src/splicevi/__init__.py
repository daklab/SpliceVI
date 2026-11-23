"""SpliceVI: a joint model of single-cell gene expression and alternative splicing (built on scvi-tools)."""
import logging

from .splicevi import SPLICEVI
from .presets import preset, filter_read_support

__all__ = ["SPLICEVI", "preset", "filter_read_support"]

# Progress messages (model set-up, fine-tuning) are shown by default, as scvi-tools does for its own messages.
# Silence them with logging.getLogger("splicevi").setLevel(logging.WARNING).
_logger = logging.getLogger("splicevi")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[splicevi] %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
