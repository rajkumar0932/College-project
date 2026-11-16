"""Automatic Modulation Classification (AMC) with deep learning on raw I/Q data."""
import sys as _sys

# Windows consoles/pipes often use cp1252, which cannot print symbols such as
# "≥". Replace unprintable characters instead of crashing.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

# Class order used everywhere (the sorted RML2016.10a modulation names).
CLASSES = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]

DEFAULT_SEED = 2026
