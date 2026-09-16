"""Brain connectivity analysis package"""

# filepath: src/core/__init__.py
"""Core utilities for brain connectivity analysis"""

# You can optionally expose key functions for easy imports:
from .data_loading import load_metadata, load_sc_matrix

__all__ = [
    'load_metadata', 'load_sc_matrix',
]