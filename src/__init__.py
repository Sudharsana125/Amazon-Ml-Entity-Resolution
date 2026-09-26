"""
Root src re-export for Business Entity Resolution Preprocessing Module.
Phase 1: Data Understanding & Text Normalization.
"""

import sys
import os

# Ensure student_resource is on sys.path
_curr = os.path.dirname(os.path.abspath(__file__))
_sr = os.path.join(os.path.dirname(_curr), "student_resource")
if _sr not in sys.path:
    sys.path.insert(0, _sr)

from student_resource.src.preprocessing import (
    normalize_text,
    normalize_business_name,
    normalize_address,
    normalize_country,
    extract_name_tokens,
    extract_address_tokens,
    preprocess_dataframe,
)

__all__ = [
    "normalize_text",
    "normalize_business_name",
    "normalize_address",
    "normalize_country",
    "extract_name_tokens",
    "extract_address_tokens",
    "preprocess_dataframe",
]
