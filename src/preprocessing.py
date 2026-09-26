"""
Root src/preprocessing.py forwarding to student_resource/src/preprocessing.py
"""

import sys
import os

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
    LEGAL_SUFFIX_MAP,
    ADDRESS_ABBREVIATIONS,
    COUNTRY_MAP,
)

__all__ = [
    "normalize_text",
    "normalize_business_name",
    "normalize_address",
    "normalize_country",
    "extract_name_tokens",
    "extract_address_tokens",
    "preprocess_dataframe",
    "LEGAL_SUFFIX_MAP",
    "ADDRESS_ABBREVIATIONS",
    "COUNTRY_MAP",
]
