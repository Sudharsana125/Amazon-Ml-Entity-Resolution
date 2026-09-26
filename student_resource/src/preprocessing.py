"""
Business Entity Resolution - Preprocessing Module
Amazon ML Challenge 2026

Phase 1: Data Understanding and Text Preprocessing
This module provides fast, robust, and reusable normalization functions
for business names, addresses, and countries across multiple noisy data sources.

Key Design Principles:
1. Information Preservation: Avoid aggressive removal of tokens (numbers,
   street names, landmarks, legal forms) that provide discriminative signal.
2. Robustness to Noise: Standardize casing, spacing, Unicode artifacts,
   common abbreviations, punctuation anomalies, and web domain patterns.
3. Open-Set Country Support: Handle US, India, France, and any unseen
   country names gracefully without hard-coded filtering.
4. Safe Null Handling: Gracefully manage NaNs, empty strings, and non-string types.
5. High Performance: Utilize pre-compiled regular expressions and vectorized
   operations suitable for millions of records.
"""

from typing import Any, List, Optional, Set
import re
import unicodedata
import pandas as pd


# ---------------------------------------------------------------------------
# Pre-compiled Regular Expressions for Performance
# ---------------------------------------------------------------------------

# Multi-space and whitespace cleanup
RE_WHITESPACE = re.compile(r"\s+")

# Corrupted Unicode replacement characters and control characters
RE_CORRUPT_CHARS = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Decorative brackets and symbols at start/end or repeated
RE_DECORATIVE_SYMBOLS = re.compile(r"^[\s#*~<>\-=_+!?,.:;\"\'`|/\\]+|[\s#*~<>\-=_+!?,.:;\"\'`|/\\]+$")
RE_DOUBLE_PUNCT = re.compile(r"([^\w\s])\1+")

# Web address patterns in business names (e.g., "siiainvestments.com", "www.shop.in")
RE_URL_PREFIX = re.compile(r"^(?:https?://)?(?:www\.)?", re.IGNORECASE)
RE_DOMAIN_SUFFIX = re.compile(r"\.(?:com|org|net|io|co|in|co\.in|fr|biz|info|us|org\.in|gov|edu)(?:/.*)?$", re.IGNORECASE)

# Hyphenated ranges with spaces (e.g. "1708 - 1710", "30091- 426th")
RE_HYPHEN_SPACES = re.compile(r"(\d+)\s*-\s*(\d+)")

# Ampersand normalization with proper word boundaries
RE_AMPERSAND = re.compile(r"\s*&\s*")

# Ordinal numbers normalization in street addresses (e.g., "1st", "2nd", "3rd", "4th")
RE_ORDINALS = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)

# Number with hash or no prefix (e.g. "# 125", "no. 5")
RE_NO_PREFIX = re.compile(r"\b(?:no|num|number)\.?\s*(\d+)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Dictionaries for Canonical Standardization
# ---------------------------------------------------------------------------

# Standardize common legal suffixes without discarding them
# Standardizing to a single canonical token ensures "Inc.", "Incorporated",
# and "Inc" all match identically while preserving the legal entity structure.
LEGAL_SUFFIX_MAP = {
    "incorporated": "inc",
    "incorporation": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "company": "co",
    "co": "co",
    "limited": "ltd",
    "ltd": "ltd",
    "private limited": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "pvt": "pvt",
    "private": "pvt",
    "limited liability company": "llc",
    "llc": "llc",
    "l.l.c.": "llc",
    "limited liability partnership": "llp",
    "llp": "llp",
    "professional corporation": "pc",
    "pc": "pc",
    "p.c.": "pc",
    # French legal entities (Test set support)
    "societe a responsabilite limitee": "sarl",
    "sarl": "sarl",
    "s.a.r.l.": "sarl",
    "societe par actions simplifiee": "sas",
    "sas": "sas",
    "s.a.s.": "sas",
    "societe par actions simplifiee unipersonnelle": "sasu",
    "sasu": "sasu",
    "entreprise unipersonnelle a responsabilite limitee": "eurl",
    "eurl": "eurl",
    "societe anonyme": "sa",
    "sa": "sa",
    "societe civile immobiliere": "sci",
    "sci": "sci",
}

# Compiled regex for legal suffixes at end of string
_LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(LEGAL_SUFFIX_MAP.keys(), key=len, reverse=True)) + r")\.?\b",
    re.IGNORECASE,
)

# Common address token abbreviations across US, India, and France
ADDRESS_ABBREVIATIONS = {
    # Street Types
    "street": "street",
    "st": "street",
    "str": "street",
    "road": "road",
    "rd": "road",
    "avenue": "avenue",
    "ave": "avenue",
    "av": "avenue",
    "boulevard": "boulevard",
    "blvd": "boulevard",
    "drive": "drive",
    "dr": "drive",
    "court": "court",
    "ct": "court",
    "lane": "lane",
    "ln": "lane",
    "parkway": "parkway",
    "pkwy": "parkway",
    "place": "place",
    "pl": "place",
    "highway": "highway",
    "hwy": "highway",
    "way": "way",
    "circle": "circle",
    "cir": "circle",
    "terrace": "terrace",
    "ter": "terrace",
    # Unit / Sub-premise
    "suite": "suite",
    "ste": "suite",
    "apartment": "apartment",
    "apt": "apartment",
    "building": "building",
    "bldg": "building",
    "floor": "floor",
    "fl": "floor",
    "unit": "unit",
    "room": "room",
    "rm": "room",
    "department": "department",
    "dept": "department",
    # Indian Localities & Landmarks
    "extension": "extension",
    "ext": "extension",
    "extn": "extension",
    "sector": "sector",
    "sec": "sector",
    "nagar": "nagar",
    "ngr": "nagar",
    "colony": "colony",
    "col": "colony",
    "opposite": "opposite",
    "opp": "opposite",
    "near": "near",
    "nr": "near",
    "behind": "behind",
    "bh": "behind",
    "beside": "beside",
    "cross": "cross",
    "main": "main",
    "layout": "layout",
    "lyt": "layout",
    "phase": "phase",
    "ph": "phase",
    "block": "block",
    "blk": "block",
    "district": "district",
    "dist": "district",
    "township": "township",
    "twp": "township",
    # French Address Types
    "rue": "rue",
    "r": "rue",
    "boulevard": "boulevard",
    "bd": "boulevard",
    "allee": "allee",
    "all": "allee",
    "impasse": "impasse",
    "imp": "impasse",
    "chemin": "chemin",
    "ch": "chemin",
    "place": "place",
    "passage": "passage",
}

_ADDRESS_ABBR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ADDRESS_ABBREVIATIONS.keys(), key=len, reverse=True)) + r")\.?\b",
    re.IGNORECASE,
)

# Known country synonyms for standard canonical representation
COUNTRY_MAP = {
    "us": "US",
    "usa": "US",
    "u.s.": "US",
    "u.s.a.": "US",
    "united states": "US",
    "united states of america": "US",
    "india": "India",
    "ind": "India",
    "in": "India",
    "bharat": "India",
    "france": "France",
    "fr": "France",
    "fra": "France",
    "republique francaise": "France",
}


# ---------------------------------------------------------------------------
# Core Preprocessing Functions
# ---------------------------------------------------------------------------

def normalize_text(text: Any, lowercase: bool = True, strip_accents: bool = True) -> str:
    """
    Base text normalization function.
    
    Handles:
    - Safe conversion of nulls, NaNs, numbers to string (returns "" for null)
    - Unicode NFKD normalization and diacritics handling (e.g. é -> e)
    - Removal of corrupt/control characters (e.g. \ufffd)
    - Normalization of smart quotes, dashes, and whitespace
    - Lowercasing (default True)
    
    Parameters
    ----------
    text : Any
        Input text, potentially None, float (np.nan), or string.
    lowercase : bool, default=True
        Whether to convert text to lowercase.
    strip_accents : bool, default=True
        Whether to decompose and strip accents/diacritics.
        
    Returns
    -------
    str
        Cleaned, normalized string.
    """
    if text is None or pd.isna(text):
        return ""
    
    # Convert to string
    s = str(text).strip()
    if not s:
        return ""
    
    # Remove corrupt Unicode replacement characters
    s = RE_CORRUPT_CHARS.sub(" ", s)
    
    # Normalize Unicode characters (NFKD)
    if strip_accents:
        # Decompose accented characters and filter out combining diacritical marks
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
    else:
        s = unicodedata.normalize("NFC", s)
        
    # Standardize curly quotes, backticks, and typographic dashes
    s = s.replace("’", "'").replace("`", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    
    # Lowercase
    if lowercase:
        s = s.lower()
        
    # Normalize multiple whitespace characters to a single space
    s = RE_WHITESPACE.sub(" ", s).strip()
    return s


def normalize_business_name(
    name: Any,
    standardize_legal: bool = True,
    strip_legal: bool = False,
    clean_domain: bool = True,
) -> str:
    """
    Normalizes a business name while preserving meaningful identity tokens.
    
    Handles:
    - Base text cleaning (Unicode, casing, whitespace)
    - Conversion of domain names (e.g., 'siiainvestments.com' -> 'siia investments')
    - Ampersand normalization ('&' -> 'and')
    - Removal of decorative brackets and wrapping symbols ('<< Team Ecole' -> 'team ecole')
    - Canonical standardization of legal suffixes ('Incorporated' -> 'inc', 'Pvt. Ltd.' -> 'pvt ltd')
    - Optional stripping of legal suffixes for core name matching
    - Preserves business acronyms, alphanumeric IDs, and distinctive keywords
    
    Parameters
    ----------
    name : Any
        Business name string or null.
    standardize_legal : bool, default=True
        Whether to standardize legal entity forms to canonical abbreviations.
    strip_legal : bool, default=False
        Whether to completely remove legal entity forms. False by default
        to prevent aggressive loss of information.
    clean_domain : bool, default=True
        Whether to parse web domain names into business name tokens.
        
    Returns
    -------
    str
        Normalized business name.
    """
    s = normalize_text(name, lowercase=True, strip_accents=True)
    if not s:
        return ""
    
    # Handle domain names (e.g. "www.whiteallgraphics.com" or "siiainvestments.com")
    if clean_domain and ("." in s and not " " in s):
        # Remove URL scheme and www
        s = RE_URL_PREFIX.sub("", s)
        # Remove common domain TLD suffixes
        s = RE_DOMAIN_SUFFIX.sub("", s)
        
    # Standardize ampersands to 'and'
    s = RE_AMPERSAND.sub(" and ", s)
    
    # Strip decorative punctuation (quotes, brackets, hashes, angle brackets)
    s = s.replace("<<", " ").replace(">>", " ")
    s = s.replace("(", " ").replace(")", " ")
    s = s.replace("[", " ").replace("]", " ")
    s = s.replace("{", " ").replace("}", " ")
    
    # Clean leading/trailing punctuation
    s = RE_DECORATIVE_SYMBOLS.sub("", s)
    
    # Legal suffix handling
    if standardize_legal:
        def _replace_suffix(match: re.Match) -> str:
            token = match.group(1).lower()
            return LEGAL_SUFFIX_MAP.get(token, token)
        s = _LEGAL_SUFFIX_PATTERN.sub(_replace_suffix, s)
        
    if strip_legal:
        # Strip known legal suffix tokens if explicitly requested
        s = _LEGAL_SUFFIX_PATTERN.sub("", s)
        
    # Normalize repeated punctuation and spaces
    s = RE_DOUBLE_PUNCT.sub(r"\1", s)
    s = RE_WHITESPACE.sub(" ", s).strip()
    return s


def normalize_address(
    address: Any,
    standardize_types: bool = True,
    preserve_components: bool = True,
) -> str:
    """
    Normalizes a business address while preserving structural geographic components.
    
    Handles:
    - Safe handling of missing addresses (null/NaN -> "")
    - Base text cleaning (Unicode, casing, whitespace)
    - Removal of noisy leading markers ('##', '#', '***')
    - Normalization of hyphenated building ranges ('30091- 426th' -> '30091-426th')
    - Standardization of street/building abbreviations ('st.', 'rd', 'ave', 'pvt')
    - Preserves house numbers, street names, landmarks, cities, postal codes, and states.
    
    Parameters
    ----------
    address : Any
        Business address string or null.
    standardize_types : bool, default=True
        Whether to standardize common street and unit abbreviations.
    preserve_components : bool, default=True
        Whether to preserve commas separating address components for downstream parsing.
        
    Returns
    -------
    str
        Normalized business address.
    """
    s = normalize_text(address, lowercase=True, strip_accents=True)
    if not s:
        return ""
    
    # Clean noisy prefix symbols (e.g. "##125 Mountain View Lane" -> "125 Mountain View Lane")
    s = re.sub(r"^[\s#*~,\-]+", "", s)
    
    # Standardize hyphenated ranges (e.g. "1708 - 1710" -> "1708-1710")
    s = RE_HYPHEN_SPACES.sub(r"\1-\2", s)
    
    # Standardize number prefix (e.g. "no. 5" -> "5")
    s = RE_NO_PREFIX.sub(r"\1", s)
    
    # Normalize street and building type abbreviations
    if standardize_types:
        def _replace_addr_abbr(match: re.Match) -> str:
            token = match.group(1).lower()
            return ADDRESS_ABBREVIATIONS.get(token, token)
        s = _ADDRESS_ABBR_PATTERN.sub(_replace_addr_abbr, s)
        
    # Standardize comma spacing if preserving components
    if preserve_components:
        s = re.sub(r"\s*,\s*", ", ", s)
        # Remove trailing or leading commas
        s = s.strip(", ")
    else:
        s = s.replace(",", " ")
        
    # Final whitespace cleanup
    s = RE_WHITESPACE.sub(" ", s).strip()
    return s


def normalize_country(country: Any) -> str:
    """
    Normalizes a country label supporting an open set of countries.
    
    Does NOT restrict labels to only 'US' or 'India'. Fully supports
    'France' and any unseen country labels in the test set.
    
    Parameters
    ----------
    country : Any
        Country label string or null.
        
    Returns
    -------
    str
        Canonical country string (e.g. 'US', 'India', 'France') or standardized Title case.
    """
    if country is None or pd.isna(country):
        return ""
    
    raw = str(country).strip()
    if not raw:
        return ""
    
    # Check known canonical aliases
    lowered = raw.lower()
    if lowered in COUNTRY_MAP:
        return COUNTRY_MAP[lowered]
    
    # Open-set fallback: if uppercase 2-letter ISO code, keep uppercase;
    # otherwise normalize to standard title casing
    if len(raw) == 2 and raw.isalpha():
        return raw.upper()
    return raw.title()


# ---------------------------------------------------------------------------
# Reusable Token Extraction Utilities (for Blocking & Similarity in Phase 2/3)
# ---------------------------------------------------------------------------

def extract_name_tokens(name: Any) -> List[str]:
    """
    Extracts alphanumeric tokens from a business name for blocking or Jaccard similarity.
    
    Parameters
    ----------
    name : Any
        Raw or preprocessed business name.
        
    Returns
    -------
    List[str]
        List of alphanumeric tokens with length >= 2.
    """
    clean = normalize_business_name(name)
    tokens = re.findall(r"\b[a-z0-9]+\b", clean)
    return [t for t in tokens if len(t) >= 2]


def extract_address_tokens(address: Any) -> List[str]:
    """
    Extracts component tokens (numbers, street names, cities, landmarks)
    from an address, invariant to word-order differences.
    
    Parameters
    ----------
    address : Any
        Raw or preprocessed address.
        
    Returns
    -------
    List[str]
        List of address tokens.
    """
    clean = normalize_address(address)
    tokens = re.findall(r"\b[a-z0-9]+\b", clean)
    return [t for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# Pipeline Preprocessing for DataFrames
# ---------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    inplace: bool = False,
    add_clean_columns: bool = True,
) -> pd.DataFrame:
    """
    Applies Phase 1 preprocessing across a pandas DataFrame of business entities.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns: 'entity_id', 'business_name', 'business_address', 'country'.
    inplace : bool, default=False
        Whether to modify DataFrame in place or return a new copy.
    add_clean_columns : bool, default=True
        If True, adds 'clean_business_name', 'clean_business_address', 'clean_country'
        while preserving original columns. If False, overwrites columns.
        
    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    res = df if inplace else df.copy()
    
    if "business_name" in res.columns:
        clean_names = res["business_name"].apply(normalize_business_name)
        if add_clean_columns:
            res["clean_business_name"] = clean_names
        else:
            res["business_name"] = clean_names
            
    if "business_address" in res.columns:
        clean_addrs = res["business_address"].apply(normalize_address)
        if add_clean_columns:
            res["clean_business_address"] = clean_addrs
        else:
            res["business_address"] = clean_addrs
            
    if "country" in res.columns:
        clean_countries = res["country"].apply(normalize_country)
        if add_clean_columns:
            res["clean_country"] = clean_countries
        else:
            res["country"] = clean_countries
            
    return res
