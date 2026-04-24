"""ISO alpha-2 country code → set of scripts used on street signage.

Only non-Latin-primary and multi-script countries are listed explicitly; all
others fall back to DEFAULT_SCRIPTS = {"Latin"} via country_scripts().

Script bucket names must match those produced by src.ocr.classify_char().
"""

from __future__ import annotations

DEFAULT_SCRIPTS: frozenset[str] = frozenset({"Latin"})


COUNTRY_SCRIPTS: dict[str, frozenset[str]] = {
    # Cyrillic-primary
    "RU": frozenset({"Cyrillic", "Latin"}),
    "UA": frozenset({"Cyrillic", "Latin"}),
    "BY": frozenset({"Cyrillic", "Latin"}),
    "BG": frozenset({"Cyrillic", "Latin"}),
    "RS": frozenset({"Cyrillic", "Latin"}),
    "MK": frozenset({"Cyrillic", "Latin"}),
    "ME": frozenset({"Cyrillic", "Latin"}),
    "KZ": frozenset({"Cyrillic", "Latin"}),
    "KG": frozenset({"Cyrillic", "Latin"}),
    "MN": frozenset({"Cyrillic", "Latin"}),
    "TJ": frozenset({"Cyrillic", "Latin"}),
    # Arabic-primary
    "SA": frozenset({"Arabic", "Latin"}),
    "AE": frozenset({"Arabic", "Latin"}),
    "EG": frozenset({"Arabic", "Latin"}),
    "JO": frozenset({"Arabic", "Latin"}),
    "LB": frozenset({"Arabic", "Latin"}),
    "SY": frozenset({"Arabic", "Latin"}),
    "IQ": frozenset({"Arabic", "Latin"}),
    "MA": frozenset({"Arabic", "Latin"}),
    "TN": frozenset({"Arabic", "Latin"}),
    "DZ": frozenset({"Arabic", "Latin"}),
    "LY": frozenset({"Arabic", "Latin"}),
    "QA": frozenset({"Arabic", "Latin"}),
    "KW": frozenset({"Arabic", "Latin"}),
    "BH": frozenset({"Arabic", "Latin"}),
    "OM": frozenset({"Arabic", "Latin"}),
    "YE": frozenset({"Arabic", "Latin"}),
    "PK": frozenset({"Arabic", "Latin"}),
    "IR": frozenset({"Arabic", "Latin"}),
    "AF": frozenset({"Arabic", "Latin"}),
    # CJK / Asian scripts
    "CN": frozenset({"CJK", "Latin"}),
    "HK": frozenset({"CJK", "Latin"}),
    "TW": frozenset({"CJK", "Latin"}),
    "JP": frozenset({"CJK", "Hiragana", "Katakana", "Latin"}),
    "KR": frozenset({"Hangul", "CJK", "Latin"}),
    "KP": frozenset({"Hangul", "CJK", "Latin"}),
    # Devanagari / South Asian
    "IN": frozenset({"Devanagari", "Latin"}),
    "NP": frozenset({"Devanagari", "Latin"}),
    "BD": frozenset({"Latin"}),  # Bengali script not in our OCR language set
    # SE Asia
    "TH": frozenset({"Thai", "Latin"}),
    "LA": frozenset({"Latin"}),
    "MM": frozenset({"Latin"}),
    "KH": frozenset({"Latin"}),
    # Middle East / Europe
    "IL": frozenset({"Hebrew", "Latin"}),
    "GR": frozenset({"Latin"}),  # Greek script not in our OCR language set
    "AM": frozenset({"Latin"}),  # Armenian not in our OCR language set
    "GE": frozenset({"Latin"}),  # Georgian not in our OCR language set
}


def country_scripts(cc: str | None) -> frozenset[str]:
    """Return the set of scripts used on street signage for ISO alpha-2 code cc."""
    if not cc:
        return DEFAULT_SCRIPTS
    return COUNTRY_SCRIPTS.get(cc.upper(), DEFAULT_SCRIPTS)


def script_bonus(detected: set[str], accepted: frozenset[str]) -> float:
    """Asymmetric re-rank bonus for one (query image, candidate country) pair.

    Rules:
      - Only Latin detected -> 0 everywhere (no signal: Mapillary images
        worldwide tend to have some Latin, so "Latin-only" doesn't discriminate
        between Latin-primary and non-Latin-primary countries — boosting
        Latin-primary candidates empirically flips correct non-Latin
        predictions to wrong Latin ones).
      - At least one non-Latin script detected:
        * Any detected non-Latin script appears in country's accepted set -> +1
        * None do -> -1  (e.g. Cyrillic detected and candidate is USA)

    `detected` is the set of script names returned by ScriptDetector.detect()
    (keys of its `scripts` dict). `accepted` is what country_scripts(cc) returns.
    """
    non_latin_detected = detected - {"Latin"}
    if not non_latin_detected:
        return 0.0
    if non_latin_detected & accepted:
        return 1.0
    return -1.0
