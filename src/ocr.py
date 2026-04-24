from __future__ import annotations

from threading import Lock
from typing import Iterable

import numpy as np
from PIL import Image


# Languages we want EasyOCR to recognize. 'en' is always bundled with each
# non-Latin reader so Latin text on mixed-script signs stays readable.
DEFAULT_LANGUAGES: tuple[str, ...] = ("en", "ru", "ar", "hi", "th", "ja", "ko", "ch_sim")


# EasyOCR's recognition-model families. A single easyocr.Reader can only hold
# languages that share one family (plus 'en'), so we spin up one Reader per
# family and merge their results in detect().
_FAMILY_TO_LANGS: dict[str, frozenset[str]] = {
    "latin": frozenset({"en"}),
    "cyrillic": frozenset({"ru", "rs_cyrillic", "be", "bg", "uk", "mn", "abq", "ady", "kbd", "lbe", "lez", "dar", "inh", "che", "ava", "tab", "tjk"}),
    "arabic": frozenset({"ar", "fa", "ug", "ur"}),
    "devanagari": frozenset({"hi", "mr", "ne", "bho", "mai", "gom", "sck", "new", "bh", "pi", "ang"}),
    "bengali": frozenset({"bn", "as"}),
    "chinese_sim": frozenset({"ch_sim"}),
    "chinese_tra": frozenset({"ch_tra"}),
    "japanese": frozenset({"ja"}),
    "korean": frozenset({"ko"}),
    "thai": frozenset({"th"}),
    "tamil": frozenset({"ta"}),
    "telugu": frozenset({"te"}),
    "kannada": frozenset({"kn"}),
}


# Unicode block ranges → script bucket name. Must match the keys used in
# src.country_scripts (Latin, Cyrillic, Arabic, Devanagari, Thai, Hebrew,
# Hangul, Hiragana, Katakana, CJK).
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0041, 0x005A, "Latin"),
    (0x0061, 0x007A, "Latin"),
    (0x00C0, 0x024F, "Latin"),
    (0x1E00, 0x1EFF, "Latin"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0500, 0x052F, "Cyrillic"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0750, 0x077F, "Arabic"),
    (0x08A0, 0x08FF, "Arabic"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0590, 0x05FF, "Hebrew"),
    (0xAC00, 0xD7AF, "Hangul"),
    (0x3130, 0x318F, "Hangul"),
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x4E00, 0x9FFF, "CJK"),
    (0x3400, 0x4DBF, "CJK"),
    (0xF900, 0xFAFF, "CJK"),
)


def classify_char(ch: str) -> str | None:
    """Return the script bucket for a single character, or None to ignore (digits, punct, space, etc.)."""
    if not ch or not ch.strip():
        return None
    if ch.isdigit():
        return None
    cp = ord(ch)
    for lo, hi, name in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    return None


def _script_share(text_items: Iterable[tuple[str, float]]) -> dict[str, float]:
    """Given (text, conf) pairs, return confidence-weighted share per script bucket."""
    acc: dict[str, float] = {}
    for text, conf in text_items:
        for ch in text:
            s = classify_char(ch)
            if s is None:
                continue
            acc[s] = acc.get(s, 0.0) + float(conf)
    total = sum(acc.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in acc.items()}


def _group_languages(languages: Iterable[str]) -> dict[str, list[str]]:
    """Group requested langs into EasyOCR model families. Each value list is passed to one Reader."""
    requested = set(languages)
    has_en = "en" in requested
    groups: dict[str, list[str]] = {}

    for family, family_langs in _FAMILY_TO_LANGS.items():
        if family == "latin":
            continue
        in_family = sorted(requested & family_langs)
        if in_family:
            langs = in_family + (["en"] if has_en else [])
            groups[family] = langs

    if has_en and not groups:
        groups["latin"] = ["en"]
    elif has_en and groups:
        # If we built at least one non-Latin reader, 'en' is already bundled in each;
        # spinning up a separate Latin-only reader would just duplicate Latin detections.
        pass

    unknown = requested - {"en"}
    for family_langs in _FAMILY_TO_LANGS.values():
        unknown -= family_langs
    if unknown:
        raise ValueError(f"OCR languages not mapped to any family: {sorted(unknown)}")

    return groups


class ScriptDetector:
    _instance: "ScriptDetector | None" = None
    _lock = Lock()

    def __init__(self, languages: Iterable[str] = DEFAULT_LANGUAGES, gpu: bool | None = None):
        import easyocr  # lazy: heavy import

        if gpu is None:
            try:
                import torch

                gpu = bool(torch.cuda.is_available())
            except Exception:
                gpu = False
        self.languages = tuple(languages)
        groups = _group_languages(self.languages)
        self.readers: list[tuple[str, "easyocr.Reader"]] = []
        for family, langs in groups.items():
            print(f"[ocr] loading {family} reader: {langs}")
            reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
            self.readers.append((family, reader))
        if not self.readers:
            raise ValueError(f"No usable OCR readers built from languages={self.languages}")

    @classmethod
    def get(cls, languages: Iterable[str] = DEFAULT_LANGUAGES, gpu: bool | None = None) -> "ScriptDetector":
        langs = tuple(languages)
        with cls._lock:
            if cls._instance is None or cls._instance.languages != langs:
                cls._instance = cls(languages=langs, gpu=gpu)
            return cls._instance

    @staticmethod
    def _iou(a: list, b: list) -> float:
        """IoU of two 4-point EasyOCR bboxes, axis-aligned approximation via their enclosing rectangles."""
        ax = [p[0] for p in a]; ay = [p[1] for p in a]
        bx = [p[0] for p in b]; by = [p[1] for p in b]
        ax0, ax1, ay0, ay1 = min(ax), max(ax), min(ay), max(ay)
        bx0, bx1, by0, by1 = min(bx), max(bx), min(by), max(by)
        ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
        iy0, iy1 = max(ay0, by0), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _dedupe(cls, items: list[dict], iou_thresh: float = 0.5) -> list[dict]:
        """Merge overlapping detections from different readers; keep the highest-confidence one per cluster."""
        items_sorted = sorted(items, key=lambda d: -float(d["conf"]))
        kept: list[dict] = []
        for item in items_sorted:
            collision = False
            for k in kept:
                if cls._iou(item["bbox"], k["bbox"]) >= iou_thresh:
                    collision = True
                    break
            if not collision:
                kept.append(item)
        return kept

    def detect(self, image: Image.Image, min_conf: float = 0.4, min_chars: int = 3) -> dict:
        arr = np.asarray(image.convert("RGB"))
        all_items: list[dict] = []
        for _family, reader in self.readers:
            try:
                raw = reader.readtext(arr, detail=1)
            except Exception:
                continue
            for item in raw:
                if len(item) < 3:
                    continue
                bbox, text, conf = item[0], item[1], float(item[2])
                all_items.append({"bbox": bbox, "text": str(text), "conf": conf})

        all_items = self._dedupe(all_items, iou_thresh=0.5)
        kept = [(i["text"], i["conf"]) for i in all_items if i["conf"] >= min_conf]
        scripts = _script_share(kept)
        dominant: str | None = None
        char_count = sum(
            sum(1 for ch in t if classify_char(ch) is not None) for t, _ in kept
        )
        if scripts and char_count >= min_chars:
            dominant = max(scripts.items(), key=lambda kv: kv[1])[0]

        return {"scripts": scripts, "text": all_items, "dominant": dominant}
