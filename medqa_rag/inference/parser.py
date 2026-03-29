from __future__ import annotations

import re


FINAL_PATTERNS = [
    re.compile(r"(?i)\bfinal answer\b[^A-D]*\b([A-D])\b"),
    re.compile(r"(?i)\banswer\b[^A-D]*\b([A-D])\b"),
    re.compile(r"(?mi)^\s*([A-D])\s*$"),
    re.compile(r"(?i)\boption\s+([A-D])\b"),
]


def parse_answer_letter(text: str | None) -> str | None:
    if not text:
        return None

    for pattern in FINAL_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()

    standalone = re.findall(r"\b([A-D])\b", text.upper())
    if standalone:
        return standalone[-1]
    return None
