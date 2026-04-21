"""Lightweight LaTeX math to Unicode converter for terminal display.

Uses pylatexenc for core LaTeX-to-text conversion (commands, delimiters,
fractions, sqrt, etc.) and adds Unicode super/subscript conversion on top
since pylatexenc leaves ^/_ as plain text.
"""

from __future__ import annotations

import re

from pylatexenc.latex2text import LatexNodes2Text

_L2T = LatexNodes2Text()

# ---------------------------------------------------------------------------
# Superscript / subscript Unicode maps
# ---------------------------------------------------------------------------

_SUPERSCRIPTS: dict[str, str] = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "n": "ⁿ",
    "i": "ⁱ",
    "T": "ᵀ",
}

_SUBSCRIPTS: dict[str, str] = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
}


def _convert_script(content: str, table: dict[str, str]) -> str:
    """Convert a string using a character table. Returns "" if any char is missing."""
    converted = []
    for ch in content:
        if ch in table:
            converted.append(table[ch])
        else:
            return ""
    return "".join(converted)


def _replace_superscripts(text: str) -> str:
    """Replace ^{...} and ^x with Unicode superscripts where possible."""

    def _sub_run(m: re.Match[str]) -> str:
        """Convert a run of superscript-eligible characters after ^."""
        content = m.group(1)
        result = _convert_script(content, _SUPERSCRIPTS)
        return result if result else f"^{content}"

    # Match ^ followed by a run of digits/signs (e.g., ^-1, ^20, ^2)
    text = re.sub(r"\^([0-9+\-][0-9]*)", _sub_run, text)
    # Single letter: ^n, ^T, ^i, etc.
    text = re.sub(r"\^([a-zA-Z])", _sub_run, text)
    return text


def _replace_subscripts(text: str) -> str:
    """Replace _x with Unicode subscripts where possible (digits only)."""

    def _sub_single(m: re.Match[str]) -> str:
        ch = m.group(1)
        return _SUBSCRIPTS.get(ch, f"_{ch}")

    text = re.sub(r"_([0-9])", _sub_single, text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def latex_to_unicode(text: str) -> str:
    """Convert LaTeX math notation in *text* to Unicode approximations.

    Uses pylatexenc for the core conversion (commands, delimiters, structural
    patterns), then applies Unicode super/subscript conversion for the ^/_
    notation that pylatexenc leaves as plain text.

    Plain text without any LaTeX passes through untouched (fast path).
    """
    if not text or ("\\" not in text and "$" not in text and "^" not in text):
        return text

    # pylatexenc handles: \(...\), \[...\], $...$, $$...$$, \sin, \sigma,
    # \frac{a}{b}, \sqrt{x}, \hat{x}, \text{...}, \mathrm{...}, etc.
    text = _L2T.latex_to_text(text)

    # pylatexenc leaves ^/_ as plain ASCII; convert to Unicode
    text = _replace_superscripts(text)
    text = _replace_subscripts(text)

    return text
