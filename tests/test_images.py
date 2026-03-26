"""Tests for image encoding utilities."""

import base64
from pathlib import Path

from auto_scientist.images import ImageData, encode_images_from_paths

# Minimal valid 1x1 red PNG (67 bytes)
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_encode_single_png(tmp_path: Path) -> None:
    """Encoding a PNG returns correct base64 and media type."""
    png = tmp_path / "plot.png"
    png.write_bytes(TINY_PNG)

    result = encode_images_from_paths([png])

    assert len(result) == 1
    img = result[0]
    assert isinstance(img, ImageData)
    assert img.media_type == "image/png"
    assert base64.b64decode(img.data) == TINY_PNG


def test_encode_missing_file_skipped(tmp_path: Path) -> None:
    """Non-existent paths are silently skipped."""
    missing = tmp_path / "does_not_exist.png"

    result = encode_images_from_paths([missing])

    assert result == []


def test_encode_multiple_files(tmp_path: Path) -> None:
    """Multiple files are all encoded."""
    for name in ("a.png", "b.png", "c.png"):
        (tmp_path / name).write_bytes(TINY_PNG)

    result = encode_images_from_paths([tmp_path / n for n in ("a.png", "b.png", "c.png")])

    assert len(result) == 3
    assert all(img.media_type == "image/png" for img in result)


def test_media_type_detection_jpg(tmp_path: Path) -> None:
    """JPEG files get image/jpeg media type."""
    jpg = tmp_path / "plot.jpg"
    jpg.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header

    result = encode_images_from_paths([jpg])

    assert len(result) == 1
    assert result[0].media_type == "image/jpeg"


def test_mixed_existing_and_missing(tmp_path: Path) -> None:
    """Only existing files are encoded; missing ones are skipped."""
    existing = tmp_path / "real.png"
    existing.write_bytes(TINY_PNG)
    missing = tmp_path / "gone.png"

    result = encode_images_from_paths([existing, missing])

    assert len(result) == 1
    assert result[0].media_type == "image/png"
