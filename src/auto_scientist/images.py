"""Image encoding utilities for multimodal LLM calls."""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class ImageData(NamedTuple):
    """Pre-encoded image for multimodal API calls."""

    data: str  # base64-encoded bytes
    media_type: str  # e.g. "image/png"


def encode_images_from_paths(paths: list[Path]) -> list[ImageData]:
    """Read image files and return base64-encoded ImageData list.

    Skips files that don't exist or can't be read, logging a warning.
    """
    images: list[ImageData] = []
    for path in paths:
        if not path.exists():
            logger.warning(f"Image file not found, skipping: {path}")
            continue
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        raw = path.read_bytes()
        images.append(ImageData(data=base64.b64encode(raw).decode(), media_type=mime))
    return images
