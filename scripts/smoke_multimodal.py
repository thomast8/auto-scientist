"""Smoke test: verify multimodal image input works for each provider.

Usage:
    uv run python scripts/smoke_multimodal.py [plot.png]

If no PNG is provided, creates a minimal synthetic plot via matplotlib.
Tests Anthropic, OpenAI, and Google (skips any with missing API keys).
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from auto_scientist.images import encode_images_from_paths


def _create_test_plot(path: Path) -> None:
    """Generate a minimal valid PNG (1x1 red pixel) with no external deps."""
    import struct
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        raw = chunk_type + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1 RGB
    idat = zlib.compress(b"\x00\xff\x00\x00")  # filter=none, R=255 G=0 B=0

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )
    print(f"Created minimal test PNG: {path}")


PROMPT = "Describe what you see in this plot in 1-2 sentences. Mention the axes, trend, and any noise."


async def test_anthropic(images):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  SKIP (no ANTHROPIC_API_KEY)")
        return
    from auto_scientist.models.anthropic_client import query_anthropic

    result = await query_anthropic("claude-sonnet-4-6", PROMPT, images=images)
    print(f"  OK: {result[:200]}")


async def test_openai(images):
    if not os.environ.get("OPENAI_API_KEY"):
        print("  SKIP (no OPENAI_API_KEY)")
        return
    from auto_scientist.models.openai_client import query_openai

    result = await query_openai("gpt-4.1", PROMPT, images=images)
    print(f"  OK: {result[:200]}")


async def test_google(images):
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("  SKIP (no GOOGLE_API_KEY/GEMINI_API_KEY)")
        return
    from auto_scientist.models.google_client import query_google

    result = await query_google("gemini-2.5-flash", PROMPT, images=images)
    print(f"  OK: {result[:200]}")


async def main():
    # Get or create a test image
    if len(sys.argv) > 1:
        png_path = Path(sys.argv[1])
        if not png_path.exists():
            print(f"File not found: {png_path}")
            sys.exit(1)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        png_path = Path(tmp.name)
        tmp.close()
        _create_test_plot(png_path)

    images = encode_images_from_paths([png_path])
    print(f"Encoded {len(images)} image(s), {len(images[0].data)} base64 chars\n")

    for name, fn in [("Anthropic", test_anthropic), ("OpenAI", test_openai), ("Google", test_google)]:
        print(f"[{name}]")
        try:
            await fn(images)
        except Exception as e:
            print(f"  FAIL: {e}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
