"""Lab notebook XML utilities.

The lab notebook is an XML file that records the investigation history.
Each entry has a version, source agent, title, and narrative content.
The orchestrator owns the file structure; agents produce plain text content.
"""

import logging
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)

NOTEBOOK_FILENAME = "lab_notebook.xml"

_HEADER = '<?xml version="1.0" encoding="utf-8"?>\n'


def _strip_markdown_header(text: str) -> str:
    """Remove leading ## vXX - ... markdown header if present."""
    return re.sub(r"^##\s+v\d+\s*-\s*", "", text.strip())


def format_entry(content: str, version: str, source: str) -> str:
    """Wrap narrative content in an XML <entry> element.

    The first line of content becomes the title.
    Remaining lines become the <content> body.
    XML-special characters are escaped.
    """
    content = _strip_markdown_header(content)
    lines = content.split("\n", 1)
    title = escape(lines[0].strip())
    body = escape(lines[1].strip()) if len(lines) > 1 and lines[1].strip() else ""

    parts = [f'<entry version="{escape(version)}" source="{escape(source)}">']
    parts.append(f"  <title>{title}</title>")
    if body:
        parts.append(f"  <content>\n{body}\n  </content>")
    else:
        parts.append("  <content/>")
    parts.append("</entry>")
    return "\n".join(parts)


def append_entry(
    notebook_path: Path,
    content: str,
    version: str,
    source: str,
) -> None:
    """Append an entry to the notebook XML file, creating it if needed."""
    entry_xml = format_entry(content, version, source)

    if not notebook_path.exists():
        notebook_path.write_text(f"{_HEADER}<lab_notebook>\n{entry_xml}\n</lab_notebook>\n")
        return

    text = notebook_path.read_text()
    # Insert before the closing </lab_notebook> tag
    closing = "</lab_notebook>"
    idx = text.rfind(closing)
    if idx == -1:
        # Malformed file, append with wrapper
        notebook_path.write_text(f"{_HEADER}<lab_notebook>\n{entry_xml}\n</lab_notebook>\n")
        return

    new_text = text[:idx] + entry_xml + "\n" + text[idx:]
    notebook_path.write_text(new_text)


def read_notebook(notebook_path: Path) -> str:
    """Read the notebook file for prompt injection. Returns '' if missing."""
    if notebook_path.exists():
        return notebook_path.read_text()
    return ""


def parse_notebook_entries(notebook_path: Path) -> list[dict[str, str]]:
    """Parse lab_notebook.xml into a list of entry dicts.

    Each returned dict has keys: ``version``, ``source``, ``title``, ``content``.
    Returns ``[]`` if the file is missing, empty, or malformed. Preserves the
    order in which entries appear in the file.

    Missing-file is silent (the first iteration legitimately has no notebook
    yet). Read errors and parse errors are logged at WARNING/ERROR so a
    corrupted notebook is visible instead of silently downgrading every
    subsequent agent prompt to "(no notebook entries yet)".
    """
    if not notebook_path.exists():
        return []
    try:
        text = notebook_path.read_text().strip()
    except OSError as exc:
        logger.warning(
            f"Failed to read lab notebook at {notebook_path}: {exc}. "
            "Falling back to empty entry list - downstream agents will plan "
            "as if no prior notebook history existed."
        )
        return []
    if not text:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        logger.error(
            f"Lab notebook {notebook_path} is malformed XML: {exc}. "
            "Falling back to empty entry list - this almost certainly "
            "indicates a corrupted file from a partial write or crash, "
            "and any prior reasoning history is now invisible to agents. "
            "Investigate before continuing the run."
        )
        return []

    entries: list[dict[str, str]] = []
    for entry_el in root.iter("entry"):
        version = entry_el.get("version", "")
        source = entry_el.get("source", "")
        title_el = entry_el.find("title")
        content_el = entry_el.find("content")
        title = (title_el.text or "").strip() if title_el is not None else ""
        content = (content_el.text or "").strip() if content_el is not None else ""
        entries.append(
            {
                "version": version,
                "source": source,
                "title": title,
                "content": content,
            }
        )
    return entries
