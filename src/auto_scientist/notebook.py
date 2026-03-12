"""Lab notebook XML utilities.

The lab notebook is an XML file that records the investigation history.
Each entry has a version, source agent, title, and narrative content.
The orchestrator owns the file structure; agents produce plain text content.
"""

import re
from pathlib import Path
from xml.sax.saxutils import escape

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
    notebook_path: Path, content: str, version: str, source: str,
) -> None:
    """Append an entry to the notebook XML file, creating it if needed."""
    entry_xml = format_entry(content, version, source)

    if not notebook_path.exists():
        notebook_path.write_text(
            f"{_HEADER}<lab_notebook>\n{entry_xml}\n</lab_notebook>\n"
        )
        return

    text = notebook_path.read_text()
    # Insert before the closing </lab_notebook> tag
    closing = "</lab_notebook>"
    idx = text.rfind(closing)
    if idx == -1:
        # Malformed file, append with wrapper
        notebook_path.write_text(
            f"{_HEADER}<lab_notebook>\n{entry_xml}\n</lab_notebook>\n"
        )
        return

    new_text = text[:idx] + entry_xml + "\n" + text[idx:]
    notebook_path.write_text(new_text)


def read_notebook(notebook_path: Path) -> str:
    """Read the notebook file for prompt injection. Returns '' if missing."""
    if notebook_path.exists():
        return notebook_path.read_text()
    return ""
