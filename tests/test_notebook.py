"""Tests for the notebook XML utility module."""

from auto_scientist.notebook import (
    NOTEBOOK_FILENAME,
    append_entry,
    format_entry,
    read_notebook,
)


class TestNotebookFilename:
    def test_filename_is_xml(self):
        assert NOTEBOOK_FILENAME == "lab_notebook.xml"


class TestFormatEntry:
    def test_basic_entry(self):
        xml = format_entry("Some narrative text", version="v00", source="scientist")
        assert '<entry version="v00" source="scientist">' in xml
        assert "<title>Some narrative text</title>" in xml
        assert "</entry>" in xml

    def test_multiline_extracts_title_from_first_line(self):
        content = "Polynomial Fitting\n\nFirst iteration explored data..."
        xml = format_entry(content, version="v01", source="scientist")
        assert "<title>Polynomial Fitting</title>" in xml
        assert "First iteration explored data..." in xml

    def test_escapes_xml_special_chars_in_content(self):
        content = "Title\n\nRMSE < 0.5 & R² > 0.95"
        xml = format_entry(content, version="v02", source="scientist")
        assert "&lt;" in xml
        assert "&amp;" in xml
        assert "&gt;" in xml

    def test_escapes_xml_special_chars_in_title(self):
        content = "RMSE < 0.5 & better"
        xml = format_entry(content, version="v02", source="scientist")
        assert "<title>RMSE &lt; 0.5 &amp; better</title>" in xml

    def test_single_line_has_empty_content(self):
        xml = format_entry("Just a title", version="v00", source="scientist")
        assert "<title>Just a title</title>" in xml
        assert "<content/>" in xml

    def test_revision_source(self):
        xml = format_entry("Post-debate revision\n\nAdjusted...", version="v01", source="revision")
        assert 'source="revision"' in xml

    def test_strips_leading_markdown_header(self):
        """If the scientist accidentally includes ## vXX prefix, strip it."""
        content = "## v03 - Interaction features\n\nv02 was incremental..."
        xml = format_entry(content, version="v03", source="scientist")
        assert "<title>Interaction features</title>" in xml
        assert "## v03" not in xml


class TestAppendEntry:
    def test_creates_new_file_if_missing(self, tmp_path):
        notebook_path = tmp_path / NOTEBOOK_FILENAME
        append_entry(notebook_path, "First entry\n\nNarrative", "v00", "scientist")

        content = notebook_path.read_text()
        assert content.startswith("<?xml")
        assert "<lab_notebook>" in content
        assert "</lab_notebook>" in content
        assert '<entry version="v00" source="scientist">' in content
        assert "<title>First entry</title>" in content

    def test_appends_to_existing_file(self, tmp_path):
        notebook_path = tmp_path / NOTEBOOK_FILENAME
        append_entry(notebook_path, "First\n\nContent", "v00", "scientist")
        append_entry(notebook_path, "Second\n\nMore content", "v01", "scientist")

        content = notebook_path.read_text()
        assert content.count("<entry") == 2
        assert content.count("</entry>") == 2
        assert "<title>First</title>" in content
        assert "<title>Second</title>" in content
        # Only one root element
        assert content.count("<lab_notebook>") == 1
        assert content.count("</lab_notebook>") == 1

    def test_preserves_existing_entries(self, tmp_path):
        notebook_path = tmp_path / NOTEBOOK_FILENAME
        append_entry(notebook_path, "Entry A\n\nAlpha", "ingestion", "ingestor")
        append_entry(notebook_path, "Entry B\n\nBeta", "v00", "scientist")
        append_entry(notebook_path, "Entry C\n\nGamma", "v00", "revision")

        content = notebook_path.read_text()
        assert content.count("<entry") == 3
        assert 'source="ingestor"' in content
        assert 'source="scientist"' in content
        assert 'source="revision"' in content

    def test_handles_xml_special_chars(self, tmp_path):
        notebook_path = tmp_path / NOTEBOOK_FILENAME
        append_entry(notebook_path, "Title\n\nRMSE < 0.5 & x > 3", "v00", "scientist")

        content = notebook_path.read_text()
        assert "&lt;" in content
        assert "&amp;" in content


class TestReadNotebook:
    def test_returns_content_when_exists(self, tmp_path):
        notebook_path = tmp_path / NOTEBOOK_FILENAME
        append_entry(notebook_path, "My entry\n\nSome text", "v00", "scientist")

        content = read_notebook(notebook_path)
        assert "<lab_notebook>" in content
        assert "<entry" in content

    def test_returns_empty_when_missing(self, tmp_path):
        notebook_path = tmp_path / NOTEBOOK_FILENAME
        assert read_notebook(notebook_path) == ""

    def test_returns_full_xml(self, tmp_path):
        notebook_path = tmp_path / NOTEBOOK_FILENAME
        append_entry(notebook_path, "A\n\nFirst", "v00", "scientist")
        append_entry(notebook_path, "B\n\nSecond", "v01", "scientist")

        content = read_notebook(notebook_path)
        assert "<title>A</title>" in content
        assert "<title>B</title>" in content
