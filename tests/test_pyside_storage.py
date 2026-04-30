from __future__ import annotations

import json

import pytest

from pyside_app.notebook_examples import get_desktop_notebook_examples
from pyside_app.storage import NotebookDocument, NotebookStorage


def test_storage_round_trips_notebook_document(tmp_path):
    storage = NotebookStorage()
    path = tmp_path / "sample_notebook.json"
    document = NotebookDocument(
        cells=[
            {"id": "cell-1", "type": "code", "source": "x = 1", "outputs": [{"kind": "value", "data": {"text": "1"}}]},
            {"id": "cell-2", "type": "markdown", "source": "# Heading", "outputs": []},
        ],
        metadata={"title": "Desktop Notebook"},
    )

    storage.save(path, document)
    loaded = storage.load(path)

    assert loaded.metadata["title"] == "Desktop Notebook"
    assert loaded.cells[0]["source"] == "x = 1"
    assert loaded.cells[0]["outputs"][0]["data"]["text"] == "1"
    assert loaded.cells[1]["type"] == "markdown"
    assert loaded.metadata["version"] == 6


def test_storage_round_trips_column_panel_width_and_layout(tmp_path):
    storage = NotebookStorage()
    path = tmp_path / "layout_notebook.json"
    document = NotebookDocument(
        cells=[
            {
                "id": "cell-1",
                "type": "code",
                "column": "right",
                "panel_width": 620,
                "source": "temperature = 873",
                "outputs": [],
            },
            {
                "id": "cell-2",
                "type": "plot",
                "column": "left",
                "panel_width": 540,
                "source": "",
                "plot_spec": {"x": [0, 1], "y": [1, 2]},
                "outputs": [{"kind": "html", "data": {"html": "<b>plot</b>", "ignored": "drop-me"}}],
            },
        ],
        metadata={"title": "Layout Notebook", "version": 999},
        layout={"left_column_width_pct": 63},
    )

    storage.save(path, document)
    payload = json.loads(path.read_text(encoding="utf-8"))
    loaded = storage.load(path)

    assert payload["metadata"]["version"] == storage.version
    assert payload["cells"][0]["column"] == "right"
    assert payload["cells"][0]["panel_width"] == 620
    assert payload["layout"]["left_column_width_pct"] == 63
    assert loaded.cells[0]["column"] == "right"
    assert loaded.cells[0]["panel_width"] == 620
    assert loaded.cells[1]["plot_spec"] == {"x": [0, 1], "y": [1, 2]}
    assert loaded.cells[1]["outputs"][0]["data"] == {"html": "<b>plot</b>"}
    assert loaded.layout["left_column_width_pct"] == 63
    assert loaded.metadata["version"] == storage.version


def test_storage_rejects_missing_version_on_load(tmp_path):
    storage = NotebookStorage()
    path = tmp_path / "missing_version.json"
    path.write_text(json.dumps({"cells": [], "metadata": {"title": "Legacy"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing notebook JSON version"):
        storage.load(path)


def test_storage_rejects_unsupported_newer_version_on_load(tmp_path):
    storage = NotebookStorage()
    path = tmp_path / "newer_version.json"
    path.write_text(
        json.dumps(
            {
                "version": storage.version + 1,
                "cells": [],
                "metadata": {"version": storage.version + 1, "title": "Future Notebook"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported newer notebook JSON version"):
        storage.load(path)


def test_storage_rejects_version_mismatch_between_payload_and_metadata(tmp_path):
    storage = NotebookStorage()
    path = tmp_path / "mismatch.json"
    path.write_text(
        json.dumps(
            {
                "version": storage.version,
                "cells": [],
                "metadata": {"version": storage.version - 1, "title": "Mismatch Notebook"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="version mismatch"):
        storage.load(path)


def test_storage_builds_default_document():
    document = NotebookStorage().default_document()

    assert len(document.cells) == 1
    assert document.cells[0]["type"] == "code"
    assert document.cells[0]["column"] == "left"
    assert document.cells[0]["panel_width"] is None
    assert document.cells[0]["source"] == ""
    assert document.metadata["version"] == 6
    assert document.layout["left_column_width_pct"] == 50


def test_desktop_notebook_examples_expose_scientific_starter_snippets():
    examples = get_desktop_notebook_examples()
    example_ids = {example.id for example in examples}

    assert len(examples) >= 5
    assert all(example.id for example in examples)
    assert all(example.title for example in examples)
    assert all(example.description for example in examples)
    assert all(example.cells for example in examples)
    assert all(any(cell["type"] == "markdown" for cell in example.cells) for example in examples)
    assert all(any(cell["type"] == "code" for cell in example.cells) for example in examples)
    assert any("numpy" in example.primary_source for example in examples)
    assert any(example.category == "materials" for example in examples)
    assert "diffusion_explicit_1d" in example_ids


def test_desktop_notebook_examples_return_independent_copies():
    examples = get_desktop_notebook_examples()
    examples[0].title = "Changed in test"

    fresh_examples = get_desktop_notebook_examples()

    assert fresh_examples[0].title != "Changed in test"
