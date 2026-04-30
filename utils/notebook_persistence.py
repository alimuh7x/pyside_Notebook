"""Helpers for notebook JSON save/load payloads."""

from __future__ import annotations


NOTEBOOK_JSON_VERSION = 6


def serialize_notebook_cells(cells: list[dict] | None, layout: dict | None = None) -> dict:
    """Build the canonical JSON payload for notebook save/export."""
    normalized = []
    for cell in cells or []:
        row = {
            "id": cell.get("id") or "",
            "type": cell.get("type", "code"),
            "column": cell.get("column", "left"),
            "panel_width": cell.get("panel_width"),
            "source": cell.get("source", ""),
        }
        if cell.get("type") == "plot":
            row["plot_spec"] = dict(cell.get("plot_spec") or {})
        normalized.append(row)
    return {
        "version": NOTEBOOK_JSON_VERSION,
        "layout": {"left_column_width_pct": int((layout or {}).get("left_column_width_pct", 50))},
        "cells": normalized,
    }


def deserialize_notebook_cells(payload: dict | None) -> list[dict]:
    """Normalize saved JSON into rerunnable notebook cells."""
    cells = []
    for cell in (payload or {}).get("cells", []) or []:
        restored = {
            "id": cell.get("id") or "",
            "type": cell.get("type", "code"),
            "column": cell.get("column", "left"),
            "panel_width": cell.get("panel_width"),
            "source": cell.get("source", ""),
            "outputs": [],
            "dirty": False,
        }
        if restored["type"] == "plot":
            restored["plot_spec"] = dict(cell.get("plot_spec") or {})
        cells.append(restored)
    return cells
