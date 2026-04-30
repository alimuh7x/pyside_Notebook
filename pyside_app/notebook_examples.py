from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyside_app.storage import NotebookDocument, NotebookStorage


def get_notebook_examples_dir() -> Path:
    directory = Path(__file__).resolve().parent.parent / "NotebookExamples"
    print(f"[debug][notebook-examples] examples_dir path={str(directory)!r}", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def suggest_example_filename(title: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in title.strip().lower())
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    filename = (collapsed or "desktop_notebook") + ".json"
    print(f"[debug][notebook-examples] suggest_filename title={title!r} filename={filename!r}", flush=True)
    return filename


@dataclass
class NotebookExample:
    id: str
    title: str
    category: str
    description: str
    tags: list[str] = field(default_factory=list)
    cells: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    layout: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def primary_source(self) -> str:
        for cell in self.cells:
            if cell.get("type") == "code" and str(cell.get("source") or "").strip():
                return str(cell.get("source") or "")
        for cell in self.cells:
            source = str(cell.get("source") or "")
            if source.strip():
                return source
        return ""


def _fallback_description(document: NotebookDocument) -> str:
    for cell in document.cells:
        if cell.get("type") == "markdown":
            text = " ".join(line.strip().lstrip("#").strip() for line in str(cell.get("source") or "").splitlines())
            if text:
                return text[:180]
        source = str(cell.get("source") or "").strip()
        if source:
            return source.splitlines()[0][:180]
    return "Notebook example"


def get_desktop_notebook_examples() -> list[NotebookExample]:
    print("[debug][notebook-examples] get_examples:start", flush=True)
    storage = NotebookStorage()
    examples_dir = get_notebook_examples_dir()
    examples: list[NotebookExample] = []
    for path in sorted(examples_dir.glob("*.json")):
        print(f"[debug][notebook-examples] get_examples:file path={str(path)!r}", flush=True)
        try:
            document = storage.load(path)
        except Exception as exc:  # pragma: no cover - defensive debug path
            print(f"[debug][notebook-examples] get_examples:error path={str(path)!r} error={exc!r}", flush=True)
            continue
        metadata = dict(document.metadata or {})
        title = str(metadata.get("title") or path.stem.replace("_", " ").replace("-", " ").title())
        category = str(metadata.get("category") or "general")
        description = str(metadata.get("description") or _fallback_description(document))
        tags = [str(tag) for tag in (metadata.get("tags") or [])]
        examples.append(
            NotebookExample(
                id=path.stem,
                title=title,
                category=category,
                description=description,
                tags=tags,
                cells=deepcopy(document.cells),
                metadata=metadata,
                layout=dict(document.layout or {}),
                path=path,
            )
        )
    print(f"[debug][notebook-examples] get_examples:count count={len(examples)}", flush=True)
    print(f"[debug][notebook-examples] get_examples:ids ids={[example.id for example in examples]}", flush=True)
    return deepcopy(examples)
