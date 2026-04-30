from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from pyside_app.notebook_examples import get_desktop_notebook_examples
from pyside_app.storage import NotebookDocument, NotebookStorage


def main() -> None:
    print("[verify][pyside-storage] start", flush=True)
    storage = NotebookStorage()
    examples = get_desktop_notebook_examples()
    example_ids = [example.id for example in examples]
    print(f"[verify][pyside-storage] examples_count={len(examples)}", flush=True)
    print(f"[verify][pyside-storage] first_example={examples[0].id}", flush=True)
    print(f"[verify][pyside-storage] example_ids={example_ids}", flush=True)

    if "diffusion_explicit_1d" not in example_ids:
        raise SystemExit("[verify][pyside-storage] FAIL: diffusion example missing")

    document = NotebookDocument(
        cells=[
            {
                "id": "cell-verify-1",
                "type": "code",
                "column": "right",
                "panel_width": 610,
                "source": examples[0].primary_source,
                "outputs": [{"kind": "value", "data": {"text": "ok"}}],
            }
        ],
        metadata={"title": "Verify Desktop Notebook"},
        layout={"left_column_width_pct": 62},
    )
    print(f"[verify][pyside-storage] document_layout={document.layout}", flush=True)
    print(f"[verify][pyside-storage] document_cell={document.cells[0]}", flush=True)

    with TemporaryDirectory(prefix="pyside_storage_verify_") as tmp_dir:
        path = Path(tmp_dir) / "verify_notebook.json"
        print(f"[verify][pyside-storage] save_path={path}", flush=True)
        storage.save(path, document)
        loaded = storage.load(path)
        print(f"[verify][pyside-storage] loaded_metadata={loaded.metadata}", flush=True)
        print(f"[verify][pyside-storage] loaded_layout={loaded.layout}", flush=True)
        print(f"[verify][pyside-storage] loaded_cell={loaded.cells[0]}", flush=True)

        if loaded.cells[0]["column"] != "right":
            raise SystemExit("[verify][pyside-storage] FAIL: column was not preserved")
        if loaded.cells[0]["panel_width"] != 610:
            raise SystemExit("[verify][pyside-storage] FAIL: panel_width was not preserved")
        if loaded.layout.get("left_column_width_pct") != 62:
            raise SystemExit("[verify][pyside-storage] FAIL: layout was not preserved")
        if loaded.metadata.get("version") != storage.version:
            raise SystemExit("[verify][pyside-storage] FAIL: version was not normalized")

    print("[verify][pyside-storage] PASS", flush=True)


if __name__ == "__main__":
    main()
