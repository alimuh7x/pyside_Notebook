from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.notebook_persistence import NOTEBOOK_JSON_VERSION, deserialize_notebook_cells, serialize_notebook_cells


@dataclass
class NotebookDocument:
    cells: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    layout: dict[str, Any] = field(default_factory=lambda: {"left_column_width_pct": 50})


class NotebookStorage:
    version = NOTEBOOK_JSON_VERSION

    def default_document(self) -> NotebookDocument:
        print("[debug][storage] default_document", flush=True)
        return NotebookDocument(
            cells=[{"id": "cell-1", "type": "code", "column": "left", "panel_width": None, "source": "", "outputs": []}],
            metadata={"version": self.version, "title": "Untitled Notebook"},
            layout={"left_column_width_pct": 50},
        )

    def _extract_version(self, payload: dict[str, Any]) -> int:
        print("[debug][storage] extract_version:start", flush=True)
        payload_version = payload.get("version")
        metadata = dict(payload.get("metadata") or {})
        metadata_version = metadata.get("version")
        print(f"[debug][storage] extract_version:payload_version value={payload_version!r}", flush=True)
        print(f"[debug][storage] extract_version:metadata_version value={metadata_version!r}", flush=True)
        if payload_version is None and metadata_version is None:
            print("[debug][storage] extract_version:error missing version", flush=True)
            raise ValueError("missing notebook JSON version in payload metadata")
        if payload_version is not None and metadata_version is not None and payload_version != metadata_version:
            print("[debug][storage] extract_version:error mismatch", flush=True)
            raise ValueError(
                f"notebook JSON version mismatch between payload ({payload_version}) and metadata ({metadata_version})"
            )
        resolved_version = payload_version if payload_version is not None else metadata_version
        print(f"[debug][storage] extract_version:resolved value={resolved_version!r}", flush=True)
        if not isinstance(resolved_version, int):
            print("[debug][storage] extract_version:error invalid type", flush=True)
            raise ValueError(f"invalid notebook JSON version {resolved_version!r}")
        if resolved_version > self.version:
            print("[debug][storage] extract_version:error newer version", flush=True)
            raise ValueError(
                f"unsupported newer notebook JSON version {resolved_version}; current version is {self.version}"
            )
        if resolved_version < self.version:
            print("[debug][storage] extract_version:error older version", flush=True)
            raise ValueError(
                f"unsupported older notebook JSON version {resolved_version}; expected version {self.version}"
            )
        return resolved_version

    def save(self, path: str | Path, document: NotebookDocument) -> None:
        target = Path(path)
        payload = serialize_notebook_cells(document.cells, document.layout)
        payload["metadata"] = {**dict(document.metadata or {}), "version": self.version}
        runtime_outputs = {}
        for cell in document.cells:
            outputs = []
            for output in cell.get("outputs", []) or []:
                data = dict(output.get("data") or {})
                outputs.append(
                    {
                        "kind": output.get("kind", "value"),
                        "data": {key: value for key, value in data.items() if key in {"text", "html"}},
                    }
                )
            if outputs:
                runtime_outputs[cell.get("id") or ""] = outputs
        if runtime_outputs:
            payload["runtime_outputs"] = runtime_outputs
        print(f"[debug][storage] save:path path={str(target)!r}", flush=True)
        print(f"[debug][storage] save:cells count={len(document.cells)}", flush=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> NotebookDocument:
        source = Path(path)
        print(f"[debug][storage] load:path path={str(source)!r}", flush=True)
        payload = json.loads(source.read_text(encoding="utf-8"))
        resolved_version = self._extract_version(payload)
        cells = deserialize_notebook_cells(payload)
        runtime_outputs = dict(payload.get("runtime_outputs") or {})
        for cell in cells:
            cell["outputs"] = list(runtime_outputs.get(cell.get("id") or "", []))
        document = NotebookDocument(
            cells=cells,
            metadata={**dict(payload.get("metadata") or {}), "version": resolved_version},
            layout=dict(payload.get("layout") or {"left_column_width_pct": 50}),
        )
        print(f"[debug][storage] load:cells count={len(document.cells)}", flush=True)
        print(f"[debug][storage] load:layout layout={document.layout}", flush=True)
        return document
