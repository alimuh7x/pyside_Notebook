from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pyside_app.notebook_tab import NotebookTab


class VerifyLspClient(QObject):
    completions_ready = Signal(str, list)
    diagnostics_ready = Signal(str, list)
    availability_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.started = False
        self.opened_documents: list[tuple[str, str]] = []
        self.changed_documents: list[tuple[str, str]] = []
        self.requested_completions: list[tuple[str, int, int]] = []

    def start(self) -> None:
        print("[verify][editor-intelligence] lsp:start", flush=True)
        self.started = True
        self.availability_changed.emit(True)

    def open_document(self, uri: str, text: str) -> None:
        print(f"[verify][editor-intelligence] lsp:open uri={uri!r} text={text!r}", flush=True)
        self.opened_documents.append((uri, text))

    def change_document(self, uri: str, text: str) -> None:
        print(f"[verify][editor-intelligence] lsp:change uri={uri!r} text={text!r}", flush=True)
        self.changed_documents.append((uri, text))

    def request_completion(self, uri: str, line: int, character: int) -> None:
        print(
            f"[verify][editor-intelligence] lsp:completion uri={uri!r} line={line} character={character}",
            flush=True,
        )
        self.requested_completions.append((uri, line, character))


app = QApplication.instance() or QApplication([])
fake_lsp = VerifyLspClient()
tab = NotebookTab(lsp_client=fake_lsp)
cell = tab.cells[0]
editor = cell.editor

editor.setPlainText("")
QTest.keyClick(editor, Qt.Key.Key_Tab)
print(f"[verify][editor-intelligence] tab_text={editor.toPlainText()!r}", flush=True)

tab.execution_engine.execute("alpha = 42")
tab._refresh_completion_words()
print(f"[verify][editor-intelligence] has_alpha={'alpha' in editor.completion_words()}", flush=True)

fake_lsp.diagnostics_ready.emit(
    editor.document_uri,
    [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 4}}, "message": "Example error"}],
)
QApplication.processEvents()
print(f"[verify][editor-intelligence] diagnostics={editor.diagnostic_messages()!r}", flush=True)
print(f"[verify][editor-intelligence] opened={len(fake_lsp.opened_documents)} changed={len(fake_lsp.changed_documents)}", flush=True)
