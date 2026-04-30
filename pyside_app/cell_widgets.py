from __future__ import annotations

import re
import uuid
from typing import Any, Callable

import markdown
from PySide6.QtCore import QStringListModel, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from pyside_app.editor_intelligence import NotebookLspClient
from pyside_app.execution_engine import ExecutionResult
from pyside_app.markdown_preview import MarkdownPreview
from pyside_app.plot_view import PlotView


class AutoResizePlainTextEdit(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.document().documentLayout().documentSizeChanged.connect(self._update_height)
        self.textChanged.connect(self._update_height)
        self._update_height()

    def setPlainText(self, text: str) -> None:
        super().setPlainText(text)
        self._update_height()

    def _update_height(self, *_args: object) -> None:
        line_spacing = self.fontMetrics().lineSpacing()
        document_height = max(1, self.blockCount()) * line_spacing
        frame = self.frameWidth() * 2
        margins = self.contentsMargins().top() + self.contentsMargins().bottom()
        padding = 12
        height = max(120, int(document_height + frame + margins + padding))
        print(f"[debug][auto-resize-editor] update_height height={height}", flush=True)
        self.setFixedHeight(height)


class NotebookCodeEditor(AutoResizePlainTextEdit):
    def __init__(
        self,
        parent: QWidget | None = None,
        lsp_client: NotebookLspClient | None = None,
        document_uri: str = "",
        completion_words: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.document_uri = document_uri
        self._lsp_client = lsp_client
        self._namespace_completion_words: list[str] = sorted(set(completion_words or []))
        self._lsp_completion_words: list[str] = []
        self._diagnostic_messages: list[str] = []
        self._completer_model = QStringListModel(self)
        self._completer = QCompleter(self._completer_model, self)
        self._completer.setWidget(self)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.activated.connect(self._insert_completion)
        self._completer.popup().setStyleSheet(
            """
            QListView {
                background:#ffffff;
                color:#0f1b2b;
                border:1px solid #d1dce8;
                border-radius:8px;
                outline:0;
                padding:4px;
            }
            QListView::item {
                background:#ffffff;
                color:#0f1b2b;
                padding:6px 10px;
                border-radius:4px;
            }
            QListView::item:hover {
                background:#c7def5;
                color:#0f1b2b;
            }
            QListView::item:selected {
                background:#dbeafe;
                color:#001f41;
            }
            """
        )
        print("[debug][code-editor] completer_popup_style background='#ffffff' border='#d1dce8'", flush=True)
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(120)
        self._sync_timer.setSingleShot(True)
        self._sync_timer.timeout.connect(self._sync_document_to_lsp)
        self._completion_timer = QTimer(self)
        self._completion_timer.setInterval(90)
        self._completion_timer.setSingleShot(True)
        self._completion_timer.timeout.connect(self._request_completions_from_lsp)
        self.textChanged.connect(self._handle_text_changed)
        self.set_completion_words(self._namespace_completion_words)
        print(
            f"[debug][code-editor] init uri={self.document_uri!r} words={len(self._namespace_completion_words)} "
            f"has_lsp={self._lsp_client is not None}",
            flush=True,
        )
        if self._lsp_client is not None and self.document_uri:
            self.attach_lsp_client(self._lsp_client, self.document_uri)

    def attach_lsp_client(self, lsp_client: NotebookLspClient, document_uri: str) -> None:
        self._lsp_client = lsp_client
        self.document_uri = document_uri
        print(f"[debug][code-editor] attach_lsp uri={self.document_uri!r}", flush=True)
        self._open_document_with_lsp()

    def completion_words(self) -> list[str]:
        words = self._completer_model.stringList()
        print(f"[debug][code-editor] completion_words count={len(words)}", flush=True)
        return words

    def set_completion_words(self, words: list[str]) -> None:
        self._namespace_completion_words = sorted(set(words))
        self._update_completion_model()

    def set_lsp_completions(self, words: list[str]) -> None:
        self._lsp_completion_words = sorted(set(word for word in words if word))
        print(f"[debug][code-editor] set_lsp_completions count={len(self._lsp_completion_words)}", flush=True)
        self._update_completion_model()
        self._show_completion_popup()

    def diagnostic_messages(self) -> list[str]:
        print(f"[debug][code-editor] diagnostic_messages count={len(self._diagnostic_messages)}", flush=True)
        return list(self._diagnostic_messages)

    def apply_diagnostics(self, diagnostics: list[dict[str, Any]]) -> None:
        print(f"[debug][code-editor] apply_diagnostics count={len(diagnostics)}", flush=True)
        self._diagnostic_messages = [str(item.get("message", "")) for item in diagnostics if item.get("message")]
        selections: list[QTextEdit.ExtraSelection] = []
        for diagnostic in diagnostics:
            diagnostic_range = diagnostic.get("range") or {}
            start = diagnostic_range.get("start") or {}
            end = diagnostic_range.get("end") or {}
            start_position = self._position_from_lsp(int(start.get("line", 0)), int(start.get("character", 0)))
            end_position = self._position_from_lsp(int(end.get("line", 0)), int(end.get("character", 0)))
            selection = QTextEdit.ExtraSelection()
            selection.format.setUnderlineColor(QColor("#b60021"))
            selection.format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.SpellCheckUnderline)
            cursor = self.textCursor()
            cursor.setPosition(start_position)
            cursor.setPosition(max(start_position, end_position), QTextCursor.MoveMode.KeepAnchor)
            selection.cursor = cursor
            selections.append(selection)
            print(
                f"[debug][code-editor] apply_diagnostics:item start={start_position} end={end_position} "
                f"message={diagnostic.get('message', '')!r}",
                flush=True,
            )
        self.setExtraSelections(selections)
        tooltip = "\n".join(self._diagnostic_messages)
        self.setToolTip(tooltip)
        print(f"[debug][code-editor] apply_diagnostics:done tooltip={tooltip!r}", flush=True)

    def keyPressEvent(self, event: Any) -> None:
        if self._completer.popup().isVisible() and event.key() in {Qt.Key.Key_Tab, Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            completion = self._completer.currentCompletion()
            print(f"[debug][code-editor] keypress:accept_completion completion={completion!r}", flush=True)
            if completion:
                self._insert_completion(completion)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Backtab:
            print("[debug][code-editor] keypress:backtab", flush=True)
            self._unindent_selection()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Tab:
            print("[debug][code-editor] keypress:tab", flush=True)
            self._indent_selection()
            event.accept()
            return
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Space:
            print("[debug][code-editor] keypress:ctrl_space", flush=True)
            self._request_completions_from_lsp()
            self._show_completion_popup()
            event.accept()
            return
        super().keyPressEvent(event)
        if event.text() and (event.text().isalnum() or event.text() == "_"):
            self._completion_timer.start()
            self._show_completion_popup()
        elif event.key() in {Qt.Key.Key_Backspace, Qt.Key.Key_Delete}:
            self._completion_timer.start()
            self._show_completion_popup()

    def _handle_text_changed(self) -> None:
        print(f"[debug][code-editor] text_changed uri={self.document_uri!r}", flush=True)
        self._sync_timer.start()
        if self._current_completion_prefix():
            self._completion_timer.start()

    def _update_completion_model(self) -> None:
        words = sorted(set(self._namespace_completion_words) | set(self._lsp_completion_words))
        self._completer_model.setStringList(words)
        print(f"[debug][code-editor] update_completion_model count={len(words)}", flush=True)

    def _indent_selection(self) -> None:
        cursor = self.textCursor()
        if not cursor.hasSelection():
            cursor.insertText("    ")
            self.setTextCursor(cursor)
            print("[debug][code-editor] indent_selection inserted_spaces=4", flush=True)
            return
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.beginEditBlock()
        cursor.setPosition(start)
        start_block = cursor.blockNumber()
        cursor.setPosition(end)
        end_block = cursor.blockNumber()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        for _index in range(start_block, end_block + 1):
            cursor.insertText("    ")
            if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock):
                break
        cursor.endEditBlock()
        print(f"[debug][code-editor] indent_selection start_block={start_block} end_block={end_block}", flush=True)

    def _unindent_selection(self) -> None:
        cursor = self.textCursor()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.beginEditBlock()
        cursor.setPosition(start)
        start_block = cursor.blockNumber()
        cursor.setPosition(end)
        end_block = cursor.blockNumber()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        for _index in range(start_block, end_block + 1):
            current_block = cursor.block().text()
            remove_count = min(4, len(current_block) - len(current_block.lstrip(" ")))
            print(f"[debug][code-editor] unindent_selection:block remove_count={remove_count}", flush=True)
            for _count in range(remove_count):
                cursor.deleteChar()
            if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock):
                break
        cursor.endEditBlock()
        print(f"[debug][code-editor] unindent_selection start_block={start_block} end_block={end_block}", flush=True)

    def _current_completion_prefix(self) -> str:
        cursor = self.textCursor()
        text = self.toPlainText()
        position = cursor.position()
        start = position
        while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
            start -= 1
        prefix = text[start:position]
        print(f"[debug][code-editor] current_prefix prefix={prefix!r}", flush=True)
        return prefix

    def _show_completion_popup(self) -> None:
        prefix = self._current_completion_prefix()
        if not prefix:
            self._completer.popup().hide()
            print("[debug][code-editor] show_completion_popup hide_empty_prefix", flush=True)
            return
        self._completer.setCompletionPrefix(prefix)
        popup = self._completer.popup()
        if popup.model().rowCount() == 0:
            popup.hide()
            print("[debug][code-editor] show_completion_popup hide_no_rows", flush=True)
            return
        rect = self.cursorRect()
        rect.setWidth(max(240, popup.sizeHintForColumn(0) + 24))
        self._completer.complete(rect)
        print(f"[debug][code-editor] show_completion_popup prefix={prefix!r}", flush=True)

    def _insert_completion(self, completion: str) -> None:
        prefix = self._current_completion_prefix()
        cursor = self.textCursor()
        for _count in range(len(prefix)):
            cursor.deletePreviousChar()
        cursor.insertText(completion)
        self.setTextCursor(cursor)
        self._completer.popup().hide()
        print(f"[debug][code-editor] insert_completion completion={completion!r}", flush=True)

    def _sync_document_to_lsp(self) -> None:
        if self._lsp_client is None or not self.document_uri:
            print("[debug][code-editor] sync_document_to_lsp skipped", flush=True)
            return
        print(f"[debug][code-editor] sync_document_to_lsp uri={self.document_uri!r}", flush=True)
        self._lsp_client.change_document(self.document_uri, self.toPlainText())

    def _request_completions_from_lsp(self) -> None:
        if self._lsp_client is None or not self.document_uri:
            print("[debug][code-editor] request_completions skipped", flush=True)
            return
        cursor = self.textCursor()
        line = cursor.blockNumber()
        character = cursor.positionInBlock()
        print(
            f"[debug][code-editor] request_completions uri={self.document_uri!r} line={line} character={character}",
            flush=True,
        )
        self._lsp_client.request_completion(self.document_uri, line, character)

    def _open_document_with_lsp(self) -> None:
        if self._lsp_client is None or not self.document_uri:
            print("[debug][code-editor] open_document_with_lsp skipped", flush=True)
            return
        print(f"[debug][code-editor] open_document_with_lsp uri={self.document_uri!r}", flush=True)
        self._lsp_client.open_document(self.document_uri, self.toPlainText())

    def _position_from_lsp(self, line: int, character: int) -> int:
        document = self.document()
        block = document.findBlockByNumber(line)
        if not block.isValid():
            print(f"[debug][code-editor] position_from_lsp invalid_line={line}", flush=True)
            return document.characterCount() - 1
        position = min(block.position() + character, block.position() + block.length() - 1)
        print(
            f"[debug][code-editor] position_from_lsp line={line} character={character} position={position}",
            flush=True,
        )
        return position


class AutoResizeTextBrowser(QTextBrowser):
    def __init__(self, parent: QWidget | None = None, minimum_height: int = 48) -> None:
        super().__init__(parent)
        self._minimum_height = minimum_height
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().documentLayout().documentSizeChanged.connect(self._update_height)
        self._update_height()

    def setPlainText(self, text: str) -> None:
        super().setPlainText(text)
        self._update_height()

    def setHtml(self, text: str) -> None:
        super().setHtml(text)
        self._update_height()

    def _update_height(self, *_args: object) -> None:
        document_height = self.document().documentLayout().documentSize().height()
        text_lines = max(1, len(self.toPlainText().splitlines()) or 1)
        document_height = max(document_height, text_lines * self.fontMetrics().lineSpacing())
        frame = self.frameWidth() * 2
        margins = self.contentsMargins().top() + self.contentsMargins().bottom()
        padding = 12
        height = max(self._minimum_height, int(document_height + frame + margins + padding))
        print(f"[debug][auto-resize-browser] update_height height={height}", flush=True)
        self.setFixedHeight(height)


class PythonHighlighter(QSyntaxHighlighter):
    RULES = [
        (
            r"\b(def|class|return|import|from|as|if|elif|else|for|while|with|try|except|finally|pass|break|continue|and|or|not|in|is|lambda|yield|raise|del|global|nonlocal|True|False|None)\b",
            QColor("#001f41"),
            True,
        ),
        (r"\b(np|pd|go|scipy|math)\b", QColor("#7c3aed"), False),
        (r"#[^\n]*", QColor("#94a3b8"), False),
        (r'"[^"\n]*"|\'[^\'\n]*\'', QColor("#15803d"), False),
        (r"\b\d+\.?\d*\b", QColor("#b60021"), False),
    ]
    _COMPILED_RULES: list[tuple[re.Pattern[str], QTextCharFormat]] = []

    @classmethod
    def _build_rules(cls) -> list[tuple[re.Pattern[str], QTextCharFormat]]:
        if cls._COMPILED_RULES:
            return cls._COMPILED_RULES
        for pattern, color, bold in cls.RULES:
            fmt = QTextCharFormat()
            fmt.setForeground(color)
            if bold:
                fmt.setFontWeight(QFont.Weight.Bold)
            cls._COMPILED_RULES.append((re.compile(pattern), fmt))
        return cls._COMPILED_RULES

    def __init__(self, document: Any) -> None:
        super().__init__(document)
        self._rules = self._build_rules()

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self._rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)


class OutputArea(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

    def clear_outputs(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def set_result(self, result: ExecutionResult) -> None:
        print(
            f"[debug][output-area] set_result outputs={len(result.outputs)} "
            f"has_error={result.error is not None}",
            flush=True,
        )
        self.clear_outputs()
        for output in result.outputs:
            if output.kind == "plotly":
                plot = PlotView(self)
                plot.set_html(output.data["html"])
                self._layout.addWidget(plot)
            elif output.data.get("html"):
                browser = AutoResizeTextBrowser(self, minimum_height=64)
                browser.setHtml(output.data["html"])
                self._layout.addWidget(browser)
            else:
                browser = AutoResizeTextBrowser(self, minimum_height=48)
                browser.setPlainText(output.data.get("text", ""))
                self._layout.addWidget(browser)
        if result.error:
            error = AutoResizeTextBrowser(self, minimum_height=72)
            error.setPlainText(result.error)
            error.setStyleSheet("color: #b00020;")
            self._layout.addWidget(error)


class NotebookCellWidget(QFrame):
    def __init__(
        self,
        cell_type: str = "code",
        source: str = "",
        column: str = "left",
        lsp_client: NotebookLspClient | None = None,
        document_uri: str = "",
        completion_words: list[str] | None = None,
        on_run: Callable[["NotebookCellWidget"], None] | None = None,
        on_delete: Callable[["NotebookCellWidget"], None] | None = None,
        on_move: Callable[["NotebookCellWidget", int], None] | None = None,
        on_move_column: Callable[["NotebookCellWidget", str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.cell_id = f"cell-{uuid.uuid4().hex[:8]}"
        self.cell_type = cell_type
        self.column = column
        self.on_run = on_run
        self.on_delete = on_delete
        self.on_move = on_move
        self.on_move_column = on_move_column
        self.preview_mode = cell_type == "code" or (cell_type == "markdown" and bool(source.strip()))
        self.last_result: ExecutionResult | None = None
        accent = "#001f41" if cell_type == "code" else "#7c3aed"
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame {"
            " background: #ffffff;"
            " border: 1px solid #d1dce8;"
            f" border-left: 4px solid {accent};"
            " border-radius: 0px 8px 8px 0px;"
            "} "
            "QWidget { background: #ffffff; } "
            "QPlainTextEdit { border: none; background: #ffffff; font-family: 'Consolas'; font-size: 15px; color: #0f1b2b; } "
            "QTextBrowser { border: 1px solid #e2e8f0; background: #f8fbfe; font-size: 14px; color: #475569; }"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(8)
        toolbar = QHBoxLayout()
        self.title_label = QLabel("Code Cell" if cell_type == "code" else "Markdown Cell", self)
        self.title_label.setStyleSheet(
            f"color: {accent};"
            " font-size: 13px;"
            " font-weight: 700;"
            " text-transform: uppercase;"
            " letter-spacing: 0.05em;"
        )
        toolbar.addWidget(self.title_label)
        toolbar.addStretch(1)

        if cell_type == "markdown":
            self.toggle_btn = QPushButton("Preview", self)
            self.toggle_btn.setStyleSheet(
                "QPushButton {"
                f" background:{accent}; color:white; border:none; border-radius:5px; padding:3px 10px; font-weight:600;"
                "} "
                "QPushButton:hover { background:#6d28d9; }"
            )
            self.toggle_btn.clicked.connect(self.toggle_preview)
            toolbar.addWidget(self.toggle_btn)
        else:
            self.run_btn = QPushButton("Run Cell", self)
            self.run_btn.setStyleSheet(
                "QPushButton {"
                " background:#001f41; color:white; border:none; border-radius:5px; padding:3px 10px; font-weight:600;"
                "} "
                "QPushButton:hover { background:#0d3567; }"
            )
            self.run_btn.clicked.connect(lambda: self.on_run and self.on_run(self))
            toolbar.addWidget(self.run_btn)

        self.move_up_btn = QPushButton("Up", self)
        self.move_up_btn.setStyleSheet(
            "QPushButton {"
            " background:#f1f5f9; color:#475569; border:1px solid #d1dce8; border-radius:5px; padding:3px 8px;"
            "} "
            "QPushButton:hover { background:#cbd5e1; }"
        )
        self.move_up_btn.clicked.connect(lambda: self.on_move and self.on_move(self, -1))
        toolbar.addWidget(self.move_up_btn)
        self.move_down_btn = QPushButton("Down", self)
        self.move_down_btn.setStyleSheet(
            "QPushButton {"
            " background:#f1f5f9; color:#475569; border:1px solid #d1dce8; border-radius:5px; padding:3px 8px;"
            "} "
            "QPushButton:hover { background:#cbd5e1; }"
        )
        self.move_down_btn.clicked.connect(lambda: self.on_move and self.on_move(self, 1))
        toolbar.addWidget(self.move_down_btn)

        self.move_left_btn = QPushButton("Left", self)
        self.move_left_btn.setStyleSheet(
            "QPushButton {"
            " background:#f1f5f9; color:#475569; border:1px solid #d1dce8; border-radius:5px; padding:3px 8px;"
            "} "
            "QPushButton:hover { background:#cbd5e1; }"
        )
        self.move_left_btn.clicked.connect(lambda: self.on_move_column and self.on_move_column(self, "left"))
        toolbar.addWidget(self.move_left_btn)
        self.move_right_btn = QPushButton("Right", self)
        self.move_right_btn.setStyleSheet(
            "QPushButton {"
            " background:#f1f5f9; color:#475569; border:1px solid #d1dce8; border-radius:5px; padding:3px 8px;"
            "} "
            "QPushButton:hover { background:#cbd5e1; }"
        )
        self.move_right_btn.clicked.connect(lambda: self.on_move_column and self.on_move_column(self, "right"))
        toolbar.addWidget(self.move_right_btn)

        self.delete_btn = QPushButton("Delete", self)
        self.delete_btn.setStyleSheet(
            "QPushButton {"
            " background:rgba(220,38,38,0.12); color:#b91c1c; border:1px solid rgba(220,38,38,0.25); border-radius:5px; padding:3px 8px;"
            "} "
            "QPushButton:hover { background:#b60021; color:white; }"
        )
        self.delete_btn.clicked.connect(lambda: self.on_delete and self.on_delete(self))
        toolbar.addWidget(self.delete_btn)
        self.clear_btn = QPushButton("Clear Output", self)
        self.clear_btn.setStyleSheet(
            "QPushButton {"
            " background:#f1f5f9; color:#475569; border:1px solid #d1dce8; border-radius:5px; padding:3px 8px;"
            "} "
            "QPushButton:hover { background:#cbd5e1; }"
        )
        self.clear_btn.clicked.connect(self.clear_output)
        toolbar.addWidget(self.clear_btn)
        outer.addLayout(toolbar)

        body = QHBoxLayout()
        body.setSpacing(10)
        if cell_type == "code":
            self.editor = NotebookCodeEditor(self, lsp_client=lsp_client, document_uri=document_uri, completion_words=completion_words)
        else:
            self.editor = AutoResizePlainTextEdit(self)
        self.editor.setPlainText(source)
        if cell_type == "code":
            self._highlighter = PythonHighlighter(self.editor.document())
        body.addWidget(self.editor, 1)
        if cell_type == "code":
            self.run_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self.editor)
            self.run_shortcut.activated.connect(lambda: self.on_run and self.on_run(self))
            self.inline_result = AutoResizeTextBrowser(self, minimum_height=120)
            self.inline_result.setMinimumWidth(280)
            self.inline_result.setMaximumWidth(440)
            self.inline_result.setStyleSheet(
                "QTextBrowser {"
                " border: none;"
                " border-left: 2px solid #e2e8f0;"
                " background: #f8fbfe;"
                " color: #475569;"
                " font-family: 'Roboto Condensed', 'Segoe UI', sans-serif;"
                " font-size: 14px;"
                " padding: 4px 8px;"
                "}"
            )
            self.inline_result.hide()
            body.addWidget(self.inline_result, 0)
        outer.addLayout(body)

        self.preview = MarkdownPreview(self) if cell_type == "markdown" else AutoResizeTextBrowser(self, minimum_height=120)
        self.preview.hide()
        outer.addWidget(self.preview)

        self.output_area = OutputArea(self)
        if cell_type in {"markdown", "code"}:
            self.output_area.hide()
        outer.addWidget(self.output_area)
        self._sync_column_buttons()
        if self.cell_type == "markdown" and self.preview_mode:
            self._show_markdown_preview()

    def _sync_column_buttons(self) -> None:
        print(f"[debug][cell-widget] sync_column_buttons cell_id={self.cell_id!r} column={self.column!r}", flush=True)
        self.move_left_btn.setEnabled(self.column != "left")
        self.move_right_btn.setEnabled(self.column != "right")

    def set_column(self, column: str) -> None:
        self.column = column
        self._sync_column_buttons()

    def source(self) -> str:
        return self.editor.toPlainText()

    def set_result(self, result: ExecutionResult) -> None:
        self.last_result = result
        if self.cell_type == "code" and hasattr(self, "inline_result"):
            self.inline_result.setPlainText("")
            graph_outputs = [
                output
                for output in result.outputs
                if output.kind == "plotly" or (output.kind == "html" and "data:image" in output.data.get("html", ""))
            ]
            rich_outputs = [output for output in result.outputs if output not in graph_outputs]
            if rich_outputs or result.error:
                self.output_area.show()
                self.output_area.set_result(
                    ExecutionResult(outputs=rich_outputs, stdout=result.stdout, stderr=result.stderr, error=result.error)
                )
            else:
                self.output_area.clear_outputs()
                self.output_area.hide()
        else:
            self.output_area.set_result(result)

    def clear_output(self) -> None:
        print(f"[debug][cell-widget] clear_output cell_id={self.cell_id!r}", flush=True)
        self.last_result = None
        if self.cell_type == "code" and hasattr(self, "inline_result"):
            self.inline_result.setPlainText("")
        self.output_area.clear_outputs()
        self.output_area.hide()

    def toggle_preview(self) -> None:
        if self.cell_type != "markdown":
            return
        showing_preview = self.preview.isHidden()
        print(f"[debug][cell-widget] toggle_preview cell_id={self.cell_id!r} preview={showing_preview}", flush=True)
        if showing_preview:
            self._show_markdown_preview()
        else:
            self._show_markdown_editor()

    def _show_markdown_preview(self) -> None:
        print(f"[debug][cell-widget] show_markdown_preview cell_id={self.cell_id!r}", flush=True)
        if isinstance(self.preview, MarkdownPreview):
            self.preview.set_markdown(self.source())
        else:
            html = markdown.markdown(
                self.source(),
                extensions=["fenced_code", "tables", "toc"],
            )
            self.preview.setHtml(html)
        self.preview.show()
        self.editor.hide()
        self.preview_mode = True
        self.toggle_btn.setText("Edit")

    def _show_markdown_editor(self) -> None:
        print(f"[debug][cell-widget] show_markdown_editor cell_id={self.cell_id!r}", flush=True)
        self.preview.hide()
        self.editor.show()
        self.preview_mode = False
        self.toggle_btn.setText("Preview")
