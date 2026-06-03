from __future__ import annotations

import math as _math
import re
import uuid
from typing import Any, Callable

import markdown
from PySide6.QtCore import QStringListModel, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from pyside_app.controls import CheckableComboBox
from pyside_app.editor_intelligence import NotebookLspClient
from pyside_app.execution_engine import ExecutionResult, detect_cell_parameters
from pyside_app.markdown_preview import MarkdownPreview
from pyside_app.plot_view import PlotView


class AutoResizePlainTextEdit(QPlainTextEdit):
    """Plain-text editor that grows vertically to fit its document."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Configure the editor for auto-resize and no internal scrollbars."""
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.document().documentLayout().documentSizeChanged.connect(self._update_height)
        self.textChanged.connect(self._update_height)
        self._update_height()

    def setPlainText(self, text: str) -> None:
        """Set editor contents and immediately recompute widget height."""
        super().setPlainText(text)
        self._update_height()

    def _update_height(self, *_args: object) -> None:
        """Resize the editor to match its current number of text lines."""
        line_spacing = self.fontMetrics().lineSpacing()
        document_height = max(1, self.blockCount()) * line_spacing
        frame = self.frameWidth() * 2
        margins = self.contentsMargins().top() + self.contentsMargins().bottom()
        padding = 12
        height = max(120, int(document_height + frame + margins + padding))
        print(f"[debug][auto-resize-editor] update_height height={height}", flush=True)
        self.setFixedHeight(height)


class NotebookCodeEditor(AutoResizePlainTextEdit):
    """Code editor with namespace completions, LSP hooks, and diagnostics support."""

    def __init__(
        self,
        parent: QWidget | None = None,
        lsp_client: NotebookLspClient | None = None,
        document_uri: str = "",
        completion_words: list[str] | None = None,
    ) -> None:
        """Initialize the code editor, completer, and optional LSP integration."""
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
                background:#2c313a;
                color:#d7dae0;
                border:1px solid #3e4451;
                border-radius:8px;
                outline:0;
                padding:4px;
            }
            QListView::item {
                background:#2c313a;
                color:#d7dae0;
                padding:6px 10px;
                border-radius:4px;
            }
            QListView::item:hover {
                background:#3e4451;
                color:#61afef;
            }
            QListView::item:selected {
                background:#282c34;
                color:#61afef;
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
        """Attach an LSP client and open this editor as a tracked document."""
        self._lsp_client = lsp_client
        self.document_uri = document_uri
        print(f"[debug][code-editor] attach_lsp uri={self.document_uri!r}", flush=True)
        self._open_document_with_lsp()

    def completion_words(self) -> list[str]:
        """Return the currently visible completion word list."""
        words = self._completer_model.stringList()
        print(f"[debug][code-editor] completion_words count={len(words)}", flush=True)
        return words

    def set_completion_words(self, words: list[str]) -> None:
        """Replace namespace-derived completion words."""
        self._namespace_completion_words = sorted(set(words))
        self._update_completion_model()

    def set_lsp_completions(self, words: list[str]) -> None:
        """Update completion candidates returned by the LSP service."""
        self._lsp_completion_words = sorted(set(word for word in words if word))
        print(f"[debug][code-editor] set_lsp_completions count={len(self._lsp_completion_words)}", flush=True)
        self._update_completion_model()
        self._show_completion_popup()

    def diagnostic_messages(self) -> list[str]:
        """Return the current diagnostic messages shown for this editor."""
        print(f"[debug][code-editor] diagnostic_messages count={len(self._diagnostic_messages)}", flush=True)
        return list(self._diagnostic_messages)

    def apply_diagnostics(self, diagnostics: list[dict[str, Any]]) -> None:
        """Underline diagnostic ranges and expose their messages as a tooltip."""
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
        """Handle completion acceptance, indentation shortcuts, and popup refresh."""
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
        """Debounce document sync and completion refresh after edits."""
        print(f"[debug][code-editor] text_changed uri={self.document_uri!r}", flush=True)
        self._sync_timer.start()
        if self._current_completion_prefix():
            self._completion_timer.start()

    def _update_completion_model(self) -> None:
        """Merge namespace and LSP completion sources into one popup model."""
        words = sorted(set(self._namespace_completion_words) | set(self._lsp_completion_words))
        self._completer_model.setStringList(words)
        print(f"[debug][code-editor] update_completion_model count={len(words)}", flush=True)

    def _indent_selection(self) -> None:
        """Indent the current line or selected block by four spaces."""
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
        """Remove up to four leading spaces from each selected line."""
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
        """Return the identifier fragment immediately before the cursor."""
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
        """Open or hide the completion popup based on the current prefix."""
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
        """Replace the current prefix with the chosen completion item."""
        prefix = self._current_completion_prefix()
        cursor = self.textCursor()
        for _count in range(len(prefix)):
            cursor.deletePreviousChar()
        cursor.insertText(completion)
        self.setTextCursor(cursor)
        self._completer.popup().hide()
        print(f"[debug][code-editor] insert_completion completion={completion!r}", flush=True)

    def _sync_document_to_lsp(self) -> None:
        """Push the current buffer contents to the LSP client."""
        if self._lsp_client is None or not self.document_uri:
            print("[debug][code-editor] sync_document_to_lsp skipped", flush=True)
            return
        print(f"[debug][code-editor] sync_document_to_lsp uri={self.document_uri!r}", flush=True)
        self._lsp_client.change_document(self.document_uri, self.toPlainText())

    def _request_completions_from_lsp(self) -> None:
        """Ask the LSP client for completions at the current cursor position."""
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
        """Register this editor document with the LSP client."""
        if self._lsp_client is None or not self.document_uri:
            print("[debug][code-editor] open_document_with_lsp skipped", flush=True)
            return
        print(f"[debug][code-editor] open_document_with_lsp uri={self.document_uri!r}", flush=True)
        self._lsp_client.open_document(self.document_uri, self.toPlainText())

    def _position_from_lsp(self, line: int, character: int) -> int:
        """Convert an LSP line/character pair into a QTextDocument offset."""
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
    """Read-only text or HTML viewer that resizes to its content."""

    def __init__(self, parent: QWidget | None = None, minimum_height: int = 48) -> None:
        """Create the browser with automatic vertical growth enabled."""
        super().__init__(parent)
        self._minimum_height = minimum_height
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().documentLayout().documentSizeChanged.connect(self._update_height)
        self._update_height()

    def setPlainText(self, text: str) -> None:
        """Set plain text and refresh the widget height."""
        super().setPlainText(text)
        self._update_height()

    def setHtml(self, text: str) -> None:
        """Set HTML content and refresh the widget height."""
        super().setHtml(text)
        self._update_height()

    def _update_height(self, *_args: object) -> None:
        """Resize the browser to fit its rendered document height."""
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
    """Apply lightweight syntax highlighting rules to Python source text."""

    # Atom One Dark palette — (pattern, hex-color, bold, italic)
    RULES = [
        # definition keywords — purple
        (r"\b(def|class|lambda|global|nonlocal|import|from|as|with)\b",
         "#c678dd", True, False),
        # control flow — golden yellow
        (r"\b(if|elif|else|for|while|try|except|finally|pass|break|continue)\b",
         "#e5c07b", True, False),
        # special values / return / boolean keywords — cyan
        (r"\b(return|yield|raise|del|and|or|not|in|is|True|False|None|assert|async|await)\b",
         "#56b6c2", True, False),
        # self / cls — red-pink
        (r"\b(self|cls)\b", "#e06c75", False, False),
        # built-in functions — light blue
        (r"\b(print|len|range|type|list|dict|int|float|str|bool|tuple|set|frozenset"
         r"|abs|round|min|max|sum|sorted|reversed|enumerate|zip|map|filter|any|all"
         r"|open|input|repr|id|hash|dir|vars|getattr|setattr|hasattr|isinstance|issubclass"
         r"|super|property|staticmethod|classmethod|callable|iter|next)\b",
         "#61afef", False, False),
        # scientific imports / aliases — light blue
        (r"\b(np|pd|go|plt|sp|scipy|math|os|sys|re|json|copy|time|datetime|pathlib|typing)\b",
         "#61afef", False, False),
        # decorators — light blue
        (r"@[\w\.]+", "#61afef", False, False),
        # numpy array creation — teal (distinct from builtins)
        (r"\b(zeros|ones|linspace|arange|eye|empty|full|stack|concatenate"
         r"|vstack|hstack|meshgrid|zeros_like|ones_like|empty_like|full_like"
         r"|tile|repeat|fromiter|array)\b",
         "#2bbac5", False, False),
        # numpy math / signal functions — teal
        (r"\b(sin|cos|tan|arcsin|arccos|arctan|arctan2|sinh|cosh|tanh"
         r"|exp|log|log2|log10|sqrt|square|cbrt|sign|floor|ceil"
         r"|clip|where|argmin|argmax|nanmin|nanmax|nanmean|nanstd|nansum"
         r"|allclose|isnan|isinf|isfinite|inf|nan|pi|gradient|interp"
         r"|dot|cross|matmul|cumsum|cumprod|diff|fft|rfft|ifft|argsort|unique)\b",
         "#2bbac5", False, False),
        # array / dataframe attributes and methods (after dot) — orange
        (r"\.(shape|dtype|T|size|ndim|real|imag|flat|nbytes"
         r"|flatten|reshape|ravel|squeeze|transpose"
         r"|copy|astype|tolist|item|fill"
         r"|values|index|columns|iloc|loc|at|iat"
         r"|mean|std|var|prod|cumsum|cumprod"
         r"|sort_values|value_counts|nunique|trace|diagonal)\b",
         "#d19a66", False, False),
        # identifiers subscripted with [ — array / sequence access — warm teal
        (r"\b[A-Za-z_]\w*(?=\s*\[)", "#56b6c2", False, False),
        # assignment-target identifiers (left of any = form) — light green
        (r"\b[A-Za-z_]\w*(?=\s*(?:\+|-|\*\*?|//|/|%|@|&|\||\^|<<|>>)?=(?!=))",
         "#98c379", False, False),
        # augmented assignment operators — yellow (must come before arithmetic rule)
        (r"\+=|-=|\*\*=|//=|\*=|/=|%=|@=|&=|\|=|\^=", "#e5c07b", False, False),
        # plain assignment = — yellow (skip ==, !=, <=, >=, and augmented forms)
        (r"(?<![=!<>+\-\*/%&|^~])=(?!=)", "#e5c07b", False, False),
        # identifier on the right-hand side of = — green (matches assignment target color)
        (r"(?<![=!<>+\-\*/%&|^~])=(?!=)\s*([A-Za-z_]\w*)", "#98c379", False, False, True),
        # arithmetic operators — purple
        (r"\*\*|//|[+\-\*/%](?!=)", "#c678dd", False, False),
        # comparison operators — cyan
        (r"==|!=|<=|>=", "#56b6c2", False, False),
        # colon: slice, dict, annotation — cyan
        (r":", "#56b6c2", False, False),
        # comma separator — muted gold so [save_id, :] reads distinctly
        (r",", "#e5c07b", False, False),
        # identifier as first index/arg after [ — colors save_id in phi[save_id, :]
        (r"\[\s*([A-Za-z_]\w*)", "#d19a66", False, False, True),
        # identifier after comma — colors each subsequent index/arg
        (r",\s*([A-Za-z_]\w*)", "#d19a66", False, False, True),
        # square brackets — vivid purple
        (r"[\[\]]", "#c678dd", False, False),
        # parentheses — vivid teal
        (r"[()]", "#56b6c2", False, False),
        # curly braces — vivid orange
        (r"[{}]", "#d19a66", False, False),
        # numbers (int, float, hex, sci notation) — orange
        (r"\b(0x[0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?)\b", "#d19a66", False, False),
        # strings — green (applied after numpy so strings override false positives)
        (r'"[^"\n]*"', "#98c379", False, False),
        (r"'[^'\n]*'", "#98c379", False, False),
        # comments — dark gray italic, last so they override everything
        (r"#[^\n]*", "#5c6370", False, True),
    ]
    _COMPILED_RULES: list[tuple[re.Pattern[str], QTextCharFormat, bool]] = []

    @classmethod
    def _build_rules(cls) -> list[tuple[re.Pattern[str], QTextCharFormat, bool]]:
        """Compile and cache the regex-based syntax highlighting rules."""
        if cls._COMPILED_RULES:
            return cls._COMPILED_RULES
        for rule in cls.RULES:
            pattern, color, bold, italic = rule[:4]
            use_group = rule[4] if len(rule) > 4 else False
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            if bold:
                fmt.setFontWeight(QFont.Weight.Bold)
            if italic:
                fmt.setFontItalic(True)
            cls._COMPILED_RULES.append((re.compile(pattern), fmt, use_group))
        return cls._COMPILED_RULES

    def __init__(self, document: Any) -> None:
        """Attach the syntax highlighter to a document."""
        super().__init__(document)
        self._rules = self._build_rules()

    def highlightBlock(self, text: str) -> None:
        """Apply all configured highlighting rules to one text block."""
        for pattern, fmt, use_group in self._rules:
            for match in pattern.finditer(text):
                if use_group and match.lastindex:
                    start, end = match.start(1), match.end(1)
                else:
                    start, end = match.start(), match.end()
                self.setFormat(start, end - start, fmt)


class OutputArea(QWidget):
    """Stack rendered execution outputs for a single notebook cell."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the vertical output container."""
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

    def clear_outputs(self) -> None:
        """Remove and delete all currently rendered output widgets."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def set_result(self, result: ExecutionResult) -> None:
        """Render all outputs and any error from one execution result."""
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


_SLIDER_STYLE = (
    "QSlider::groove:horizontal { height:5px; background:#3e4451; border-radius:3px; }"
    "QSlider::handle:horizontal { width:16px; height:16px; margin:-6px 0;"
    " background:#61afef; border:2px solid #21252b; border-radius:8px; }"
    "QSlider::sub-page:horizontal { background:#61afef; border-radius:3px; }"
)


def _nice_step(lo: float, hi: float, is_int: bool, max_steps: int = 20) -> float:
    """Round step up to the nearest 1/2/5 multiple so values land on round numbers.

    Targets ~10 intervals; guarantees at most max_steps intervals.
    """
    if is_int:
        raw = max(1.0, (hi - lo) / max_steps)
        return float(int(_math.ceil(raw)))
    span = hi - lo
    if span <= 0:
        return 0.1
    # Start from a target of 10 intervals, round UP to nearest 1/2/5 multiple.
    raw = span / 10.0
    exp = _math.floor(_math.log10(raw))
    m = raw / (10 ** exp)
    if m <= 1.0:
        nice = 1.0
    elif m <= 2.0:
        nice = 2.0
    elif m <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    step = nice * (10 ** exp)
    # Clamp to max_steps by bumping to next nice level if needed.
    while (hi - lo) / step > max_steps:
        if nice == 1.0:
            nice = 2.0
        elif nice == 2.0:
            nice = 5.0
        elif nice == 5.0:
            nice = 10.0
            exp += 1
            nice = 1.0
        step = nice * (10 ** exp)
    return step


class AutoParamPanel(QWidget):
    """Interactive slider panel for exploring code-cell parameters."""

    def __init__(
        self,
        params: dict[str, dict[str, Any]],
        rerun_fn: Callable,
        source: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._params = dict(params)
        self._rerun_fn = rerun_fn
        self._source = source
        self._hidden_params: set[str] = set()
        self._overrides: dict[str, Any] = {n: s["value"] for n, s in params.items()}
        self._sliders: dict[str, tuple[QSlider, QLineEdit]] = {}
        self._steps: dict[str, float] = {}   # nice step per param
        self._row_widgets: dict[str, QWidget] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 4)
        root.setSpacing(4)

        # ── header row ────────────────────────────────────────────────────────
        _hdr_btn_ss = (
            "QPushButton { font-size:13px; padding:2px 10px; border:1px solid #3e4451;"
            " border-radius:6px; background:#2c313a; color:#d7dae0; }"
            "QPushButton:hover { background:#3e4451; border-color:#4a5568; color:#d7dae0; }"
        )
        header_row = QHBoxLayout()
        header_row.setSpacing(5)

        # 1. Hide all
        hide_all_btn = QPushButton("Hide all", self)
        hide_all_btn.setFixedHeight(22)
        hide_all_btn.setStyleSheet(_hdr_btn_ss)
        hide_all_btn.clicked.connect(self._hide_all_parameters)
        header_row.addWidget(hide_all_btn)

        # 2. Multi-select dropdown for params
        self._show_param_combo = CheckableComboBox(self, placeholder="Show hidden…")
        self._show_param_combo.setFixedHeight(24)
        self._show_param_combo.setMinimumWidth(200)
        self._show_param_combo.setStyleSheet(
            "QComboBox { font-size:13px; padding:2px 8px; border:1px solid #3e4451;"
            " border-radius:6px; background:#2c313a; color:#d7dae0; }"
            "QComboBox:hover { border-color:#61afef; }"
        )
        self._show_param_combo.checkedItemsChanged.connect(self._on_show_hidden_changed)
        header_row.addWidget(self._show_param_combo)

        header_row.addStretch(1)

        # 3. Reset all (visible only)
        reset_all_btn = QPushButton("Reset all", self)
        reset_all_btn.setFixedHeight(22)
        reset_all_btn.setStyleSheet(_hdr_btn_ss)
        reset_all_btn.clicked.connect(self._reset_all_parameters)
        header_row.addWidget(reset_all_btn)

        # 4. Manual run
        run_btn = QPushButton("Run", self)
        run_btn.setFixedHeight(22)
        run_btn.setStyleSheet(
            "QPushButton { font-size:13px; padding:2px 14px; border:none;"
            " border-radius:6px; background:#16a34a; color:#ffffff; font-weight:600; }"
            "QPushButton:hover { background:#15803d; }"
        )
        run_btn.clicked.connect(self._run_current_parameters)
        header_row.addWidget(run_btn)

        # 6. Status label
        self._status_lbl = QLabel("", self)
        self._status_lbl.setStyleSheet("font-size:13px; color:#5c6370; padding-left:4px;")
        header_row.addWidget(self._status_lbl)

        root.addLayout(header_row)

        # ── slider rows ───────────────────────────────────────────────────────
        _btn_ss = (
            "QPushButton { font-size:16px; font-weight:700; padding:0 6px; border:1px solid #3e4451;"
            " border-radius:6px; background:#2c313a; color:#d7dae0; min-width:28px; }"
            "QPushButton:hover { background:#1d4ed8; border-color:#1d4ed8; color:#ffffff; }"
        )
        _hide_ss = (
            "QPushButton { font-size:12px; padding:0 5px; border:1px solid #3e4451;"
            " border-radius:5px; background:#2c313a; color:#5c6370; }"
            "QPushButton:hover { background:#2d1f1f; border-color:#e06c75; color:#e06c75; }"
        )

        for name, spec in params.items():
            lo = float(spec["min"])
            hi = float(spec["max"])
            cur_val = float(spec["value"])
            step = _nice_step(lo, hi, bool(spec["is_int"]))
            self._steps[name] = step
            n_steps = max(1, min(20, int(round((hi - lo) / step))))

            row_w = QWidget(self)
            row_w.setStyleSheet("background:transparent;")
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)

            hide_btn = QPushButton("Hide", row_w)
            hide_btn.setFixedHeight(20)
            hide_btn.setFixedWidth(34)
            hide_btn.setStyleSheet(_hide_ss)
            hide_btn.clicked.connect(lambda _=False, n=name: self._hide_parameter(n))
            row.addWidget(hide_btn)

            lbl = QLabel(name, row_w)
            lbl.setFixedWidth(110)
            lbl.setStyleSheet("font-size:15px; color:#d7dae0; font-weight:600;")
            lbl.setToolTip(name)
            row.addWidget(lbl)

            minus_btn = QPushButton("−", row_w)
            minus_btn.setFixedSize(28, 28)
            minus_btn.setStyleSheet(_btn_ss)
            minus_btn.clicked.connect(lambda _=False, n=name: self._step_value(n, -1))
            row.addWidget(minus_btn)

            slider = QSlider(Qt.Orientation.Horizontal, row_w)
            slider.setRange(0, n_steps)
            slider.setValue(self._pos_for_value(cur_val, lo, step, n_steps))
            slider.setStyleSheet(_SLIDER_STYLE)
            slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row.addWidget(slider, 1)

            plus_btn = QPushButton("+", row_w)
            plus_btn.setFixedSize(28, 28)
            plus_btn.setStyleSheet(_btn_ss)
            plus_btn.clicked.connect(lambda _=False, n=name: self._step_value(n, +1))
            row.addWidget(plus_btn)

            val_edit = QLineEdit(row_w)
            val_edit.setFixedWidth(72)
            val_edit.setText(self._fmt(cur_val, bool(spec["is_int"]), step))
            val_edit.setStyleSheet(
                "QLineEdit { font-family:monospace; font-size:15px; font-weight:700; color:#61afef;"
                " background:#2c313a; border:1px solid #3e4451; border-radius:6px; padding:1px 5px; }"
                "QLineEdit:focus { background:#282c34; border-color:#61afef; }"
            )

            slider.valueChanged.connect(
                lambda v, n=name, ve=val_edit: self._on_slider_moved(n, v, ve)
            )
            val_edit.editingFinished.connect(
                lambda n=name, ve=val_edit: self._on_manual_value(n, ve)
            )
            row.addWidget(val_edit)

            reset_btn = QPushButton("Reset", row_w)
            reset_btn.setFixedHeight(20)
            reset_btn.setStyleSheet(_btn_ss)
            reset_btn.clicked.connect(lambda _=False, n=name: self._reset_parameter(n))
            row.addWidget(reset_btn)

            self._sliders[name] = (slider, val_edit)
            self._row_widgets[name] = row_w
            root.addWidget(row_w)
            # hidden by default
            self._hidden_params.add(name)
            row_w.hide()

        self._refresh_hidden_combo()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(value: float, is_int: bool, step: float = 0.0) -> str:
        if is_int:
            return str(int(round(value)))
        if step > 0:
            decimals = max(0, -int(_math.floor(_math.log10(step))))
            text = f"{value:.{decimals}f}"
            # strip trailing zeros (0.30 → 0.3, 1.00 → 1)
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return text
        if abs(value) >= 1e4 or (abs(value) < 1e-3 and value != 0.0):
            return f"{value:.3g}"
        return f"{value:.4g}"

    @staticmethod
    def _pos_for_value(value: float, lo: float, step: float, n_steps: int) -> int:
        if step <= 0:
            return 0
        return max(0, min(n_steps, int(round((value - lo) / step))))

    def _val_for_pos(self, name: str, pos: int) -> float:
        spec = self._params[name]
        lo = float(spec["min"])
        step = self._steps[name]
        raw = lo + pos * step
        if spec["is_int"]:
            return float(int(round(raw)))
        # snap to step precision
        if step > 0:
            decimals = max(0, -int(_math.floor(_math.log10(step))))
            return round(raw, decimals)
        return raw

    def _set_value(self, name: str, value: float, *, trigger_run: bool = True) -> None:
        spec = self._params.get(name)
        if spec is None:
            return
        lo, hi = float(spec["min"]), float(spec["max"])
        step = self._steps[name]
        n_steps = self._sliders[name][0].maximum()
        value = max(lo, min(hi, value))
        if spec["is_int"]:
            value = float(int(round(value)))
        elif step > 0:
            decimals = max(0, -int(_math.floor(_math.log10(step))))
            value = round(value, decimals)
        self._overrides[name] = value
        slider, val_edit = self._sliders[name]
        slider.blockSignals(True)
        slider.setValue(self._pos_for_value(value, lo, step, n_steps))
        slider.blockSignals(False)
        val_edit.setText(self._fmt(value, bool(spec["is_int"]), step))
        _ = trigger_run  # auto-run removed; run is manual only

    def update_params(self, params: dict[str, dict[str, Any]], source: str) -> None:
        self._params = dict(params)
        self._source = source
        for name, spec in params.items():
            if name not in self._sliders:
                continue
            step = self._steps.get(name, _nice_step(float(spec["min"]), float(spec["max"]), bool(spec["is_int"])))
            self._steps[name] = step
            cur_val = self._overrides.get(name, float(spec["value"]))
            self._set_value(name, float(cur_val), trigger_run=False)
        self._refresh_hidden_combo()

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_slider_moved(self, name: str, pos: int, val_edit: QLineEdit) -> None:
        spec = self._params.get(name)
        if spec is None:
            return
        val = self._val_for_pos(name, pos)
        self._overrides[name] = val
        step = self._steps.get(name, 0.0)
        val_edit.setText(self._fmt(val, bool(spec["is_int"]), step))

    def _on_manual_value(self, name: str, val_edit: QLineEdit) -> None:
        try:
            val = float(val_edit.text())
        except ValueError:
            return
        self._set_value(name, val, trigger_run=True)

    def _step_value(self, name: str, direction: int) -> None:
        """Move value by one nice step up (+1) or down (-1)."""
        current = float(self._overrides.get(name, self._params[name]["value"]))
        step = self._steps.get(name, 1.0)
        self._set_value(name, current + direction * step, trigger_run=True)

    def _on_show_hidden_changed(self) -> None:
        checked = set(self._show_param_combo.checked_values())
        for name in self._params:
            row = self._row_widgets.get(name)
            if name in checked and name in self._hidden_params:
                self._hidden_params.discard(name)
                self._overrides.setdefault(name, self._params[name]["value"])
                if row is not None:
                    row.show()
            elif name not in checked and name not in self._hidden_params:
                self._hidden_params.add(name)
                if row is not None:
                    row.hide()
        # do NOT call _refresh_hidden_combo() — it clears the combo while the
        # popup is open, which closes it and appears to remove the item

    def _refresh_hidden_combo(self) -> None:
        self._show_param_combo.blockSignals(True)
        self._show_param_combo.clear()
        for name in self._params:
            checked = name not in self._hidden_params  # visible → ticked
            self._show_param_combo.add_check_item(name, name, checked=checked)
        self._show_param_combo.blockSignals(False)

    def _reset_parameter(self, name: str) -> None:
        spec = self._params.get(name)
        if spec is None:
            return
        self._set_value(name, float(spec["value"]), trigger_run=False)
        self._status_lbl.setText(f"reset {name}")

    def _hide_parameter(self, name: str) -> None:
        self._hidden_params.add(name)
        row = self._row_widgets.get(name)
        if row is not None:
            row.hide()
        self._status_lbl.setText(f"hidden {name}")
        self._refresh_hidden_combo()

    def _show_parameter(self, name: str) -> None:
        self._hidden_params.discard(name)
        row = self._row_widgets.get(name)
        if row is not None:
            row.show()
        self._overrides.setdefault(name, self._params[name]["value"])
        self._status_lbl.setText(f"shown {name}")
        self._refresh_hidden_combo()

    def _hide_all_parameters(self) -> None:
        for name in list(self._params):
            if name not in self._hidden_params:
                self._hide_parameter(name)

    def _reset_all_parameters(self) -> None:
        for name in list(self._params):
            if name not in self._hidden_params:
                self._reset_parameter(name)
        self._status_lbl.setText("reset all")

    def _show_all_parameters(self) -> None:
        for name, spec in self._params.items():
            self._hidden_params.discard(name)
            row = self._row_widgets.get(name)
            if row is not None:
                row.show()
            self._overrides.setdefault(name, spec["value"])
        self._status_lbl.setText("shown all")
        self._refresh_hidden_combo()

    def _run_current_parameters(self) -> None:
        if self._rerun_fn is None:
            return
        active = {n: v for n, v in self._overrides.items() if n not in self._hidden_params}
        self._rerun_fn(self._source, active, self._on_result)
        self._status_lbl.setText("running…")

    def _on_result(self, result: ExecutionResult) -> None:
        self._status_lbl.setText("done" if not result.error else "error")


class NotebookCellWidget(QFrame):
    """Full notebook cell widget for code or markdown content."""

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
        rerun_fn: Callable | None = None,
    ) -> None:
        """Build the editor, toolbar, preview area, and output area for one cell."""
        super().__init__(parent)
        self.cell_id = f"cell-{uuid.uuid4().hex[:8]}"
        self.cell_type = cell_type
        self.column = column
        self.on_run = on_run
        self.on_delete = on_delete
        self.on_move = on_move
        self.on_move_column = on_move_column
        self._rerun_fn = rerun_fn
        self.preview_mode = cell_type == "code" or (cell_type == "markdown" and bool(source.strip()))
        self.last_result: ExecutionResult | None = None
        accent = "#61afef" if cell_type == "code" else "#c678dd"
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame {"
            " background: #282c34;"
            " border: 1px solid #3e4451;"
            f" border-left: 4px solid {accent};"
            " border-radius: 0px 8px 8px 0px;"
            "} "
            "QWidget { background: #282c34; } "
            "QPlainTextEdit {"
            " border: none;"
            " background: #282c34;"
            " font-family: 'Cascadia Code', 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;"
            " font-size: 14px;"
            " color: #d7dae0;"
            " padding: 6px 4px;"
            "} "
            "QTextBrowser { border: 1px solid #3e4451; background: #21252b; font-size: 14px; color: #d7dae0; }"
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
                " background:#1e3d28; color:#98c379; border:1px solid #2d5a3d; border-radius:5px; padding:3px 10px; font-weight:600; font-size:11px;"
                "} "
                "QPushButton:hover { background:#2d5a3d; }"
            )
            self.run_btn.clicked.connect(lambda: self.on_run and self.on_run(self))
            toolbar.addWidget(self.run_btn)

        _cell_btn_ss = (
            "QPushButton {"
            " background:#3e4451; color:#d7dae0; border:1px solid #4a5568; border-radius:5px; padding:3px 8px; font-size:11px;"
            "} "
            "QPushButton:hover { background:#4a5568; color:#d7dae0; }"
        )
        self.move_up_btn = QPushButton("Up", self)
        self.move_up_btn.setStyleSheet(_cell_btn_ss)
        self.move_up_btn.clicked.connect(lambda: self.on_move and self.on_move(self, -1))
        toolbar.addWidget(self.move_up_btn)
        self.move_down_btn = QPushButton("Down", self)
        self.move_down_btn.setStyleSheet(_cell_btn_ss)
        self.move_down_btn.clicked.connect(lambda: self.on_move and self.on_move(self, 1))
        toolbar.addWidget(self.move_down_btn)

        self.move_left_btn = QPushButton("Left", self)
        self.move_left_btn.setStyleSheet(_cell_btn_ss)
        self.move_left_btn.clicked.connect(lambda: self.on_move_column and self.on_move_column(self, "left"))
        toolbar.addWidget(self.move_left_btn)
        self.move_right_btn = QPushButton("Right", self)
        self.move_right_btn.setStyleSheet(_cell_btn_ss)
        self.move_right_btn.clicked.connect(lambda: self.on_move_column and self.on_move_column(self, "right"))
        toolbar.addWidget(self.move_right_btn)

        self.delete_btn = QPushButton("Delete", self)
        self.delete_btn.setStyleSheet(
            "QPushButton {"
            " background:rgba(220,38,38,0.12); color:#e06c75; border:1px solid rgba(220,38,38,0.3); border-radius:5px; padding:3px 8px; font-size:11px;"
            "} "
            "QPushButton:hover { background:#b60021; color:white; }"
        )
        self.delete_btn.clicked.connect(lambda: self.on_delete and self.on_delete(self))
        toolbar.addWidget(self.delete_btn)
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
                " font-family: 'Inter', sans-serif;"
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
        """Enable or disable move buttons based on the current column."""
        print(f"[debug][cell-widget] sync_column_buttons cell_id={self.cell_id!r} column={self.column!r}", flush=True)
        self.move_left_btn.setEnabled(self.column != "left")
        self.move_right_btn.setEnabled(self.column != "right")

    def set_column(self, column: str) -> None:
        """Update the stored column assignment for this cell."""
        self.column = column
        self._sync_column_buttons()

    def source(self) -> str:
        """Return the current source text from the cell editor."""
        return self.editor.toPlainText()

    def set_result(self, result: ExecutionResult) -> None:
        """Apply an execution result to the cell's inline and rich output areas."""
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

        if self.cell_type == "code" and not result.error and hasattr(self, "_on_run_success"):
            self._on_run_success(self.source(), result)

    def clear_output(self) -> None:
        """Clear all outputs currently associated with this cell."""
        print(f"[debug][cell-widget] clear_output cell_id={self.cell_id!r}", flush=True)
        self.last_result = None
        if self.cell_type == "code" and hasattr(self, "inline_result"):
            self.inline_result.setPlainText("")
        self.output_area.clear_outputs()
        self.output_area.hide()

    def toggle_preview(self) -> None:
        """Toggle a markdown cell between editor and rendered preview modes."""
        if self.cell_type != "markdown":
            return
        showing_preview = self.preview.isHidden()
        print(f"[debug][cell-widget] toggle_preview cell_id={self.cell_id!r} preview={showing_preview}", flush=True)
        if showing_preview:
            self._show_markdown_preview()
        else:
            self._show_markdown_editor()

    def _show_markdown_preview(self) -> None:
        """Render markdown source and show the preview widget."""
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
        """Return a markdown cell from preview mode back to edit mode."""
        print(f"[debug][cell-widget] show_markdown_editor cell_id={self.cell_id!r}", flush=True)
        self.preview.hide()
        self.editor.show()
        self.preview_mode = False
        self.toggle_btn.setText("Preview")
