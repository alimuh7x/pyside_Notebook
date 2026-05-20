from __future__ import annotations

import math as _math
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from pyside_app.controls import AutoCloseComboBox
from pyside_app.notebook_plot_panel import QuickGraphPreviewPanel


PRIMARY_BTN = (
    "QPushButton { background:#1e3d28; color:#98c379; border:1px solid #2d5a3d; border-radius:6px; padding:2px 10px; "
    "min-height:24px; max-height:24px; font-weight:700; } "
    "QPushButton:hover { background:#2d5a3d; }"
)
SECONDARY_BTN = (
    "QPushButton { background:#374151; color:white; border-radius:6px; padding:2px 10px; "
    "min-height:24px; max-height:24px; } "
    "QPushButton:hover { background:#1f2937; }"
)
LIGHT_BTN = (
    "QPushButton { background:#3e4451; color:#d7dae0; border-radius:6px; padding:2px 10px; "
    "min-height:24px; max-height:24px; } "
    "QPushButton:hover { background:#4a5568; }"
)


class NotebookToolbar(QWidget):
    """Toolbar-only widget for notebook actions and status display."""

    run_all_requested = Signal()
    restart_requested = Signal()
    autosave_requested = Signal()
    save_requested = Signal()
    save_example_requested = Signal()
    open_requested = Signal()
    functions_requested = Signal()
    markdown_help_requested = Signal()

    def __init__(self, status_style: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][notebook-toolbar] init:start", flush=True)
        root = QHBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(0, 0, 0, 0)

        left_buttons = [
            ("run_all_button", "Run All", self.run_all_requested.emit, PRIMARY_BTN),
            ("restart_button", "Restart Kernel", self.restart_requested.emit, SECONDARY_BTN),
            ("autosave_button", "Autosave", self.autosave_requested.emit, LIGHT_BTN),
            ("save_button", "Save", self.save_requested.emit, LIGHT_BTN),
            ("save_example_button", "Save Example", self.save_example_requested.emit, LIGHT_BTN),
            ("open_button", "Open", self.open_requested.emit, LIGHT_BTN),
        ]
        right_buttons = [
            ("functions_toggle_button", "Functions", self.functions_requested.emit, LIGHT_BTN),
            ("markdown_toggle_button", "Markdown Help", self.markdown_help_requested.emit, LIGHT_BTN),
        ]

        for attr, text, slot, style in left_buttons:
            print(f"[debug][notebook-toolbar] add_left attr={attr}", flush=True)
            button = self._create_toolbar_button(text, slot, style)
            setattr(self, attr, button)
            root.addWidget(button)

        root.addStretch()

        for attr, text, slot, style in right_buttons:
            print(f"[debug][notebook-toolbar] add_right attr={attr}", flush=True)
            button = self._create_toolbar_button(text, slot, style)
            setattr(self, attr, button)
            root.addWidget(button)

        self.status_label = QLabel("Ready", self)
        self.status_label.setStyleSheet(status_style)
        root.addWidget(self.status_label)
        print("[debug][notebook-toolbar] init:done", flush=True)

    def _toolbar_button(self, label: str, handler: object, style: str) -> QPushButton:
        print(f"[debug][notebook-toolbar] toolbar_button label={label!r}", flush=True)
        button = QPushButton(label, self)
        button.setStyleSheet(style + " QPushButton { font-size:14px; font-weight:700; }")
        button.clicked.connect(handler)
        return button

    def _create_toolbar_button(self, text: str, slot: object, style: str) -> QPushButton:
        print(f"[debug][notebook-toolbar] create_toolbar_button text={text!r}", flush=True)
        return self._toolbar_button(text, slot, style)

    def button(self, name: str) -> QPushButton:
        print(f"[debug][notebook-toolbar] button name={name!r}", flush=True)
        return getattr(self, name)


#-------------------------------------------------------------------------------------------------------------------------
# -- Note: SidebarWidgetget
#-------------------------------------------------------------------------------------------------------------------------

class SidebarWidget(QFrame):
    """Sidebar showing only the live variables browser."""

    def __init__(
        self,
        text_style: str,
        heading_style: str,
        namespace_skip: set[str],
        summarize_value: Callable[[object], tuple[str, str]],
        combo_style: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        print("[debug][sidebar-widget] init:start", flush=True)
        self._namespace_skip = namespace_skip
        self._summarize_value = summarize_value
        self._parameter_specs: dict[str, dict[str, Any]] = {}
        self._parameter_source = ""
        self._parameter_rerun_fn: Callable | None = None
        self._parameter_overrides: dict[str, Any] = {}
        self._parameter_steps: dict[str, float] = {}
        self.parameter_value_edits: dict[str, QLineEdit] = {}
        self.parameter_minus_buttons: dict[str, QPushButton] = {}
        self.parameter_plus_buttons: dict[str, QPushButton] = {}
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background:#282c34; border:1px solid #3e4451; }"
            "QLabel { background:transparent; border:none; " + text_style + " }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.examples_bar = self._build_examples_row(combo_style)
        layout.addWidget(self.examples_bar)

        self.variables_panel = self._build_variables_panel(heading_style)
        layout.addWidget(self.variables_panel, 1)

        print("[debug][sidebar-widget] init:done", flush=True)

    def _build_examples_row(self, combo_style: str) -> "ExamplesBar":
        bar = ExamplesBar(combo_style=combo_style, parent=self)
        return bar

    def _build_variables_panel(self, heading_style: str) -> QWidget:
        print("[debug][sidebar-widget] build_variables_panel", flush=True)
        wrapper = QWidget(self)
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 6, 0, 0)
        wrapper_layout.setSpacing(8)
        variables_header = QHBoxLayout()
        variables_label = QLabel("Variables", wrapper)
        variables_label.setStyleSheet(heading_style)
        variables_header.addWidget(variables_label)
        variables_header.addStretch(1)
        self.parameter_reset_all_button = QPushButton("Reset all", wrapper)
        self.parameter_reset_all_button.setFixedHeight(24)
        self.parameter_reset_all_button.setStyleSheet(LIGHT_BTN)
        self.parameter_reset_all_button.clicked.connect(self._reset_all_parameters)
        variables_header.addWidget(self.parameter_reset_all_button)
        self.parameter_run_button = QPushButton("Run", wrapper)
        self.parameter_run_button.setFixedHeight(24)
        self.parameter_run_button.setStyleSheet(PRIMARY_BTN)
        self.parameter_run_button.clicked.connect(self._run_current_parameters)
        variables_header.addWidget(self.parameter_run_button)
        wrapper_layout.addLayout(variables_header)

        self.variables_browser = VariablesParameterTable(wrapper)
        self.variables_browser.setMinimumHeight(360)
        self.variables_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.variables_browser.setStyleSheet(
            "QTableWidget {"
            " border:1px solid #3e4451;"
            " background:#282c34;"
            " font-family:'Inter', sans-serif;"
            " font-size:12px;"
            " color:#d7dae0;"
            "}"
            "QHeaderView::section {"
            " background:#2c313a; color:#e5c07b; border:1px solid #3e4451;"
            " padding:5px 7px; font-size:14px; font-weight:700;"
            "}"
            "QTableWidget::item { border-bottom:1px solid #3e4451; padding:4px 7px; }"
        )
        wrapper_layout.addWidget(self.variables_browser, 1)
        self._refresh_variables_panel({})
        return wrapper

    @staticmethod
    def _fmt_parameter_value(value: float, is_int: bool, step: float = 0.0) -> str:
        if is_int:
            return str(int(round(value)))
        if step > 0:
            decimals = max(0, -int(_math.floor(_math.log10(step))))
            text = f"{value:.{decimals}f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return text
        if abs(value) >= 1e4 or (abs(value) < 1e-3 and value != 0.0):
            return f"{value:.3g}"
        return f"{value:.4g}"

    @staticmethod
    def _nice_parameter_step(lo: float, hi: float, is_int: bool, max_steps: int = 20) -> float:
        if is_int:
            raw = max(1.0, (hi - lo) / max_steps)
            return float(int(_math.ceil(raw)))
        span = hi - lo
        if span <= 0:
            return 0.1
        raw = span / 10.0
        exp = _math.floor(_math.log10(raw))
        nice = raw / (10 ** exp)
        if nice <= 1.0:
            rounded = 1.0
        elif nice <= 2.0:
            rounded = 2.0
        elif nice <= 5.0:
            rounded = 5.0
        else:
            rounded = 10.0
        step = rounded * (10 ** exp)
        while (hi - lo) / step > max_steps:
            if rounded == 1.0:
                rounded = 2.0
            elif rounded == 2.0:
                rounded = 5.0
            elif rounded == 5.0:
                rounded = 10.0
            else:
                rounded = 1.0
                exp += 1
            step = rounded * (10 ** exp)
        return step

    def set_parameter_controls(self, params: dict[str, dict[str, Any]], source: str, rerun_fn: Callable | None) -> None:
        print(f"[debug][sidebar-widget] set_parameter_controls count={len(params)} source_len={len(source)}", flush=True)
        self._parameter_specs = {name: dict(spec) for name, spec in params.items()}
        self._parameter_source = source
        self._parameter_rerun_fn = rerun_fn
        next_overrides: dict[str, Any] = {}
        next_steps: dict[str, float] = {}
        for name, spec in self._parameter_specs.items():
            default = spec["value"]
            next_overrides[name] = self._parameter_overrides.get(name, default)
            next_steps[name] = self._nice_parameter_step(float(spec["min"]), float(spec["max"]), bool(spec["is_int"]))
            print(
                f"[debug][sidebar-widget] parameter:init name={name!r} default={default!r} "
                f"pending={next_overrides[name]!r} step={next_steps[name]!r}",
                flush=True,
            )
        self._parameter_overrides = next_overrides
        self._parameter_steps = next_steps
        self._refresh_variables_panel({})

    def clear_parameter_controls(self) -> None:
        print("[debug][sidebar-widget] clear_parameter_controls", flush=True)
        self._parameter_specs = {}
        self._parameter_source = ""
        self._parameter_rerun_fn = None
        self._parameter_overrides = {}
        self._parameter_steps = {}
        self.parameter_value_edits.clear()
        self.parameter_minus_buttons.clear()
        self.parameter_plus_buttons.clear()
        self._refresh_variables_panel({})

    def _refresh_variables_panel(self, namespace: dict[str, object]) -> None:
        print("[debug][sidebar-widget] refresh_variables_panel", flush=True)
        print(f"[debug][sidebar-widget] refresh_variables_panel_namespace count={len(namespace)}", flush=True)
        rows_by_name: dict[str, str] = {}
        for name, value in sorted(namespace.items()):
            if name in self._namespace_skip or name.startswith("_"):
                continue
            if isinstance(value, ModuleType) or callable(value):
                continue
            if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
                continue
            summary, _type_name = self._summarize_value(value)
            rows_by_name[name] = summary
        for name, spec in self._parameter_specs.items():
            if name not in rows_by_name:
                step = self._parameter_steps.get(name, 0.0)
                rows_by_name[name] = self._fmt_parameter_value(float(spec["value"]), bool(spec["is_int"]), step)
        rows = [(name, rows_by_name[name]) for name in sorted(rows_by_name)]
        print(
            f"[debug][sidebar-widget] refresh_variables_panel rows={len(rows)} "
            f"params={list(self._parameter_specs)}",
            flush=True,
        )
        self._rebuild_variables_table(rows)

    def _rebuild_variables_table(self, rows: list[tuple[str, str]]) -> None:
        print(f"[debug][sidebar-widget] rebuild_variables_table rows={len(rows)}", flush=True)
        self.parameter_value_edits.clear()
        self.parameter_minus_buttons.clear()
        self.parameter_plus_buttons.clear()
        table = self.variables_browser
        table.clear()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Variable", "Value", "Parameters"])
        table.setRowCount(max(1, len(rows)))
        table.set_plain_rows([("Variable", "Value", "Parameters"), *rows])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table.verticalHeader().setVisible(False)
        if not rows:
            table.setItem(0, 0, QTableWidgetItem("Run code to see variables here."))
            table.setItem(0, 1, QTableWidgetItem(""))
            table.setItem(0, 2, QTableWidgetItem("-"))
            return
        for row_index, (name, summary) in enumerate(rows):
            print(
                f"[debug][sidebar-widget] rebuild_variables_table:row index={row_index} "
                f"name={name!r} is_param={name in self._parameter_specs}",
                flush=True,
            )
            name_item = QTableWidgetItem(name)
            name_item.setForeground(Qt.GlobalColor.white)
            value_item = QTableWidgetItem(summary)
            value_item.setForeground(Qt.GlobalColor.white)
            table.setItem(row_index, 0, name_item)
            table.setItem(row_index, 1, value_item)
            if name in self._parameter_specs:
                table.setCellWidget(row_index, 2, self._build_parameter_controls(name))
            else:
                dash_item = QTableWidgetItem("-")
                dash_item.setForeground(Qt.GlobalColor.white)
                table.setItem(row_index, 2, dash_item)

    def _build_parameter_controls(self, name: str) -> QWidget:
        spec = self._parameter_specs[name]
        step = self._parameter_steps.get(name, 0.0)
        pending = float(self._parameter_overrides.get(name, spec["value"]))
        row = QWidget(self.variables_browser)
        row.setStyleSheet("QWidget { background:transparent; border:none; }")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(3)
        button_style = (
            "QPushButton { font-size:15px; font-weight:700; padding:0 6px; border:1px solid #3e4451;"
            " border-radius:5px; background:#2c313a; color:#ffffff; min-width:24px; }"
            "QPushButton:hover { background:#1d4ed8; border-color:#1d4ed8; color:#ffffff; }"
        )
        minus_btn = QPushButton("-", row)
        minus_btn.setFixedSize(24, 24)
        minus_btn.setStyleSheet(button_style)
        minus_btn.clicked.connect(lambda _=False, n=name: self._step_parameter(n, -1))
        layout.addWidget(minus_btn)
        edit = QLineEdit(row)
        edit.setFixedWidth(72)
        edit.setText(self._fmt_parameter_value(pending, bool(spec["is_int"]), step))
        edit.setStyleSheet(
            "QLineEdit { font-family:Consolas, monospace; font-size:14px; font-weight:700; color:#61afef;"
            " background:#2c313a; border:1px solid #3e4451; border-radius:5px; padding:1px 5px; }"
            "QLineEdit:focus { background:#282c34; border-color:#61afef; }"
        )
        edit.editingFinished.connect(lambda n=name, e=edit: self._on_parameter_edit_finished(n, e))
        layout.addWidget(edit)
        plus_btn = QPushButton("+", row)
        plus_btn.setFixedSize(24, 24)
        plus_btn.setStyleSheet(button_style)
        plus_btn.clicked.connect(lambda _=False, n=name: self._step_parameter(n, 1))
        layout.addWidget(plus_btn)
        self.parameter_minus_buttons[name] = minus_btn
        self.parameter_value_edits[name] = edit
        self.parameter_plus_buttons[name] = plus_btn
        print(
            f"[debug][sidebar-widget] parameter_controls:built name={name!r} pending={pending!r} step={step!r}",
            flush=True,
        )
        return row

    def _set_parameter_value(self, name: str, value: float) -> None:
        spec = self._parameter_specs.get(name)
        if spec is None:
            print(f"[debug][sidebar-widget] parameter:set skipped_missing name={name!r}", flush=True)
            return
        lo, hi = float(spec["min"]), float(spec["max"])
        step = self._parameter_steps.get(name, 0.0)
        if value > hi:
            spec["max"] = value
            print(
                f"[debug][sidebar-widget] parameter:expand_max name={name!r} old_max={hi!r} new_max={value!r}",
                flush=True,
            )
        elif value < lo:
            spec["min"] = value
            print(
                f"[debug][sidebar-widget] parameter:expand_min name={name!r} old_min={lo!r} new_min={value!r}",
                flush=True,
            )
        if spec["is_int"]:
            value = float(int(round(value)))
        elif step > 0:
            decimals = max(0, -int(_math.floor(_math.log10(step))))
            value = round(value, decimals)
        self._parameter_overrides[name] = value
        text = self._fmt_parameter_value(value, bool(spec["is_int"]), step)
        edit = self.parameter_value_edits.get(name)
        if edit is not None:
            edit.setText(text)
        print(f"[debug][sidebar-widget] parameter:set name={name!r} value={value!r} text={text!r}", flush=True)

    def _step_parameter(self, name: str, direction: int) -> None:
        current = float(self._parameter_overrides.get(name, self._parameter_specs[name]["value"]))
        step = self._parameter_steps.get(name, 1.0)
        next_value = current + direction * step
        print(
            f"[debug][sidebar-widget] parameter:step name={name!r} direction={direction} "
            f"current={current!r} step={step!r} next={next_value!r}",
            flush=True,
        )
        self._set_parameter_value(name, next_value)

    def _on_parameter_edit_finished(self, name: str, edit: QLineEdit) -> None:
        print(f"[debug][sidebar-widget] parameter:edit_finished name={name!r} text={edit.text()!r}", flush=True)
        try:
            value = float(edit.text())
        except ValueError:
            self._set_parameter_value(name, float(self._parameter_overrides.get(name, self._parameter_specs[name]["value"])))
            return
        self._set_parameter_value(name, value)

    def _reset_all_parameters(self) -> None:
        print(f"[debug][sidebar-widget] parameter:reset_all count={len(self._parameter_specs)}", flush=True)
        for name, spec in self._parameter_specs.items():
            self._set_parameter_value(name, float(spec["value"]))
        print("[debug][sidebar-widget] parameter:reset_all:done", flush=True)

    def _run_current_parameters(self) -> None:
        print(
            f"[debug][sidebar-widget] parameter:run_start count={len(self._parameter_overrides)} "
            f"source_len={len(self._parameter_source)}",
            flush=True,
        )
        if self._parameter_rerun_fn is None:
            print("[debug][sidebar-widget] parameter:run_skipped reason='no_rerun_fn'", flush=True)
            return
        active = dict(self._parameter_overrides)
        self._parameter_rerun_fn(self._parameter_source, active, self._on_parameter_result)

    def _on_parameter_result(self, result: object) -> None:
        error = getattr(result, "error", None)
        print(f"[debug][sidebar-widget] parameter:run_result error={bool(error)}", flush=True)


class VariablesParameterTable(QTableWidget):
    """Interactive variables table with a QTextBrowser-like text export."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plain_rows: list[tuple[str, ...]] = []
        self.setShowGrid(False)
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def set_plain_rows(self, rows: list[tuple[str, ...]]) -> None:
        self._plain_rows = rows

    def toPlainText(self) -> str:
        text = "\n".join(" ".join(str(part) for part in row if part) for row in self._plain_rows)
        print(f"[debug][variables-parameter-table] toPlainText length={len(text)}", flush=True)
        return text

#-------------------------------------------------------------------------------------------------------------------------
# -- Note: ExamplesBar
#-------------------------------------------------------------------------------------------------------------------------

class ExamplesBar(QWidget):
    """Compact top-right bar: example dropdown + Insert button."""

    insert_example_requested = Signal()

    def __init__(
        self,
        combo_style: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.examples: list[object] = []
        self.setStyleSheet("background:#282c34;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self.example_combo = AutoCloseComboBox(self)
        self.example_combo.setStyleSheet(combo_style)
        self.example_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.example_combo, 1)

        self.insert_btn = QPushButton("Insert", self)
        self.insert_btn.setStyleSheet(
            "QPushButton { background:#1a2f42; color:#61afef; border:1px solid #2a4a60; border-radius:6px; padding:5px 14px;"
            " font-weight:600; font-size:14px; min-height:26px; }"
            "QPushButton:hover { background:#2a4a60; }"
        )
        self.insert_btn.clicked.connect(self.insert_example_requested.emit)
        layout.addWidget(self.insert_btn)

    def set_examples(self, examples: Sequence[object], select_title: str | None = None) -> None:
        self.examples = list(examples)
        self.example_combo.blockSignals(True)
        self.example_combo.clear()

        # non-selectable "Examples" header at index 0
        self.example_combo.addItem("Examples", None)
        header_item = self.example_combo.model().item(0)
        if header_item is not None:
            header_item.setFlags(header_item.flags() & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsSelectable)
            font = header_item.font()
            font.setBold(True)
            header_item.setFont(font)

        selected_index = 1
        for i, ex in enumerate(self.examples):
            self.example_combo.addItem(ex.title, ex.id)  # type: ignore[attr-defined]
            if select_title and ex.title == select_title:  # type: ignore[attr-defined]
                selected_index = i + 1

        self.example_combo.blockSignals(False)
        if self.example_combo.count() > 1:
            self.example_combo.setCurrentIndex(selected_index)


#-------------------------------------------------------------------------------------------------------------------------
# -- Note: HelpPanelsWidget
#-------------------------------------------------------------------------------------------------------------------------

class HelpPanelsWidget(QWidget):
    """Container for notebook help panels."""

    def __init__(self, functions_html: str, markdown_html: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][help-panels] init:start", flush=True)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        self.setMaximumWidth(460)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.functions_panel = QTextBrowser(self)
        self.functions_panel.setHtml(functions_html)
        self.functions_panel.hide()
        self.functions_panel.setMinimumHeight(180)
        self.functions_panel.setStyleSheet("QTextBrowser { font-size:14px; color:#d7dae0; background:#282c34; }")
        root.addWidget(self.functions_panel)

        self.markdown_help_panel = QTextBrowser(self)
        self.markdown_help_panel.setHtml(markdown_html)
        self.markdown_help_panel.hide()
        self.markdown_help_panel.setMinimumHeight(180)
        self.markdown_help_panel.setStyleSheet("QTextBrowser { font-size:14px; color:#d7dae0; background:#282c34; }")
        root.addWidget(self.markdown_help_panel)
        print("[debug][help-panels] init:done", flush=True)

    def toggle_functions_panel(self) -> None:
        visible = not self.functions_panel.isVisible()
        print(f"[debug][help-panels] toggle_functions visible={visible}", flush=True)
        self.functions_panel.setVisible(visible)

    def toggle_markdown_help(self) -> None:
        visible = not self.markdown_help_panel.isVisible()
        print(f"[debug][help-panels] toggle_markdown_help visible={visible}", flush=True)
        self.markdown_help_panel.setVisible(visible)

#-------------------------------------------------------------------------------------------------------------------------
# -- Note: NotebookColumnsWidget
#-------------------------------------------------------------------------------------------------------------------------

class NotebookColumnsWidget(QWidget):
    """Widget responsible for the notebook cell column layout."""

    add_code_requested = Signal(str)
    add_markdown_requested = Signal(str)
    insert_example_requested = Signal(str)

    def __init__(
        self,
        heading_style: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        print("[debug][columns-widget] init:start", flush=True)
        self.setStyleSheet("background:#282c34;")
        wrapper_layout = QVBoxLayout(self)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(8)
        header = QHBoxLayout()
        label = QLabel("Notebook", self)
        label.setStyleSheet(heading_style)
        header.addWidget(label)
        header.addStretch(1)
        header.addWidget(
            self._toolbar_button(
                "Add Code",
                self._request_add_code,
                "QPushButton { background:#2c313a; color:#d7dae0; border-radius:6px; padding:4px 10px; font-weight:600; } "
                "QPushButton:hover { background:#3e4451; }",
            )
        )
        header.addWidget(
            self._toolbar_button(
                "Add Markdown",
                self._request_add_markdown,
                "QPushButton { background:#3e4451; color:#d7dae0; border-radius:6px; padding:4px 10px; } "
                "QPushButton:hover { background:#4a5568; }",
            )
        )
        wrapper_layout.addLayout(header)

        self.container = QWidget(self)
        self.container.setStyleSheet("background:#282c34;")
        print("[debug][columns-widget] build_column container_style column='left' background='#ffffff'", flush=True)
        self.cells_layout = QVBoxLayout(self.container)
        self.cells_layout.setContentsMargins(0, 0, 0, 0)
        self.cells_layout.setSpacing(8)
        self.cells_layout.addStretch(1)
        wrapper_layout.addWidget(self.container, 1)
        print("[debug][columns-widget] init:done", flush=True)

    def _toolbar_button(self, label: str, handler: object, style: str) -> QPushButton:
        print(f"[debug][columns-widget] toolbar_button label={label!r}", flush=True)
        button = QPushButton(label, self)
        button.setStyleSheet(style + " QPushButton { font-size:14px; font-weight:700; }")
        button.clicked.connect(handler)
        return button

    def _request_add_code(self) -> None:
        print("[debug][columns-widget] request_add_code column='left'", flush=True)
        self.add_code_requested.emit("left")

    def _request_add_markdown(self) -> None:
        print("[debug][columns-widget] request_add_markdown column='left'", flush=True)
        self.add_markdown_requested.emit("left")

    def _request_insert_example(self) -> None:
        print("[debug][columns-widget] request_insert_example column='left'", flush=True)
        self.insert_example_requested.emit("left")

    def rebuild_cells(self, cells: Sequence[QWidget]) -> None:
        print(f"[debug][columns-widget] rebuild_cells count={len(cells)}", flush=True)
        while self.cells_layout.count():
            item = self.cells_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for cell in cells:
            self.cells_layout.addWidget(cell)
        self.cells_layout.addStretch(1)


class GraphPanelWidget(QWidget):
    """Right-column graph container: holds one or more QuickGraphPreviewPanel cards."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][graph-panel-widget] init:start", flush=True)
        self._namespace: dict[str, Any] = {}
        self._current_label = "baseline"
        self._panels: list[QuickGraphPreviewPanel] = []
        self._cards: list[QWidget] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)

        # "Add Graph Panel" lives in whichever panel header is current
        self._add_btn = QPushButton("+ Add Graph Panel", self)
        self._add_btn.setStyleSheet(
            "QPushButton { background:#eff6ff; color:#1d4ed8; border:1px solid #93c5fd;"
            " border-radius:6px; padding:4px 12px; font-weight:600; font-size:14px; }"
            "QPushButton:hover { background:#dbeafe; }"
        )
        self._add_btn.clicked.connect(self._add_panel)

        # ── panels ────────────────────────────────────────────────────────────
        self._panels_layout = QVBoxLayout()
        self._panels_layout.setContentsMargins(0, 0, 0, 0)
        self._panels_layout.setSpacing(10)
        root.addLayout(self._panels_layout)
        root.addStretch(1)

        self._add_panel()
        self.quick_preview_panel = self._panels[0]   # backward compat
        print("[debug][graph-panel-widget] init:done", flush=True)

    # ── internal ──────────────────────────────────────────────────────────────

    def _add_panel(self) -> None:
        n = len(self._panels) + 1
        panel = QuickGraphPreviewPanel(self)
        if self._namespace:
            panel.set_namespace(self._namespace)
            panel.set_current_label(self._current_label)
            panel.load_saved_runs_from_store()

        # ── card wrapper ──────────────────────────────────────────────────────
        card = QFrame(self)
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet(
            "QFrame { background:#282c34; border:1px solid #3e4451; border-radius:8px; }"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        # header
        hdr = QWidget(card)
        hdr.setStyleSheet(
            "QWidget { background:#2c313a; border-bottom:1px solid #3e4451;"
            " border-top-left-radius:8px; border-top-right-radius:8px; }"
            "QLabel { background:transparent; border:none; }"
            "QPushButton { background:transparent; border:none; }"
        )
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(10, 4, 6, 4)
        hdr_layout.setSpacing(6)

        title_lbl = QLabel(f"Graph Panel {n}", hdr)
        title_lbl.setStyleSheet("font-weight:700; font-size:14px; color:#e5c07b; background:transparent; border:none;")
        hdr_layout.addWidget(title_lbl)
        hdr_layout.addStretch(1)

        _btn_ss = (
            "QPushButton { font-size:14px; padding:2px 10px; border:1px solid #3e4451;"
            " border-radius:6px; background:#3e4451; color:#d7dae0; font-weight:500; }"
            "QPushButton:hover { background:#4a5568; border-color:#61afef; color:#61afef; }"
        )

        clear_btn = QPushButton("Clear history", hdr)
        clear_btn.setFixedHeight(22)
        clear_btn.setStyleSheet(_btn_ss)
        clear_btn.clicked.connect(panel.clear_history)
        hdr_layout.addWidget(clear_btn)

        dl_btn = QPushButton("Download PNG", hdr)
        dl_btn.setFixedHeight(22)
        dl_btn.setStyleSheet(_btn_ss)
        dl_btn.clicked.connect(panel.download_png)
        hdr_layout.addWidget(dl_btn)

        # migrate "+ Add Graph Panel" into this panel's header
        self._add_btn.setParent(hdr)
        hdr_layout.addWidget(self._add_btn)

        remove_btn = QPushButton("✕", hdr)
        remove_btn.setFixedSize(20, 20)
        remove_btn.setStyleSheet(
            "QPushButton { color:#64748b; font-size:13px; font-weight:700; background:transparent; border:none; }"
            "QPushButton:hover { color:#dc2626; }"
        )
        remove_btn.clicked.connect(lambda: self._remove_panel(card, panel))
        hdr_layout.addWidget(remove_btn)
        card_layout.addWidget(hdr)

        # panel body
        card_layout.addWidget(panel)

        self._panels_layout.addWidget(card)
        self._panels.append(panel)
        self._cards.append(card)
        self._update_remove_buttons()

    def _remove_panel(self, card: QWidget, panel: "QuickGraphPreviewPanel") -> None:
        if len(self._panels) <= 1:
            return
        idx = self._panels.index(panel)
        self._panels.pop(idx)
        self._cards.pop(idx)
        card.hide()
        card.deleteLater()
        self.quick_preview_panel = self._panels[0]
        self._update_remove_buttons()
        self._renumber_panels()

    def _update_remove_buttons(self) -> None:
        only_one = len(self._cards) <= 1
        for card in self._cards:
            hdr = card.layout().itemAt(0).widget()
            if hdr:
                btn = hdr.findChild(QPushButton)
                if btn:
                    btn.setEnabled(not only_one)

    def _renumber_panels(self) -> None:
        for i, card in enumerate(self._cards):
            hdr = card.layout().itemAt(0).widget()
            if hdr:
                lbl = hdr.findChild(QLabel)
                if lbl:
                    lbl.setText(f"Graph Panel {i + 1}")

    # ── public API ────────────────────────────────────────────────────────────

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        print(f"[debug][graph-panel-widget] set_namespace count={len(namespace)}", flush=True)
        self._namespace = namespace
        self._current_label = "baseline"
        for panel in self._panels:
            panel.set_namespace(namespace)

    def set_current_label(self, label: str) -> None:
        """Override the label applied to the current namespace snapshot in all panels."""
        self._current_label = label or "baseline"
        for panel in self._panels:
            panel.set_current_label(self._current_label)

    def add_run(self, namespace_snapshot: dict[str, Any], label: str) -> None:
        """Record a slider run in all panels for history trace overlay."""
        self._namespace = dict(namespace_snapshot)
        self._current_label = label or "current"
        for panel in self._panels:
            panel.add_run(namespace_snapshot, label)

    def clear_history(self) -> None:
        """Clear run history from all panels. Called on kernel restart."""
        for panel in self._panels:
            panel.clear_history()

    def set_latest_plot(self, title: str, html_value: str) -> None:
        print(
            f"[debug][graph-panel-widget] set_latest_plot title={title!r} html_length={len(html_value)}",
            flush=True,
        )
        self.quick_preview_panel.set_latest_plot(title, html_value)


def _fmt_scalar(v: float) -> str:
    """Format a scalar value compactly."""
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0.0):
        return f"{v:.3g}"
    return f"{v:.4g}"


def summarize_namespace_value(value: object) -> tuple[str, str]:
    """Return a rich summary for one namespace value."""
    if isinstance(value, np.ndarray):
        dtype_name = str(value.dtype)
        shape_str = "×".join(str(d) for d in value.shape) if value.ndim > 0 else "scalar"
        if value.size == 0:
            return f"[empty] shape=({shape_str}) dtype={dtype_name}", "ndarray"
        try:
            flat = value.ravel().astype(float)
            mn, mx, mean = float(flat.min()), float(flat.max()), float(flat.mean())
            stats = f"min={_fmt_scalar(mn)}  max={_fmt_scalar(mx)}  mean={_fmt_scalar(mean)}"
        except Exception:
            stats = ""
        summary = f"({shape_str}) {dtype_name}"
        if stats:
            summary += f"  ·  {stats}"
        return summary, "ndarray"
    if isinstance(value, pd.DataFrame):
        rows, cols = value.shape
        col_names = ", ".join(str(c) for c in value.columns[:6])
        if len(value.columns) > 6:
            col_names += ", …"
        return f"{rows}×{cols}  [{col_names}]", "DataFrame"
    if isinstance(value, pd.Series):
        try:
            stats = f"min={_fmt_scalar(float(value.min()))}  max={_fmt_scalar(float(value.max()))}"
        except Exception:
            stats = ""
        return f"len={len(value)} dtype={value.dtype}" + (f"  ·  {stats}" if stats else ""), "Series"
    if isinstance(value, (list, tuple)):
        if len(value) <= 6 and all(not isinstance(item, (list, tuple, dict)) for item in value):
            return repr(value), type(value).__name__
        return f"{type(value).__name__} len={len(value)}", type(value).__name__
    if isinstance(value, dict):
        keys = list(value.keys())
        if len(keys) <= 4:
            preview = ", ".join(str(key) for key in keys)
            return "{" + preview + "}", "dict"
        return f"dict keys={len(keys)}", "dict"
    if isinstance(value, str):
        return (value if len(value) <= 40 else value[:37] + "..."), "str"
    if isinstance(value, complex):
        return str(value), "complex"
    if isinstance(value, bool):
        return str(value), "bool"
    if isinstance(value, int):
        return str(value), "int"
    if isinstance(value, float):
        return _fmt_scalar(value), "float"
    if isinstance(value, go.Figure):
        n_traces = len(value.data)
        return f"Plotly Figure  {n_traces} trace{'s' if n_traces != 1 else ''}", "plot"
    return type(value).__name__, type(value).__name__
