from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QElapsedTimer
from PySide6.QtCore import QModelIndex
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtTest import QTest

from main import configure_desktop_graphics
from pyside_app.controls import AutoCloseComboBox, CheckableComboBox
from pyside_app.main_window import MainWindow
from pyside_app.execution_engine import ExecutionOutput, ExecutionResult
from pyside_app.markdown_preview import build_markdown_preview_html, katex_assets_dir, MarkdownPreview
from pyside_app.notebook_tab import NotebookTab
from pyside_app.plot_view import PlotView


class FakeLspClient(QObject):
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
        print("[debug][fake-lsp] start", flush=True)
        self.started = True
        self.availability_changed.emit(True)

    def is_available(self) -> bool:
        print("[debug][fake-lsp] is_available", flush=True)
        return True

    def open_document(self, uri: str, text: str) -> None:
        print(f"[debug][fake-lsp] open_document uri={uri!r} text={text!r}", flush=True)
        self.opened_documents.append((uri, text))

    def change_document(self, uri: str, text: str) -> None:
        print(f"[debug][fake-lsp] change_document uri={uri!r} text={text!r}", flush=True)
        self.changed_documents.append((uri, text))

    def request_completion(self, uri: str, line: int, character: int) -> None:
        print(
            f"[debug][fake-lsp] request_completion uri={uri!r} line={line} character={character}",
            flush=True,
        )
        self.requested_completions.append((uri, line, character))


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_exposes_tabbed_notebook_and_graphs_shell():
    _app()

    window = MainWindow()

    assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert window.centralWidget() is window.window_shell
    assert window.title_bar.window() is window
    assert window.content_layout.indexOf(window.workspace_tabs) >= 0
    assert window.workspace_tabs.count() == 2
    assert window.workspace_tabs.tabText(0) == "Notebook"
    assert window.workspace_tabs.tabText(1) == "Graphs"
    assert window.windowTitle() == "Calculation Notebook Desktop"


def test_notebook_tab_uses_workspace_scroll_instead_of_inner_cell_scroll_area():
    _app()

    tab = NotebookTab()

    assert hasattr(tab, "workspace_scroll")
    assert tab.workspace_scroll.widget() is tab.workspace_content
    assert not hasattr(tab, "left_scroll")


def test_quick_preview_header_and_controls_do_not_expand_vertically():
    _app()

    tab = NotebookTab()
    tab.resize(1400, 1000)
    tab.show()
    QApplication.processEvents()

    quick = tab.quick_preview_panel

    assert quick.status_label.height() <= 40
    assert quick.series_controls.height() <= 120


def test_main_window_title_bar_exposes_window_controls_and_labels():
    _app()

    window = MainWindow()

    assert window.title_bar.title_label.text() == "calculationNotebook"
    assert window.title_bar.subtitle_label.text() == "Desktop"
    assert not window.title_bar.minimize_button.icon().isNull()
    assert not window.title_bar.maximize_button.icon().isNull()
    assert not window.title_bar.close_button.icon().isNull()


def test_title_bar_passive_labels_do_not_intercept_drag_events():
    _app()

    window = MainWindow()

    assert window.title_bar.icon_label.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert window.title_bar.title_label.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert window.title_bar.subtitle_label.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def test_title_bar_prefers_native_system_move_when_available():
    _app()

    window = MainWindow()
    calls: list[str] = []

    class FakeWindowHandle:
        def startSystemMove(self) -> bool:
            calls.append("startSystemMove")
            return True

    window.title_bar._window_handle = lambda: FakeWindowHandle()

    assert window.title_bar._begin_system_move() is True
    assert calls == ["startSystemMove"]


def test_notebook_tab_runs_code_through_shared_engine():
    _app()
    tab = NotebookTab()

    result = tab.execution_engine.execute("value = 5\nvalue * 8")

    assert result.error is None
    assert result.outputs[-1].data["text"] == "40"


def test_code_editor_tab_inserts_four_spaces():
    _app()
    tab = NotebookTab()
    editor = tab.cells[0].editor
    editor.setFocus()
    editor.setPlainText("")

    QTest.keyClick(editor, Qt.Key.Key_Tab)

    assert editor.toPlainText() == "    "


def test_code_editor_shift_tab_unindents_selected_lines():
    _app()
    tab = NotebookTab()
    editor = tab.cells[0].editor
    editor.setPlainText("    alpha\n    beta")
    cursor = editor.textCursor()
    cursor.setPosition(0)
    cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)

    QTest.keyClick(editor, Qt.Key.Key_Backtab)

    assert editor.toPlainText() == "alpha\nbeta"


def test_notebook_tab_wires_code_editors_to_lsp_and_namespace_words():
    _app()
    fake_lsp = FakeLspClient()
    tab = NotebookTab(lsp_client=fake_lsp)
    cell = tab.cells[0]

    assert fake_lsp.started is True
    assert fake_lsp.opened_documents
    assert cell.editor.document_uri.startswith("file:///desktop-notebook/")
    assert "np" in cell.editor.completion_words()

    tab.execution_engine.execute("alpha = 42")
    tab._refresh_completion_words()

    assert "alpha" in cell.editor.completion_words()


def test_code_editor_applies_diagnostics_from_lsp():
    _app()
    fake_lsp = FakeLspClient()
    tab = NotebookTab(lsp_client=fake_lsp)
    cell = tab.cells[0]
    cell.editor.setPlainText("bad =\n")

    fake_lsp.diagnostics_ready.emit(
        cell.editor.document_uri,
        [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 3},
                },
                "message": "Syntax error",
            }
        ],
    )
    QApplication.processEvents()

    assert cell.editor.diagnostic_messages() == ["Syntax error"]
    assert len(cell.editor.extraSelections()) == 1


def test_code_editor_completer_popup_uses_light_style():
    _app()
    tab = NotebookTab()
    editor = tab.cells[0].editor

    popup_style = editor._completer.popup().styleSheet()

    assert "background:#ffffff" in popup_style
    assert "border:1px solid #d1dce8" in popup_style


def test_notebook_tab_serializes_cell_outputs_into_document():
    _app()
    tab = NotebookTab()
    cell = tab.cells[0]
    result = tab.execution_engine.execute("6 * 7")
    cell.set_result(result)

    document = tab.to_document()

    assert document.cells[0]["outputs"]
    assert document.cells[0]["outputs"][0]["kind"] == "value"


def test_notebook_tab_load_document_restores_saved_outputs():
    _app()
    tab = NotebookTab()
    tab.load_document(
        tab.storage.default_document().__class__(
            cells=[
                {
                    "id": "cell-1",
                    "type": "code",
                    "source": "x = 1",
                    "outputs": [{"kind": "value", "data": {"text": "1"}}],
                }
            ],
            metadata={"version": 6},
        )
    )

    assert tab.cells[0].last_result is not None
    assert tab.cells[0].last_result.outputs[0].data["text"] == "1"


def test_notebook_tab_load_document_restores_saved_plot_preview():
    _app()
    tab = NotebookTab()
    tab.load_document(
        tab.storage.default_document().__class__(
            cells=[
                {
                    "id": "cell-1",
                    "type": "code",
                    "source": "import plotly.graph_objects as go\nfig = go.Figure()\nfig",
                    "outputs": [{"kind": "plotly", "data": {"html": "<div>saved-plot</div>", "text": "Plotly Figure"}}],
                }
            ],
            metadata={"version": 6},
        )
    )

    assert "saved-plot" in tab.graph_state.latest_plot_html
    assert "latest executed notebook plot" in tab.graph_workspace.cards()[0].status_label.text().lower()


def test_notebook_tab_can_reorder_cells():
    _app()
    tab = NotebookTab()
    first_id = tab.cells[0].cell_id
    tab.add_code_cell()
    second_id = tab.cells[1].cell_id

    tab.move_cell(tab.cells[1], -1)

    assert tab.cells[0].cell_id == second_id
    assert tab.cells[1].cell_id == first_id


def test_notebook_tab_autosave_writes_document(tmp_path):
    _app()
    tab = NotebookTab()
    tab.cells[0].editor.setPlainText("autosave_value = 123")
    path = tmp_path / "autosave.json"

    tab.autosave_now(path)

    assert path.exists()
    assert "autosave_value = 123" in path.read_text(encoding="utf-8")


def test_variables_panel_shows_scalar_assignments():
    _app()
    tab = NotebookTab()
    tab.execution_engine.execute("answer = 42")
    tab._refresh_variables_panel()

    variables_text = tab.variables_browser.toPlainText()
    assert "answer" in variables_text
    assert "42" in variables_text


def test_code_cell_has_visual_accent_and_python_highlighter():
    _app()
    tab = NotebookTab()
    cell = tab.cells[0]

    assert hasattr(cell, "_highlighter")
    assert "#001f41" in cell.title_label.styleSheet()
    assert "border-left: 4px solid #001f41" in cell.styleSheet()
    assert cell.inline_result.isHidden()


def test_notebook_toolbar_buttons_are_styled_by_role():
    _app()
    tab = NotebookTab()

    assert "#001f41" in tab.run_all_button.styleSheet()
    assert "#e2e8f0" in tab.save_button.styleSheet()


def test_auto_close_combo_emits_activation_debug_signal():
    _app()
    combo = AutoCloseComboBox()
    calls: list[int] = []

    combo.addItem("One", 1)
    combo.addItem("Two", 2)
    combo.activated.connect(lambda index: calls.append(index))
    combo.activated.emit(1)

    assert calls == [1]


def test_auto_close_combo_tracks_current_index_changes():
    _app()
    combo = AutoCloseComboBox()
    calls: list[int] = []
    combo.addItem("One", 1)
    combo.addItem("Two", 2)
    combo.currentIndexChanged.connect(lambda index: calls.append(index))

    combo.setCurrentIndex(1)

    assert combo.currentIndex() == 1
    assert calls[-1] == 1


def test_combo_popup_views_enable_mouse_tracking_for_hover_highlight():
    _app()
    combo = AutoCloseComboBox()
    checkable = CheckableComboBox()

    assert combo.view().hasMouseTracking() is True
    assert combo.view().viewport().hasMouseTracking() is True
    assert checkable.view().hasMouseTracking() is True
    assert checkable.view().viewport().hasMouseTracking() is True


def test_checkable_combo_keeps_popup_open_when_toggling_items(monkeypatch):
    _app()
    combo = CheckableComboBox()
    calls: list[str] = []
    combo.add_check_item("phi", "phi")
    combo.add_check_item("phi_exact", "phi_exact")

    monkeypatch.setattr(combo, "hidePopup", lambda: calls.append("hide"))
    combo._toggle_clicked_item(combo.model().index(0, 0))
    combo._toggle_clicked_item(combo.model().index(1, 0))

    assert combo.checked_values() == ["phi", "phi_exact"]
    assert calls == []


def test_checkable_combo_emits_when_item_added_checked():
    _app()
    combo = CheckableComboBox()
    calls: list[str] = []
    combo.checkedItemsChanged.connect(lambda: calls.append("changed"))

    combo.add_check_item("phi", "phi", checked=True)

    assert calls == ["changed"]


def test_sidebar_uses_example_combo_and_updates_preview():
    _app()
    tab = NotebookTab()

    assert hasattr(tab, "example_combo")
    assert tab.example_combo.count() >= 5
    index = tab.example_combo.findText("1D Diffusion Explicit Scheme", Qt.MatchFlag.MatchExactly)
    assert index >= 0
    tab.example_combo.setCurrentIndex(index)

    assert "1D Diffusion Explicit Scheme" in tab.example_preview.toPlainText()


def test_help_buttons_move_to_top_toolbar_and_variables_panel_is_taller():
    _app()
    tab = NotebookTab()

    assert tab.functions_toggle_button.parent() is not tab.help_panels_container
    assert tab.markdown_toggle_button.parent() is not tab.help_panels_container
    assert tab.functions_toggle_button.text() == "Functions"
    assert tab.markdown_toggle_button.text() == "Markdown Help"
    assert tab.help_panels_container.maximumWidth() == 460
    assert tab.variables_browser.minimumHeight() >= 360


def test_insert_example_loads_markdown_and_code_cells_from_json_examples():
    _app()
    tab = NotebookTab()
    index = tab.example_combo.findText("1D Diffusion Explicit Scheme", Qt.MatchFlag.MatchExactly)
    assert index >= 0
    tab.example_combo.setCurrentIndex(index)

    tab.insert_example()

    assert any(cell.cell_type == "markdown" for cell in tab.cells)
    assert any(cell.cell_type == "code" and "import numpy as np" in cell.source() for cell in tab.cells)


def test_markdown_help_panel_exposes_examples_for_common_syntax():
    _app()
    tab = NotebookTab()

    markdown_help = tab._markdown_help_html()

    assert "Markdown Help" in markdown_help
    assert "### Subsection" in markdown_help
    assert "- item one" in markdown_help
    assert "[OpenPhase](https://example.com)" in markdown_help
    assert "```python" in markdown_help


def test_markdown_preview_html_includes_local_katex_assets_and_render_hook():
    assets = katex_assets_dir()
    html = build_markdown_preview_html("Inline math $x^2$ and block $$a^2+b^2=c^2$$")

    assert (assets / "katex.min.css").exists()
    assert (assets / "katex.min.js").exists()
    assert (assets / "auto-render.min.js").exists()
    assert "renderMathInElement" in html
    assert "katex.min.css" in html
    assert "$$a^2+b^2=c^2$$" in html


def test_markdown_cells_use_dedicated_markdown_preview_widget():
    _app()
    tab = NotebookTab()
    tab.add_markdown_cell(source="Euler: $e^{i\\pi}+1=0$")
    cell = next(item for item in tab.cells if item.cell_type == "markdown")

    assert isinstance(cell.preview, MarkdownPreview)
    assert not cell.preview.isHidden()
    assert cell.editor.isHidden()

    cell.toggle_preview()

    assert cell.preview.isHidden()
    assert not cell.editor.isHidden()


def test_markdown_cells_with_source_start_in_preview_mode():
    _app()
    tab = NotebookTab()
    tab.add_markdown_cell(source="# Title\n\nText with $x^2$")
    cell = next(item for item in reversed(tab.cells) if item.cell_type == "markdown")

    assert cell.preview_mode is True
    assert not cell.preview.isHidden()
    assert cell.editor.isHidden()
    assert cell.toggle_btn.text() == "Edit"


def test_markdown_preview_shrinks_for_short_content():
    _app()
    preview = MarkdownPreview()

    preview._apply_height(26)

    assert preview.height() < 120
    assert preview._view.height() < 120


def test_functions_panel_includes_usage_details_and_examples():
    _app()
    tab = NotebookTab()

    html = tab._functions_reference_html()

    assert "How To Use" in html
    assert "<table" in html
    assert "What It Does" in html
    assert "diffusion_dt(dx, D, f=0.5)" in html
    assert "plane_stress(E, nu, exx, eyy, exy=0.0)" in html
    assert "5 * mm" in html
    assert "reynolds(rho, u, L, mu)" in html


def test_code_cell_starts_with_hidden_output_area():
    _app()
    tab = NotebookTab()

    assert tab.cells[0].output_area.isHidden()


def test_code_cell_sends_stdout_to_bottom_output_area():
    _app()
    tab = NotebookTab()
    result = tab.execution_engine.execute("print('hello')\n42")
    tab.cells[0].set_result(result)

    assert tab.cells[0].inline_result.toPlainText() == ""
    assert not tab.cells[0].output_area.isHidden()


def test_code_cell_editor_and_hidden_inline_result_disable_internal_scrollbars():
    _app()
    tab = NotebookTab()
    cell = tab.cells[0]
    cell.editor.setPlainText("\n".join(f"line {index}" for index in range(60)))
    cell.inline_result.setPlainText("\n".join(f"value {index}" for index in range(20)))

    assert cell.editor.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert cell.editor.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert cell.inline_result.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert cell.inline_result.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert cell.editor.height() > 120
    assert cell.inline_result.height() > 120
    assert cell.inline_result.isHidden()


def test_notebook_tab_run_cell_async_updates_variables_panel():
    _app()
    tab = NotebookTab()
    tab.cells[0].editor.setPlainText("x = 5\ny = 10\nresult = x + y")

    tab.run_cell(tab.cells[0])

    timer = QElapsedTimer()
    timer.start()
    app = QApplication.instance()
    while timer.elapsed() < 3000 and "result" not in tab.variables_browser.toPlainText():
        app.processEvents()

    variables_text = tab.variables_browser.toPlainText()
    assert "result" in variables_text
    assert "15" in variables_text
    assert tab.status_label.text() == "Done"


def test_notebook_tab_uses_less_aggressive_autosave_interval():
    _app()
    tab = NotebookTab()

    assert tab.autosave_timer.interval() >= 8000


def test_plot_view_has_real_embedded_height():
    _app()
    view = PlotView()

    assert view.minimumHeight() >= 420
    assert view._view.minimumHeight() >= 420


def test_plot_view_wrapper_hides_internal_scroll_and_loads_local_mathjax():
    _app()
    view = PlotView()
    view.set_html("<div style='width:1200px;height:900px;'>wide plot</div>")

    wrapped_html = view._html_path.read_text(encoding="utf-8")

    assert "overflow: hidden" in wrapped_html
    assert "tex-svg.js" in wrapped_html


def test_notebook_tab_uses_multi_graph_workspace_for_latest_plot():
    _app()
    tab = NotebookTab()
    result = tab.execution_engine.execute("import plotly.graph_objects as go\nfig = go.Figure()\nfig.add_scatter(y=[1,2,3])\nfig")
    tab.cells[0].set_result(result)
    tab._refresh_graphs_panel()

    assert not tab.graph_workspace.isHidden()
    assert tab.graph_workspace.card_count() == 1
    first_card = tab.graph_workspace.cards()[0]
    assert first_card.mode_combo.currentData() == "series"
    assert hasattr(first_card, "x_combo")
    assert hasattr(first_card, "y_combo")
    assert hasattr(first_card, "evolution_matrix_combo")
    assert not first_card.plot_view.isHidden()
    assert "latest executed notebook plot" in first_card.status_label.text().lower()


def test_notebook_fonts_are_larger_than_old_defaults():
    _app()
    tab = NotebookTab()

    assert "font-size:15px" in tab.example_preview.styleSheet()
    assert "font-size: 15px" in tab.cells[0].styleSheet()


def test_variables_panel_summarizes_large_arrays():
    _app()
    tab = NotebookTab()
    tab.execution_engine.execute("import numpy as np\nu = np.arange(50)")
    tab._refresh_variables_panel()

    variables_text = tab.variables_browser.toPlainText()
    assert "u" in variables_text
    assert "shape=(50,)" in variables_text


def test_configure_desktop_graphics_sets_software_rendering():
    _app()

    config = configure_desktop_graphics()

    assert config["QT_OPENGL"] == "software"
    assert config["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert "--disable-gpu" in config["QTWEBENGINE_CHROMIUM_FLAGS"]


def test_graph_panel_uses_source_line_as_title():
    _app()
    tab = NotebookTab()
    tab.cells[0].editor.setPlainText("plt.plot([0, 1], [0, 1])\nplt.show()")
    tab.cells[0].last_result = ExecutionResult(
        outputs=[
            ExecutionOutput(
                kind="plotly",
                data={"html": "<div>Plotly.newPlot('id', [], {})</div>", "text": "Plotly Figure"},
            )
        ]
    )

    tab._refresh_graphs_panel()

    assert tab.graph_state.latest_plot_title == "plt.plot([0, 1], [0, 1])"
    assert tab.graph_workspace.cards()[0].status_label.text()


def test_graph_panel_uses_plot_view_for_html_image_fallback():
    _app()
    tab = NotebookTab()
    tab.cells[0].last_result = ExecutionResult(
        outputs=[
            ExecutionOutput(
                kind="html",
                data={"html": "<img src='data:image/png;base64,AAAA' alt='fallback' />", "text": "Matplotlib Figure"},
            )
        ]
    )

    tab._refresh_graphs_panel()

    assert isinstance(tab.graph_workspace.cards()[0].plot_view, PlotView)
    assert "fallback" in tab.graph_state.latest_plot_html


def test_notebook_graph_workspace_falls_back_to_latest_plot_when_first_card_has_no_selection():
    _app()
    tab = NotebookTab()
    tab.execution_engine.execute("import numpy as np\nx = np.arange(4.0)\ny = x**2")
    tab.cells[0].last_result = ExecutionResult(
        outputs=[
            ExecutionOutput(
                kind="plotly",
                data={"html": "<div>latest-plot-fallback</div>", "text": "Plotly Figure"},
            )
        ]
    )

    tab._refresh_graphs_panel()
    first_card = tab.graph_workspace.cards()[0]
    first_card.y_combo.set_checked_values([])
    first_card.refresh_plot()

    assert "latest executed notebook plot" in first_card.status_label.text().lower()
    assert first_card.current_figure() is not None


def test_notebook_graph_workspace_renders_series_plot_from_namespace_dropdowns():
    _app()
    tab = NotebookTab()
    tab.execution_engine.execute("import numpy as np\nx = np.arange(5)\nphi = x**2")
    tab._refresh_graphs_panel()

    first_card = tab.graph_workspace.cards()[0]
    assert first_card.x_combo.findData("x") >= 0
    assert "phi" in first_card.y_combo.checked_values()
    figure = first_card.current_figure()
    assert len(figure.data) == 1
    assert figure.data[0].name == "phi"
    assert figure.layout.height >= 520
    assert figure.layout.width is None


def test_notebook_graph_workspace_switches_to_evolution_mode():
    _app()
    tab = NotebookTab()
    tab.execution_engine.execute(
        "import numpy as np\n"
        "x = np.linspace(0.0, 1.0, 4)\n"
        "time = np.array([0.0, 1.0, 2.0])\n"
        "history = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]])"
    )
    tab._refresh_graphs_panel()

    first_card = tab.graph_workspace.cards()[0]
    first_card.mode_combo.setCurrentIndex(first_card.mode_combo.findData("evolution"))
    first_card.evolution_matrix_combo.setCurrentIndex(
        first_card.evolution_matrix_combo.findData("history")
    )
    first_card.evolution_time_combo.setCurrentIndex(
        first_card.evolution_time_combo.findData("time")
    )
    first_card.evolution_value_combo.setCurrentIndex(
        first_card.evolution_value_combo.findData("x")
    )
    first_card.evolution_step_slider.setValue(2)
    first_card.refresh_plot()

    figure = first_card.current_figure()
    assert first_card.mode_combo.currentData() == "evolution"
    assert len(figure.data) == 1
    assert list(figure.data[0].y) == [3.0, 4.0, 5.0, 6.0]


def test_notebook_graph_workspace_defaults_to_evolution_for_2d_only_namespace():
    _app()
    tab = NotebookTab()
    tab.execution_engine.execute("import numpy as np\nhistory = np.arange(12.0).reshape(3, 4)")
    tab._refresh_graphs_panel()

    first_card = tab.graph_workspace.cards()[0]
    assert first_card.mode_combo.currentData() == "evolution"
    assert first_card.evolution_matrix_combo.currentData() == "history"
    assert "updated" in first_card.status_label.text().lower() or "evolution" in first_card.status_label.text().lower()


def test_graph_panel_reuses_existing_widgets_for_unchanged_plot():
    _app()
    tab = NotebookTab()
    html = "<div>Plotly.newPlot('id', [], {})</div>"
    tab.cells[0].last_result = ExecutionResult(
        outputs=[ExecutionOutput(kind="plotly", data={"html": html, "text": "Plotly Figure"})]
    )
    tab._refresh_graphs_panel()
    first_widget = tab.graph_workspace.cards()[0].plot_view

    tab.add_code_cell()
    tab.cells[1].last_result = ExecutionResult(outputs=[ExecutionOutput(kind="value", data={"text": "42"})])
    tab._refresh_graphs_panel()

    assert tab.graph_workspace.cards()[0].plot_view is first_widget
    assert tab.graph_state.latest_plot_html == html


def test_notebook_graph_workspace_adds_and_removes_independent_cards():
    _app()
    tab = NotebookTab()
    tab.execution_engine.execute("import numpy as np\nx = np.arange(5)\nphi = x**2\nforce = x + 1")
    tab._refresh_graphs_panel()

    assert tab.graph_workspace.card_count() == 1
    tab.graph_workspace.add_graph_card()
    assert tab.graph_workspace.card_count() == 2

    first_card, second_card = tab.graph_workspace.cards()
    first_card.x_combo.setCurrentIndex(first_card.x_combo.findData("x"))
    first_card.y_combo.set_checked_values(["phi"])
    first_card.refresh_plot()
    second_card.x_combo.setCurrentIndex(second_card.x_combo.findData("x"))
    second_card.y_combo.set_checked_values(["force"])
    second_card.refresh_plot()

    first_names = [trace.name for trace in first_card.current_figure().data]
    second_names = [trace.name for trace in second_card.current_figure().data]
    assert first_names == ["phi"]
    assert second_names == ["force"]

    tab.graph_workspace.remove_graph_card(second_card)
    assert tab.graph_workspace.card_count() == 1


def test_notebook_tab_uses_namespace_snapshot_methods_for_ui_refreshes():
    _app()
    tab = NotebookTab()
    calls: list[str] = []

    def fake_get_namespace() -> dict[str, object]:
        print("[debug][fake-engine] get_namespace", flush=True)
        calls.append("namespace")
        return {"alpha": 1}

    tab.execution_engine.namespace = None  # type: ignore[assignment]
    tab.execution_engine.get_namespace = fake_get_namespace  # type: ignore[method-assign]

    words = tab._completion_words()
    tab._refresh_variables_panel()
    tab._refresh_graphs_panel()

    assert "alpha" in words
    assert "alpha" in tab.variables_browser.toPlainText()
    assert len(calls) >= 3


def test_plot_view_removes_temp_html_file_when_destroyed():
    _app()
    view = PlotView()
    view.set_html("<div>cleanup me</div>")
    html_path = view._html_path

    assert html_path.exists()

    view.close()

    assert not html_path.exists()
