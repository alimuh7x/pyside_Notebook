# calculationNotebook Desktop

`calculationNotebook` now includes a PySide6 desktop application that replaces the old Dash UI with a local notebook-style workflow.

The new desktop app keeps the scientific backend logic from `utils/` and adds:

- Real Python cell execution with a shared namespace
- Markdown cells with edit/preview toggle
- Plotly result rendering inside the desktop app
- JSON save/open plus autosave
- A notebook-focused desktop shell

The legacy Dash code is still present in the repository during migration, but the new primary entrypoint is `main.py`.

## Install

Use Python 3.10+.

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
py -3.12 -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```

## Run

Start the desktop app:

```bash
source myenv/bin/activate
python main.py
```

For headless/offscreen environments:

```bash
QT_QPA_PLATFORM=offscreen QTWEBENGINE_DISABLE_SANDBOX=1 python main.py
```

## Architecture

Desktop entrypoint:

- `main.py` initializes `QApplication` and opens the main window

Desktop modules:

- `pyside_app/main_window.py` builds the top-level notebook window
- `pyside_app/notebook_tab.py` manages notebook cells, execution, save/open, autosave, and kernel reset
- `pyside_app/cell_widgets.py` contains reusable code/markdown cell widgets and output rendering
- `pyside_app/plot_view.py` renders Plotly HTML inside Qt
- `pyside_app/execution_engine.py` executes real Python code with captured stdout/errors and last-expression display
- `pyside_app/execution_worker.py` runs notebook execution off the UI thread
- `pyside_app/storage.py` saves and loads notebook JSON documents

Reusable backend retained from the original project:

- `utils/notebook_eval.py` scientific helpers, constants, units, and evaluator logic
- `utils/formula_parser.py` safe formula parsing and vectorized evaluation
- `utils/notebook_persistence.py` existing notebook payload conventions

## Migration Notes

- The Dash UI has not been deleted yet.
- The new desktop app is built alongside the old Dash code.
- `utils/` remains the main reusable scientific backend.
- The desktop notebook uses real Python execution via `exec()` and `eval()` for the last expression.
- Plotly rendering is now embedded through Qt instead of a browser-served Dash app.

## Verification

Desktop-specific tests:

```bash
QT_QPA_PLATFORM=offscreen QTWEBENGINE_DISABLE_SANDBOX=1 \
python -m pytest -s \
  tests/test_pyside_execution_engine.py \
  tests/test_pyside_storage.py \
  tests/test_pyside_formula_plot.py \
  tests/test_pyside_gui_smoke.py
```

Core backend regression tests still relevant to the preserved scientific logic:

```bash
python -m pytest -s \
  tests/test_notebook_eval.py \
  tests/test_notebook_eval_source.py \
  tests/test_mechanical_loads_explorer.py \
  tests/test_mechanical_load_history.py \
  tests/test_initializations_explorer.py
```

## Known Limitations

- The desktop notebook is a practical Jupyter-style workflow, but it is not a full Jupyter kernel protocol implementation.
- The desktop shell is notebook-only for now. The formula plotting module still exists in the repo, but it is not exposed in the main window until that UI returns.
- The old Dash-specific source tests are intentionally not the main verification path for the desktop app.
- `QWebEngineView` may require `QTWEBENGINE_DISABLE_SANDBOX=1` in restricted/headless environments.

## Next Improvements

- Richer inline table rendering for pandas and numpy outputs
- Reintroduce formula plotting once the notebook workflow is stabilized
- Drag-and-drop notebook cell reordering
- Per-cell execution history and timestamps
- Import/export helpers for legacy notebook payloads
