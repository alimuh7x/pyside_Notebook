# AGENTS.md

This file gives repo-wide instructions for Codex and other coding agents working in OPView.

## Scope

- These rules apply to the whole project.
- Keep changes focused on the user request. Do not refactor unrelated areas.
- Prefer small, local edits over broad rewrites.

## Project Structure

| Module           | Purpose                                                                                                                                                                                                                                                                             |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/`           | App creation and global state. `AppContext` in `context.py` holds runtime state (loaded projects, caches). `OPViewApp` in `opview_app.py` wires all managers together. Do not store feature-specific state here.                                                                    |
| `config/`        | Static configuration only. Global constants go in `constants.py`. Tab/dataset definitions go in `tabs.py`. Persistent user settings (saved to `~/.opview_settings.json`) go in `settings_store.py`. Do not put runtime state here.                                                  |
| `ui/`            | Layout construction only — returns Dash component trees, no business logic. `layout.py` builds the top-level shell; feature panels live in their own files (e.g. `formula_graphs.py`, `mechanical_loads_explorer.py`).                                                              |
| `callbacks/`     | Dash callback wiring and app state handling. Each feature has its own manager (e.g. `formula_manager.py`, `notebook_manager.py`). Keep computation out of callbacks — delegate to `utils/`.                                                                                         |
| `utils/`         | Pure, reusable logic with no Dash dependency. Parsing, evaluation, fitting, file I/O, and numeric analysis live here. Functions here must be testable in isolation.                                                                                                                 |
| `viewer/`        | Self-contained VTK slice viewer. `state.py` defines `ViewerState` (serialisable per-tab state). `panel.py` owns the viewer's callbacks. `layout.py` builds the viewer layout. `defaults.py` holds viewer-specific defaults. Do not mix viewer state with the rest of the app state. |
| `comparisonmgr/` | Multi-file heatmap comparison feature. `ui_builders.py` builds comparison layouts, `callbacks.py` wires comparison callbacks, `manager.py` is the orchestrator. Follows the same ui/callbacks/utils separation as the rest of the app.                                              |
| `assets/`        | Static files served by Dash. CSS in `style.css`. PNG icons (see Icons section). JS helpers for client-side behavior. Do not put Python logic here.                                                                                                                                  |

**Placement rules:**

- New feature code → closest existing module, not a new parallel structure
- Shared pure logic → `utils/`, not inside a callback or UI builder
- New global constants → `config/constants.py`
- New tab or dataset definition → `config/tabs.py`

## Coding Rules

- Follow existing OPView patterns before introducing new abstractions.
- Prefer helper functions for non-trivial logic instead of embedding large blocks inside callbacks.
- Keep computation and validation out of UI builders when possible.
- Keep computation and parsing out of callbacks when possible.
- Normalize and validate user input defensively.
- Fail with clear, user-facing error messages instead of silent fallback when correctness matters.
- Use descriptive names and short docstrings for non-obvious helpers.
- Avoid adding dependencies unless they are clearly necessary.
- Do not rename, move, or split files unless the task needs it.
- Do not replace working callback flows with a new framework or pattern.
- Keep comments sparse and useful.
- Preserve backward-compatible behavior unless the user asked for a behavior change.

## Dash and UI Rules

- Preserve the current separation between layout code and callback code.
- Match the existing style of Dash components, state dictionaries, and callback helpers.
- Extend existing panel/state structures instead of inventing a second state model.
- Reuse existing styling conventions and constants where possible.
- Do not introduce large visual redesigns unless the task explicitly asks for them.

## UI Layout Conventions

**Single-field controls** (dropdowns, numeric inputs, steppers):

- Label on top, input below — not label-left / value-right.
- One field per block.

**Multi-field rows** (when one item has several related fields):

- Render as a table-like row: item label first, then related fields as columns with headers.
- Do not split related fields into separate stacked blocks unless explicitly asked.
- Example:
  
  ```
  Component | Type   | Value
  XX        | Stress | [-] 350e6 [+]
  YY        | None   | [-] 0    [+]
  ZZ        | Strain | [-] 0.01 [+]
  ```

## Component Size Standards

All values below come from the existing codebase. Use them for all new UI to keep the interface consistent.

### Typography

| Element                              | Font size          | Weight  | Extra                                                                        |
| ------------------------------------ | ------------------ | ------- | ---------------------------------------------------------------------------- |
| App title (top header bar)           | `1.5rem` (24px)    | 700     | uppercase, white, `letter-spacing: 0.02em`                                   |
| Dataset / panel big title            | `1.2rem` (19.2px)  | 700     | `.dataset-title`, `color: var(--text-main)`                                  |
| Sidebar tab label                    | `1.1rem` (18.15px) | 600     | `.custom-tab`, white 70% opacity                                             |
| Sidebar nav title (group label)      | `0.85rem` (14px)   | —       | `.sidebar-title`, uppercase, `letter-spacing: 0.4em`, white 60% opacity      |
| Section title (inside feature panel) | `14px`             | 700     | `SECTION_TITLE_STYLE`, uppercase, `letter-spacing: 0.04em`, `color: #355070` |
| Load / sub-section header            | `13px`             | 700     | `LOAD_HEADER_STYLE`, `color: #1e3a5f`, border-bottom                         |
| Body base font                       | `16.5px`           | —       | `body`, Roboto Condensed                                                     |
| Field labels (above inputs)          | `12px`             | 500–700 | `LABEL_STYLE`, `color: #64748b`                                              |
| Component labels (BC row)            | `12px`             | 700     | `COMPONENT_LABEL_STYLE`, `color: #355070`                                    |
| Sub-labels / hint text               | `11px`             | —       | italic, `color: #94a3b8`                                                     |
| Summary stat value (big number)      | `22px`             | 700     | `color: #1e3a5f` — used in load status cards                                 |

### Shared control height

All interactive controls share one height. Use the CSS variable — never hardcode:

```css
--control-height: 32px
```

In Python inline styles: `"minHeight": "32px"` or `"height": "32px"`.

### Text inputs

| Property          | Value                                                              |
| ----------------- | ------------------------------------------------------------------ |
| Min height        | `32px` (via `--control-height`)                                    |
| Font size         | `0.85rem` (~13.6px) — CSS global; Python `INPUT_STYLE` uses `13px` |
| Padding           | `8px 11px` (CSS global for `input[type]`)                          |
| Border            | `1.5px solid #d1dce8`                                              |
| **Border radius** | **`7px`**                                                          |
| Background        | `linear-gradient(180deg, #ffffff 0%, #fafbfc 100%)`                |
| Min width         | `0` (let flex container control width)                             |
| Max width         | none (fill container)                                              |
| Flex              | `width: 100%` within its container                                 |


> Note: Some feature panels define a local `INPUT_STYLE` dict with slightly different padding (`5px 8px` or `6px 10px`) and border-radius (`6px` or `10px`). These are inconsistencies in the existing code. For new code, follow the CSS values above (`7px` radius, `8px 11px` padding).

### Dropdowns

There are two dropdown sizes used in OPView:

**Big dropdown** — full-width, standalone field (e.g. trigger type selector):

| Property  | Value                       |
| --------- | --------------------------- |
| Font size | `13px`                      |
| Width     | `100%` of container         |
| Flex      | `1` (fills available space) |
| Min width | `0`                         |

**Small dropdown** — compact, inline in a row (e.g. BC type in component rows):

| Property  | Value                                                 |
| --------- | ----------------------------------------------------- |
| Font size | `12px`                                                |
| Flex      | `1 1 170px`                                           |
| Min width | `145px` — never go below this or text will be clipped |

**Both sizes share these CSS properties:**

| Property                    | Value                                         |
| --------------------------- | --------------------------------------------- |
| Min height                  | `32px` (via `--control-height`)               |
| **Border radius (control)** | **`7px`**                                     |
| **Border radius (menu)**    | **`10px`**                                    |
| Min width (menu)            | `100%` of the control                         |
| Max width (menu)            | `98vw`                                        |
| Menu width                  | `max-content` — expands to fit longest option |

Never constrain dropdown menu width. The menu must always be wide enough to show the full option text.

### Stepper inputs (`[-]` / `[+]` buttons)

| Property                 | Value                                                    |
| ------------------------ | -------------------------------------------------------- |
| Input min height         | `36px`                                                   |
| Input min width          | `150px`                                                  |
| Button width             | `34px` (fixed, `flex: 0 0 auto`)                         |
| Button font size         | `18px`                                                   |
| **Button border radius** | **`10px 0 0 10px`** (left) / **`0 10px 10px 0`** (right) |
| Step magnitude           | Preserve decimal — `1→2`, `0.1→0.2`, `0.01→0.02`         |


### Standard buttons (`.btn` class)

| Property          | Value                     |
| ----------------- | ------------------------- |
| Padding           | `6px 14px`                |
| Font size         | `0.8rem`                  |
| Font weight       | `600`                     |
| **Border radius** | **`8px`** (`--radius-sm`) |
| Flex              | `0 0 auto`                |

### Icon-only buttons (`.icon-btn` class)

| Property          | Value                 |
| ----------------- | --------------------- |
| Width × Height    | `30px × 30px` (fixed) |
| **Border radius** | **`6px`**             |
| Font size         | `0.9rem`              |
| Flex              | `0 0 auto`            |

### Compact reset button (`.reset-btn` class)

| Property          | Value                 |
| ----------------- | --------------------- |
| Width × Height    | `32px × 32px` (fixed) |
| **Border radius** | **`4px`**             |
| Padding           | `0`                   |
| Flex              | `0 0 auto`            |

### Panels and containers

| Component                    | Min width                   | Max width | Flex              |
| ---------------------------- | --------------------------- | --------- | ----------------- |
| Sidebar panel                | `240px` (fixed `0 0 240px`) | `240px`   | `flex: 0 0 240px` |
| Feature panel / card         | `200px`                     | none      | `flex: 1 1 300px` |
| Control block (inside panel) | `200px`                     | none      | `flex: 1`         |
| Trigger / sub-block          | `220px`                     | `240px`   | `flex: 1 1 240px` |
| Notebook / wide panel        | `400px`                     | `960px`   | —                 |

**Border radius for panels:** `0px` (flat edges — `--radius-lg` and `--radius-md` are both `0px`). Panels do not have rounded corners.

## Colors and Buttons

- Primary: `#001f41` (deep blue — backgrounds, sidebar, button fill)
- Accent / active: `#b60021` (red — active states, accents)
- Hover: `#95041a` (darker red — button hover)
- Accent dark: `#9f051d` (CSS `--accent-dark` — deepest red variant)
- Buttons: white text, soft shadow, `8px` rounded corners. Do not introduce new colors for buttons.

## Icons

All icons live in `assets/`. Use via CSS background-image or `html.Img src='/assets/<file>'`.

| Purpose               | Default         | Hover                |
| --------------------- | --------------- | -------------------- |
| Close / remove item   | `remove.png`    | `remove_hover.png`   |
| Add / create item     | `plus.png`      | `plus_hover.png`     |
| Toggle on/off         | `switch-on.png` | `switch-off.png`     |
| Add row (alternative) | `addition.png`  | `addition_hover.png` |

CSS classes already defined for common icons:

- `.opview-image-close-btn` — remove/close (uses `remove.png`)
- `.opview-image-add-btn` — add/plus (uses `plus.png`)
- `.mechanical-remove-load-btn` — remove in mechanical loads panel

Other available icons: `bar-chart.png`, `color-scale.png`, `download.png`, `Reset.png`, `Horizontal.png`, `Vertical.png`, `OP_Logo.png`, `OP_Logo_main.png`.

## Formula and Data Logic

- Keep formula evaluation safe and explicit.
- Prefer pure helper functions for parsing, math, interpolation, and numeric analysis.
- Keep allowed-symbol or allowed-function lists centralized when working on formula features.
- When adding analysis features, make edge-case handling explicit for invalid, empty, non-finite, or shape-mismatched data.

## Testing

Run tests:

```
source myenv/bin/activate && python -m pytest tests/
```

- Test new functions in `utils/` — that is the only layer that is unit-testable in isolation.
- Use `pytest` style (`assert`) for new test files.
- Test edge cases explicitly: empty input, non-finite values, shape mismatches.
- Do not write tests for `ui/` or `callbacks/` — they require a running Dash server.

## Debugging

Every new feature must print enough to verify it is working without opening the GUI.

- Print inputs before the operation: what data came in, what parameters were received.
- Print outputs after the operation: what was computed, what will be returned or stored.
- Print inside every exception path: the error, the input that caused it, and where it happened.
- Use `print(f"[debug][<feature-name>] ...")` so lines are easy to grep and remove later.

Example:

```python
print(f"[debug][load-summary] inputs: loads={loads}")
result = build_loads_summary(loads)
print(f"[debug][load-summary] result: {result}")
```

## Verification Scripts

For every new feature, write a short standalone script that calls the new code with realistic sample inputs and prints the outputs. Run it and read the output to confirm the feature works — no browser needed.

- Put it in `tests/` named `verify_<feature>.py`.
- Call `utils/` functions directly — they have no Dash dependency.
- For callback-level logic, call the underlying method directly on the manager instance with mock inputs.
- After writing the script, run it and read the output. Confirm the results look correct before marking the feature done.
- Never rely on the browser to verify. If it cannot be confirmed from the terminal, the feature is not done.

Example:

```python
# tests/verify_load_summary.py
from utils.mechanical_loads_explorer import build_loads_summary, default_load_structure

load = default_load_structure()
load["bc_types"] = ["Stress", "None", "Strain", "None", "None", "None"]
load["bc_values"] = [350e6, 0.0, 0.01, 0.0, 0.0, 0.0]

print(f"[verify][load-summary] input: {load}")
result = build_loads_summary([load])
print(f"[verify][load-summary] result: {result}")
```

Run it, read the output, and confirm the result is correct:

```
source myenv/bin/activate && python tests/verify_load_summary.py
```
