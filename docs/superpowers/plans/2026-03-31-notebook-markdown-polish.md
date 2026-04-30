# Notebook Markdown Polish Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve notebook markdown controls and presentation with a small, low-risk polish pass.

**Architecture:** Keep the current markdown renderer stack and add only thin UI/client callback changes. The work is limited to notebook UI composition, markdown toggle wiring, and preview CSS so it remains incremental.

**Tech Stack:** Python, Dash, Monaco, clientside callbacks, CSS, KaTeX, marked

---

### Task 1: Lock Down Markdown UI Expectations

**Files:**
- Modify: `tests/test_calculation_notebook_ui_source.py`
- Modify: `tests/test_notebook_manager_source.py`

- [ ] **Step 1: Write the failing test**

Add source-level checks for:
- explicit markdown `Edit` control
- markdown help button/panel
- richer markdown preview CSS selectors

- [ ] **Step 2: Run test to verify it fails**

Run: `myenv/bin/python -m pytest tests/test_calculation_notebook_ui_source.py tests/test_notebook_manager_source.py -k 'markdown' -v`
Expected: FAIL because the new controls/styles do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement only the controls, help panel, and CSS needed to satisfy the new expectations.

- [ ] **Step 4: Run test to verify it passes**

Run: `myenv/bin/python -m pytest tests/test_calculation_notebook_ui_source.py tests/test_notebook_manager_source.py -k 'markdown' -v`
Expected: PASS

### Task 2: Implement Markdown Controls and Help Overlay

**Files:**
- Modify: `ui/calculation_notebook.py`
- Modify: `callbacks/notebook_manager.py`

- [ ] **Step 1: Add explicit markdown edit control**

Wire a markdown `Edit` button into the existing clientside preview logic.

- [ ] **Step 2: Add lightweight markdown help overlay**

Reuse the floating-panel pattern already present for the functions reference.

- [ ] **Step 3: Verify control wiring**

Run: `myenv/bin/python -m pytest tests/test_calculation_notebook_ui_source.py tests/test_notebook_manager_source.py -k 'markdown' -v`
Expected: PASS

### Task 3: Improve Markdown Preview Styling

**Files:**
- Modify: `assets/style.css`
- Modify: `tests/test_calculation_notebook_ui_source.py`

- [ ] **Step 1: Add improved markdown preview styling**

Style headings, links, lists, blockquotes, tables, horizontal rules, inline code, and fenced code blocks.

- [ ] **Step 2: Verify style-related source tests**

Run: `myenv/bin/python -m pytest tests/test_calculation_notebook_ui_source.py -k 'markdown' -v`
Expected: PASS

### Task 4: Final Verification

**Files:**
- Modify: `ui/calculation_notebook.py`
- Modify: `callbacks/notebook_manager.py`
- Modify: `assets/style.css`

- [ ] **Step 1: Run targeted notebook markdown tests**

Run: `myenv/bin/python -m pytest tests/test_calculation_notebook_ui_source.py tests/test_notebook_manager_source.py -k 'markdown' -v`
Expected: PASS

- [ ] **Step 2: Manual verification**

1. Run `myenv/bin/python OPView.py`
2. Create a markdown cell.
3. Use `Preview`, `Edit`, and markdown help.
4. Confirm headings, lists, tables, code blocks, and math render cleanly.
