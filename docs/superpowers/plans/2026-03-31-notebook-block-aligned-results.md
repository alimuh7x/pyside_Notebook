# Notebook Block-Aligned Results Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fragile line-based matrix/result alignment with a block-aligned execution view that keeps Monaco as the editor but renders source/result pairs as shared-height blocks.

**Architecture:** Keep one Monaco editor per code cell for editing and persistence. After execution, derive structured expression blocks from the evaluated rows and render a separate aligned notebook/result view where each source block and result block share one height, including inline matrix tables up to `6x6`.

**Tech Stack:** Dash, Python callback/state management, custom notebook evaluator, Monaco client JS, pytest source/runtime tests

---

### Task 1: Define structured execution blocks

**Files:**
- Modify: `utils/notebook_eval.py`
- Modify: `callbacks/notebook_manager.py`
- Test: `tests/test_notebook_eval.py`
- Test: `tests/test_notebook_manager_source.py`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- scalar expressions produce one execution block with `source`, `source_span`, `result_kind`, `result_span`
- multiline matrix literals produce one execution block with correct `source_span`
- control-flow blocks remain a single logical block

- [ ] **Step 2: Run tests to verify they fail**

Run: `myenv/bin/python -m pytest tests/test_notebook_eval.py tests/test_notebook_manager_source.py -k 'execution_block or block_metadata' -v`
Expected: FAIL because execution blocks do not exist yet

- [ ] **Step 3: Write minimal implementation**

Add a helper that converts evaluated rows into structured execution blocks. Preserve:
- source text per block
- source span
- rendered result payload
- matrix/table metadata for `<= 6x6`

- [ ] **Step 4: Run tests to verify they pass**

Run: `myenv/bin/python -m pytest tests/test_notebook_eval.py tests/test_notebook_manager_source.py -k 'execution_block or block_metadata' -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utils/notebook_eval.py callbacks/notebook_manager.py tests/test_notebook_eval.py tests/test_notebook_manager_source.py
git commit -m "feat: add structured notebook execution blocks"
```

### Task 2: Render a block-aligned notebook/results view

**Files:**
- Modify: `ui/calculation_notebook.py`
- Modify: `callbacks/notebook_manager.py`
- Test: `tests/test_calculation_notebook_ui_source.py`
- Test: `tests/test_notebook_manager_source.py`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- code cells render a dedicated execution view container
- execution view renders paired source/result blocks
- inline matrix tables occupy their own block without depending on row-span gutter hacks

- [ ] **Step 2: Run tests to verify they fail**

Run: `myenv/bin/python -m pytest tests/test_calculation_notebook_ui_source.py tests/test_notebook_manager_source.py -k 'execution view or block aligned' -v`
Expected: FAIL because the UI still renders the old gutter-only model

- [ ] **Step 3: Write minimal implementation**

Implement a new renderer for executed code blocks:
- left column: rendered source block text
- right column: rendered result block
- shared block height
- inline matrix tables up to `6x6`
- compact summary for larger matrices

Keep the visual treatment continuous, not card-per-expression.

- [ ] **Step 4: Run tests to verify they pass**

Run: `myenv/bin/python -m pytest tests/test_calculation_notebook_ui_source.py tests/test_notebook_manager_source.py -k 'execution view or block aligned' -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/calculation_notebook.py callbacks/notebook_manager.py tests/test_calculation_notebook_ui_source.py tests/test_notebook_manager_source.py
git commit -m "feat: render block-aligned notebook results"
```

### Task 3: Reduce Monaco to editing only

**Files:**
- Modify: `assets/monaco_notebook.js`
- Modify: `callbacks/notebook_manager.py`
- Test: `tests/test_notebook_manager_source.py`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- Monaco no longer relies on result view zones for notebook/result alignment
- notebook state refresh still works for editor sync and variable decorations

- [ ] **Step 2: Run tests to verify they fail**

Run: `myenv/bin/python -m pytest tests/test_notebook_manager_source.py -k 'view zones or monaco alignment' -v`
Expected: FAIL because view-zone alignment is still present

- [ ] **Step 3: Write minimal implementation**

Remove or reduce `_nbApplyResultZones` as the primary alignment mechanism. Keep Monaco for:
- editing
- syntax highlighting
- keyboard commands
- source sync

Do not use Monaco to simulate result heights anymore.

- [ ] **Step 4: Run tests to verify they pass**

Run: `myenv/bin/python -m pytest tests/test_notebook_manager_source.py -k 'view zones or monaco alignment' -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assets/monaco_notebook.js callbacks/notebook_manager.py tests/test_notebook_manager_source.py
git commit -m "refactor: decouple monaco from result alignment"
```

### Task 4: Wire callbacks to the new execution view

**Files:**
- Modify: `callbacks/notebook_manager.py`
- Modify: `ui/calculation_notebook.py`
- Test: `tests/test_notebook_manager_source.py`
- Test: `tests/test_calculation_notebook_ui_source.py`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- run-cell and run-all callbacks store structured execution blocks in notebook state
- rendered code-cell output uses those blocks instead of flat result rows
- multiline matrix inputs no longer create false overlap with unrelated notebook rows

- [ ] **Step 2: Run tests to verify they fail**

Run: `myenv/bin/python -m pytest tests/test_notebook_manager_source.py tests/test_calculation_notebook_ui_source.py -k 'structured execution or multiline matrix' -v`
Expected: FAIL because callbacks still populate flat result lines only

- [ ] **Step 3: Write minimal implementation**

Update callback outputs so each code cell stores:
- raw Monaco source
- structured execution blocks
- derived summaries for plots/variables

Render the aligned execution view from that structured state.

- [ ] **Step 4: Run tests to verify they pass**

Run: `myenv/bin/python -m pytest tests/test_notebook_manager_source.py tests/test_calculation_notebook_ui_source.py -k 'structured execution or multiline matrix' -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add callbacks/notebook_manager.py ui/calculation_notebook.py tests/test_notebook_manager_source.py tests/test_calculation_notebook_ui_source.py
git commit -m "feat: drive aligned notebook view from structured execution state"
```

### Task 5: Final verification

**Files:**
- Modify: `assets/monaco_notebook.js`
- Modify: `callbacks/notebook_manager.py`
- Modify: `ui/calculation_notebook.py`
- Modify: `utils/notebook_eval.py`
- Test: `tests/test_notebook_eval.py`
- Test: `tests/test_notebook_manager_source.py`
- Test: `tests/test_calculation_notebook_ui_source.py`

- [ ] **Step 1: Run focused regression suite**

Run:
`myenv/bin/python -m pytest tests/test_notebook_eval.py tests/test_notebook_manager_source.py tests/test_calculation_notebook_ui_source.py -k 'notebook or matrix or markdown or quick_start' -v`

Expected: PASS for the focused notebook slice

- [ ] **Step 2: Run syntax verification for modified Python files**

Run:
`myenv/bin/python -m py_compile callbacks/notebook_manager.py ui/calculation_notebook.py utils/notebook_eval.py`

Expected: no output

- [ ] **Step 3: Manual browser verification**

Check:
- single-line `3x3` matrix aligns with its own source block
- multiline matrix input does not steal unrelated rows
- `<= 6x6` matrices render inline
- `> 6x6` matrices stay compact
- Quick Start still inserts into code

- [ ] **Step 4: Commit**

```bash
git add assets/monaco_notebook.js callbacks/notebook_manager.py ui/calculation_notebook.py utils/notebook_eval.py tests/test_notebook_eval.py tests/test_notebook_manager_source.py tests/test_calculation_notebook_ui_source.py
git commit -m "feat: replace notebook row hacks with block-aligned results"
```
