# Notebook JSON Save/Load Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make notebook JSON save/load preserve the full rerunnable notebook input state and restore it cleanly for rerun.

**Architecture:** Add small serializer/deserializer helpers in the notebook callback manager and a save-time sync path that captures current Monaco editor sources before exporting JSON. Keep the file format limited to canonical notebook inputs so load remains simple and reruns stay authoritative.

**Tech Stack:** Python, Dash callbacks, Monaco-backed notebook UI, pytest/unittest

---

### Task 1: Lock Down Serialization Expectations

**Files:**
- Modify: `tests/test_notebook_manager_source.py`
- Test: `tests/test_notebook_manager_source.py`

- [ ] **Step 1: Write the failing test**

Add regression coverage for:
- a serializer helper that emits a versioned payload with all notebook cells
- a deserializer helper that rebuilds rerunnable cells without persisted outputs
- save flow wiring that synchronizes editor sources before export

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebook_manager_source.py -v`
Expected: FAIL because the helpers and save-sync wiring do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add notebook JSON serialization/deserialization helpers and wire save export through the synchronized snapshot.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebook_manager_source.py -v`
Expected: PASS

### Task 2: Implement Save-Time Notebook Snapshot

**Files:**
- Modify: `callbacks/notebook_manager.py`
- Modify: `ui/calculation_notebook.py`
- Test: `tests/test_notebook_manager_source.py`

- [ ] **Step 1: Write the failing test**

Cover that save no longer depends solely on stale `notebook-cells-store`, and that current editor state is captured before export.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebook_manager_source.py -v`
Expected: FAIL on the new save-sync assertions.

- [ ] **Step 3: Write minimal implementation**

Add a save-sync store and clientside callback to collect current Monaco editor values, then export from the synchronized cells snapshot.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebook_manager_source.py -v`
Expected: PASS

### Task 3: Verify Load Restores Rerunnable State

**Files:**
- Modify: `callbacks/notebook_manager.py`
- Test: `tests/test_notebook_manager_source.py`

- [ ] **Step 1: Write the failing test**

Assert that JSON load rebuilds cells with source/type/id preserved while outputs stay empty and state resets for rerun.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebook_manager_source.py -v`
Expected: FAIL if deserialization still leaks transient state or misses cell fields.

- [ ] **Step 3: Write minimal implementation**

Normalize loaded cell payloads into canonical rerunnable notebook state.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebook_manager_source.py -v`
Expected: PASS

### Task 4: Final Verification

**Files:**
- Modify: `callbacks/notebook_manager.py`
- Modify: `ui/calculation_notebook.py`
- Test: `tests/test_notebook_manager_source.py`

- [ ] **Step 1: Run targeted verification**

Run: `pytest tests/test_notebook_manager_source.py -v`
Expected: PASS

- [ ] **Step 2: Run a broader notebook regression slice**

Run: `pytest tests/test_calculation_notebook_ui_source.py tests/test_notebook_eval.py -v`
Expected: PASS

- [ ] **Step 3: Manual verification**

1. Run `myenv/bin/python OPView.py`
2. Edit multiple notebook cells, including markdown and code.
3. Save `.json`.
4. Reload `.json`.
5. Press run and confirm the restored notebook reruns from the loaded state.
