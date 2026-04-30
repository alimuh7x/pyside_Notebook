# Notebook Output Rules Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formalize notebook result row-span rules so scalars, vectors, matrices, errors, and empty outputs render with predictable inline alignment.

**Architecture:** Keep the current Monaco-plus-right-gutter layout. Centralize row-span and output-kind decisions in the notebook manager, then let the existing gutter renderer consume explicit metadata instead of ad hoc branching.

**Tech Stack:** Dash, Python, Monaco, pytest

---

### Task 1: Lock output rules with tests

**Files:**
- Modify: `tests/test_notebook_manager_source.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run the targeted pytest command and verify failure**
- [ ] **Step 3: Implement the minimal callback helper changes**
- [ ] **Step 4: Re-run the targeted pytest command and verify pass**

### Task 2: Apply the rule helper to gutter output

**Files:**
- Modify: `callbacks/notebook_manager.py`
- Test: `tests/test_notebook_manager_source.py`

- [ ] **Step 1: Route `_build_result_lines()` through one row-span rule helper**
- [ ] **Step 2: Keep current multiline-input behavior unchanged**
- [ ] **Step 3: Re-run focused tests**

### Task 3: Verify UI still matches the normalized result metadata

**Files:**
- Modify: `tests/test_calculation_notebook_ui_source.py`

- [ ] **Step 1: Add or update UI source assertions only if needed**
- [ ] **Step 2: Run notebook manager/UI source pytest slice**
