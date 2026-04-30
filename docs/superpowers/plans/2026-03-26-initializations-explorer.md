# Initializations Explorer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new top-level `Initializations Explorer` tab to OPView with a first 2D `QuasiRandomNuclei` explorer that lets users adjust sliders and see the final nuclei pattern.

**Architecture:** Keep the feature isolated in new UI/callback/utility files. Put the quasi-random nuclei logic into a pure helper so it can be tested independently, then build a small Dash/Plotly explorer on top and wire it into the top-level tab switcher.

**Tech Stack:** Dash, Plotly, Python standard library, existing OPView callback/layout structure

---

### Task 1: Add and test the pure quasi-random nuclei helper

**Files:**
- Create: `tests/test_initializations_explorer.py`
- Create: `utils/initializations_explorer.py`

- [ ] **Step 1: Write the failing tests**

Write tests for:
- deterministic seed behavior
- effective origin fallback
- zero-deviation behavior
- threshold/filtering behavior
- boundary-safe final points

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_initializations_explorer -v`
Expected: FAIL because helper module does not exist yet

- [ ] **Step 3: Write minimal implementation**

Create a pure helper that:
- accepts `Nx`, `Ny`, `offset`, `spacing`, `deviation`, `threshold`, `seed`
- simulates final nuclei points using local RNG
- returns final points and simple counts

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_initializations_explorer -v`
Expected: PASS

### Task 2: Build the explorer UI and figure helpers

**Files:**
- Create: `ui/initializations_explorer.py`

- [ ] **Step 1: Add builder helpers**

Implement:
- control panel builder
- stats/explanation builder
- Plotly figure builder

- [ ] **Step 2: Keep v1 focused**

Ensure the figure shows:
- only final nuclei points
- faint grid/domain background

- [ ] **Step 3: Verify syntax**

Run: `python3 -m py_compile ui/initializations_explorer.py utils/initializations_explorer.py`
Expected: PASS

### Task 3: Wire the new top-level tab into the layout

**Files:**
- Modify: `ui/layout.py`
- Modify: `callbacks/tab_manager.py`

- [ ] **Step 1: Add the tab**

Add `Initializations Explorer` to the top-level tabs in `ui/layout.py`.

- [ ] **Step 2: Add content container**

Add a dedicated `initializations-content` container in `ui/layout.py`.

- [ ] **Step 3: Update top-level tab rendering**

Update `callbacks/tab_manager.py` so the new content is shown only when the explorer tab is active.

- [ ] **Step 4: Update sidebar visibility logic**

Hide the main sidebar selectors while on the explorer tab.

### Task 4: Register the new callback manager

**Files:**
- Create: `callbacks/initializations_explorer_manager.py`
- Modify: `callbacks/manager.py`
- Modify: `callbacks/__init__.py`
- Modify: `ui/__init__.py`

- [ ] **Step 1: Add a minimal callback manager**

Create the manager class even if v1 has little or no dynamic callback logic.

- [ ] **Step 2: Register and export it**

Update imports/exports so the manager is registered consistently with the rest of OPView.

### Task 5: Final verification

**Files:**
- Verify all touched files

- [ ] **Step 1: Run tests**

Run: `python3 -m unittest tests.test_initializations_explorer -v`
Expected: PASS

- [ ] **Step 2: Run syntax verification**

Run: `python3 -m py_compile ui/initializations_explorer.py utils/initializations_explorer.py callbacks/initializations_explorer_manager.py ui/layout.py callbacks/tab_manager.py callbacks/manager.py callbacks/__init__.py ui/__init__.py`
Expected: PASS

- [ ] **Step 3: Report outcome**

Summarize what changed, what was verified, and any remaining runtime-only risks.
