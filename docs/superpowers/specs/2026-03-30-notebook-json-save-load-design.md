# Notebook JSON Save/Load Design

## Goal

Make Calculation Notebook JSON save/load preserve the full notebook input state needed to rerun the notebook after loading, without persisting computed results.

## Scope

- Save only the current Calculation Notebook state.
- Persist enough information to reconstruct the notebook inputs exactly after load.
- Do not persist computed outputs or derived variables as source of truth.
- Loading a saved file should restore notebook inputs; pressing run should regenerate results from that restored state.

## Current Problem

The current save callback writes a minimal JSON payload from `notebook-cells-store` that only includes `id`, `type`, and `source`. In practice, the store can be stale relative to the Monaco editors, so recently edited notebook content may be missing from the saved JSON even after the user has run the notebook. Load also rebuilds state from that limited payload, which means saved files are not a reliable rerunnable snapshot of the notebook input state.

## Chosen Approach

Use a versioned notebook JSON schema and save from a synchronized snapshot of the current notebook cells.

- Add serializer/deserializer helpers in the notebook manager.
- Capture the current Monaco editor contents before save so the payload reflects the notebook as the user sees it.
- Persist only canonical notebook inputs:
  - cell order
  - cell id
  - cell type
  - cell source
- Exclude outputs, variables, array variables, and other derived execution artifacts from the file.

## Load Behavior

- Accept the versioned JSON payload going forward.
- Reconstruct notebook cells with normalized defaults for missing optional fields.
- Reset outputs and execution-derived state on load.
- Leave the notebook ready for rerun.

## Compatibility

- Continue accepting legacy `.txt` notebook loads.
- Continue accepting older JSON payloads that contain a top-level `cells` array.
- New saves use the canonical versioned schema.

## Validation

- Saving a notebook with multiple code and markdown cells preserves order, ids, types, and source text.
- Saving uses the latest editor content rather than stale store content.
- Loading the saved JSON rebuilds notebook state with empty outputs and clean rerunnable cells.
