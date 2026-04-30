# Notebook Markdown Polish Design

## Goal

Improve notebook markdown UX without changing the renderer stack or attempting Jupyter-level parity.

## Scope

- Keep `marked` + `KaTeX` + Monaco.
- Add explicit markdown edit/preview controls.
- Improve markdown preview styling for headings, lists, blockquotes, tables, links, and code blocks.
- Add a lightweight markdown help panel with supported syntax examples.

## Non-Goals

- No renderer replacement.
- No `.ipynb` compatibility work.
- No rich markdown plugin ecosystem.
- No notebook execution model changes.

## Chosen Approach

- Extend markdown cell controls with a visible `Edit` action in addition to `Preview`.
- Reuse the existing floating panel pattern already used by the functions reference for markdown help.
- Improve only CSS and small clientside callback wiring so the change stays low-risk and easy to maintain.

## Expected Outcome

- Markdown cells feel more explicit and notebook-like.
- Preview looks more polished with minimal code churn.
- Users can discover supported markdown/math syntax directly from the notebook UI.
