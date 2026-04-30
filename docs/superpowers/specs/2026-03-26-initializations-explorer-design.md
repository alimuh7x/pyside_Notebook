# Initializations Explorer Design

## Goal

Add a new top-level OPView tab called `Initializations Explorer` that helps users visually understand OpenPhase initialization methods. The first supported method is a 2D teaching/inspection view of `Initializations::QuasiRandomNuclei(...)`.

## Scope

### Included in v1

- New top-level tab: `Initializations Explorer`
- Method selector inside the tab
- First method: `QuasiRandomNuclei`
- 2D controls for:
  - `Nx`, `Ny`
  - `offsetX`, `offsetY`
  - `spacingX`, `spacingY`
  - `deviationX`, `deviationY`
  - `threshold`
  - `seed`
- Plotly visualization showing only the final nuclei points
- Faint domain/grid background so spacing and offset remain understandable
- Simple stats:
  - final nuclei count
- Short plain-language explanation of the current settings
- Implementation kept isolated from Formula Plot, Custom Graph, and Calculation Notebook

### Excluded from v1

- 3D visualization
- Intermediate layers (candidate points, accepted-before-deviation points, rejected points)
- Direct coupling to OpenPhase runtime objects
- Additional initialization methods beyond `QuasiRandomNuclei`

## User Experience

The tab is intended as a teaching and exploration tool. Users adjust sliders and immediately see how the final planted nuclei pattern changes. The visual should answer questions like:

- What does larger spacing do?
- How does offset move the lattice origin?
- How much does deviation scatter the nuclei?
- What happens when threshold is increased?
- How does changing the seed affect the final pattern?

The page should feel lightweight and intuitive, like an interactive explanation rather than a heavy workflow.

## Data and Logic Model

The 2D simulator will mirror the OpenPhase function logic for the x/y plane:

1. Compute effective lattice origin:
   - `distx = offsetX if offsetX < Nx and spacingX < Nx else 0`
   - `disty = offsetY if offsetY < Ny and spacingY < Ny else 0`
2. Loop over grid candidates:
   - `for i in range(distx, Nx, spacingX)`
   - `for j in range(disty, Ny, spacingY)`
3. Draw a random number `chance`
4. Accept only when `chance > threshold`
5. Draw random deviations `di`, `dj` within the same bounds used by the OpenPhase code
6. Keep the final point only if it remains inside the domain

To keep behavior deterministic in Dash, the Python implementation will use a local seeded RNG instead of global `rand()` state.

## Architecture

### New files

- `ui/initializations_explorer.py`
  - builds the tab layout
  - builds the Plotly figure
  - generates the explanation block
- `callbacks/initializations_explorer_manager.py`
  - wires UI interactions if needed
  - v1 can stay callback-light if the figure is generated directly from component inputs
- `utils/initializations_explorer.py`
  - pure simulation helper for quasi-random nuclei generation
  - safe to unit test independently
- `tests/test_initializations_explorer.py`
  - focused tests for the simulation helper

### Existing files to update

- `ui/layout.py`
  - add the new top-level tab
  - add the new content container
- `callbacks/tab_manager.py`
  - show/hide the new tab content
  - hide sidebar selectors while on the explorer tab
- `callbacks/manager.py`
  - register the new callback manager
- `callbacks/__init__.py`
  - export the new manager
- `ui/__init__.py`
  - export the new builder helpers

## Visual Design

The page should follow the OPView visual language:

- left controls card
- main scatter plot to the right
- short explanation and stats below the controls
- simple colors and minimal ornament

The plot should show:

- a faint rectangular domain
- subtle grid lines
- final nuclei as clear markers

## Testing Strategy

The risky part is the simulation logic, so v1 tests will focus on the pure helper:

- correct effective origin handling
- deterministic output for a given seed
- threshold behavior
- boundary rejection behavior
- deviation-free behavior when deviations are zero

UI integration will be verified with syntax checks and focused runtime smoke checks.
