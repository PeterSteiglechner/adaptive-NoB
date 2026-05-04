# ANB Model — Adaptive Networks of Beliefs

Simulation code accompanying our recent project on Adaptive Networks of Beliefs (ANB)

## Setup & Run

This project uses [uv](https://docs.astral.sh/uv/) to manage python packages (see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/))

```bash
uv sync
uv run 2026-04_ANBmodel.py
```

> **Alternative (classic venv):** Create and activate a virtual environment manually, install dependencies via `pip`, and run `2026-04_ANBmodel.py` directly.


## Configuration

In `2026-04_ANBmodel.py`, adjust the following variables before running:

| Mode | `detail` | `seeds` | `pressures` |
|------|----------|---------|-------------|
| Single detailed simulation | `True` | `[1]` | `[4]` |
| Ensemble | `False` | `range(100)` | `[1,2,4,8,16]` |

Per default, this simulates the model across four BN configurations: *static-low*, *static-high*, *adaptive→static*, and *fully adaptive*.

For testing, choose a smaller number of random seeds.

Detailed mode stores all time steps.

## Analysis Pipeline

Run scripts in order after simulations (with settings as in the table above) are complete:

1. **`2026-04_timeseries.py`** — Time series figure from a single simulation
2. **`2026-04_analyseResponses.py`** — Single simulation with external pressure + ensemble response frequencies across BN configurations
3. **`2026-04_metric.py`** — Table and figure of agent-level metrics (belief networks, belief content, social embeddedness, dissonance)
4. **`2026-04_social.py`** — Response frequencies and BN changes under varying social adaptation
   - ⚠️ Requires additional simulations — use the `param_combis` list in this file to run with `mu > 0`. Set `pressures = [4]` and `detail = False`,
5. **`2026-04_sa.py`** — Sensitivity analysis: response frequencies under one-at-a-time parameter variation
   - ⚠️ Requires additional simulations — vary `beta`, `link_prob`, `M`, `N`, and `tau` separately. Set `mu = 0`, `pressures = [4]`, and `detail = False`.
