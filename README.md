# Bangkok CCTV Image Analysis

A university Data Science and Data Engineering project that will turn raw
Bangkok CCTV images into structured traffic observations, integrate external
data collected through web scraping, analyze traffic patterns, and present
insights through an interactive Streamlit dashboard.

This repository contains the initial scaffold only. No datasets, model weights,
predictions, scraping, or analysis results are included. Workflow functions raise
`NotImplementedError` until implemented; only the dashboard landing page runs.

## Team responsibilities

| Owner | Responsibility |
| --- | --- |
| Developer 1 | `engineering/`: ingestion, detection, preprocessing, validation |
| Developer 2 | `scraping/` and `analysis/`: external data and traffic analysis |
| Developer 3 | `dashboard/` and integration |
| Shared | Configuration, documentation, testing, and presentation |

Use small branches and peer-reviewed changes. Agree on the provisional data
schema before integrating modules.

## Windows installation

Use Python 3.11 or newer. Run these commands in PowerShell from this repository:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Replace `-3.11` with an installed supported version if needed.
If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place
of `python` in the commands; activation is optional.

## Structure

- `data/raw/cctv/` and `data/raw/external/`: original local inputs.
- `data/interim/` and `data/processed/`: intermediate and curated outputs.
- `src/bangkok_cctv/`: importable package using a src layout.
- `engineering/`, `scraping/`, `analysis/`, `dashboard/`: team-owned modules.
- `configs/config.yaml`: paths relative to the repository root.
- `models/`: locally supplied model weights; ignored by Git.
- `notebooks/`: exploratory notebooks.
- `scripts/`: future workflow entry points; currently unimplemented.
- `tests/`: basic module import tests.
- `docs/`: architecture and provisional dataset schema.
- `presentation/`: course presentation materials.

Dataset directories preserve only their `.gitkeep` files in Git.
Keep secrets in a local `.env`; `.env.example` is a template only.
Automatic environment-variable or configuration loading is not implemented.

## Run the dashboard

From the repository root after installation:

```powershell
python -m streamlit run src/bangkok_cctv/dashboard/app.py
```

The landing page describes the project without loading data or model weights.
The modules under `dashboard/pages/` are placeholders for future views.

## Validation

```powershell
python -m pytest
python -m ruff check .
```

## Planned data pipeline

1. Ingest locally provided CCTV images while preserving originals.
2. Detect vehicles using an agreed model and locally supplied weights.
3. Preprocess observations and validate counts, identifiers, and timestamps.
4. Collect approved external sources and retain their provenance.
5. Integrate compatible external observations with traffic records.
6. Explore temporal and spatial patterns using validated processed data.
7. Present verified findings in the dashboard.

See `docs/architecture.md` and `docs/data_dictionary.md`.
