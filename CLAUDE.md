# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ethoscopy is a Python data analysis toolbox for behavioral time series data from Ethoscope (Drosophila monitoring system). It's built as a specialized pandas DataFrame subclass that maintains linked metadata and provides analysis/visualization tools for circadian rhythms, sleep patterns, and behavioral states using HMM.

## Development Environment

### Python Environment
- **Use Python 3.12+** (project requires >=3.12,<4.0)
- **Always use virtual environment** (`.venv` as mentioned in user's global instructions)
- **Install in development mode**: `pip install -e .`

### Package Management
- Uses **pyproject.toml** for modern Python packaging (not setup.py for production)
- **Hatchling** as build backend
- **Rye** for dependency management (managed = true in pyproject.toml)

### Key Dependencies
- Core: pandas >=2.2.2, numpy >=2.0.0
- Visualization: plotly >=5.22.0, seaborn >=0.13.2
- Analysis: hmmlearn >=0.3.2, astropy >=7.0, pywavelets >=1.6.0
- Dev: ipykernel for Jupyter notebook support

## Build and Development Commands

### Installation
```bash
# Install in development mode
pip install -e .

# For Jupyter notebook development
pip install -e ".[dev]"
```

### Testing
- **Pytest suite in `tests/`**: `python -m pytest tests/ -q` (~230 tests, seconds to run)
- **Caveat**: `pytest.ini` declares its section as `[tool:pytest]`, which is the
  *setup.cfg* spelling — pytest silently ignores the file, so `addopts` (coverage,
  `--strict-markers`) and the `markers` registrations do **not** apply. Renaming the
  section to `[pytest]` would switch on `--cov-fail-under=70` and `--strict-markers`
  for the whole suite; check it passes before doing so.
- Tutorial notebooks in `tutorial_notebook/` complement the suite for end-to-end checks

### Docker Environment
- **Docker available** for JupyterHub deployment
- Build: `JUPYTER_HUB_TAG=5.3.0 ETHOSCOPE_LAB_TAG=1.0 docker compose build`
- Run: `docker compose up -d` (from Docker/ directory)

### Opening databases: WAL mode and read-only mounts

Ethoscopes record in **WAL** (Write-Ahead Logging) mode and the results tree is
normally mounted read-only (`:ro` in Docker). That combination has broken loading
repeatedly — as `database disk image is malformed`, and as
`unable to open database file` on `SELECT * FROM ROI_MAP`.

**How `_connect_db()` handles it** (load.py)

ethoscopy never writes to these files, so the connection is *always* read-only —
this also stops a load from checkpointing or truncating raw data as a side effect.
It walks an ordered ladder of open modes, `_READ_STRATEGIES`, and **probes each
one with a real statement** before returning it:

| rung | reads | notes |
|---|---|---|
| `mode=ro` | every recoverable state, incl. WAL with `-shm`, corrupt `-shm`, hot rollback journal | the only mode that sees data still in an uncheckpointed `-wal` |
| `immutable=1` | WAL database whose sidecars are absent on a read-only mount | ignores the `-wal`; warns via `_warn_if_wal_ignored()` if it would hide committed rows |

Two traps worth not rediscovering:

- **Never use `mode=ro&nolock=1`.** SQLite rejects it for *every* WAL database.
- **`sqlite3.connect()` never touches the file**, so a bad URI opens "successfully"
  and only fails on the first query — past any `try/except` around the connect
  call. That is why every rung is probed, and it is how the `nolock=1` bug survived
  a fallback that looked like it covered the case.
- `immutable=1` will also "open" a **missing** path, *creating* an empty database
  and reporting `no such table: ROI_MAP`. `_connect_db()` checks the file exists first.

`read_single_roi_optimized()` retries a failed read with `_connect_db(path,
degraded=True)`, which skips to the last rung — a retry with the default ladder
would just pick the same failing mode again.

**Optional**: convert databases to DELETE mode to avoid the question entirely
```bash
python3 scripts/convert_wal_to_delete.py /mnt/ethoscope_data/results --verbose
./scripts/convert_databases.sh /mnt/ethoscope_data/results   # bash wrapper
```

**Tests**: `tests/test_load_wal.py` builds real SQLite files in each on-disk state
(read-only and writable directories) rather than mocking sqlite3 — the failures
being guarded against come from SQLite itself and only appear when a statement runs.

**See Also**: `Docker/README.md` for detailed database preparation instructions

## Architecture and Code Structure

### Core Architecture
- **behavpy_core**: Base DataFrame subclass with core functionality (xmv, curate, summary)
- **behavpy**: Main user-facing class for backward compatibility
- **Canvas System**: Supports both 'plotly' and 'seaborn' for visualization backends

### Key Components
1. **Data Loading** (`load.py`): FTP download, ethoscope database loading, metadata linking
2. **Analysis** (`analyse.py`): Sleep annotation, velocity detection, stimulus response
3. **Visualization**: Split between plotly (`behavpy_plotly.py`) and seaborn (`behavpy_seaborn.py`)
4. **Specialized Classes**:
   - `behavpy_HMM_class.py`: Hidden Markov Model analysis
   - `behavpy_periodogram_class.py`: Circadian rhythm analysis
5. **Utilities** (`misc/`): General functions, validation, tutorial data

### Data Structure
- **Dual DataFrame design**: Main data + linked metadata via shared 'id' column
- **Metadata cleaning**: Automatically removes columns like 'path', 'file_name', 'file_size', 'machine_id'
- **Index requirement**: Data must have 'id' as index name

### Import Pattern
```python
import ethoscopy as etho
df = etho.behavpy(data, metadata, check=True, canvas='plotly', palette='Set2')
```

## Key Design Patterns

### Version 2.0 Migration
- **Backward compatible** with pre-2.0 pickled data
- **Unified class structure**: All analysis under single `behavpy()` class
- **Canvas selection**: Choose between plotly/seaborn at initialization
- **New concat function**: Use `etho.concat()` instead of `behavpy_object.concat()`

### Plotting Architecture
- **Dual backend support**: Plotly (interactive) vs Seaborn (static)
- **Built-in statistical processing**: Z-score normalization, bootstrapping for plotly
- **Seaborn backend**: Uses seaborn's internal statistics
- **Performance consideration**: Plotly can create large notebook files

### Analysis Capabilities
- **Sleep detection**: Movement-based sleep annotation
- **Circadian analysis**: Periodograms (Lomb-Scargle, Fourier, Wavelet)
- **HMM behavioral states**: Using hmmlearn with visualization tools
- **Data curation**: Dead specimen removal, interpolation, filtering

## Development Guidelines

### Code Organization
- **src/ethoscopy/**: Main package code
- **tutorial_notebook/**: Jupyter notebooks for testing and examples
- **Docker/**: JupyterHub deployment configuration
- **Keep files under 500 lines** (per user's global instructions)

### Testing Approach
- **Use Jupyter notebooks** for validation and testing
- **Tutorial notebooks serve as integration tests**
- **Always test with both canvas options** (plotly/seaborn)

### Documentation
- **Docstrings**: Google style required for all functions
- **Tutorial notebooks**: Primary documentation method
- **README.md**: Keep updated with installation and basic usage
