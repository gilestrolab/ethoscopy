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
- **No formal test suite found** - project relies on Jupyter notebooks for validation
- Use the tutorial notebooks in `tutorial_notebook/` for testing functionality
- Run notebooks: `jupyter notebook tutorial_notebook/`

### Docker Environment
- **Docker available** for JupyterHub deployment
- Build: `JUPYTER_HUB_TAG=5.3.0 ETHOSCOPE_LAB_TAG=1.0 docker compose build`
- Run: `docker compose up -d` (from Docker/ directory)

### Troubleshooting: Database "Malformed" Errors

**Problem**: Intermittent "database disk image is malformed" errors when loading ethoscope data in Docker

**Root Cause**: SQLite databases in WAL (Write-Ahead Logging) mode on read-only Docker mounts

**Solution Options**:
1. **Convert databases to DELETE mode** (recommended for immediate fix)
   ```bash
   # Using the conversion script (from project root)
   python3 scripts/convert_wal_to_delete.py /mnt/ethoscope_data/results --verbose

   # Or using bash wrapper
   ./scripts/convert_databases.sh /mnt/ethoscope_data/results
   ```

2. **Improved connection handling** (v2.0.4+)
   - The `_connect_db()` function in `load.py` now detects WAL mode automatically
   - Uses `mode=ro&nolock=1` URI parameters for WAL databases on read-only mounts
   - Includes retry logic in `read_single_roi_optimized()` for resilience

**Key Implementation Details** (load.py):
- Lines 18-70: `_connect_db()` - Smart connection with WAL detection and appropriate parameters
- Lines 943-979: Retry logic in `read_single_roi_optimized()` - Handles transient errors with fresh connections

**Testing After Changes**:
```python
import ethoscopy as etho
metadata = etho.link_meta_index('metadata.csv', '/mnt/ethoscope_results')
data = etho.load_ethoscope(metadata, reference_hour=9.0, FUN=etho.sleep_annotation)
# Should load all ROIs without "malformed" errors
```

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
