# ethoscopy

[![CI/CD Pipeline](https://github.com/gilestrolab/ethoscopy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gilestrolab/ethoscopy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gilestrolab/ethoscopy/branch/main/graph/badge.svg)](https://codecov.io/gh/gilestrolab/ethoscopy)
[![PyPI version](https://img.shields.io/pypi/v/ethoscopy.svg)](https://pypi.org/project/ethoscopy/)
[![Python versions](https://img.shields.io/badge/python-3.12-blue.svg)](https://pypi.org/project/ethoscopy/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A data-analysis toolbox utilising Pandas, Seaborn, and Plotly to curate, clean, analyse, and visualise behavioural time series data. Whilst the toolbix was created around the data produced from an Ethoscope (a Drosophila monitoring system), if the users data follows the same structure for time series data all methods can be utilised.

Head to the [tutorial](https://bookstack.lab.gilest.ro/books/ethoscopy) for an in-depth walk through. The Jupyter notebooks in [`tutorial_notebook/`](tutorial_notebook/) are the authoritative, runnable examples.

For more information on the Ethoscope system, click [here](https://www.notion.so/The-ethoscope-60952be38787404095aa99be37c42a27)
    - If using in conjenction with Ethoscope data this software contains functions for loading the Ethoscope data into ethocopy from .db files both locally and in remote ftp servers.

At its core ethoscopy is a subclass of the data manipulation tool Pandas. The dataframe object has been altered to contain a linked metadata dataframe which contains experimental information. This secondary dataframe can be used to filter the data containing dataframe, as well as a store of information per specimen during analysis.

Ethoscopy contains methods to perform common analytical techniques per specimen in the data table, such as removing dead specimens, interpolating missing values, or calculating sleep from movement. Addtionally, specialist anlysing tools have been implemented for analysing circadian rhythm, such as periodograms, and for generating hidden Markov models (HMM) to understand latent behavioural states. HMMs are trained utilising hmmlearn in the background and come accompanied with a range of visualisation tools to understand the generated model.

### -- Update to 2.0 --

This new update sees a whole refactoring of the code base to make everything more streamline and keep the package up to date with the new versions of pandas and numpy. Gone are seperate classes for periodograms and HMM based analysis, all are under one class behavpy(). Addtioanlly, now the user can choose between plotter packages, Seaborn and Plotly, and choose a desired colour pallete. The previous used package Plotly can balloon the size of jupyter notebooks, putting a strain on storage, despite being great for data exploration. If you just want static plots, use Seaborn. But be wary of comparison, the backend for Plotly plots is all calculated in ethoscopy applying z-score and bootstrapping to quantification plots, whereas Seaborn based plots will use the Seaborn internal tools for errors and averaging.

The latest update is backwards compatible with all previously saved behavpy dataframes. However, post loading they should be re-initiated as the new behavpy class. See in the getting started a demonstration of what to do.

Addtionally, the concat method ( behavpy_object.concat() ) for combining dataframes has been shifted to a function that is imported automatically. Call etho.concat(df1, df2) or etho.concat(*[df1, df2]) instead. There are other minor changes to method and argument names, which are reflected in their docstrings and in the tutorial.

## Getting Started

Ethoscopy can be installed via pip from [PyPi](https://pypi.org/project/ethoscopy/)

We recommned installing ethoscopy into a virtual environment due to specific package versions.

```bash
python pip install ethoscopy
```

## Example of use

Ethoscopy is primarily made to work in a Jupyter notebook environment and should be imported in as so:

```bash
import ethoscopy as etho
```

Generate a behavpy dataframe object as so:

```bash
data = pandas_dataframe
metadata = pandas_dataframe

df = etho.behavpy(data, metadata, check = True, canvas = 'plotly', palette = 'Set2')

# select only the data from specimens in experimental group 2
filtered_df = df.xmv('experimental_column', 'group_2')
```

Loading and re-initialising old data (saved pre 2.0)
```bash
import ethoscopy as etho
import pandas as pd

df = pd.read_pickle('path/to/your/file.pkl')
df = etho.behavpy(df, df.meta, check = True, canvas = 'plotly', palette = 'Set2')
```

## Tutorial data

The six pickle files used by the tutorial notebooks (~36 MB total, dominated by `overview_data.pkl` at ~31 MB) are **intentionally not shipped with the PyPI wheel** to keep `pip install ethoscopy` lean. Fetch them once with:

```python
import ethoscopy as etho
etho.download_tutorial_data()   # idempotent — skips files already on disk
```

By default this populates `~/.cache/ethoscopy/tutorial_data/` (user-writable on every platform, no root required). After that, `get_tutorial('overview')`, `get_tutorial('circadian')` and `get_HMM('M'|'F')` Just Work.

**Lookup order.** At load time, ethoscopy checks three locations in this order:

1. `<site-packages>/ethoscopy/misc/tutorial_data/` — used by dev/editable installs and by the project's Docker image (which pre-populates it during build; see `Docker/Dockerfile`).
2. `$ETHOSCOPY_TUTORIAL_DATA_DIR` if set — useful for shared clusters where one admin-maintained copy serves many users.
3. `~/.cache/ethoscopy/tutorial_data/` — the default for `download_tutorial_data()`.

**Manual download.** The canonical URLs live at
<https://github.com/gilestrolab/ethoscopy/tree/main/src/ethoscopy/misc/tutorial_data>.
Place the six `*.pkl` files in any of the three locations above.

## Development and Testing

Ethoscopy includes a comprehensive test suite to ensure code reliability and prevent regressions.

### Running Tests

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=ethoscopy --cov-report=term-missing
```

Or use the convenience script:
```bash
python run_tests.py
```

### Test Categories

- **Unit tests**: Fast tests for individual functions (`pytest -m unit`)
- **Integration tests**: Workflow testing (`pytest -m integration`)
- **Performance tests**: Long-running tests (`pytest -m slow`)

For detailed testing information, see [TESTING.md](TESTING.md).

## License

This project is licensed under the [GNU-3 license](LICENSE)
