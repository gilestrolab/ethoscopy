"""
Ethoscopy: A Python package for behavioral analysis of Drosophila data.

This package provides tools for loading, analyzing and visualizing ethoscope data.
"""

import importlib.metadata

from ethoscopy.analyse import (
    max_velocity_detector,
    sleep_annotation,
    stimulus_prior,
    stimulus_response,
)
from ethoscopy.behavpy import behavpy
from ethoscopy.load import (
    download_from_remote_dir,
    link_meta_index,
    load_ethoscope,
    load_ethoscope_diagnostics,
    load_ethoscope_metadata,
)
from ethoscopy.misc.general_functions import concat
from ethoscopy.misc.get_tutorials import download_tutorial_data, get_tutorial

__version__ = importlib.metadata.version("ethoscopy")

__all__ = [
    "behavpy",
    "download_from_remote_dir",
    "download_tutorial_data",
    "get_tutorial",
    "link_meta_index",
    "load_ethoscope",
    "load_ethoscope_diagnostics",
    "load_ethoscope_metadata",
    "max_velocity_detector",
    "sleep_annotation",
    "stimulus_response",
    "stimulus_prior",
    "concat",
]
