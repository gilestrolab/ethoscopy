from ethoscopy.behavpy import behavpy
from ethoscopy.load import download_from_remote_dir, link_meta_index, load_ethoscope, load_ethoscope_metadata
from ethoscopy.analyse import max_velocity_detector, sleep_annotation, stimulus_response, stimulus_prior
from ethoscopy.misc.static_functions import concat

import importlib.metadata

__version__ = importlib.metadata.version("ethoscopy")