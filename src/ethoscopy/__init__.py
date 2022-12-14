from ethoscopy.misc.save_fig import save_figure
from ethoscopy.behavpy_class import behavpy
from ethoscopy.behavpy_HMM_class import behavpy_HMM
from ethoscopy.behavpy_periodogram_class import behavpy_periodogram
from ethoscopy.load import download_from_remote_dir, link_meta_index, load_ethoscope
from ethoscopy.analyse import max_velocity_detector, sleep_annotation, puff_mago, find_motifs, isolate_activity_lengths

import importlib.metadata

__version__ = importlib.metadata.version("ethoscopy")