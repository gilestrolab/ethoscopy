import ethoscopy as etho
print(etho.__version__)
from ethoscopy.load import download_from_remote_dir, link_meta_index, load_ethoscope
from ethoscopy.analyse import max_velocity_detector

from functools import partial

# meta = r'C:\Users\lab\Documents\Projects\testing_hmm\odours_v2\test_CS.csv' # replace with your own
# remote = 'ftp://turing.lab.gilest.ro/auto_generated_data/ethoscope_results'
# local = r'C:\Users\lab\Documents\ethoscope_databases' # replace with your own

# download_from_remote_dir(meta, remote, local)

# meta = link_meta_index(meta, remote, local)
# print(meta)

# data = load_ethoscope(meta, reference_hour = 9.0, FUN = partial(max_velocity_detector, time_window_length = 60, raw = True))
# print(data)

# df = etho.set_behavpy(meta, data)

# df.display()