import ethoscopy as etho
print(etho.__version__)

from functools import partial

meta = r'C:\Users\lab\Documents\Projects\testing_hmm\odours_v2\test_CS.csv' # replace with your own
remote = 'ftp://turing.lab.gilest.ro/auto_generated_data/ethoscope_results' #ftp server
local = r'C:\Users\lab\Documents\ethoscope_databases' # replace with your own

etho.download_from_remote_dir(meta, remote, local)

# meta = etho.link_meta_index(meta, remote, local)
# print(meta)

# data = etho.load_ethoscope(meta, reference_hour = 9.0, FUN = partial(etho.max_velocity_detector, time_window_length = 60, raw = True))
# print(data)

# df = etho.set_behavpy(meta, data)

# df.display()