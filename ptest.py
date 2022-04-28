import ethoscopy as etho 
import pandas as pd 

metadata = pd.read_pickle('ac_meta.pkl')
data = pd.read_pickle('ac_puff.pkl')
df = etho.set_behavpy(metadata, data)
df = df.xmv('status', 'ok')
df.display()
