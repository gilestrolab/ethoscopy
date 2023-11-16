**ethoscopy**

Head to: https://bookstack.lab.gilest.ro/books/ethoscopy for an in-depth tutorial on how to use ethoscopy

A data-analysis toolbox utilising the python language for use with data collected from 'Ethoscopes', a Drosophila video monitoring system.

For more information on the ethoscope system: https://www.notion.so/The-ethoscope-60952be38787404095aa99be37c42a27

Ethoscopy is made to work alongside this system, working as a post experiment analysis toolkit. 

Ethoscopy provides the tools to download experimental data from a remote ftp servers as setup in ethoscope tutorial above. Downloaded data can be curated during the pipeline in a range of ways, all formatted using the pandas data structure.

Further the ethoscopy package provides behavpy a subclassed version of pandas that combines metadata with the data for easy manipulation. Behavpy can be used independently of the Ethoscope system data if following the same structure. Within behavpy there are a range of methods to curate your data and then to generate plots using the plotly plotting package. Additionally, there are methods to analyse bout length, contiguous sleep, and many circadian analysis methods including periodograms.

Within Behavpy is a wrapped version of the python package hmmlearn, a tool for creating hidden markov models. With the update the user can easy train new HMMs from their data and use bult in methods to create graphs analysing the decoded dataset.

-- Update to 1.3.0 --

The latest update now allows the user to choose the colour palette from plotly to be applied to all plotting methods, see the updated tutorial on how to do this. 
Adtioanlly, the .pivot() method has been changed to .analyse_column() to avoid overwriting the original pandas method. Also, puff_mago and find_motifs analysing functions have been renamed to stimulus_response and stimulus_prior respectively.