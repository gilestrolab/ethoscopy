**ethoscopy**

A data-analysis toolbox utilising the python language for use with data collected from 'Ethoscopes', a Drosophila video monitoring system.

For more information on the ethoscope system: https://www.notion.so/The-ethoscope-60952be38787404095aa99be37c42a27

Ethoscopy is made to work alongside this system, working as a post experiment analysis toolkit. 

Ethoscopy provides the tools to download epxerimental data from a remote ftp servers as setup in ethoscope tutorial above. Downloaded data can be curated during the pipeline in a range of ways, all fromatted using the pandas data structure.

Further the ethoscopy package provides behavpy a subclassed version of pandas that combines metadata with the data for easy manipulation. Behavpy can be used independantly of the Ethoscope system data if following the same structure.

**TO COME** 
The addition of a hidden markov model to train the data on will be added (hmmlearn - https://hmmlearn.readthedocs.io/en/latest/). Here you can set the architecture and train a HMM of your choice. There are several plotting functions avaiable alongside side it to explore the hidden markov model, using plotly as graphing tool of choice.
