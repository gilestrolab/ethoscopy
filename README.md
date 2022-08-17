**ethoscopy**

A data-analysis toolbox utilising the python language for use with data collected from 'Ethoscopes', a Drosophila video monitoring system.

For more information on the ethoscope system: https://www.notion.so/The-ethoscope-60952be38787404095aa99be37c42a27

Ethoscopy is made to work alongside this system, working as a post experiment analysis toolkit. 

Ethoscopy provides the tools to download epxerimental data from a remote ftp servers as setup in ethoscope tutorial above. Downloaded data can be curated during the pipeline in a range of ways, all fromatted using the pandas data structure.

Further the ethoscopy package provides behavpy a subclassed version of pandas that combines metadata with the data for easy manipulation. Behavpy can be used independantly of the Ethoscope system data if following the same structure.

Within Behavpy is a wrapped version of the python package hmmlearn, a tool for creating hidden markov models. With the update the user can easy train new HMMs from their data.

**To Come**
Plotting functions to complement the addtioanal hmmlearn update

