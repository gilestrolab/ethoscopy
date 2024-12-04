**ethoscopy**

A data-analysis toolbox utilising Pandas, Seaborn, and Plotly to curate, clean, analyse, and visualse behavioural time series data. Whilst the toolbix was created around the data produced from an Ethoscope (a Drosophila monitoring system), if the users data follows the same structure for time series data all methods can be utilised.

Head to the [tutorial](https://bookstack.lab.gilest.ro/books/ethoscopy) for an in-depth walk through.

For more information on the Ethoscope system, click [here](https://www.notion.so/The-ethoscope-60952be38787404095aa99be37c42a27)
    - If using in conjenction with Ethoscope data this software contains functions for loading the Ethoscope data into ethocopy from .db files both locally and in remote ftp servers.

At its core ethoscopy is a subclass of the data manipulation tool Pandas. The dataframe object has been altered to contain a linked metadata dataframe which contains experimental information. This secondary dataframe can be used to filter the data containing dataframe, as well as a store of information per specimen during analysis.

Ethoscopy contains methods to perform common analytical techniques per specimen in the data table, such as removing dead specimens, interpolating missing values, or calculating sleep from movement. Addtionally, specialist anlysing tools have been implemented for analysing circadian rhythm, such as periodograms, and for generating hidden Markov models (HMM) to understand latent behavioural states. HMMs are trained utilising hmmlearn in the background and come accompanied with a range of visualisation tools to understand the generated model.

-- Update to 2.0.0 --

This new update sees a whole refactoring of the code base to make everything more streamline and keep the package up to date with the new versions of pandas and numpy. Gone are seperate classes for periodograms and HMM based analysis, all are under one class behavpy(). Addtioanlly, now the user can choose between plotter packages, Seaborn and Plotly, and choose a desired colour pallete. The previous used package Plotly can balloon the size of jupyter notebooks, putting a strain on storage, despite being great for data exploration. If you just want static plots, use Seaborn. But be wary of comparison, the backend for Plotly plots is all calculated in ethoscopy applying z-score and bootstrapping to quantification plots, whereas Seaborn based plots will use the Seaborn internal standard error tools.

The latest update is backwards compatible with all previously saved behavpy dataframes. However, post loading they should be re-initiated as the new behavpy class. 

Addtionally, behavpy_object.concat() for combining dataframes has been shifted to a function that is imported automatically. Call etho.concat(df1, df2) or etho.concat(*[df1, df2]) for conbine dataframes and their metadata. There are other minor changes to method and argument names, which are reflected in their docstring and the tutorial. 

## Getting Started

Ethoscopy can be installed via pip from [PyPi](https://pypi.org/project/ethoscopy/)

We recommned installing ethoscopy into a virtual environment due to specific pacakge versions.

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

## License

This project is licensed under the [GNU-3 license](LICENSE)