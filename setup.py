#!/usr/bin/env python

from setuptools import setup, find_packages
import importlib.metadata

VERSION = importlib.metadata.version("ethoscopy")
DESCRIPTION = 'A python based toolkit to download and anlyse data from the Ethoscope hardware system.'

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

setup(
        name = "ethoscopy", 
        version = VERSION,
        author = "Laurence Blackhurst",
        author_email = "l.blackhurst19@imperial.ac.uk",
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        packages = find_packages('src'),
        package_dir = {'' : 'src'},
        install_requires=['pandas >= 1.4', 'numpy >= 1.22',],
        keywords=['python', 'ethomics', 'ethoscope', 'sleep'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: MacOS :: MacOS X"
        ],
        python_requires = ">=3.6"
)
