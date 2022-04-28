from setuptools import setup, find_packages

VERSION = '0.1.0' 
DESCRIPTION = 'ethoscopy - a python toolbox for the Ethoscope'

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

setup(
        name = "ethoscopy", 
        version = VERSION,
        author = "Laurence Blackhurst",
        author_email = "l.blackhurst19@imperial.ac.uk",
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        packages = find_packages(),
        install_requires=['pandas', 'numpy', 'sqlite3', 'ftplib', 'urllib', 'math', 'pickle'],
        keywords=['python', 'ethomics', 'ethoscope', 'sleep'],
        classifiers= [
            "Development Status :: 1 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: UNIX",
            "Operating System :: MacOS :: MacOS X"
        ],
        python_requires = ">=3.6"
)
