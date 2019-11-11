This repository holds code for the paper *Earthmover-Based Manifold Learning for Analyzing Molecular Conformation Spaces* by Nathan Zelesko, Amit Moscovich, Joe Kileel, and Amit Singer. earthmover_main.py can be used (along with the files in the modules folder) to generate the type of data and figures seen in the paper. 

Prqrequisites:
- An installation of Python 3 with SciPy and scikit-learn. If you don't have ths installed, install the Anaconda python distribution.
- The PyWavelets package (included in Anaconda)
- The mrcfile module, to install it run `pip install mrcfile`.

You can then run './produce_all_figures.py' to compute and produce all the figures.

Note that this script must be run from the base project directory, otherwise it can't find the data file `rotating_shaft_res6.mrc`.

