This repository holds code for the paper *Earthmover-Based Manifold Learning for Analyzing Molecular Conformation Spaces* by Nathan Zelesko, Amit Moscovich, Joe Kileel, and Amit Singer. earthmover_main.py can be used (along with the files in the modules folder) to generate the type of data and figures seen in the paper. 

One simply needs to set up directories where the computations and figures will be stored and insert the file paths where instructed at the beginning of the earthmover_main.py file. The computations directory must include two subfolders:
- Noiseless Data
- STD = 0.01644027
Each with all of the following subfolders:
- Angles
- Colors
- Euclidean Distance Matrices
- Raw Data
- Times
- WEMD Distance Matrices

The figures directory must include the two subfolders:
- EMD-embeddings-CoifmanLafon
- Euclidean-embeddings-CoifmanLafon

Finally, the file path for the directory containing rotating_shaft_res6.mrc must also be inserted at the beginning of earthmover_main.py.

Once these directories are set up, one can use the function initial_calculations() to perform the initial data generation and computation of distance matrices. The function figure_all() then generates all the figures seen in the paper at once from this data. The function generate_all() will generate both the data and the figures.
