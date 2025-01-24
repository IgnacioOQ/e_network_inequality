# Basic Code Structure

Besides the imports.py, the code is organized in groups:

1. Basic Model

This includes agents.py, model.py, and basic_model_testing.ipynb notebook.

To note: The code is constructed so that it works both for directed and undirected networks. Agents get observations from who they are pointing to (i.e. their neighbots). 
This means that the directed networks we are going to use are inverse of the direction flow: A points to B if A cites B (and hence A gets observations from B).

2. Networks

- network_utils.py where the plotting functions, as well as some cleaning functions and the network statistic functions are defined.
- network_generation.py defines the generative models that we are going to use.
- network_randomization.py is a single file containing the network randomization function that is at the core of our project.
- getting_citations_networks.ipynb (still unclean) is a self contained notebook that we used to generate the empirical networks used for the study.
- networ_testing.ipynb tests the functions defined here.

  
3. 
