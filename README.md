# Basic Code Structure

Besides the imports.py, the code is organized in groups:

1. Basic Model

This includes agents.py, model.py, and basic_model_testing.ipynb notebook.

To note: The code is constructed so that it works both for directed and undirected networks. Agents get observations from who they are pointing to (i.e. their neighbots). 
This means that the directed networks we are going to use are inverse of the direction flow: A points to B if A cites B (and hence A gets observations from B).

2. Networks

- network_utils.py where the plotting functions, as well as some cleaning functions and the network statistic functions are defined.

3. 
