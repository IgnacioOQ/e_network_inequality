# Basic Code Structure

Besides the imports.py, the code is organized in groups:

1. Basic Model

This includes agents.py, model.py, and basic_model_testing.ipynb notebook.

To note: The code is constructed so that it works both for directed and undirected networks. Agents get observations from who they are pointing to (i.e. their neighbots). 
This means that the directed networks we are going to use are inverse of the direction flow: A points to B if A cites B (and hence A gets observations from B).

2. Networks

- network_utils.py where the plotting functions, as well as some cleaning functions and the network statistic functions are defined.
  - Since we want to make it compatible with both directed and undirected most functions have a "directed" input parameter set as default as True.
  - Since we constructed the networks inverse to the direction of flow, the relevant degree (for gini, entropy, etc.) is the in_degree.
  - Since some network properties might not be defined, there are some ad-hoc definitions to deal with this.
  - Advanced network plotting functions are not defined.
- network_generation.py defines the generative models that we are going to use.
- network_randomization.py is a single file containing the network randomization function that is at the core of our project.
- getting_citations_networks.ipynb (still unclean) is a self contained notebook that we used to generate the empirical networks used for the study.
- Both the (clean) peptic ulcer and the perceptron networks are in the repository.
- network_testing.ipynb tests the functions defined here.

3. Simulations

- TBU
- Simulations will be done on Google Colab to make use of parallelization. So I will just put the notebook here.

4. Results Analysis

- Here the regression analysis.
