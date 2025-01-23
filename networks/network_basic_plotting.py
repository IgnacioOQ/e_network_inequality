# Plotting functions
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Plotting Functions
def plot_network_degree_distribution(G, directed=True, title='title'):
    if directed:
        degrees = np.array([degree for node, degree in G.in_degree()])
    else:
        degrees = np.array([degree for node, degree in G.degree()])
    # Create the histogram with a KDE
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.histplot(degrees, kde=False, bins=150, stat="count")
    # Calculate the mean
    mean_value = np.mean(degrees)
    print(mean_value)
    print(np.median(degrees))

    # Plot a vertical line at the mean value
    plt.axvline(mean_value, color='b', linestyle='--', linewidth=2)
    plt.text(mean_value + 0.1, plt.ylim()[1] * 0.9, 'Mean: {:.2f}'.format(mean_value), color='b')

    plt.title('Timeline Smooth Histogram for: ' + title)
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.xticks(fontsize=8,rotation=20)
    plt.show()
    
def plot_loglog(G,directed=True,m=10):
    if directed:
        # Get the in-degree of all nodes
        in_degrees = [d for _, d in G.in_degree()]

        # Compute the histogram
        max_degree = max(in_degrees)
        degree_freq = [in_degrees.count(i) for i in range(max_degree + 1)]
    else:
        degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    plt.figure(figsize=(12, 8))
    plt.loglog(degrees[m:], degree_freq[m:],'go-')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Log-Log plot of the degree distribution')