If I recall correctly, Max and I thought it would be best to use the following stopping condition:

* Stop after a fixed number of steps. The literature tends to use 10,000 steps, but we might use a larger number of steps (for instance, 1,000,000) because we know that our simulations converge more slowly. In any case, as a minimal requirement, our stopping condition should be such that it successfully replicates Zollman's original work (I believe 2007 uses beta agents).

  * We could, perhaps, think of a way to stop a run sooner if we have good evidence that it has already converged. One simple option is to run a minimum number of steps and, after that, check every 10k steps whether anything has changed. (This would be akin to our previous tolerance stopping, except that we would increase the number of steps between the prior and post credences.) I am a bit unsure if this would work. 

* Note: We can use cloud services from Utrecht University for this. 


# **Variation methods**

The revised variation method has the following structure:

The method builds a variant of the input network by adding/rewiring `n_edges` edges while trying to preserve two properties of the original: the **degree distribution** and the **clustering coefficient**.

It proceeds in three phases:

1. **Setup**: Optionally removes the same number of edges as will be added (keeping density fixed). Sets the target degree distribution, original distribution for densify or uniform distribution for equalization.  
2. **Main loop**: Adds edges one at a time. Each iteration takes one of two paths: if clustering has fallen below target, it adds an edge to push clustering back up (the interim clustering branch, more info below); otherwise it samples an edge to push the degree distribution to the target (the degree branch, more info below).  
3. **Post clustering**: After all edges are added, performs degree-preserving edge swaps to correct any remaining gap between the achieved and target clustering (post clustering loop, more info below).

Both the interim clustering branch and the post clustering loop are optional.

## **Status**

* ✅ PUD: interim and post clustering both work  
* ✅ Tobacco: both interim and post clustering work for densify, but post clustering works best for equalize  
  * Equalize  
    * Interim clustering ⇒ correlation 0.21 (weak) \[Gini 0.62 —0.76\]  
    * Post clustering ⇒ correlation \-0.025 (insignificant) \[Gini: 0.55—0.76\]  
* ⏯️ Ego depletion: only post clustering works for densify, still working on equalize...

## **Degree branch**

### **Two approaches: independent and conditional approach**

* **Independent approach:** Draw `attempts` sources independently (weighted by out-degree), draw `attempts` targets independently (weighted by in-degree), pair them up, then throw out any pairs that are self-loops or already-existing edges. Pick one surviving pair at random.  
* **Conditional approach:** Draw one source (weighted by out-degree). Look at that specific source's already-existing connections and exclude them. Draw a target from what remains, still weighted by in-degree. Return immediately on the first success.

The structural difference is: independent approach draws source and target *independently* and filters afterwards; conditional approach draws the target *given* the source.

### **Hyperparameter `p_conditional`**

Probability in \[0, 1\] of using the conditional (sequential) sampling approach in the degree branch. With probability `1 - p_conditional` the independent approach is used instead.

---

### **Why the approaches produce opposite correlations**

* In short: independent approach's rejection filter introduces an *implicit equalizing pressure* that grows with n\_edges; conditional's per-source filtering avoids this and lets preferential attachment run cleanly, which concentrates degree.

---

* Both approaches use the same original degree weights throughout — weights never update as edges are added. The divergence comes from how they handle the fact that high-degree nodes (hubs) tend to already be densely connected.  
* **Why independent → negative correlation:**  
  * When n\_edges is large, many hub-to-hub connections already exist by the time later edges are being placed. Because sources and targets are drawn independently, proposals for hub-to-hub edges keep getting generated — but then discarded as duplicates. The pairs that *survive* the filter are systematically biased toward connections involving less-connected nodes, because those slots aren't taken yet. So the more edges you add, the more the surviving edges are pushed toward peripheral nodes, flattening the degree distribution and lowering the Gini.  
* **Why conditional → positive correlation:**  
  * When a hub is drawn as source, the conditional approach excludes only *that hub's* existing connections from the target pool. The remaining available targets are still weighted by in-degree, so other high-degree nodes (those the hub hasn't connected to yet) are still preferentially selected. There is no global rejection pressure accumulating across attempts — the very first attempt almost always succeeds. So preferential attachment runs unimpeded throughout: more edges consistently go to high-degree nodes, concentrating degree further and raising the Gini with larger n\_edges.

## **Clustering loop**

* **Interim clustering branch**: When clustering dips below target during edge-adding, instead of sampling from the full network, the algorithm picks a random node and samples a new edge between that node's existing neighbours (predecessors and successors). Since every candidate node in that pool already shares a common connection through the chosen node, any edge added between them completes a triangle, directly raising clustering. The source and target within the neighbourhood are still chosen weighted by degree, so higher-degree neighbours are more likely to be picked. If no valid edge exists within the neighbourhood (e.g. it is fully connected), the loop simply moves on to the next iteration.  
* **Post clustering loop**: After all edges are added, the algorithm fine-tunes clustering without changing any node's degree by performing degree-preserving swaps: two edges are removed and their endpoints reconnected in the opposite pairing. Each swap is only accepted if it moves clustering closer to the target, and the process is guided by biased edge selection — preferring triangle-participating edges when clustering is too high, and non-triangle edges when too low.

---

* Notes  
  * I have tried and implemented an interim declustering loop. However, this appeared to be giving worse results. Also, it wouldn’t solve our current issues: the solution for the ego depletion network requires clustering, not declustering.  
  * I am a bit worried about the post clustering loop: it feels like it may mess up the network structure in unexpected ways.  
  * Given a specific network, there might be a limit to how far one can change the degree Gini without changing the clustering. That is, it might be that, beyond a certain point, lowering the degree Gini necessarily yields a lower clustering coefficient.

## **To do**

* Milder equalization  
  * Currently, the equalization method targets the uniform degree distribution. One idea is to parametrize the equalization method so that the target distribution is a (linear?) combination of the uniform and the original degree distribution.  
* Hyperparameter optimization  
  * Create a variation method with several hyperparameters such as `p_conditional` , but perhaps also `p_max_edges`, and also `p_max_rewired`.  
  * Then, use [hyperopt](https://hyperopt.github.io/hyperopt/) to do hyperparameter optimalization to find the best hyperparameters of the variation method where the objective is to have a low correlation between network statistics.

