# GNN Taxonomy for Operations Research and Industrial Engineering

> **Updated:** August 2026  
> This document extends the model list in the main README. The goal is not to collect every architecture name in graph learning, but to distinguish the model families, graph structures, and techniques that are genuinely relevant to operations research, optimization, and industrial engineering.

## 1. First distinction: architecture family, graph structure, or technique?

Three different concepts are often mixed together:

1. **Architecture family:** GCN, GAT, GIN, R-GCN, GGNN, PNA, etc.
2. **Graph/problem structure:** directed graph, heterogeneous graph, hypergraph, signed graph, line graph, etc.
3. **Technique:** rewiring, diffusion, positional encoding, graph sparsification, sampling, normalization, etc.

This repository keeps them conceptually separate.

For example, **Directed GNN** is not one unique layer name; it is a class of direction-aware message-passing designs. **Graph rewiring** is usually not a separate GNN family; it is a technique used to improve information flow or reduce oversquashing.

---

# 2. Core and directly useful families for OR

## 2.1 GCN

Graph Convolutional Networks perform local neighborhood aggregation using normalized adjacency information.

A simplified layer is

\[
H^{(l+1)} = \sigma\left(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)}\right).
\]

Useful as:

- a simple baseline,
- a node-scoring model,
- a graph embedding model,
- a differentiable combinatorial optimizer component.

Relevant problems:

- Max-Cut,
- MIS,
- graph partitioning,
- small combinatorial graph problems.

**Caution:** the smoothing behavior and implicit homophily bias can be inappropriate for conflict-heavy problems such as Max-Cut.

## 2.2 GraphSAGE

GraphSAGE uses learned neighborhood aggregation and supports inductive generalization.

Useful when:

- graphs are large,
- training and test graphs differ,
- neighbor sampling is important,
- network optimization instances vary in size.

Typical OR applications include large infrastructure networks, logistics graphs, and scalable candidate scoring.

## 2.3 GAT

Graph Attention Networks learn neighborhood attention weights.

Useful for:

- routing,
- scheduling,
- allocation,
- heterogeneous local importance.

**Caution:** attention does not automatically solve long-range dependency problems; large dense graphs can also make attention expensive.

## 2.4 GIN

Graph Isomorphism Networks were designed for strong neighborhood discrimination under the message-passing framework.

Useful when structural differences between local graph patterns matter.

Potential OR uses:

- graph combinatorial problems,
- graph-level instance embeddings,
- structural candidate ranking.

## 2.5 MPNN

Message Passing Neural Networks are a general template:

\[
m_v^{(l)} = \operatorname{AGG}_{u\in N(v)} \phi(h_u^{(l)}, h_v^{(l)}, e_{uv}),
\]

\[
h_v^{(l+1)} = \psi(h_v^{(l)},m_v^{(l)}).
\]

They are particularly useful in OR because edge attributes often have direct semantic meaning:

- distance,
- travel time,
- cost,
- capacity,
- setup time,
- coefficient values,
- physical parameters.

## 2.6 PNA — Principal Neighbourhood Aggregation

PNA combines multiple aggregators such as mean, max, min, and standard deviation, often with degree-aware scaling.

This can be useful when graph degree distributions vary substantially between instances.

Potential OR settings:

- network design,
- heterogeneous infrastructure graphs,
- large instance families with different sparsity levels.

PNA is best viewed as a stronger MPNN-style baseline rather than a completely different optimization paradigm.

---

# 3. Structural GNN families that matter strongly in optimization

## 3.1 Directed GNNs

In a directed graph, `u -> v` and `v -> u` have different semantics.

This is common in OR:

- supply-chain flows,
- one-way transportation networks,
- precedence networks,
- project scheduling,
- production routes,
- network flow,
- telecommunications,
- energy transport.

A simple direction-aware design uses separate incoming and outgoing messages:

\[
m_v^{in}=\operatorname{AGG}_{u:(u,v)\in E}\phi_{in}(h_u),
\]

\[
m_v^{out}=\operatorname{AGG}_{u:(v,u)\in E}\phi_{out}(h_u),
\]

\[
h_v' = \psi(h_v,m_v^{in},m_v^{out}).
\]

This is not a mathematical invention for this repository. Direction-aware graph convolution is an established research area; one example is **Directed Graph Convolutional Network** by Tong et al.

Reference: https://arxiv.org/abs/2004.13970

**Repository status:** core / important.

## 3.2 Bipartite GNNs

A bipartite GNN is especially natural when two entity classes interact but same-type edges are absent or secondary.

Examples:

```text
variable <-> constraint
variable <-> clause
worker <-> task
facility <-> customer
machine <-> operation
```

MILP variable-constraint bipartite graphs are one of the most important optimization applications.

**Repository status:** core / important.

## 3.3 Heterogeneous GNNs

Heterogeneous graphs contain multiple node and/or edge types.

Examples:

```text
supplier -> plant -> warehouse -> customer
operation -> machine
job -> operation -> resource
```

They are useful when collapsing all entity types into one homogeneous node class would destroy semantics.

**Repository status:** core / important.

## 3.4 Relational GNN / R-GCN

R-GCN assigns relation-specific transformations.

A common formulation is

\[
h_i^{(l+1)} = \sigma\left(
W_0^{(l)}h_i^{(l)} +
\sum_{r\in R}\sum_{j\in N_i^r}
\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}
\right).
\]

Job-shop example:

```text
operation --precedes--> operation
operation --uses--> machine
machine --compatible_with--> operation
```

Supply-chain example:

```text
supplier --supplies--> plant
plant --feeds--> warehouse
warehouse --ships_to--> customer
```

Reference: Schlichtkrull et al., **Modeling Relational Data with Graph Convolutional Networks**  
https://arxiv.org/abs/1703.06103

**Repository status:** core / important.

## 3.5 Hypergraph Neural Networks

A hyperedge can connect more than two nodes simultaneously.

This is a natural representation when one decision or constraint links a group of entities:

- Set Cover,
- Set Packing,
- SAT/CSP,
- group-resource constraints,
- multi-way compatibility,
- higher-order allocation.

For Set Cover:

```text
element = node
candidate set = hyperedge
```

The incidence matrix keeps the original many-to-many structure directly.

**Repository status:** core for higher-order incidence problems.

## 3.6 Edge-centric GNNs and line-graph GNNs

Many OR decisions live on edges rather than nodes:

\[
x_{ij}=1
\]

may mean “select arc `(i,j)`.”

Relevant problems:

- TSP/VRP,
- shortest path,
- network design,
- transmission expansion,
- transportation,
- communication routing.

A line graph transforms each original edge into a node. Two line-graph nodes are adjacent when the corresponding original edges share an endpoint.

This allows standard node-based GNN machinery to reason about edge decisions.

**Caution:** line graphs can be substantially larger or denser than the original graph.

**Repository status:** important, especially for arc-decision models.

## 3.7 Signed GNNs

Signed graphs distinguish positive and negative relations.

Potential optimization uses include:

- conflict graphs,
- compatibility/incompatibility systems,
- antagonistic network relations,
- selected formulations of partitioning and Max-Cut-like problems.

Signed GNNs are real model families, but they are more situational for general industrial engineering than directed or relational GNNs.

**Repository status:** established but situational.

---

# 4. Global information, expressivity, and long-range dependence

## 4.1 Graph Transformers

Graph Transformers combine attention with graph structure and positional/structural information.

They are attractive when a decision depends on global context:

- TSP/VRP,
- assignment,
- global scheduling,
- network design,
- facility/customer interaction.

The main issue is scalability: naive full attention can be quadratic in node count.

**Repository status:** core for neural combinatorial optimization and global interactions.

## 4.2 Spectral GNNs

Spectral GNNs use graph Laplacian or frequency-domain structure.

Natural problem areas:

- graph partitioning,
- Max-Cut,
- clustering,
- power systems,
- diffusion-like physical networks.

They are conceptually connected to classical spectral relaxations used in optimization.

**Repository status:** established and useful in selected problem classes.

## 4.3 Higher-order / k-GNN / subgraph GNN

Standard MPNNs have known expressivity limits related to Weisfeiler-Lehman tests.

Higher-order models use tuples, subgraphs, motifs, or more expressive local structures.

Potential optimization relevance:

- symmetry-heavy combinatorial problems,
- graph structures that standard MPNNs cannot distinguish,
- small/medium hard graph instances.

**Caution:** computational cost may scale poorly, sometimes roughly with `n^k` depending on the construction.

**Repository status:** advanced but legitimate.

## 4.4 Graph rewiring

Rewiring changes the computational graph used by the GNN to improve information flow.

This matters because optimization decisions may depend on distant nodes and standard local message passing can suffer from **oversquashing**.

Possible techniques include:

- spectral rewiring,
- curvature-inspired rewiring,
- learned rewiring,
- adding long-range virtual edges,
- diffusion-based connectivity.

Important distinction:

> Rewiring is usually a technique applied to a GNN, not a standalone GNN family.

OR research question:

> Can we preserve the original optimization model while using a better computational graph for the neural policy?

**Repository status:** important technique / research topic.

## 4.5 Diffusion and propagation enhancements

Diffusion operators, personalized PageRank-style propagation, multi-scale propagation, and residual/global pathways can extend the effective receptive field.

These are useful to study when solver decisions require global problem information.

**Repository status:** supporting technique.

---

# 5. Space, geometry, and time

## 5.1 Temporal / Dynamic GNNs

The graph or its features change over time:

\[
G_1,G_2,\ldots,G_t.
\]

Relevant OR problems:

- dynamic VRP,
- online scheduling,
- traffic control,
- cloud-resource allocation,
- telecom routing,
- dynamic supply chains,
- disruption management.

Temporal GNNs are often combined with RL or model-predictive control.

**Repository status:** core for online/dynamic problems.

## 5.2 Equivariant / Geometric GNNs

These models preserve geometric symmetries such as translation, rotation, or Euclidean transformations.

Relevant applications:

- robotics and motion planning,
- 3D placement,
- physical layout,
- molecule/material design,
- geometric facility layouts,
- some power/wireless systems.

For purely algebraic MILPs without geometry, these architectures may add unnecessary complexity.

**Repository status:** important when the problem is genuinely geometric.

---

# 6. Iterative and algorithmic GNNs

## 6.1 Gated Graph Neural Networks (GGNN)

GGNNs use recurrent gating, commonly GRU-style updates, across message-passing iterations:

\[
h_v^{t+1}=\operatorname{GRU}(h_v^t,m_v^t).
\]

Potential OR uses:

- iterative constraint propagation,
- SAT/CSP reasoning,
- shortest-path reasoning,
- learned dynamic programming,
- scheduling state propagation,
- neural algorithmic reasoning.

**Repository status:** established and useful for iterative reasoning.

## 6.2 Neural Algorithmic Reasoning

The objective is to learn algorithm-like state transitions rather than a static prediction.

Examples of classical algorithms that can inspire supervision:

- Bellman-Ford,
- BFS,
- Prim,
- Floyd-Warshall,
- dynamic programming recurrences.

This is promising for generalization but is not a universal replacement for exact OR algorithms.

**Repository status:** research-relevant.

---

# 7. Generative and representation-learning extensions

## 7.1 Graph Autoencoders / Variational Graph Autoencoders

These models learn latent graph representations.

Potential optimization roles:

- instance embeddings,
- scenario clustering,
- solution-space representations,
- anomaly detection in networks,
- candidate generation.

They are usually not optimization solvers by themselves.

## 7.2 Generative graph models

Generative graph models can create:

- synthetic optimization instances,
- scenario graphs,
- candidate network designs,
- diverse solution neighborhoods.

Potential value:

- stress testing,
- data augmentation,
- training distribution expansion,
- simulation of rare network structures.

**Repository status:** adjacent / specialized.

---

# 8. Advanced topological models

These are real research directions, but they should not be presented as the default choice for ordinary industrial optimization.

## 8.1 Simplicial Neural Networks

A simplicial complex can contain nodes, edges, triangles, and higher-dimensional simplices.

Conceptually:

```text
node <-> edge <-> face <-> higher-order simplex
```

Potential relevance:

- flow/topology-sensitive physical systems,
- higher-order interactions with orientation,
- advanced network physics.

**Repository status:** advanced research topic.

## 8.2 Cell-complex neural networks

Cell complexes generalize graph/simplicial representations and can model higher-dimensional cells with richer topology.

Potential relevance exists for complex infrastructure and physical systems, but conventional MILP/scheduling/routing projects rarely require them.

**Repository status:** advanced research topic.

## 8.3 Sheaf Neural Networks

Sheaf-based graph learning allows information at nodes/edges to live in different vector spaces with learned or structured compatibility maps.

Potential relevance:

- heterogeneous physical variables,
- multi-commodity or multi-domain systems,
- networks where edge compatibility is richer than scalar adjacency.

This is mathematically legitimate research, but currently far less standard for OR practice than bipartite GNNs, R-GCNs, Graph Transformers, or directed GNNs.

**Repository status:** advanced research topic; not a default industrial recommendation.

---

# 9. Practical maturity classification for this repository

## Tier A — Directly useful and should be understood

```text
GCN
GraphSAGE
GAT
GIN
MPNN
PNA
Bipartite GNN
Heterogeneous GNN
R-GCN / relational GNN
Directed GNN
Hypergraph GNN
Graph Transformer
Temporal GNN
```

## Tier B — Important in the right problem

```text
Spectral GNN
Higher-order / k-GNN
Edge-centric / line-graph GNN
Signed GNN
GGNN
Equivariant / geometric GNN
```

## Tier C — Techniques rather than standalone architecture families

```text
rewiring
diffusion
positional encoding
sampling
sparsification
global tokens / global features
```

## Tier D — Real but research-heavy for general OR practice

```text
simplicial neural networks
cell-complex neural networks
sheaf neural networks
advanced topological deep learning
```

The key point is not to ask whether a model exists mathematically. The relevant engineering question is:

> **Does the model encode a structural property that is actually present in the optimization problem, and does that representation improve the complete optimization pipeline against strong baselines?**

---

# 10. Recommended model-to-problem mapping

| Problem | Natural graph structure | Strong starting model |
|---|---|---|
| TSP / CVRP | geometric directed/complete graph | MPNN / Graph Transformer |
| Dynamic VRP | temporal directed graph | Temporal GNN + RL |
| MILP branching | variable-constraint bipartite graph | Bipartite GNN |
| Job Shop | relational/disjunctive graph | Heterogeneous GNN / R-GCN / attention |
| Supply-chain design | layered directed relational graph | R-GCN / directed heterogeneous GNN |
| Min-cost flow | directed graph | Directed GNN |
| Set Cover | incidence hypergraph | Hypergraph GNN |
| Max-Cut | heterophilic graph | heterophily-aware / spectral / multi-filter GNN |
| MIS / clique / coloring | combinatorial graph | GIN / MPNN / higher-order when needed |
| SAT / CSP | variable-clause bipartite/hypergraph | bipartite GNN / GGNN / Hypergraph GNN |
| 3D layout / robotics | geometric graph | equivariant GNN |
| Large graph candidate screening | sparse large graph | GraphSAGE / PNA / scalable PyG stack |

---

# 11. Research and benchmark discipline

Using a more exotic GNN family is justified only if it improves the relevant OR metric.

Always measure:

- objective value or optimality gap,
- feasibility,
- time to solution,
- total neural inference overhead,
- search-tree size when applicable,
- robustness to instance size,
- robustness to distribution shift,
- comparison with strong classical baselines.

A sophisticated graph architecture that improves classification accuracy but makes the end-to-end optimizer slower is not automatically an optimization improvement.

The repository therefore prioritizes **structurally justified hybrid designs** over architecture novelty for its own sake.
