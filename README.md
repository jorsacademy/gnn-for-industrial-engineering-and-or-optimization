# Graph Neural Networks for Industrial Engineering, Operations Research, and Optimization

> **Updated:** August 2026  
> This repository is an English guide to using Graph Neural Networks (GNNs) in operations research (OR), industrial engineering, and optimization.

GNNs are not limited to node classification and link prediction. When an optimization problem naturally contains a network, graph, variable-constraint structure, precedence relation, machine-operation relation, facility-customer network, or higher-order incidence structure, a GNN can learn representations that are useful for optimization decisions.

The central position of this repository is:

> **A GNN does not need to replace a classical optimizer. In many real applications, the stronger design is to use the GNN as a learned component that guides, accelerates, or narrows the search performed by a classical solver.**

Accordingly, the repository covers not only `graph -> GNN -> solution`, but also GNN + local search, GNN + reinforcement learning, GNN + branch-and-bound, GNN + cutting planes, GNN + warm starts, GNN + candidate screening, GNN + CP/MIP, and differentiable optimization.

See [APPLICATIONS.md](APPLICATIONS.md) for the applied examples and [advanced_topics/gnn_taxonomy.md](advanced_topics/gnn_taxonomy.md) for the extended model taxonomy.

---

## Contents

1. [Why GNNs for optimization?](#1-why-gnns-for-optimization)
2. [How to represent an optimization problem as a graph](#2-how-to-represent-an-optimization-problem-as-a-graph)
3. [Which GNN family fits which problem?](#3-which-gnn-family-fits-which-problem)
4. [What role can the GNN play?](#4-what-role-can-the-gnn-play)
5. [Operations-research problem classes](#5-operations-research-problem-classes)
6. [Libraries and frameworks](#6-libraries-and-frameworks)
7. [Recommended technology stacks](#7-recommended-technology-stacks)
8. [How to evaluate GNN-assisted optimization](#8-how-to-evaluate-gnn-assisted-optimization)
9. [Suggested learning path](#9-suggested-learning-path)
10. [Repository examples](#10-repository-examples)
11. [Resources](#11-resources)

---

# 1. Why GNNs for optimization?

Many OR problems already have graph structure:

- **TSP / VRP:** cities, customers, and depots are nodes; travel connections are edges.
- **Network design:** facilities, warehouses, customers, stations, or infrastructure assets form nodes and arcs.
- **Production scheduling:** operations are nodes; precedence and machine-conflict relations are edges.
- **MILP:** variables and constraints can be represented as two node types in a bipartite graph.
- **Set Cover / Set Packing:** sets and elements form a bipartite graph or a hypergraph.
- **SAT / CSP:** variables and clauses/constraints form a bipartite incidence graph.
- **Supply chains:** suppliers, plants, warehouses, distribution centers, and customers form a layered graph.
- **Energy and transportation systems:** the physical system is often already a directed or undirected network.

A classical optimization problem can be written as

```text
min f(x)

s.t.
    g_i(x) <= 0
    h_j(x)  = 0
    x in X
```

A GNN does not have to predict the complete optimizer `x*`. Learning only one expensive decision can already be useful:

- which variable to branch on,
- which valid cut to add,
- which edge/node/candidate to keep,
- which local-search move to try,
- which initial solution to provide,
- which variables may be fixed,
- which neighborhood to explore,
- which job, vehicle, machine, or route decision should be made next.

This perspective is closely related to **learning-augmented optimization**, **learning to optimize**, and **neural combinatorial optimization**.

---

# 2. How to represent an optimization problem as a graph

For optimization, graph representation is often as important as the neural architecture.

## 2.1 TSP / VRP

```text
Node         = city / customer / depot
Edge         = feasible travel connection
Node feature = coordinates, demand, time window, service time
Edge feature = distance, travel time, cost, traffic, capacity-related data
```

TSP is frequently represented as a complete graph. For large instances, sparsification such as k-nearest-neighbor graphs may be useful.

## 2.2 Job Shop Scheduling

```text
Node        = operation
Edge type 1 = precedence relation
Edge type 2 = machine conflict / disjunctive relation
Node feature = processing time, job ID, machine ID, release time, state information
```

This naturally motivates heterogeneous, relational, or disjunctive graph models.

## 2.3 MILP as a variable-constraint bipartite graph

For

```text
min c^T x
s.t. A x <= b
     x_i integer for selected i
```

construct

```text
Node type 1 = variables
Node type 2 = constraints
Edge        = variable j is connected to constraint i if A_ij != 0
Edge feature = coefficient A_ij
```

Possible variable features include:

- objective coefficient,
- LP relaxation value,
- reduced cost,
- lower/upper bounds,
- integrality/binary indicator,
- pseudo-cost,
- fractionality,
- incumbent-relative information.

Possible constraint features include:

- right-hand side,
- slack,
- dual value,
- constraint type,
- activity and violation information.

This representation is particularly important for learned branching, cut selection, variable fixing, and primal heuristics.

## 2.4 Hypergraph representations

If a single relation simultaneously connects many variables or entities, pairwise edges may destroy useful higher-order structure.

For example,

```text
x1 + x7 + x13 + x20 <= 2
```

can be interpreted as a hyperedge connecting four variables.

Hypergraphs are natural for:

- Set Cover / Set Packing,
- SAT / CSP,
- multi-resource allocation,
- some scheduling models,
- higher-order network design,
- group constraints and incidence systems.

## 2.5 Directed and relational graphs

Direction is part of the problem semantics in many OR systems:

```text
supplier -> plant -> warehouse -> customer
operation A -> operation B
source -> transshipment node -> sink
```

Likewise, different edge types may mean different things. A relational GNN can assign different transformations to `supplies`, `feeds`, `ships_to`, `precedes`, or `uses_machine` relations.

---

# 3. Which GNN family fits which problem?

There is no universally best GNN. Choose the architecture according to the graph structure and the optimization decision being learned.

| GNN family | Main strength | OR / optimization use | Main caution |
|---|---|---|---|
| GCN | Simple message passing | Max-Cut, MIS, graph heuristics | Homophily bias can be inappropriate |
| GraphSAGE | Inductive learning and sampling | Large networks, scalable graph learning | Aggregator choice matters |
| GAT | Learned attention weights | Routing, scheduling, allocation | Attention cost on large graphs |
| GIN | Strong structural discrimination | Combinatorial graph problems | Global dependencies remain difficult |
| MPNN | Flexible node/edge message design | Cost/capacity/distance networks | Oversquashing |
| PNA | Multiple aggregators and degree scaling | Graphs with changing degree distributions | More design complexity |
| Bipartite GNN | Two interacting node classes | MILP, SAT, assignment | Usually problem-specific |
| Heterogeneous GNN | Multiple node/edge types | MILP, JSSP, supply chains | More complex data model |
| R-GCN / relational GNN | Relation-specific transformations | Multi-relation supply chains and scheduling | Relation explosion can increase cost |
| Directed GNN | Preserves arc direction | Flow, routing, precedence, logistics | Do not silently symmetrize the graph |
| Hypergraph GNN | Higher-order incidence | Set Cover, CSP, resource systems | Tooling is less uniform |
| Edge-centric / line-graph GNN | Models decisions on arcs directly | Routing, network design, path selection | Line graphs may become large |
| Graph Transformer | Global attention | TSP/VRP, assignment, global scheduling | Naive attention can be O(n²) |
| Spectral GNN | Laplacian/spectral structure | Partitioning, Max-Cut, power networks | Spectral generalization can be delicate |
| Higher-order / k-GNN | Greater expressive power than standard MPNNs | Symmetry-heavy graph problems | Computational cost grows quickly |
| Equivariant / geometric GNN | Respects geometric symmetries | Robotics, 3D layout, physical networks | Not useful when geometry is irrelevant |
| Temporal / dynamic GNN | Time-varying graph state | Dynamic VRP, traffic, online scheduling | State design is critical |
| Heterophily-aware GNN | Handles dissimilar neighboring states | Max-Cut and conflict-style structures | Standard GCN assumptions may fail |
| Gated / recurrent GNN | Iterative message passing | Algorithmic reasoning, CSP, iterative decisions | Training stability and horizon |

See the extended taxonomy for signed GNNs, rewiring, simplicial/cell-complex networks, and sheaf neural networks.

## 3.1 Generic message passing

A large class of GNNs can be summarized as

```text
message(u -> v) = psi(h_u, h_v, e_uv)
aggregate(v)    = AGG({message(u -> v) : u in N(v)})
h_v_new         = phi(h_v, aggregate(v))
```

Optimization decisions may depend on distant regions of a graph. Deep message passing can therefore suffer from **oversmoothing** and **oversquashing**. Graph Transformers, global features, carefully designed positional encodings, diffusion, or rewiring may help when long-range information is essential.

---

# 4. What role can the GNN play?

## 4.1 Direct solution construction

```text
Graph -> GNN -> solution
```

For example, the GNN may output

```text
p_i = P(x_i = 1)
```

and a binary solution is produced by thresholding or rounding.

**Advantage:** inference can be fast.  
**Limitation:** feasibility and optimality are not guaranteed unless additional machinery enforces them.

## 4.2 Heatmaps and candidate scores

Instead of producing the final solution, the GNN can output

```text
p_ij = P((i,j) is useful in the solution)
```

or

```text
score(v)
```

followed by:

- greedy construction,
- beam search,
- 2-opt / 3-opt,
- local search,
- MCTS,
- exact or heuristic solvers.

This hybrid design is often safer than unconstrained end-to-end prediction.

## 4.3 GNN + Reinforcement Learning

```text
state graph -> GNN encoder -> policy -> action
```

Actions can include choosing the next customer, dispatching a vehicle, scheduling an operation, assigning a resource, or choosing a local-search move.

Typical rewards are related to the optimization objective, for example `-total_cost` or `-makespan`.

## 4.4 GNN + Local Search

The GNN can rank neighborhoods rather than construct the full solution:

```text
current solution -> graph representation -> GNN -> move/neighborhood score -> local search
```

For VRP this may rank swap, relocate, 2-opt, 3-opt, or ruin-and-recreate candidates.

## 4.5 GNN + Branch-and-Bound

For MILP, the GNN can learn:

> Which candidate variable should I branch on?

```text
MILP
 -> variable-constraint bipartite graph
 -> GNN
 -> branching scores
 -> SCIP / MIP solver
```

A common research design uses strong branching as an expert and trains the neural policy with imitation learning. Correctness remains with the solver; the learned component changes the search strategy.

## 4.6 GNN + Cutting Planes

The model can rank valid cuts generated by a solver. If the classical optimizer remains responsible for validity, the neural network guides selection rather than replacing the mathematical correctness mechanism.

## 4.7 Warm starts and primal heuristics

The GNN may predict a candidate assignment or variable probabilities. These can be used for:

- MIP starts,
- feasibility repair,
- local branching,
- large-neighborhood search,
- variable fixing.

## 4.8 Candidate screening and graph pruning

For network design, Set Cover, routing, or facility-location models, a GNN can identify promising arcs/sets/variables before solving a reduced model.

A safe pattern is:

```text
GNN score -> candidate reduction -> feasibility repair/fallback -> exact solver -> validated solution
```

## 4.9 Differentiable optimization / predict-then-optimize

Another paradigm is

```text
GNN -> predicted problem parameters -> differentiable optimization layer -> decision
```

This is useful when downstream decision quality matters more than prediction error alone.

---

# 5. Operations-research problem classes

## 5.1 TSP / VRP / CVRP / VRPTW

Useful models:

- attention-based GNNs,
- Graph Transformers,
- MPNNs,
- neural combinatorial optimization policies.

Typical pipelines:

```text
constructive policy
GNN heatmap + search
GNN + local search
GNN + RL
```

## 5.2 Job Shop / Flexible Job Shop / Flow Shop

Represent operations, precedence relations, machine conflicts, setup states, and dynamic shop-floor information. Heterogeneous GNNs, relational GNNs, GATs, Graph Transformers, and temporal GNNs are natural candidates.

Objectives can include makespan, tardiness, setup cost, throughput, and energy use.

## 5.3 Facility Location and Supply-Chain Network Design

```text
supplier -> plant -> warehouse -> customer
```

GNNs can support:

- facility-opening scores,
- candidate arc screening,
- warm starts,
- scenario embeddings,
- decomposition guidance,
- disruption-aware policies.

The final decision can still be produced by Pyomo/Gurobi/SCIP/HiGHS or another solver.

## 5.4 Max-Cut

Max-Cut is interesting because adjacent nodes are encouraged to be in **different** partitions. Standard homophily-oriented GNN assumptions are therefore not always ideal.

Relevant approaches include heterophily-aware GNNs, spectral models, differentiable surrogates, and QUBO-based formulations.

## 5.5 MIS / Maximum Clique / Vertex Cover / Coloring

A common pattern is

```text
GNN score -> greedy construction -> feasibility repair -> local search
```

## 5.6 MILP

Important learning targets include:

- branching,
- node selection,
- cut selection,
- primal heuristics,
- variable fixing,
- warm starts,
- presolve decisions.

The GNN is often most useful as **solver guidance**, not as a solver replacement.

## 5.7 SAT / CSP

A variable-clause or variable-constraint graph can support learned variable scores, phase prediction, clause importance, or branching guidance while CDCL/CP machinery maintains correctness.

## 5.8 Multi-echelon inventory and production networks

Temporal GNN + RL models can support replenishment, allocation, transshipment, and disruption response when the inventory system has explicit multi-echelon graph structure.

## 5.9 Network flow and transportation

Directed GNNs are especially relevant because arc direction is part of the optimization semantics. Candidate-arc screening can reduce a large flow model before an LP/MILP solver validates the result.

## 5.10 Set Cover / Set Packing

The element-set incidence structure is naturally hypergraphic. Hypergraph neural networks can score candidate sets, after which coverage repair and an exact solver can enforce feasibility.

---

# 6. Libraries and frameworks

## 6.1 PyTorch

Core deep-learning framework for custom layers, objectives, solver integration, and RL.

```bash
pip install torch
```

## 6.2 PyTorch Geometric (PyG) — default GNN research choice

https://pytorch-geometric.readthedocs.io/

PyG supports a broad range of GNN operators, heterogeneous graphs, sampling, mini-batching, compilation, distributed workflows, graph stores, and geometric data.

```bash
pip install torch_geometric
```

For most new research projects in this repository, the default stack is **PyTorch + PyG**.

## 6.3 NetworkX

https://networkx.org/

Not a neural-network library. It is useful for generating graphs, computing graph statistics, implementing classical baselines, and validating small examples.

## 6.4 RL4CO

https://github.com/ai4co/rl4co

A useful framework for neural combinatorial optimization, particularly routing and scheduling. It includes constructive autoregressive, constructive non-autoregressive, improvement, and transductive policy families.

## 6.5 SCIP + PySCIPOpt

SCIP: https://www.scipopt.org/  
PySCIPOpt: https://pyscipopt.readthedocs.io/

A key stack for learning inside mixed-integer optimization. PySCIPOpt exposes branching rules, heuristics, separators, node selectors, cut selectors, events, and other solver hooks.

```text
PyTorch + PyTorch Geometric + PySCIPOpt + SCIP
```

## 6.6 Ecole

https://doc.ecole.ai/

Ecole exposes SCIP control problems in an ML/RL-friendly environment and is historically important for learned branching experiments. Check project maintenance and SCIP/package compatibility before starting a new production project.

## 6.7 OR-Tools

https://developers.google.com/optimization

Strong classical baselines for CP-SAT, scheduling, routing, assignment, packing, and network flow.

## 6.8 Pyomo

https://pyomo.readthedocs.io/

A general Python algebraic modeling layer that is useful when a GNN predicts parameters or candidate decisions that are then passed to a mathematical optimization model.

## 6.9 Gurobi

https://www.gurobi.com/

Commercial optimizer widely used in academic and industrial benchmarks. GNN outputs may be integrated as MIP starts, candidate restrictions, variable priorities, or callback-driven heuristics subject to the solver API.

## 6.10 DGL

https://www.dgl.ai/

A mature graph-learning framework with distributed capabilities. For new NVIDIA-centered projects, the PyG ecosystem is often the more natural direction because NVIDIA's current GNN integration is centered on cuGraph-PyG.

## 6.11 cuGraph-PyG

https://docs.nvidia.com/cugraph/

Useful for GPU graph sampling, heterogeneous sampling, distributed graph storage, multi-GPU GNN training, and very large graphs.

---

# 7. Recommended technology stacks

## General GNN + optimization research

```text
Python
PyTorch
PyTorch Geometric
NetworkX
NumPy / SciPy
Jupyter
```

## TSP / CVRP / neural combinatorial optimization

```text
PyTorch
PyTorch Geometric
RL4CO
OR-Tools / PyVRP / strong heuristic baseline
```

## MILP + learned branching / cuts / heuristics

```text
PyTorch
PyTorch Geometric
PySCIPOpt
SCIP
Pyomo (if an algebraic modeling layer is useful)
```

## Scheduling

```text
PyTorch Geometric
RL4CO or custom RL
OR-Tools CP-SAT
Pyomo / Gurobi / SCIP
```

## Large-scale graph learning

```text
PyTorch
PyTorch Geometric
cuGraph-PyG
WholeGraph when needed
```

## Production-oriented hybrid pattern

```text
GNN
  |
  v
search score / warm start / candidate reduction
  |
  v
classical optimizer
  |
  v
feasible and validated solution
```

---

# 8. How to evaluate GNN-assisted optimization

Neural loss or classification accuracy is not enough.

## 8.1 Objective value

Measure the actual decision objective, e.g. `f(x_GNN)` or the objective after repair/solver completion.

## 8.2 Optimality gap

When a strong reference solution is known, use a problem-appropriate relative gap, for example

```text
gap = (f_candidate - f_best) / |f_best|
```

with the correct sign convention for minimization/maximization.

## 8.3 Feasibility rate

```text
feasible solutions / total instances
```

A model that frequently violates constraints is not practically useful even if its unconstrained objective looks good.

## 8.4 Time to solution

Measure the complete pipeline:

```text
GNN inference + repair + solver + local search
```

A learned policy can predict better choices yet still make the total solve slower because of inference overhead.

## 8.5 Search-tree metrics

For learned MIP components, also measure:

- branch-and-bound nodes,
- LP iterations,
- primal/dual progress,
- time to first feasible solution,
- time to a target gap.

## 8.6 Size generalization

Example protocol:

```text
train: 50-100 nodes
validation: 100 nodes
test: 200 / 500 / 1000 nodes
```

## 8.7 Distribution shift

Do not evaluate only on the same random generator used for training. Consider graph family, density, cost, capacity, and demand shifts as well as real data.

## 8.8 Strong classical baselines

Compare against credible baselines such as OR-Tools, SCIP, Gurobi, CP-SAT, HiGHS, strong local search, problem-specific heuristics, and tuned metaheuristics. A comparison only against random or weak greedy baselines is insufficient.

---

# 9. Suggested learning path

### Stage 1 — Graph foundations

Adjacency, degree, paths, Laplacians, directed graphs, incidence matrices, message passing.

### Stage 2 — PyTorch Geometric

`Data`, `HeteroData`, `edge_index`, `GCNConv`, `SAGEConv`, `GATConv`, `RGCNConv`, `HypergraphConv`, batching.

### Stage 3 — Small combinatorial optimization

Max-Cut, MIS, Vertex Cover. Start with differentiable objectives or small exact-label datasets.

### Stage 4 — Hybrid heuristics

```text
GNN score -> greedy / repair / local search
```

### Stage 5 — MILP representation

Build variable-constraint bipartite graphs and extract LP/solver features.

### Stage 6 — Solver in the loop

```text
PyG + PySCIPOpt / SCIP
```

Experiment with branching or primal heuristics.

### Stage 7 — Relational, directed, and hypergraph models

Use graph structure that matches the actual OR semantics rather than defaulting to an undirected homogeneous graph.

### Stage 8 — RL-based combinatorial optimization

Use RL4CO or custom environments for routing, scheduling, and online decisions.

---

# 10. Repository examples

The repository includes:

```text
notebooks/01_maxcut_gnn.ipynb
notebooks/02_milp_variable_constraint_gnn.ipynb
examples/03_pyscipopt_learned_branching.py
notebooks/04_supply_chain_rgcn.ipynb
notebooks/05_directed_gnn_network_flow.ipynb
notebooks/06_hypergraph_gnn_set_cover.ipynb
```

Together they illustrate six different uses of graph learning in optimization:

1. differentiable combinatorial objectives,
2. graph representation of mathematical programs,
3. solver-in-the-loop learned branching,
4. relational GNN candidate screening,
5. direction-aware network optimization,
6. hypergraph candidate screening for Set Cover.

See [APPLICATIONS.md](APPLICATIONS.md) for details.

---

# 11. Resources

## Official documentation

- PyTorch: https://pytorch.org/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- NetworkX: https://networkx.org/
- RL4CO: https://github.com/ai4co/rl4co
- SCIP: https://www.scipopt.org/
- PySCIPOpt: https://pyscipopt.readthedocs.io/
- Ecole: https://doc.ecole.ai/
- OR-Tools: https://developers.google.com/optimization
- Pyomo: https://pyomo.readthedocs.io/
- DGL: https://www.dgl.ai/
- NVIDIA cuGraph-PyG: https://docs.nvidia.com/cugraph/

## Useful literature-search terms

```text
neural combinatorial optimization
learning to branch
learning to cut
learning-augmented optimization
GNN for mixed integer programming
GNN for vehicle routing
GNN job shop scheduling
GNN Max-Cut
neural algorithmic reasoning
graph reinforcement learning optimization
differentiable optimization graph neural networks
hypergraph neural network set cover
directed GNN network optimization
relational GNN supply chain optimization
```

Classic starting points include **Learning Combinatorial Optimization Algorithms over Graphs**, **NeuroSAT**, attention-based routing work, R-GCN, and the variable-constraint bipartite GNN literature for learned branching.

---

# Core idea

For an industrial engineer or operations researcher, the most productive question is often not:

> **Can I solve this entire problem with a GNN?**

but rather:

> **Which expensive decision inside the optimization algorithm can a GNN predict well enough to improve the complete computational pipeline?**

That decision may be a branch variable, cut, neighborhood, route candidate, machine assignment, warm start, variable fixing decision, candidate arc/set, or search priority.

The strongest role of GNNs in operations research is usually not to discard optimization theory, but to **augment classical algorithms with learned structural heuristics while preserving explicit feasibility and solver-based verification where it matters**.
