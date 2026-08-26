# Applied Examples

This repository is not intended merely to list GNN architectures. Its purpose is to show how graph learning can play concrete roles inside operations-research and optimization workflows.

## 01 — GNN for Max-Cut

File: `notebooks/01_maxcut_gnn.ipynb`

Goals:

- represent a graph-optimization problem for a GNN,
- use a label-free/differentiable objective,
- round continuous node probabilities to a discrete solution,
- compare the result with a brute-force optimum on a small graph.

Core pattern:

```text
Graph -> GCN -> node probabilities -> differentiable Max-Cut objective -> rounding
```

## 02 — MILP as a Variable-Constraint Bipartite Graph

File: `notebooks/02_milp_variable_constraint_gnn.ipynb`

Goals:

- convert a MILP coefficient matrix `A` to a bipartite graph,
- represent variables and constraints as distinct node types,
- use `A_ij` coefficients as edge features,
- illustrate warm-start / primal-heuristic scoring,
- connect LP fractionality and GNN uncertainty to branching candidates.

Core pattern:

```text
MILP -> variable/constraint graph -> bipartite GNN -> variable score
```

## 03 — Learned Branching with PySCIPOpt

Files:

- `examples/03_pyscipopt_learned_branching.py`
- `examples/03_pyscipopt_learned_branching.md`

Goals:

- connect to the actual SCIP branch-and-bound loop,
- use strong branching as an expert,
- collect imitation-learning labels from solver states,
- represent those states as variable-constraint bipartite graphs,
- place a trained GNN back inside a custom SCIP branching rule.

Core pattern:

```text
SCIP state
 -> variable-constraint bipartite graph
 -> strong-branching expert labels
 -> GNN imitation learning
 -> custom SCIP Branchrule
```

The example follows the learned-branching research direction associated with Gasse et al. (NeurIPS 2019), but it is an educational small-scale implementation rather than an exact reproduction of that paper.

## 04 — R-GCN + MILP for Supply-Chain Candidate Arc Screening

File: `notebooks/04_supply_chain_rgcn.ipynb`

Goals:

- represent a multi-echelon supply chain as a multi-relational graph,
- model `supplier -> plant -> warehouse -> customer` relations with relation-specific message passing,
- use arcs selected by the full MILP optimum as training labels,
- score candidate arcs with an R-GCN,
- prune low-scoring arcs conservatively,
- solve the reduced MILP,
- compare objective gap, feasibility, retained-arc ratio, and solve time,
- compare against a simple cost-based screening baseline.

Core pattern:

```text
full supply-chain MILP
 -> optimal arc labels
 -> R-GCN edge scorer
 -> candidate arc screening
 -> adaptive feasibility fallback
 -> reduced MILP
```

Safety principle:

> The GNN proposes a smaller decision space; the optimization solver still verifies feasibility and produces the final decision.

## 05 — Directed GNN + Min-Cost Flow

File: `notebooks/05_directed_gnn_network_flow.ipynb`

Goals:

- generate directed min-cost-flow networks,
- label arcs used by the full LP optimum,
- compare a graph-symmetrizing baseline with a direction-aware GNN,
- separate incoming and outgoing message aggregation,
- measure optimal-arc recall,
- use GNN scores for candidate-arc screening,
- apply adaptive fallback if pruning makes the model infeasible,
- evaluate objective gap, retained-arc ratio, and solve time,
- compare with a cost-only heuristic.

Core pattern:

```text
directed min-cost-flow network
 -> full LP
 -> optimal arc labels
 -> undirected baseline vs directed GNN
 -> arc screening
 -> feasibility fallback
 -> reduced LP
```

Here direction is not cosmetic. `u -> v` and `v -> u` represent different optimization decisions.

## 06 — Hypergraph GNN + MILP for Weighted Set Cover

File: `notebooks/06_hypergraph_gnn_set_cover.ipynb`

Goals:

- represent Set Cover naturally as a hypergraph,
- use elements as nodes and candidate sets as hyperedges,
- label sets selected by the full MILP optimum,
- learn set scores with `HypergraphConv`,
- prune candidate sets,
- repair coverage before solving the reduced MILP,
- compare the Hypergraph GNN against a set-feature MLP and a cost/coverage heuristic,
- evaluate optimal-set recall, retained-set ratio, feasibility, objective gap, and solve time.

Core pattern:

```text
weighted Set Cover MILP
 -> optimal selected-set labels
 -> Hypergraph GNN
 -> candidate-set screening
 -> coverage repair
 -> reduced MILP
```

## Suggested learning order

```text
01 Max-Cut
   ↓
02 MILP bipartite representation
   ↓
03 solver-in-the-loop learned branching
   ↓
04 relational GNN + supply-chain optimization
   ↓
05 directed GNN + network-flow screening
   ↓
06 hypergraph GNN + Set Cover screening
```

These examples cover six distinct GNN roles in optimization:

1. learning through a differentiable combinatorial objective,
2. representing a mathematical program as a graph,
3. learning a decision inside an exact solver,
4. reducing a relational network-design decision space,
5. preserving direction in network optimization,
6. preserving higher-order incidence structure in Set Cover.

The recurring design principle is:

```text
learned structural signal
        ↓
candidate / search guidance
        ↓
repair or fallback when necessary
        ↓
classical optimizer
        ↓
feasible, validated decision
```
