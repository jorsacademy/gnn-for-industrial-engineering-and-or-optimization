# Learned Branching with PySCIPOpt and a Bipartite GNN

This example demonstrates a **real solver-in-the-loop** GNN workflow.

File:

```text
examples/03_pyscipopt_learned_branching.py
```

## What the example does

The script builds small weighted Maximum Independent Set (MIS) MILPs:

\[
\max \sum_i w_i x_i
\]

subject to

\[
x_u+x_v\le 1 \qquad \forall (u,v)\in E,
\]

with \(x_i\in\{0,1\}\).

It then executes the following pipeline:

```text
SCIP branch-and-bound state
        ↓
LP branching candidates
        ↓
strong branching expert
        ↓
variable-constraint bipartite graph
        ↓
GNN imitation learning
        ↓
custom PySCIPOpt Branchrule
        ↓
held-out SCIP solve
```

## Why strong branching?

Strong branching tentatively evaluates candidate branching variables and is often a strong but computationally expensive reference policy.

In learning-to-branch research, the expensive expert can be used offline to generate labels. A neural policy then attempts to approximate the expert more cheaply during later solves.

## Graph representation

Variable nodes include:

- objective coefficient,
- LP relaxation value,
- fractionality,
- local lower bound,
- local upper bound,
- original MIS graph degree.

Constraint nodes correspond to edge constraints `x_u + x_v <= 1`.

The variable-constraint edge coefficient is 1 in this MIS formulation.

## Important limitations

This script is intentionally educational.

It is **not** an exact reproduction of Gasse et al. (NeurIPS 2019), and it is not enough to claim that learned branching is better than SCIP.

A serious benchmark should include:

- many more training/test instances,
- multiple random seeds,
- larger MILPs,
- default SCIP settings as a baseline,
- inference overhead,
- search-tree nodes,
- time to first feasible solution,
- time to target MIP gap,
- size generalization,
- distribution shift,
- comparison with pseudo-cost and reliability branching.

## References

- Gasse et al., *Exact Combinatorial Optimization with Graph Convolutional Neural Networks*, NeurIPS 2019: https://arxiv.org/abs/1906.01629
- PySCIPOpt branching tutorial: https://pyscipopt.readthedocs.io/en/latest/tutorials/branchrule.html
- SCIP: https://www.scipopt.org/
