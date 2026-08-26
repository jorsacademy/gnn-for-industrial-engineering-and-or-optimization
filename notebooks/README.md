# Notebooks

The notebooks progress from basic graph optimization to solver-assisted and structure-aware models.

1. `01_maxcut_gnn.ipynb` — label-free GCN objective for Max-Cut.
2. `02_milp_variable_constraint_gnn.ipynb` — MILP variable-constraint bipartite representation and warm-start scoring.
3. `04_supply_chain_rgcn.ipynb` — R-GCN candidate-arc screening for a fixed-charge supply-chain MILP.
4. `05_directed_gnn_network_flow.ipynb` — directed vs symmetrized message passing for min-cost flow.
5. `06_hypergraph_gnn_set_cover.ipynb` — HypergraphConv candidate-set screening for weighted Set Cover.

The solver-in-the-loop branching example is a Python script under `examples/` because it is callback-driven rather than notebook-oriented.
