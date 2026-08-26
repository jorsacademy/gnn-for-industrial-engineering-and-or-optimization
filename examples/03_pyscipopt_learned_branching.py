"""Educational PySCIPOpt + strong branching + bipartite GNN example.

This script uses real SCIP branching hooks:

- getLPBranchCands()
- startStrongbranch()
- getVarStrongbranch()
- endStrongbranch()
- custom Branchrule
- branchVar()

Workflow
--------
1. Generate small weighted Maximum Independent Set (MIS) MILPs.
2. Use strong branching as an expert and collect branching states.
3. Represent each solver state as a variable-constraint bipartite graph.
4. Train a GNN to imitate the expert variable choice.
5. Put the trained GNN back into SCIP as a custom branching rule.
6. Compare default SCIP and learned branching on held-out small instances.

This is an educational implementation of the general idea popularized by
Gasse et al. (NeurIPS 2019), not an exact reproduction of their experimental
pipeline.

Paper:
https://arxiv.org/abs/1906.01629

PySCIPOpt branching tutorial:
https://pyscipopt.readthedocs.io/en/latest/tutorials/branchrule.html
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from pyscipopt import (
    Branchrule,
    Model,
    SCIP_PARAMSETTING,
    SCIP_RESULT,
)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Small MIS instances
# ---------------------------------------------------------------------------

def make_mis_instance(n: int = 16, p: float = 0.28, seed: int = 0):
    """Generate a weighted MIS instance.

    We explicitly insert an odd cycle to make a fractional LP relaxation more
    likely, which in turn makes the branching example more informative.
    """
    rng = np.random.default_rng(seed)
    graph = nx.gnp_random_graph(n=n, p=p, seed=seed)

    if n >= 5:
        cycle = [0, 1, 2, 3, 4]
        for u, v in zip(cycle, cycle[1:] + cycle[:1]):
            graph.add_edge(u, v)

    weights = rng.integers(1, 11, size=n).astype(float)
    return graph, weights


def build_scip_mis(graph: nx.Graph, weights: np.ndarray, name: str = "mis"):
    """Build weighted MIS as a minimization MILP.

    max sum_i w_i x_i
    is written as
    min -sum_i w_i x_i

    subject to x_u + x_v <= 1 for every graph edge.
    """
    model = Model(name)

    # Keeping presolve/separation off makes the educational graph-to-solver
    # mapping easier to inspect. Production benchmarks should test default
    # SCIP settings as well.
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.hideOutput()

    variables = []
    for i, weight in enumerate(weights):
        var = model.addVar(
            name=f"x_{i}",
            vtype="B",
            obj=-float(weight),
        )
        variables.append(var)

    edges = list(graph.edges())
    for k, (u, v) in enumerate(edges):
        model.addCons(
            variables[u] + variables[v] <= 1,
            name=f"edge_{k}_{u}_{v}",
        )

    model.setMinimize()
    return model, variables, edges


# ---------------------------------------------------------------------------
# 2. Solver state -> bipartite graph
# ---------------------------------------------------------------------------

def scaled(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    s = np.max(np.abs(values))
    return values / s if s > 0 else values


def frac01(value: float):
    """Fractionality scaled to [0,1], with 1 at value 0.5."""
    d = min(value - np.floor(value), np.ceil(value) - value)
    return float(np.clip(2.0 * d, 0.0, 1.0))


def candidate_index_from_name(name: str):
    if not name.startswith("x_"):
        raise ValueError(f"Unexpected variable name: {name}")
    return int(name.split("_", 1)[1])


def build_bipartite_state(
    model: Model,
    graph: nx.Graph,
    weights: np.ndarray,
    edges: list[tuple[int, int]],
    candidate_vars,
):
    """Convert the current SCIP LP state to PyG HeteroData.

    Variable features
    -----------------
    - normalized objective coefficient
    - LP value
    - fractionality
    - local lower bound
    - local upper bound
    - graph degree

    Constraint features
    -------------------
    - current LP slack for x_u + x_v <= 1
    - normalized edge-degree proxy

    Edge feature
    ------------
    - linear coefficient (1 for MIS constraints)
    """
    transformed = {
        var.name: var
        for var in model.getVars(transformed=True)
    }

    n_vars = len(weights)
    n_cons = len(edges)

    lp_values = np.zeros(n_vars)
    lbs = np.zeros(n_vars)
    ubs = np.ones(n_vars)

    for i in range(n_vars):
        var = transformed[f"x_{i}"]
        lp_values[i] = float(model.getVal(var))
        lbs[i] = float(var.getLbLocal())
        ubs[i] = float(var.getUbLocal())

    degree = np.array(
        [graph.degree(i) for i in range(n_vars)],
        dtype=float,
    )
    degree = degree / max(degree.max(), 1.0)

    objective = scaled(-weights)
    fractionality = np.array([frac01(v) for v in lp_values])

    var_x = np.column_stack([
        objective,
        lp_values,
        fractionality,
        lbs,
        ubs,
        degree,
    ]).astype(np.float32)

    if n_cons:
        slack = np.array(
            [1.0 - lp_values[u] - lp_values[v] for u, v in edges],
            dtype=float,
        )
        con_x = np.column_stack([
            slack,
            np.ones(n_cons, dtype=float) / max(n_cons, 1),
        ]).astype(np.float32)

        var_idx = []
        con_idx = []
        for j, (u, v) in enumerate(edges):
            var_idx.extend([u, v])
            con_idx.extend([j, j])

        edge_index = torch.tensor(
            [var_idx, con_idx],
            dtype=torch.long,
        )
        edge_attr = torch.ones(
            (len(var_idx), 1),
            dtype=torch.float32,
        )
    else:
        con_x = np.zeros((0, 2), dtype=np.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)

    candidate_indices = [
        candidate_index_from_name(var.name)
        for var in candidate_vars
    ]

    candidate_mask = torch.zeros(n_vars, dtype=torch.bool)
    candidate_mask[candidate_indices] = True

    data = HeteroData()
    data["variable"].x = torch.tensor(var_x)
    data["constraint"].x = torch.tensor(con_x)
    data["variable"].candidate_mask = candidate_mask

    rel = data["variable", "participates", "constraint"]
    rel.edge_index = edge_index
    rel.edge_attr = edge_attr

    return data


# ---------------------------------------------------------------------------
# 3. Strong-branching expert
# ---------------------------------------------------------------------------

def get_lp_candidates(model: Model):
    """Return LP branching candidates from PySCIPOpt.

    PySCIPOpt returns additional arrays/counts; the first item is the variable
    list and the second item contains current LP solution values.
    """
    result = model.getLPBranchCands()
    return result[0], result[1]


def strong_branch_tuple(model: Model, var, iteration_limit: int = 50):
    """Call getVarStrongbranch while tolerating minor signature differences."""
    try:
        return model.getVarStrongbranch(
            var,
            iteration_limit,
            idempotent=True,
        )
    except TypeError:
        return model.getVarStrongbranch(var, iteration_limit)


def strong_branch_score(model: Model, result_tuple, lp_obj: float):
    """Convert a strong-branch result to a scalar expert score.

    For a minimization problem, larger child dual-bound improvement is better.
    Infeasible children are treated as very strong evidence.
    """
    down = float(result_tuple[0])
    up = float(result_tuple[1])
    down_valid = bool(result_tuple[2])
    up_valid = bool(result_tuple[3])
    down_inf = bool(result_tuple[4])
    up_inf = bool(result_tuple[5])

    big_gain = 1e6
    down_gain = (
        big_gain
        if down_inf
        else max(down - lp_obj, 0.0) if down_valid else 0.0
    )
    up_gain = (
        big_gain
        if up_inf
        else max(up - lp_obj, 0.0) if up_valid else 0.0
    )

    return min(down_gain, up_gain) + 1e-3 * max(down_gain, up_gain)


@dataclass
class BranchSample:
    graph: HeteroData
    target_variable: int


class StrongBranchCollector(Branchrule):
    """Branch with strong branching and save imitation-learning samples."""

    def __init__(self, graph, weights, edges, samples, max_samples_per_solve=20):
        super().__init__()
        self.problem_graph = graph
        self.weights = weights
        self.edges = edges
        self.samples = samples
        self.max_samples_per_solve = max_samples_per_solve
        self.collected = 0

    def branchexeclp(self, allowaddcons):
        candidates, _candidate_values = get_lp_candidates(self.model)

        if not candidates:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        if len(candidates) == 1:
            self.model.branchVar(candidates[0])
            return {"result": SCIP_RESULT.BRANCHED}

        lp_obj = float(self.model.getLPObjVal())

        scores = []
        self.model.startStrongbranch()
        try:
            for var in candidates:
                result = strong_branch_tuple(self.model, var)
                scores.append(
                    strong_branch_score(self.model, result, lp_obj)
                )
        finally:
            self.model.endStrongbranch()

        best_pos = int(np.argmax(scores))
        expert_var = candidates[best_pos]

        if self.collected < self.max_samples_per_solve:
            state = build_bipartite_state(
                self.model,
                self.problem_graph,
                self.weights,
                self.edges,
                candidates,
            )
            target = candidate_index_from_name(expert_var.name)
            self.samples.append(BranchSample(state, target))
            self.collected += 1

        self.model.branchVar(expert_var)
        return {"result": SCIP_RESULT.BRANCHED}


def collect_samples(
    n_instances: int = 30,
    n_nodes: int = 16,
    start_seed: int = 1000,
):
    samples: list[BranchSample] = []

    for k in range(n_instances):
        graph, weights = make_mis_instance(
            n=n_nodes,
            seed=start_seed + k,
        )
        model, _vars, edges = build_scip_mis(
            graph,
            weights,
            name=f"expert_{k}",
        )

        branchrule = StrongBranchCollector(
            graph,
            weights,
            edges,
            samples,
        )
        model.includeBranchrule(
            branchrule,
            "strong_branch_collector",
            "Collect strong-branching imitation labels",
            priority=100000,
            maxdepth=-1,
            maxbounddist=1.0,
        )

        model.optimize()

    return samples


# ---------------------------------------------------------------------------
# 4. Bipartite GNN
# ---------------------------------------------------------------------------

def mean_aggregate(messages, index, dim_size):
    out = messages.new_zeros((dim_size, messages.size(-1)))
    out.index_add_(0, index, messages)

    count = messages.new_zeros((dim_size, 1))
    count.index_add_(
        0,
        index,
        messages.new_ones((messages.size(0), 1)),
    )

    return out / count.clamp_min(1.0)


class BipartiteBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()

        self.var_to_con = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.con_update = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.con_to_var = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.var_update = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.var_norm = nn.LayerNorm(hidden)
        self.con_norm = nn.LayerNorm(hidden)

    def forward(self, h_var, h_con, edge_index, edge_attr):
        var_idx, con_idx = edge_index

        if edge_index.numel() == 0:
            return h_var, h_con

        msg_vc = self.var_to_con(
            torch.cat([h_var[var_idx], edge_attr], dim=-1)
        )
        agg_con = mean_aggregate(
            msg_vc,
            con_idx,
            h_con.size(0),
        )
        h_con = self.con_norm(
            h_con
            + self.con_update(torch.cat([h_con, agg_con], dim=-1))
        )

        msg_cv = self.con_to_var(
            torch.cat([h_con[con_idx], edge_attr], dim=-1)
        )
        agg_var = mean_aggregate(
            msg_cv,
            var_idx,
            h_var.size(0),
        )
        h_var = self.var_norm(
            h_var
            + self.var_update(torch.cat([h_var, agg_var], dim=-1))
        )

        return h_var, h_con


class BranchingGNN(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.var_encoder = nn.Linear(6, hidden)
        self.con_encoder = nn.Linear(2, hidden)
        self.blocks = nn.ModuleList(
            [BipartiteBlock(hidden) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: HeteroData):
        h_var = F.relu(self.var_encoder(data["variable"].x))
        h_con = F.relu(self.con_encoder(data["constraint"].x))

        rel = data["variable", "participates", "constraint"]

        for block in self.blocks:
            h_var, h_con = block(
                h_var,
                h_con,
                rel.edge_index,
                rel.edge_attr,
            )

        return self.head(h_var).squeeze(-1)


def train_branching_gnn(
    samples: list[BranchSample],
    epochs: int = 40,
):
    if not samples:
        raise RuntimeError(
            "No branching samples were collected. "
            "Try larger/denser MIS instances."
        )

    model = BranchingGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for epoch in range(epochs):
        order = np.random.permutation(len(samples))
        running = 0.0

        for idx in order:
            sample = samples[idx]
            data = sample.graph.to(DEVICE)

            logits = model(data)
            mask = data["variable"].candidate_mask
            candidate_indices = torch.flatnonzero(mask)

            if len(candidate_indices) <= 1:
                continue

            candidate_logits = logits[candidate_indices]
            target_global = sample.target_variable

            target_pos = torch.flatnonzero(
                candidate_indices == target_global
            )
            if len(target_pos) != 1:
                continue

            loss = F.cross_entropy(
                candidate_logits.unsqueeze(0),
                target_pos[:1],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"epoch={epoch+1:3d} "
                f"mean_loss={running/max(len(samples),1):.4f}"
            )

    return model


# ---------------------------------------------------------------------------
# 5. Put the learned policy back into SCIP
# ---------------------------------------------------------------------------

class LearnedBranchrule(Branchrule):
    def __init__(self, graph, weights, edges, policy):
        super().__init__()
        self.problem_graph = graph
        self.weights = weights
        self.edges = edges
        self.policy = policy

    def branchexeclp(self, allowaddcons):
        candidates, _candidate_values = get_lp_candidates(self.model)

        if not candidates:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        if len(candidates) == 1:
            self.model.branchVar(candidates[0])
            return {"result": SCIP_RESULT.BRANCHED}

        state = build_bipartite_state(
            self.model,
            self.problem_graph,
            self.weights,
            self.edges,
            candidates,
        ).to(DEVICE)

        self.policy.eval()
        with torch.no_grad():
            logits = self.policy(state).cpu().numpy()

        candidate_indices = [
            candidate_index_from_name(var.name)
            for var in candidates
        ]
        best_pos = int(
            np.argmax([logits[i] for i in candidate_indices])
        )

        self.model.branchVar(candidates[best_pos])
        return {"result": SCIP_RESULT.BRANCHED}


def solve_with_policy(graph, weights, policy=None, name="test"):
    model, _vars, edges = build_scip_mis(graph, weights, name=name)

    if policy is not None:
        branchrule = LearnedBranchrule(
            graph,
            weights,
            edges,
            policy,
        )
        model.includeBranchrule(
            branchrule,
            "learned_gnn_branching",
            "GNN branching policy",
            priority=100000,
            maxdepth=-1,
            maxbounddist=1.0,
        )

    t0 = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - t0

    return {
        "time": elapsed,
        "nodes": model.getNNodes(),
        "objective": model.getObjVal(),
        "status": str(model.getStatus()),
    }


# ---------------------------------------------------------------------------
# 6. End-to-end demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Collecting strong-branching samples...")
    samples = collect_samples(
        n_instances=25,
        n_nodes=16,
        start_seed=1000,
    )
    print("Collected states:", len(samples))

    print("\nTraining branching GNN...")
    policy = train_branching_gnn(samples, epochs=40)

    print("\nHeld-out comparison:")
    rows = []

    for k in range(8):
        graph, weights = make_mis_instance(
            n=16,
            seed=9000 + k,
        )

        default = solve_with_policy(
            graph,
            weights,
            policy=None,
            name=f"default_{k}",
        )
        learned = solve_with_policy(
            graph,
            weights,
            policy=policy,
            name=f"learned_{k}",
        )

        rows.append((default, learned))

        print(
            f"instance={k:02d} "
            f"default_nodes={default['nodes']:5d} "
            f"learned_nodes={learned['nodes']:5d} "
            f"default_time={default['time']:.4f}s "
            f"learned_time={learned['time']:.4f}s"
        )

    print("\nImportant:")
    print(
        "This small educational benchmark is not evidence that learned "
        "branching is universally faster. Proper evaluation requires many "
        "instances, multiple seeds, default SCIP settings, larger problems, "
        "inference-overhead accounting, and distribution-shift tests."
    )
