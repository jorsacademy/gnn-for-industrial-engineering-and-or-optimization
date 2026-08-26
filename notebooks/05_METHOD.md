# Method note

The directed model keeps separate incoming and outgoing aggregation channels. The baseline symmetrizes connectivity.

Both models score the original directed decision arcs. The reduced LP is always re-solved by a classical optimizer, and pruning uses an adaptive feasibility fallback.
