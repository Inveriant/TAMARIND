def topological_error_per_unit_cell(
        code_distance: int,
        gate_err: float) -> float:
    return 0.1 * (100 * gate_err) ** ((code_distance + 1) / 2)


def total_topological_error(code_distance: int,
                            gate_err: float,
                            unit_cells: int) -> float:
    """
    Args:
        code_distance: Diameter of logical qubits.
        gate_err: Physical gate error rate.
        unit_cells: Number of d*d*1 spacetime cells at risk.
    """
    return unit_cells * topological_error_per_unit_cell(
        code_distance,
        gate_err)


def probability_union(*ps: float) -> float:
    t = 1
    for p in ps:
        if p >= 1:
            # This happens when e.g. using the union bound to upper bound
            # a probability by using a frequency. The frequency estimate can
            # exceed 1 error per run.
            return 1
        t *= 1 - p
    return 1 - t
