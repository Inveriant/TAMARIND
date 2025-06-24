import math
from typing import Tuple
from models.parameters import Parameters


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


def compute_distillation_error(tof_count: int,
                               params: Parameters) -> float:
    """Estimate the total chance of CCZ magic state distillation failing.

    Args:
        tof_count: Number of CCZ states to distill.
        params: Algorithm construction parameters.

    References:
        Based on spreadsheet "calculator-CCZ-2T-resources.ods" from
        https://arxiv.org/abs/1812.01238 by Gidney and Fowler
    """
    if params.use_t_t_distillation:
        return 4*9*10**-17  # From FIG 1 of https://arxiv.org/abs/1812.01238

    l1_distance = params.l1_distance
    l2_distance = params.code_distance

    # Level 0
    L0_distance = l1_distance // 2
    L0_distillation_error = params.gate_err
    L0_topological_error = total_topological_error(
        unit_cells=100,  # Estimated 100 for T injection.
        code_distance=L0_distance,
        gate_err=params.gate_err)
    L0_total_T_error = L0_distillation_error + L0_topological_error

    # Level 1
    L1_topological_error = total_topological_error(
        unit_cells=1100,  # Estimated 1000 for factory, 100 for T injection.
        code_distance=l1_distance,
        gate_err=params.gate_err)
    L1_distillation_error = 35 * L0_total_T_error**3
    L1_total_T_error = L1_distillation_error + L1_topological_error

    # Level 2
    L2_topological_error = total_topological_error(
        unit_cells=1000,  # Estimated 1000 for factory.
        code_distance=l2_distance,
        gate_err=params.gate_err)
    L2_distillation_error = 28 * L1_total_T_error**2
    L2_total_CCZ_or_2T_error = L2_topological_error + L2_distillation_error

    return tof_count * L2_total_CCZ_or_2T_error


def logical_factory_dimensions(params: Parameters) -> Tuple[int, int, float]:
    """Determine the width, height, depth of the magic state factory."""
    if params.use_t_t_distillation:
        return 12*2, 8*2, 6  # Four T2 factories

    l1_distance = params.l1_distance
    l2_distance = params.code_distance

    t1_height = 4 * l1_distance / l2_distance
    t1_width = 8 * l1_distance / l2_distance
    t1_depth = 5.75 * l1_distance / l2_distance

    ccz_depth = 5
    ccz_height = 6
    ccz_width = 3
    storage_width = 2 * l1_distance / l2_distance

    ccz_rate = 1 / ccz_depth
    t1_rate = 1 / t1_depth
    t1_factories = int(math.ceil((ccz_rate * 8) / t1_rate))
    t1_factory_column_height = t1_height * math.ceil(t1_factories / 2)

    width = int(math.ceil(t1_width * 2 + ccz_width + storage_width))
    height = int(math.ceil(max(ccz_height, t1_factory_column_height)))
    depth = max(ccz_depth, t1_depth)

    return width, height, depth


def physical_qubits_per_logical_qubit(code_distance: int) -> int:
    return (code_distance + 1)**2 * 2
