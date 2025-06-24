from models.parameters import Parameters
from quantum.error_models import total_topological_error


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
