import math
from typing import Optional

from models.parameters import Parameters, CostEstimate, DeviationProperties
from quantum.error_models import total_topological_error, probability_union
from quantum.magic_states import compute_distillation_error
from quantum.surface_code import board_logical_dimensions, physical_qubits_per_logical_qubit


def compute_deviation_properties(params: Parameters) -> DeviationProperties:
    piece_count = int(math.ceil(params.n / params.runway_sep))
    piece_len = params.runway_sep + params.deviation_padding
    reg_len = params.n + params.deviation_padding * piece_count

    # Temporarily adding carry runways into main register avoids need to
    # iterate over their bits when multiplying with that register as input.
    mul_in_bits = params.n + params.deviation_padding + 2
    # only subtract from exponent bits if using optimized windowing
    inner_loop_count = int(math.ceil((params.n_e - (params.larger_init_lookup if params.opt_win else 0)) * 2 * mul_in_bits / (
        params.exp_window * params.mul_window)))

    classical_deviation_error = inner_loop_count * piece_count / 2**params.deviation_padding
    quantum_deviation_error = 4*math.sqrt(classical_deviation_error)
    return DeviationProperties(
        piece_count=piece_count,
        piece_len=piece_len,
        reg_len=reg_len,
        inner_loop_count=inner_loop_count,
        deviation_error=quantum_deviation_error,
    )


def estimate_algorithm_cost(params: Parameters) -> Optional[CostEstimate]:
    """Determine algorithm single-shot layout and costs for given parameters."""

    post_process_error = 1e-2  # assumed to be below 1%
    dev = compute_deviation_properties(params)

    # Derive values for understanding inner loop.
    adder_depth = dev.piece_len * 2 - 1

    lookup_depth = 2 ** (params.exp_window + params.mul_window) - 1
    if params.opt_win:
        # split unlookup into two parts, cost of unary conversion (done only one per exponent window) and cost of phase correction (i.e cost of unlookup)
        unlookup_depth = math.sqrt(lookup_depth)
        unary_tof_count = math.sqrt(lookup_depth)
        unary_depth = (params.exp_window + params.mul_window)/2
        
        lookup_depth -= 2 ** (params.exp_window)
        # We call this additional toffoli count, but it's actually less if we have a larger initial lookup and use deferred unlookups
        additional_tof_count = (2 * (params.n_e - params.larger_init_lookup) * unary_tof_count / params.exp_window) + \
                     (2 ** (params.larger_init_lookup) - 1)
                     
        additional_time = (2 ** (params.larger_init_lookup) - 1) * params.code_distance * params.cycle_time / 2 + \
          (2 * (params.n_e - params.larger_init_lookup) * unary_depth / params.exp_window)*params.reaction_time
    else:
        unlookup_depth = 2 * math.sqrt(lookup_depth)
        additional_tof_count = 0
        additional_time = 0*params.cycle_time

    alpha = 1
    tof_count = alpha*(adder_depth * dev.piece_count + lookup_depth + unlookup_depth) * dev.inner_loop_count + additional_tof_count
    inner_loop_time = alpha*(
        adder_depth * params.reaction_time +
        lookup_depth * params.code_distance * params.cycle_time / 2 +
        unlookup_depth * params.code_distance * params.cycle_time / 2)
    total_time = inner_loop_time * dev.inner_loop_count + additional_time

    piece_width, piece_height, piece_distillation = board_logical_dimensions(params, dev.piece_len)
    logical_qubits = piece_width * piece_height * dev.piece_count
    distillation_area = piece_distillation * dev.piece_count

    surface_code_cycles = total_time / params.cycle_time
    topological_error = total_topological_error(
        unit_cells=(logical_qubits - distillation_area) * surface_code_cycles,
        code_distance=params.code_distance,
        gate_err=params.gate_err)

    distillation_error = compute_distillation_error(tof_count=tof_count, params=params)

    total_error = probability_union(
        topological_error,
        distillation_error,
        dev.deviation_error,
        post_process_error,
    )
    if total_error >= params.max_total_err:
        return None

    total_qubits = logical_qubits * physical_qubits_per_logical_qubit(params.code_distance)
    total_time_seconds = total_time.total_seconds()

    total_hours = total_time_seconds / 3600
    total_megaqubits = total_qubits / 1e6
    total_volume_megaqubitdays = (total_hours / 24) * total_megaqubits
    total_error = math.ceil(100 * total_error) / 100

    return CostEstimate(
        params=params,
        toffoli_count=tof_count,
        distillation_error=distillation_error,
        topological_data_error=topological_error,
        total_error=total_error,
        total_hours=total_hours,
        total_megaqubits=total_megaqubits,
        total_volume_megaqubitdays=total_volume_megaqubitdays)


def rank_estimate(costs: CostEstimate) -> float:
    # Slight preference for decreasing space over decreasing time.
    skewed_volume = costs.total_megaqubits**1.2 * costs.total_hours
    return skewed_volume / (1 - costs.total_error)


def estimate_best_problem_cost(n: int, n_e: int, gate_error_rate: float, opt_win: bool = True) -> Optional[CostEstimate]:
    from models.parameters import parameters_to_attempt
    
    estimates = [estimate_algorithm_cost(params)
                 for params in parameters_to_attempt(n, n_e, gate_error_rate, opt_win)]
    surviving_estimates = [e for e in estimates if e is not None]
    return min(surviving_estimates, key=rank_estimate, default=None)


def estimate_all_surviving_problem_cost(n: int, n_e: int, gate_error_rate: float, opt_win: bool = True) -> list[CostEstimate]:
    from models.parameters import parameters_to_attempt
    
    estimates = [estimate_algorithm_cost(params)
                 for params in parameters_to_attempt(n, n_e, gate_error_rate, opt_win)]
    surviving_estimates = [e for e in estimates if e is not None]
    return surviving_estimates
