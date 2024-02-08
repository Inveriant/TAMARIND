import datetime
import itertools
import sys

from typing import Tuple, NamedTuple, Iterable, Iterator, Optional

import math
import matplotlib.pyplot as plt

import os.path as pathutils
import numpy as np


from helpers import Parameters, TABLE_HEADER, plot_datasets, DeviationProperties, CostEstimate
from helpers import reduce_significant, fips_strength_level, fips_strength_level_rounded, probability_union, physical_qubits_per_logical_qubit


space_overhead = 1.2

l1_distances = [4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
l2_distances = range(9, 51, 2)     # we always take odd distances.. https://quantumcomputing.stackexchange.com/questions/24416/why-are-the-planar-surface-codes-in-articles-always-of-odd-distance


exp_windows = [4, 5, 6]    # Question: why 4, 5, 6 and not more? Answer: because theoretically should be log(n)/2 or something around this number.
                         # since we combine exp_windows and mul_windows the table of the lookups is 2^(exp_windows + mul_windows) and we want it to be small.

mul_windows = [4, 5, 6]
runway_seps = [128, 256, 512, 768, 1024, 1536, 2048]  # runway separation
dev_offs = range(2, 10)   # Question: ?!?1?!?


gate_error_rates = [1e-3, 1e-4]
moduli = [1024, 2048, 3072, 4096 ] #, 8192, 12288, 16384]

Cs = [0.5, 1, 1.5, 2, 2.5]   # Regevs 

glob_cycle_time = datetime.timedelta(microseconds=1)
glob_reaction_time = datetime.timedelta(microseconds=10)


def parameters_to_attempt(n: int,
                          n_e: int,
                          gate_error_rate: float) -> Iterator[Parameters]:
    
    # epsilon_budgets = [0, 0.1, 0.2, 0.3, 0.4, 0.5] if include_regev_epsilon_budget else []
  
    for d1, d2, exp_window, mul_window, runway_sep, dev_off in itertools.product(
            l1_distances,
            l2_distances,
            exp_windows,
            mul_windows,
            runway_seps,
            dev_offs):
        if mul_window > exp_window or n % runway_sep != 0:
            continue
        distill_types = [False]
        if d1 == 15 and d2 >= 31:
            distill_types.append(True)
        for b in distill_types:
            yield Parameters(
                gate_err=gate_error_rate,
                reaction_time=glob_reaction_time,
                cycle_time=glob_cycle_time,
                exp_window=exp_window,
                mul_window=mul_window,
                runway_sep=runway_sep,
                l1_distance=d1,
                code_distance=d2,
                max_total_err=0.8,
                n=n,
                n_e=n_e,
                use_t_t_distillation=b,
                deviation_padding=int(math.ceil(math.log2(n*n*n_e)) + dev_off))


def topological_error_per_unit_cell(
        code_distance: int,
        gate_err: float) -> float:
    """
    Adi: This is the logical error rate per patch of the surface code.
    These hyperparameters are coming form the numerical simulations that can be found in the flowler paper.
    These are dependent on the surface code.

    Args:
        code_distance: Diameter of logical qubits.
        gate_err: Physical gate error rate.
    """
    return 0.1 * (100 * gate_err) ** ((code_distance + 1) / 2)       # hyperparam 


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


    # you can do this in two different ways: 
    # either use_t_t_distillation==True which distills T states and you use those in the Toffoli decomposition.
    # or you use the CCZ states and use those in the Toffoli decomposition
    # In this paper https://arxiv.org/abs/1905.08916  he mentions the layout you can use for the ripple-carry adder.
    # in this way you can optimize the way to feed CCZ states in the adder and qrom (so to minimize the routing)

    if params.use_t_t_distillation:
        return 4*9*10**-17  # From FIG 1 of https://arxiv.org/abs/1812.01238    # hyperparam  

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
        unit_cells=1100,  # Estimated 1000 for factory, 100 for T injection.          # hyperparam  
        code_distance=l1_distance,
        gate_err=params.gate_err)
    L1_distillation_error = 35 * L0_total_T_error**3            # hyperparam  
    L1_total_T_error = L1_distillation_error + L1_topological_error

    # Level 2
    L2_topological_error = total_topological_error(
        unit_cells=1000,  # Estimated 1000 for factory.                # hyperparam  
        code_distance=l2_distance,
        gate_err=params.gate_err)
    L2_distillation_error = 28 * L1_total_T_error**2            # hyperparam  
    L2_total_CCZ_or_2T_error = L2_topological_error + L2_distillation_error

    return tof_count * L2_total_CCZ_or_2T_error    






def compute_deviation_properties(params: Parameters) -> DeviationProperties:
    piece_count = int(math.ceil(params.n / params.runway_sep))
    piece_len = params.runway_sep + params.deviation_padding
    reg_len = params.n + params.deviation_padding * piece_count

    # Temporarily adding carry runways into main register avoids need to
    # iterate over their bits when multiplying with that register as input.
    mul_in_bits = params.n + params.deviation_padding + 2

    # outer_loop_count = n_e  / size_of_exponent_windows     (basically n_e in regev is |\log(D)| * d  )   with 
    # this should be better called total_loops_count. 
    inner_loop_count = int(math.ceil(params.n_e * 2 * mul_in_bits / (
        params.exp_window * params.mul_window)))

    classical_deviation_error = inner_loop_count * \
        piece_count / 2**params.deviation_padding
    quantum_deviation_error = 4*math.sqrt(classical_deviation_error)
    return DeviationProperties(
        piece_count=piece_count,
        piece_len=piece_len,
        reg_len=reg_len,
        inner_loop_count=inner_loop_count,
        deviation_error=quantum_deviation_error,
    )







# Question: where does this function and constant come from?!
# Question: why one thing is l1_ distance and the other one is code distance?
def logical_factory_dimensions(params: Parameters
                               ) -> Tuple[int, int, float]:
    """Determine the width, height, depth of the magic state factory.

    All these hyperparamers are related to lattice surgery for getting the CCZ state,
    and are coming from the 
    width and height = space dimensions for lattice for using these operations
    depth = time dimension

    These numbers come from  https://arxiv.org/abs/1812.01238 
    """
    if params.use_t_t_distillation:
        return 12*2, 8*2, 6  # Four T2 factories      # hyperparam

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
    depth = max(ccz_depth, t1_depth)     # 

    return width, height, depth


def board_logical_dimensions(params: Parameters,
                             register_len: int) -> Tuple[int, int, int]:
    """Computes the dimensions of the surface code board in logical qubits.

    Assumes a single-threaded execution. For parallel execution, pass in
    parameters for an individual adder piece.

    register = length of the register for doing addition, i.e. the number of bits for representing n 
               + the qubits for coset_state + the qubits for the runways.

    Returns:
        width, height, distillation_area
    """

    factory_width, factory_height, factory_depth = (
        logical_factory_dimensions(params))                                 # before gidney this was done differently, .. in the paper he cited.

    ccz_time = factory_depth * params.cycle_time * params.code_distance          # time to distill 1 czz state.
    factory_pair_count = int(math.ceil(ccz_time / params.reaction_time / 2))   # note is a unitless quantity (s/s..)   # check time optimal quantum computation

    total_width = (factory_width + 1) * factory_pair_count + 1                      # add the fowloer on time optimal quantum computations

    # FIG. 15 Lattice surgery implementation of the CZ fixups
    cz_fixups_box_height = 3

    # FIG. 23. Implementation of the MAJ operation in lattice surgery.
    adder_height = 3

    # FIG. 31. Data layout during a parallel addition.
    routing_height = 6
    reg_height = int(math.ceil(register_len / (total_width - 2)))      # 
    total_height = sum([
        factory_height * 2,
        cz_fixups_box_height * 2,
        adder_height,
        routing_height,
        reg_height * 3,
    ])
    distillation_area = factory_height * factory_width * factory_pair_count * 2

    return total_width, total_height, distillation_area






def calculate_address_fanout_gate_sum(num_address_bits):
    """
    # question? ?????
    """
    return 2 * (2 ** num_address_bits - 1) - num_address_bits
  
def calculate_address_fanout_depth(num_address_bits):
    """
    # question? ?????
    """

    return (num_address_bits - 1) * num_address_bits






def estimate_algorithm_cost(params: Parameters) -> Optional[CostEstimate]:
    """Determine algorithm single-shot layout and costs for given parameters."""

    post_process_error = 1e-2  # assumed to be below 1%
    dev = compute_deviation_properties(params)

    # Derive values for understanding inner loop.
    adder_depth = dev.piece_len * 2 - 1
    lookup_depth = 2 ** (params.exp_window + params.mul_window) - 1
    unlookup_depth = 2 * math.sqrt(lookup_depth)

    # Derive values for understanding overall algorithm.
    piece_width, piece_height, piece_distillation = board_logical_dimensions(
        params, dev.piece_len)
    logical_qubits = piece_width * piece_height * dev.piece_count
    distillation_area = piece_distillation * dev.piece_count

    tof_count = (adder_depth * dev.piece_count
                 + lookup_depth
                 + unlookup_depth) * dev.inner_loop_count
    #print(tof_count, adder_depth, dev.piece_count, lookup_depth, unlookup_depth, dev.inner_loop_count)



    # Code distance lets us compute time taken.
    inner_loop_time = (
            adder_depth * params.reaction_time +
            # Double speed lookup.
            lookup_depth * params.code_distance * params.cycle_time / 2 +
            unlookup_depth * params.code_distance * params.cycle_time / 2)
    total_time = inner_loop_time * dev.inner_loop_count

    # Upper-bound the topological error:
    surface_code_cycles = total_time / params.cycle_time
    topological_error = total_topological_error(
        unit_cells=(logical_qubits  - distillation_area) * surface_code_cycles,
        code_distance=params.code_distance,
        gate_err=params.gate_err)

    # Account for the distillation error:
    distillation_error = compute_distillation_error(
        tof_count=tof_count,
        params=params)

    # Check the total error.
    total_error = probability_union(
        topological_error,
        distillation_error,
        dev.deviation_error,
        post_process_error,
    )
    if total_error >= params.max_total_err:
        return None

    # Great!
    total_qubits = logical_qubits * physical_qubits_per_logical_qubit(
        params.code_distance)
    total_time = total_time.total_seconds()

    # Format.
    total_hours = total_time / 60 ** 2
    total_megaqubits = total_qubits / 10 ** 6
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
        total_volume_megaqubitdays=total_volume_megaqubitdays,
        logical_qubits=logical_qubits)



def estimate_algorithm_cost_adi(params: Parameters) -> Optional[CostEstimate]:
    """Determine algorithm single-shot layout and costs for given parameters.
    This is using deferred measurement.
    """

    post_process_error = 1e-2  # assumed to be below 1%
    dev = compute_deviation_properties(params)

    # Derive values for understanding inner loop.
    adder_depth = dev.piece_len * 2 - 1
    lookup_depth = 2 ** (params.exp_window + params.mul_window) - 1
    unlookup_depth = math.sqrt(lookup_depth)
    unary_depth = math.sqrt(lookup_depth)

    # Derive values for understanding overall algorithm.
    piece_width, piece_height, piece_distillation = board_logical_dimensions(
        params, dev.piece_len)
    logical_qubits = piece_width * piece_height * dev.piece_count
    distillation_area = piece_distillation * dev.piece_count

    tof_count = (adder_depth * dev.piece_count
                 + lookup_depth) * dev.inner_loop_count + (unlookup_depth*dev.inner_loop_count + unary_depth)

    # Code distance lets us compute time taken.
    inner_loop_time = (
        adder_depth * params.reaction_time +
        # Double speed lookup.
        lookup_depth * params.code_distance * params.cycle_time / 2 +
        unlookup_depth * params.code_distance * params.cycle_time / 2)
    total_time = inner_loop_time * dev.inner_loop_count

    # Upper-bound the topological error:
    surface_code_cycles = total_time / params.cycle_time
    topological_error = total_topological_error(
        unit_cells=(logical_qubits - distillation_area) * surface_code_cycles,
        code_distance=params.code_distance,
        gate_err=params.gate_err)

    # Account for the distillation error:
    distillation_error = compute_distillation_error(
        tof_count=tof_count,
        params=params)

    # Check the total error.
    total_error = probability_union(
        topological_error,
        distillation_error,
        dev.deviation_error,
        post_process_error,
    )
    if total_error >= params.max_total_err:
        return None

    # Great!
    total_qubits = logical_qubits * physical_qubits_per_logical_qubit(
        params.code_distance)
    total_time = total_time.total_seconds()

    # Format.
    total_hours = total_time / 60 ** 2
    total_megaqubits = total_qubits / 10 ** 6
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
        total_volume_megaqubitdays=total_volume_megaqubitdays,
        logical_qubits=logical_qubits)



def estimate_algorithm_cost_qram(params: Parameters) -> Optional[CostEstimate]:
    """Determine algorithm single-shot layout and costs for given parameters."""
    
    num_address_bits = params.mul_window + params.exp_window

    post_process_error = 1e-2  # assumed to be below 1% (Flag this)
    dev = compute_deviation_properties(params)

    # Derive values for understanding inner loop.
    adder_depth = dev.piece_len * 2 - 1
    address_fanout_depth = calculate_address_fanout_depth(num_address_bits)
    address_fanout_tof_count = calculate_address_fanout_gate_sum(num_address_bits)
    lookup_depth = 2*(num_address_bits)
    lookup_tof_count = params.n*2* (2 ** (num_address_bits + 1) - 1)
    unlookup_depth = 0
    unlookup_tof_count = 0

    # Derive values for understanding overall algorithm.
    piece_width, piece_height, piece_distillation = board_logical_dimensions(
        params, dev.piece_len)
    # board_logical_dimension takes into account everything related to the factories.


    logical_qubits = piece_width * piece_height * dev.piece_count + 3*params.n*(2**(num_address_bits+1))
    distillation_area = piece_distillation * dev.piece_count

    tof_count = (adder_depth * dev.piece_count + 2*address_fanout_tof_count
                 + lookup_tof_count
                 + unlookup_tof_count) * dev.inner_loop_count

    # Code distance lets us compute time taken.
    inner_loop_time = (
            adder_depth * params.reaction_time +
            # Double speed lookup.
            # Add more explanation here.
            lookup_tof_count * params.code_distance * params.cycle_time / 2 +
            unlookup_depth * params.code_distance * params.cycle_time / 2)
    total_time = inner_loop_time * dev.inner_loop_count

    # Upper-bound the topological error:
    surface_code_cycles = total_time / params.cycle_time
    topological_error = total_topological_error(
        unit_cells=(logical_qubits  - distillation_area) * surface_code_cycles,
        code_distance=params.code_distance,
        gate_err=params.gate_err)

    # Account for the distillation error:
    distillation_error = compute_distillation_error(
        tof_count=tof_count,
        params=params)

    # Check the total error.
    total_error = probability_union(
        topological_error,
        distillation_error,
        dev.deviation_error,
        post_process_error
    )
    if total_error >= params.max_total_err:
        return None

    # Great!
    total_qubits = logical_qubits * physical_qubits_per_logical_qubit(
        params.code_distance)
    total_time = total_time.total_seconds()

    # Format.
    total_hours = total_time / 60 ** 2
    total_megaqubits = total_qubits / 10 ** 6
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
        total_volume_megaqubitdays=total_volume_megaqubitdays,
        logical_qubits=logical_qubits)



def estimate_algorithm_cost_slicedwindows(params: Parameters) -> Optional[CostEstimate]:
    """Determine algorithm single-shot layout and costs for given parameters."""

    post_process_error = 1e-2  # assumed to be below 1%
    dev = compute_deviation_properties(params)
    # Derive values for understanding inner loop.
    adder_depth = (dev.piece_len * 2 - 1)
    lookup_depth = (2 ** (params.exp_window + params.mul_window) - 1)
    
    unlookup_depth = math.sqrt(lookup_depth)
    unary_depth = math.sqrt(lookup_depth)

    # Derive values for understanding overall algorithm.
    piece_width, piece_height, piece_distillation = board_logical_dimensions(
        params, dev.piece_len)
    logical_qubits = piece_width * piece_height * dev.piece_count
    distillation_area = piece_distillation * dev.piece_count

    # Unlookup depth doesn't change. Since we are repeatedly querying the same addresses, we can keep track of phase changes
    tof_count = ((adder_depth * dev.piece_count
                 + lookup_depth) * dev.inner_loop_count + (unlookup_depth*dev.inner_loop_count + unary_depth))

    # Code distance lets us compute time taken.
    inner_loop_time = (
        dev.piece_count * adder_depth * params.reaction_time +
        # Double speed lookup.
        dev.piece_count * lookup_depth * params.code_distance * params.cycle_time / 2 +
        unlookup_depth * params.code_distance * params.cycle_time / 2)
    total_time = inner_loop_time * dev.inner_loop_count

    # Upper-bound the topological error:
    surface_code_cycles = total_time / params.cycle_time
    topological_error = total_topological_error(
        unit_cells=(logical_qubits - (params.n - params.n/dev.piece_count)  - distillation_area) * surface_code_cycles,
        code_distance=params.code_distance,
        gate_err=params.gate_err)

    # Account for the distillation error:
    distillation_error = compute_distillation_error(
        tof_count=tof_count,
        params=params)

    # Check the total error.
    total_error = probability_union(
        topological_error,
        distillation_error,
        dev.deviation_error,
        post_process_error,
    )
    if total_error >= params.max_total_err:
        return None

    # Great!
    total_qubits = (logical_qubits - (params.n - params.n/dev.piece_count)) * physical_qubits_per_logical_qubit(
        params.code_distance)
    total_time = total_time.total_seconds()

    # Format.
    total_hours = total_time / 60 ** 2
    total_megaqubits = total_qubits / 10 ** 6
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
        total_volume_megaqubitdays=total_volume_megaqubitdays,
        logical_qubits=logical_qubits - (params.n - params.n/dev.piece_count) - distillation_area)





def rank_estimate(costs: CostEstimate) -> float:
    # Slight preference for decreasing space over decreasing time.
    skewed_volume = costs.total_megaqubits**space_overhead * costs.total_hours
    return skewed_volume / (1 - costs.total_error)


def estimate_best_problem_cost(n: int, n_e: int, gate_error_rate: float) -> Optional[CostEstimate]:
    estimates = [estimate_algorithm_cost(params)
                 for params in parameters_to_attempt(n, n_e, gate_error_rate)]
    surviving_estimates = [e for e in estimates if e is not None]
    return min(surviving_estimates, key=rank_estimate, default=None)

def estimate_best_problem_cost_adi(n: int, n_e: int, gate_error_rate: float) -> Optional[CostEstimate]:
    estimates = [estimate_algorithm_cost_adi(params)
                 for params in parameters_to_attempt(n, n_e, gate_error_rate)]


    surviving_estimates = [e for e in estimates if e is not None]
    return min(surviving_estimates, key=rank_estimate, default=None)

def estimate_best_problem_cost_qram(n: int, n_e: int, gate_error_rate: float) -> Optional[CostEstimate]:
    estimates = [estimate_algorithm_cost_qram(params)
                 for params in parameters_to_attempt(n, n_e, gate_error_rate)]
    surviving_estimates = [e for e in estimates if e is not None]
    return min(surviving_estimates, key=rank_estimate, default=None)


def estimate_best_problem_cost_slicedwindows(n: int, n_e: int, gate_error_rate: float) -> Optional[CostEstimate]:
    estimates = [estimate_algorithm_cost_slicedwindows(params)
                 for params in parameters_to_attempt(n, n_e, gate_error_rate)]
    surviving_estimates = [e for e in estimates if e is not None]
    return min(surviving_estimates, key=rank_estimate, default=None)




# ------------------------------------------------------------------------------







def tabulate_cost_estimate(costs: CostEstimate):
    assert costs.params.gate_err in [1e-3, 1e-4]
    gate_error_desc = r"0.1\%" if costs.params.gate_err == 1e-3 else r"0.01\%"
    row = [
        costs.params.n,
        costs.params.n_e,
        gate_error_desc,
        costs.params.l1_distance,
        costs.params.code_distance,
        costs.params.deviation_padding -
        int(math.ceil(math.log2(costs.params.n**2*costs.params.n_e))),
        costs.params.mul_window,
        costs.params.exp_window,
        costs.params.runway_sep,
        str(math.ceil(100 * costs.total_error)) + r"\%",
        reduce_significant(costs.total_volume_megaqubitdays),
        reduce_significant(costs.total_volume_megaqubitdays /
                           (1 - costs.total_error)),
        reduce_significant(costs.total_megaqubits),
        reduce_significant(costs.total_hours),
        reduce_significant(costs.total_hours / (1 - costs.total_error)),
        costs.params.use_t_t_distillation,
        np.round(costs.toffoli_count / 10**5),
        costs.total_error,
        costs.logical_qubits,
        #reduce_significant(costs.toffoli_count / 10**3),
#    costs.toffoli_count,
    ]
    print('&'.join('${}$'.format(e).ljust(10) for e in row) + '\\\\')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------




# RSA
def eh_rsa(n, gate_error_rate, C=0) -> Optional[CostEstimate]:  # Single run.
    delta = 20  # Required to respect assumptions in the analysis.
    m = math.ceil(n / 2) - 1
    l = m - delta
    n_e = m + 2 * l
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_rsa_adi(n, gate_error_rate, C=0) -> Optional[CostEstimate]:  # Single run.
    delta = 20  # Required to respect assumptions in the analysis.     # question: what is delta?
    m = math.ceil(n / 2) - 1
    l = m - delta
    n_e = m + 2 * l
    return estimate_best_problem_cost_adi(n, n_e, gate_error_rate)

def eh_rsa_qram(n, gate_error_rate, C=0) -> Optional[CostEstimate]:  # Single run.
    delta = 20  # Required to respect assumptions in the analysis.     # question: what is delta?
    m = math.ceil(n / 2) - 1
    l = m - delta
    n_e = m + 2 * l
    return estimate_best_problem_cost_qram(n, n_e, gate_error_rate)

def eh_rsa_slicedwindows(n, gate_error_rate, C=0) -> Optional[CostEstimate]:  # Single run.
    delta = 20  # Required to respect assumptions in the analysis.     # question: what is delta?
    m = math.ceil(n / 2) - 1
    l = m - delta
    n_e = m + 2 * l
    return estimate_best_problem_cost_slicedwindows(n, n_e, gate_error_rate)



# RSA
def regev_rsa(n, gate_error_rate, C) -> Optional[CostEstimate]:  # Single run.
    delta = 20  # Required to respect assumptions in the analysis.
    m = math.ceil(n / 2) - 1
    l = m - delta
    #Let N be an n-bit number and assume that for d =√n and O(log n)-bit numbers b1, . . . , bd,
    d = n**(0.5)
    #  By taking d =√n, it suffices to take R = exp(C√n) for some constant C > 0.
    # Each value zi can take values between [0..D]. So, each zi is a log(D) bit register
    R = math.exp(C * (n**0.5))
    D = 4*R*(d**0.5)
    n_e = math.ceil(d*math.log2(D))
    return estimate_best_problem_cost(n, n_e, gate_error_rate)

# RSA
def regev_rsa_espilon_budget(n, gate_error_rate, C) -> Optional[CostEstimate]:  # Single run.
    delta = 20  # Required to respect assumptions in the analysis.
    m = math.ceil(n / 2) - 1
    l = m - delta
    #Let N be an n-bit number and assume that for d =√n and O(log n)-bit numbers b1, . . . , bd,
    d = n**(0.5)
    #  By taking d =√n, it suffices to take R = exp(C√n) for some constant C > 0.
    # Each value zi can take values between [0..D]. So, each zi is a log(D) bit register
    R = 2**(C * (n**0.5))
    D = 4*R*(d**0.5)
    n_e = math.ceil(d*math.log2(D))
    return estimate_best_problem_cost(n, n_e, gate_error_rate)

# With maximum tradeoffs.
def eh_rsa_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]:
    return estimate_best_problem_cost(n, math.ceil(n / 2), gate_error_rate)




# def eh_dlp_schnorr(n, gate_error_rate) -> Optional[CostEstimate]: # Single run.
#     z = fips_strength_level_rounded(n)
#     m = 2 * z
#     n_e = 3 * m
#     return estimate_best_problem_cost(n, n_e, gate_error_rate)


# def eh_dlp_schnorr_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]: # Multiple runs with maximal tradeoff.
#     z = fips_strength_level_rounded(n)
#     m = 2 * z
#     n_e = m
#     return estimate_best_problem_cost(n, n_e, gate_error_rate)



# Short DLP
def eh_dlp_short(n, gate_error_rate, _) -> Optional[CostEstimate]:
    z = fips_strength_level_rounded(n)
    m = 2 * z
    n_e = 3 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)

def shor_dlp_schnorr(n, gate_error_rate, _) -> Optional[CostEstimate]:
    delta = 5 # Required to reach 99% success probability.
    z = fips_strength_level_rounded(n)
    m = 2 * z + delta
    n_e = 2 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


# def eh_dlp_short_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]: # Multiple runs with maximal tradeoff.
#     z = fips_strength_level_rounded(n)
#     m = 2 * z
#     n_e = m
#     return estimate_best_problem_cost(n, n_e, gate_error_rate)


def shor_dlp_general(n, gate_error_rate, _) -> Optional[CostEstimate]:
    delta = 5 # Required to reach 99% success probability.
    m = n - 1 + delta
    n_e = 2 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_general(n, gate_error_rate, _) -> Optional[CostEstimate]: # Single run.
    m = n - 1
    n_e = 3 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


# def eh_dlp_general_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]: # Multiple runs with maximal tradeoff.
#     m = n - 1
#     n_e = m
    # return estimate_best_problem_cost(n, n_e, gate_error_rate)



# ------------------------------------------------------------------------------


def tabulate():


    datasets = [
        ("RSA, via Ekera-Håstad with s = 1 in a single run:", eh_rsa),
        ("RSA, via Ekera-Håstad ADI with s = 1 in a single run:", eh_rsa_adi),
        ("RSA, via Ekera-Håstad SlicedWindowing with s = 1 in a single run:", eh_rsa_slicedwindows),
  
    #  ("RSA, via Regev with s = 1 in a single run:", regev_rsa)
        # ("Discrete logarithms, Schnorr group, via Shor:", shor_dlp_schnorr),
        # ("Discrete logarithms, Schnorr group, via Ekera-Håstad with s = 1 in a single run:", eh_dlp_schnorr),
        # ("Discrete logarithms, short exponent, via Ekerå-Håstad with s = 1 in a single run:", eh_dlp_short),
        # ("Discrete logarithms, general, via Shor:", shor_dlp_general),
        # ("Discrete logarithms, general, via Ekerå with s = 1 in a single run:", eh_dlp_general),
    ]

    # datasets = [
    #     ("RSA, via Regev with s = 1 in a single run:", regev_rsa),
    #     ("RSA, via Ekera-Håstad with s = 1 in a single run:", eh_rsa),
    # ]

    for name, func in datasets:
        print()
        print(name)
        print('&'.join(str(e).ljust(10) for e in TABLE_HEADER) + '\\\\')
        print('\hline')
        for e in gate_error_rates:
            for n in moduli:
              for C in Cs:
                tabulate_cost_estimate(func(n, e, C))








if __name__ == '__main__':

    if len(sys.argv) > 1:
        if sys.argv[1] == 'table':
            tabulate()
        elif sys.argv[1] == 'plots':


            # PLOT 1 - RSA (original, adi1, adi2)
            selected_error = 1e-3
            datasets = [
                ('C0', 'RSA via Ekerå-Håstad - 0.1% gate error', eh_rsa, selected_error, 'o', None),
                ('C1', 'RSA via Ekerå-Håstad (improved uncomputation) - 0.1% gate error', eh_rsa_adi, selected_error, 'd', None),
                ('C2', 'RSA via Ekerå-Håstad (sliced windowing) - 0.1% gate error', eh_rsa_slicedwindows, selected_error, 'p', None),
            ]


            # # PLOT 2 - RSA (original, adi1, adi2)
            # selected_error = 1e-4
            # datasets = [
            #     ('C0', 'RSA via Ekerå-Håstad - 0.01% gate error', eh_rsa, selected_error, 'o', None),
            #     ('C1', 'RSA via Ekerå-Håstad (improved uncomputation) - 0.01% gate error', eh_rsa_adi, selected_error, 'd', None),
            #     ('C2', 'RSA via Ekerå-Håstad (sliced windowing) - 0.01% gate error', eh_rsa_slicedwindows, selected_error, 'p', None),
            # ]




                # ('C1', 'Short DLP or Schnorr DLP via EH', eh_dlp_short, 1e-3, 's', None),
                # ('C3', 'Schnorr DLP via Shor', shor_dlp_schnorr, 1e-3, 'd', None),
                # ('C2', 'General DLP via EH', eh_dlp_general, 1e-3, 'P', None),
                # ('C4', 'General DLP via Shor', shor_dlp_general, 1e-3, 'X', None),
            

            plot_datasets(datasets, sys.argv[2])

            plt.show()
