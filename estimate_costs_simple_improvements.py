import datetime
import itertools

from typing import Tuple, NamedTuple, Iterator, Optional
import numpy as np

import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import seaborn as sns
import os.path as pathutils

Parameters = NamedTuple(
    'Parameters',
    [
        # Physical gate error rate.
        ('gate_err', float),
        # Time it takes to trigger a logical measurement, error correct it,
        # and decide which measurement to do next.
        ('reaction_time', datetime.timedelta),
        # Time it takes to measure the surface code's local stabilizers.
        ('cycle_time', datetime.timedelta),
        # Window size over exponent bits. (g0 in paper)
        ('exp_window', int),
        # Window size over multiplication bits. (g1 in paper)
        ('mul_window', int),
        # Bits between runways used during parallel additions. (g2 in paper)
        ('runway_sep', int),
        # Level 2 code distance.
        ('code_distance', int),
        # Level 1 code distance.
        ('l1_distance', int),
        # Error budget.
        ('max_total_err', float),
        # Problem size.
        ('n', int),
        # Number of controlled group operations required.
        ('n_e', int),
        # Whether or not to use two levels of 15-to-1 distillation.
        ('use_t_t_distillation', bool),
        # Number of bits of padding to use for runways and modular coset.
        ('deviation_padding', int),
        # Whether or not to use optimized windowing.
        ('opt_win', bool),
        # params for direct exponentiation
        ('larger_init_lookup', int),
    ]
)


def parameters_to_attempt(n: int,
                          n_e: int,
                          gate_error_rate: float,
                          opt_win: bool) -> Iterator[Parameters]:
    if n< 1024:
      l1_distances = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
      l2_distances = range(3, 30, 2)
      exp_windows = [1,2,3,4,5]
      mul_windows = [1,2,3,4,5]
      runway_seps = [32,64,128,256]
      # don't do large initial lookup if opt window is false
      larger_init_lookup = range(1, 20) if opt_win else [0]
      dev_offs = range(2, 10)
    else:
      l1_distances = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
      l2_distances = range(9, 51, 2)
      exp_windows = [4, 5, 6]
      mul_windows = [4, 5, 6]
      runway_seps = [512, 768, 1024, 1536, 2048]
      # don't do large initial lookup if opt window is false
      larger_init_lookup = range(17, 27) if opt_win else [0]
      dev_offs = range(2, 10)

    for d1, d2, exp_window, mul_window, runway_sep, dev_off, larger_init_lkp in itertools.product(
            l1_distances,
            l2_distances,
            exp_windows,
            mul_windows,
            runway_seps,
            dev_offs,
            larger_init_lookup):
        if mul_window > exp_window or n % runway_sep != 0:
            continue
        distill_types = [False]
        if d1 == 15 and d2 >= 31:
            distill_types.append(True)
        for b in distill_types:
            yield Parameters(
              gate_err=gate_error_rate,
              reaction_time=datetime.timedelta(microseconds=10),
              cycle_time=datetime.timedelta(microseconds=1),
              exp_window=exp_window,
              mul_window=mul_window,
              runway_sep=runway_sep,
              l1_distance=d1,
              code_distance=d2,
              max_total_err=0.8,
              n=n,
              n_e=n_e,
              use_t_t_distillation=b,
              deviation_padding=int(math.ceil(math.log2(n * n * (n_e - (larger_init_lkp if opt_win else 0)))) + dev_off),
              larger_init_lookup=larger_init_lkp,
              opt_win=opt_win)


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


DeviationProperties = NamedTuple(
    'DeviationProperties',
    [
        ('piece_count', int),
        ('piece_len', int),
        ('reg_len', int),
        ('inner_loop_count', int),
        ('deviation_error', float),
    ]
)


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


def logical_factory_dimensions(params: Parameters
                               ) -> Tuple[int, int, float]:
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


def board_logical_dimensions(params: Parameters,
                             register_len: int) -> Tuple[int, int, int]:
    """Computes the dimensions of the surface code board in logical qubits.

    Assumes a single-threaded execution. For parallel execution, pass in
    parameters for an individual adder piece.

    Returns:
        width, height, distillation_area
    """

    factory_width, factory_height, factory_depth = (
        logical_factory_dimensions(params))
    ccz_time = factory_depth * params.cycle_time * params.code_distance
    factory_pair_count = int(math.ceil(ccz_time / params.reaction_time / 2))
    total_width = (factory_width + 1) * factory_pair_count + 1

    # FIG. 15 Lattice surgery implementation of the CZ fixups
    cz_fixups_box_height = 3

    # FIG. 23. Implementation of the MAJ operation in lattice surgery.
    adder_height = 3

    # FIG. 31. Data layout during a parallel addition.
    routing_height = 6
    reg_height = int(math.ceil(register_len / (total_width - 2)))
    total_height = sum([
        factory_height * 2,
        cz_fixups_box_height * 2,
        adder_height,
        routing_height,
        reg_height * 3,
    ])
    distillation_area = factory_height * factory_width * factory_pair_count * 2

    return total_width, total_height, distillation_area


CostEstimate = NamedTuple(
    'CostEstimate',
    [
        ('params', Parameters),
        ('toffoli_count', int),
        ('total_error', float),
        ('distillation_error', float),
        ('topological_data_error', float),
        ('total_hours', float),
        ('total_megaqubits', int),
        ('total_volume_megaqubitdays', float)
    ]
)


def physical_qubits_per_logical_qubit(code_distance: int) -> int:
    return (code_distance + 1)**2 * 2


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
    estimates = [estimate_algorithm_cost(params)
                 for params in parameters_to_attempt(n, n_e, gate_error_rate, opt_win)]
    surviving_estimates = [e for e in estimates if e is not None]
    return min(surviving_estimates, key=rank_estimate, default=None)
  
  
def estimate_all_surviving_problem_cost(n: int, n_e: int, gate_error_rate: float, opt_win: bool = True) -> Optional[CostEstimate]:
    estimates = [estimate_algorithm_cost(params)
                 for params in parameters_to_attempt(n, n_e, gate_error_rate, opt_win)]
    surviving_estimates = [e for e in estimates if e is not None]
    return surviving_estimates


# ------------------------------------------------------------------------------
def reduce_significant(q: float) -> float:
    """Return the value rounded to exactly 3 decimal places."""
    if q == 0:
        return 0.000
    return round(q, 3)


def fips_strength_level(n):
    # From FIPS 140-2 IG CMVP, page 110.
    #
    # This is extrapolated from the asymptotic complexity of the sieving
    # step in the general number field sieve (GNFS).
    ln = math.log
    return (1.923 * (n * ln(2))**(1/3) * ln(n * ln(2))**(2/3) - 4.69) / ln(2)


def fips_strength_level_rounded(n): # NIST-style rounding
    return 8 * round(fips_strength_level(n) / 8)


TABLE_HEADER = [
    'n',
    'n_e',
    'phys_err',
    'd1',
    'd2',
    'dev_off',
    'g_mul',
    'g_exp',
    'g_sep',
    '%',
    'volume',
    'E:volume',
    'Mqb',
    'hours',
    'E:hours',
    'tt_distill',
    'B tofs',
    'Init lookup'
]


def tabulate_cost_estimate(costs: CostEstimate):
    assert costs.params.gate_err in [1e-3, 1e-4]
    gate_error_desc = r"0.1\%" if costs.params.gate_err == 1e-3 else r"0.01\%"
    row = [
        costs.params.n,
        costs.params.n_e,
        gate_error_desc,
        costs.params.l1_distance,
        costs.params.code_distance,
        costs.params.deviation_padding - int(math.ceil(math.log2(costs.params.n**2*(costs.params.n_e-costs.params.larger_init_lookup)))),
        costs.params.mul_window,
        costs.params.exp_window,
        costs.params.runway_sep,
        str(math.ceil(100 * costs.total_error)) + r"\%",
        reduce_significant(costs.total_volume_megaqubitdays),
        reduce_significant(costs.total_volume_megaqubitdays / (1 - costs.total_error)),
        reduce_significant(costs.total_megaqubits),
        reduce_significant(costs.total_hours),
        reduce_significant(costs.total_hours / (1 - costs.total_error)),
        costs.params.use_t_t_distillation,
        reduce_significant(costs.toffoli_count / 10**9),
        costs.params.larger_init_lookup,
    ]
    print('&'.join('${}$'.format(e).ljust(10) for e in row) + '\\\\')

# ------------------------------------------------------------------------------


# RSA with simple improvements
def eh_rsa(n, gate_error_rate) -> Optional[CostEstimate]: # Single run.
    delta = 20 # Required to respect assumptions in the analysis.
    m = math.ceil(n / 2) - 1
    l = m - delta
    n_e = m + 2 * l
    return estimate_best_problem_cost(n, n_e, gate_error_rate, opt_win=True)
  
  
def eh_rsa_all_surviving_estimates(n, gate_error_rate) -> Optional[CostEstimate]: # Single run.
    delta = 20 # Required to respect assumptions in the analysis.
    m = math.ceil(n / 2) - 1
    l = m - delta
    n_e = m + 2 * l
    return estimate_all_surviving_problem_cost(n, n_e, gate_error_rate, opt_win=True)
  
# Original gidney/ekera implementation of RSA factoring
def eh_rsa_orig(n, gate_error_rate) -> Optional[CostEstimate]: # Single run.
  delta = 20 # Required to respect assumptions in the analysis.
  m = math.ceil(n / 2) - 1
  l = m - delta
  n_e = m + 2 * l
  return estimate_best_problem_cost(n, n_e, gate_error_rate, opt_win=False)


def eh_rsa_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]: # With maximum tradeoffs.
    return estimate_best_problem_cost(n, math.ceil(n / 2), gate_error_rate)


# General DLP
def shor_dlp_general(n, gate_error_rate) -> Optional[CostEstimate]:
    delta = 5 # Required to reach 99% success probability.
    m = n - 1 + delta
    n_e = 2 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_general(n, gate_error_rate) -> Optional[CostEstimate]: # Single run.
    m = n - 1
    n_e = 3 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_general_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]: # Multiple runs with maximal tradeoff.
    m = n - 1
    n_e = m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


# Schnorr DLP
def shor_dlp_schnorr(n, gate_error_rate) -> Optional[CostEstimate]:
    delta = 5 # Required to reach 99% success probability.
    z = fips_strength_level_rounded(n)
    m = 2 * z + delta
    n_e = 2 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_schnorr(n, gate_error_rate) -> Optional[CostEstimate]: # Single run.
    z = fips_strength_level_rounded(n)
    m = 2 * z
    n_e = 3 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_schnorr_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]: # Multiple runs with maximal tradeoff.
    z = fips_strength_level_rounded(n)
    m = 2 * z
    n_e = m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


# Short DLP
def eh_dlp_short(n, gate_error_rate) -> Optional[CostEstimate]:
    z = fips_strength_level_rounded(n)
    m = 2 * z
    n_e = 3 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_short_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]: # Multiple runs with maximal tradeoff.
    z = fips_strength_level_rounded(n)
    m = 2 * z
    n_e = m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)

# ------------------------------------------------------------------------------


def tabulate():
    gate_error_rates = [1e-3, 1e-4]
    moduli = [1024, 2048, 3072, 4096, 8192, 12288, 16384]

    datasets = [
        ("Our work with s = 1 in a single run:", eh_rsa),
        ("Ekera-Håstad with s = 1 in a single run:", eh_rsa_orig),
        # ("Discrete logarithms, Schnorr group, via Shor:", shor_dlp_schnorr),
        # ("Discrete logarithms, Schnorr group, via Ekera-Håstad with s = 1 in a single run:", eh_dlp_schnorr),
        # ("Discrete logarithms, short exponent, via Ekerå-Håstad with s = 1 in a single run:", eh_dlp_short),
        # ("Discrete logarithms, general, via Shor:", shor_dlp_general),
        # ("Discrete logarithms, general, via Ekerå with s = 1 in a single run:", eh_dlp_general),
    ]

    for name, func in datasets:
        print()
        print(name)
        print('&'.join(str(e).ljust(10) for e in TABLE_HEADER) + '\\\\')
        print('\hline')
        for e in gate_error_rates:
            for n in moduli:
                tabulate_cost_estimate(func(n, e))


def significant_bits(n: int) -> int:
    assert n >= 0
    high = n.bit_length()
    low = (n ^ (n - 1)).bit_length()
    return high - low + 1


def plot(key_size: int = 1024):
    # Try to load cached data first
    cache_file = f'plot_cache_{key_size}.npy'
    if pathutils.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        cached_data = np.load(cache_file, allow_pickle=True).item()
        plot_from_cache(cached_data, key_size)
        return

    # Choose bit sizes to plot if no cache exists
    min_key_size = key_size
    if key_size == 1024:
        max_steps = 64
    else:
        max_steps = 8
    bits = [min_key_size * s for s in range(1, max_steps + 1)]
    bits = [e for e in bits if significant_bits(e) <= 3]
    max_y = min_key_size * max_steps

    datasets = [
        ('C0', 'Our work - 0.1% gate error rate', eh_rsa, 1e-3, 'o'),
        ('C1', 'Our work - 0.01% gate error rate', eh_rsa, 1e-4, '*'),
        ('C2', 'Ekerå-Håstad - 0.1% gate error rate', eh_rsa_orig, 1e-3, 's'),
        ('C3', 'Ekerå-Håstad - 0.01% gate error rate', eh_rsa_orig, 1e-4, 'd'),
    ]

    # Adjust figure size for small key sizes
    if key_size <= 256:
        plt.subplots(figsize=(20, 11))  # Increased height for legend
    else:
        plt.subplots(figsize=(16, 9))
    plt.rcParams.update({'font.size': 14})

    # Dictionary to store results for caching
    cache_data = {
        'bits': bits,
        'max_y': max_y,
        'results': []
    }

    def process_dataset(color, name, func, gate_error_rate, marker):
        valid_ns = []
        hours = []
        megaqubits = []

        for n in bits:
            cost = func(n, gate_error_rate)
            if cost is not None:
                expected_hours = cost.total_hours / (1 - cost.total_error)
                hours.append(expected_hours)
                megaqubits.append(cost.total_megaqubits)
                valid_ns.append(n)

        return color, name, marker, valid_ns, hours, megaqubits

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_dataset, color, name, func, gate_error_rate, marker)
               for color, name, func, gate_error_rate, marker in datasets]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Datasets"):
            color, name, marker, valid_ns, hours, megaqubits = future.result()
            cache_data['results'].append({
                'color': color,
                'name': name,
                'marker': marker,
                'valid_ns': valid_ns,
                'hours': hours,
                'megaqubits': megaqubits
            })
            plt.plot(valid_ns, hours, color=color, label=name + ', hours', marker=marker)
            plt.plot(valid_ns, megaqubits, color=color, label=name + ', megaqubits', linestyle='--', marker=marker)

    # Save the computed data
    np.save(cache_file, cache_data)
    print(f"Cached data saved to {cache_file}")

    finish_plot(bits, max_y, key_size)


def plot_from_cache(cache_data, key_size):
    # Adjust figure size for small key sizes
    # if key_size <= 256:
    #     plt.subplots(figsize=(20, 11))  # Increased height for legend
    # else:
    plt.subplots(figsize=(16, 9))
    plt.rcParams.update({'font.size': 14})

    bits = cache_data['bits']
    max_y = cache_data['max_y']

    for result in cache_data['results']:
        plt.plot(result['valid_ns'], result['hours'], 
                color=result['color'], 
                label=result['name'] + ', hours', 
                marker=result['marker'])
        plt.plot(result['valid_ns'], result['megaqubits'],
                color=result['color'],
                label=result['name'] + ', megaqubits',
                linestyle='--',
                marker=result['marker'])

    finish_plot(bits, max_y, key_size)


def finish_plot(bits, max_y, key_size):
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(key_size, max_y)
    plt.xticks(bits, [str(e) for e in bits], rotation=90, fontsize=12)
    
    if key_size <= 256:
        yticks = [(5 if e else 1) * 10**k
                for k in range(3)  # Only go up to 10^2 = 100
                for e in range(2)][:-1]
        plt.ylim(top=500)
    else:
        yticks = [(5 if e else 1) * 10**k
                for k in range(6)
                for e in range(2)][:-1]
                
    plt.yticks(yticks, [str(e) for e in yticks], fontsize=12)
    plt.minorticks_off()
    plt.grid(True)
    plt.xlabel('Modulus length n (bits)', fontsize=16, fontweight='bold')
    plt.ylabel('Expected time (hours) and physical qubit count (megaqubits)', fontsize=16, fontweight='bold')

    # Place legend inside the axes in the upper-left corner
    plt.legend(loc='upper left', shadow=False, fontsize=14)
    plt.gcf().subplots_adjust(bottom=0.16)  # Adjust bottom to make room for ticks
    
    plt.tight_layout()

    path = pathutils.dirname(pathutils.realpath(__file__))
    path = pathutils.normpath(path + f'/assets/{str(key_size)}-rsa-dlps-extras.pdf')
    plt.savefig(path, bbox_inches='tight')


def update_cached_labels(key_size: int = 1024):
    cache_file = f'plot_cache_{key_size}.npy'
    if pathutils.exists(cache_file):
        cached_data = np.load(cache_file, allow_pickle=True).item()
        
        # Define the mapping from old to new labels
        new_names = {
            'RSA via Optimized Windowing - 0.1% gate error rate': 'Our work - 0.1% gate error rate',
            'RSA via Optimized Windowing - 0.01% gate error rate': 'Our work - 0.01% gate error rate',
            'RSA via Ekerå-Håstad - 0.1% gate error rate': 'Ekerå-Håstad - 0.1% gate error rate',
            'RSA via Ekerå-Håstad - 0.01% gate error rate': 'Ekerå-Håstad - 0.01% gate error rate'
        }
        
        # Update the names in the cached data
        for result in cached_data['results']:
            if result['name'] in new_names:
                result['name'] = new_names[result['name']]
        
        # Save the updated data back to the cache file
        np.save(cache_file, cached_data)
        print(f"Updated labels in {cache_file}")

def plot_surviving_estimates(key_sizes: list[int], gate_error_rate: float):
    """Creates a scatter plot of all surviving estimates for multiple key sizes."""
    # Set up the plot style
    plt.subplots(figsize=(16, 9))
    plt.rcParams.update({'font.size': 14})
    
    # Use consistent colors from the rest of the code
    color_map = {
        1024: 'C0',  # Blue
        2048: 'C1',  # Orange
        3072: 'C2',  # Green
        4096: 'C3',  # Red
        8192: 'C4',  # Purple
        12288: 'C5', # Brown
        16384: 'C6'  # Pink
    }
    
    # Store best estimates for annotation
    best_estimates = {}
    
    # Create empty lists for legend handles and labels
    legend_elements = []
    
    # Process each key size with tqdm progress bar
    min_time = float('inf')
    max_time = 0
    min_qubits = float('inf')
    max_qubits = 0
    
    for idx, n in tqdm(enumerate(key_sizes), total=len(key_sizes), desc="Processing key sizes"):
        surviving_estimates = eh_rsa_all_surviving_estimates(n, gate_error_rate)
        
        if not surviving_estimates:
            print(f"No surviving estimates for n={n}")
            continue
            
        expected_times = [est.total_hours / (1 - est.total_error) for est in surviving_estimates]
        qubit_counts = [est.total_megaqubits for est in surviving_estimates]
        
        # Plot points with low alpha
        plt.scatter(expected_times, qubit_counts, 
                   alpha=0.2,  # Low transparency for actual points
                   c=color_map[n],
                   s=100,
                   rasterized=False,
                   label='_nolegend_')  # Don't include in legend
        
        # Create a proxy artist for the legend with full opacity
        legend_elements.append(plt.scatter([], [], 
                                         c=color_map[n],
                                         marker='$\u26BF$',
                                         s=100,
                                         alpha=1.0,  # Full opacity for legend
                                         label=f'n={n} bits'))
        
        best = min(surviving_estimates, key=rank_estimate)
        best_estimates[n] = best
        
        best_time = best.total_hours / (1 - best.total_error)
        best_qubits = best.total_megaqubits
        plt.scatter([best_time], [best_qubits], 
                   color=color_map[n],
                   marker='$\u26BF$',  # Diamond marker for minimum estimates
                   s=200,
                   zorder=5,
                   edgecolor='black',
                   linewidth=1,
                   alpha=1.0,
                   label='_nolegend_')  # Don't add duplicate legend entries
        
        min_time = min(min_time, min(expected_times))
        max_time = max(max_time, max(expected_times))
        min_qubits = min(min_qubits, min(qubit_counts))
        max_qubits = max(max_qubits, max(qubit_counts))
    
    # Set up axes
    plt.xscale('log')
    plt.yscale('log')
    
    # Generate tick locations
    def generate_ticks(min_val, max_val):
        ticks = []
        start_exp = int(np.floor(np.log10(min_val)))
        end_exp = int(np.ceil(np.log10(max_val)))
        
        for exp in range(start_exp, end_exp + 1):
            for base in [1, 2, 5]:
                tick = base * 10**exp
                if min_val <= tick <= max_val:
                    ticks.append(tick)
        return ticks
    
    x_ticks = generate_ticks(min_time, max_time)
    y_ticks = generate_ticks(min_qubits, max_qubits)
    
    plt.xticks(x_ticks, [f'{t:.1f}' for t in x_ticks], rotation=45, fontsize=12)
    plt.yticks(y_ticks, [f'{t:.1f}' for t in y_ticks], fontsize=12)
    
    plt.minorticks_off()  # Match other plots
    plt.grid(True)  # Simple grid like other plots
    
    plt.xlabel('Expected time (hours)', fontsize=16, fontweight='bold')
    plt.ylabel('Physical qubit count (millions)', fontsize=16, fontweight='bold')
    
    # Create custom legend handles
    from matplotlib.lines import Line2D
    legend_handles = []
    for n, color in color_map.items():
        if n in best_estimates:
            # Create a solid marker for the legend
            handle = Line2D([0], [0], marker='o', color='none', 
                            markerfacecolor=color, alpha=1.0, 
                            markersize=10, label=f'n={n} bits')
            legend_handles.append(handle)

    # Add the minimum estimate marker
    min_handle = Line2D([0], [0], marker='$\u26BF$', color='gray', 
                        label='Minimum estimate', markersize=10, 
                        markerfacecolor='gray', linestyle='None')
    legend_handles.append(min_handle)
    
    # Add the legend to the plot
    plt.legend(handles=legend_handles, 
               title='Key Sizes', 
               title_fontsize=14, 
               fontsize=14, 
               loc='upper left', 
               shadow=False)
    
    plt.gcf().subplots_adjust(bottom=0.16)
    plt.tight_layout()
    
    # Save plot
    path = pathutils.dirname(pathutils.realpath(__file__))
    path = pathutils.normpath(path + f'/assets/surviving-estimates-{gate_error_rate:.0e}.png')
    plt.savefig(path, bbox_inches='tight', dpi=300, transparent=True)
    plt.savefig(path, bbox_inches='tight')
    print(f"\nPlot saved to: {path}")

if __name__ == '__main__':
    # tabulate()
    # Update all common key sizes
    # for key_size in [256, 1024]:  # add or remove sizes as needed
    #     update_cached_labels(key_size)
    # plot(key_size=256)
    # plot()
# Example usage
    key_sizes = [1024, 2048, 3072, 4096]
    plot_surviving_estimates(key_sizes, gate_error_rate=1e-3)
    plt.show()
    
    
    
    """
    RSA, via Ekera-Håstad with s = 1 in a single run:
n         &n_e       &phys_err  &d1        &d2        &dev_off   &g_mul     &g_exp     &g_sep     &%         &volume    &E:volume  &Mqb       &hours     &E:hours   &tt_distill&B tofs    &Init lookup\\
\hline
$1024$    &$1493$    &$0.1\%$   &$15$      &$27$      &$4$       &$5$       &$5$       &$1024$    &$6\%$     &$0.488$   &$0.519$   &$9.624$   &$1.217$   &$1.295$   &$False$   &$0.393$   &$18$      \\
$2048$    &$3029$    &$0.1\%$   &$17$      &$27$      &$6$       &$5$       &$5$       &$1024$    &$20\%$    &$4.419$   &$5.524$   &$21.616$  &$4.906$   &$6.133$   &$False$   &$2.656$   &$19$      \\
$3072$    &$4565$    &$0.1\%$   &$17$      &$29$      &$4$       &$4$       &$5$       &$1024$    &$9\%$     &$17.727$  &$19.48$   &$37.897$  &$11.226$  &$12.336$  &$False$   &$9.742$   &$20$      \\
$4096$    &$6101$    &$0.1\%$   &$17$      &$31$      &$8$       &$4$       &$5$       &$1024$    &$5\%$     &$46.424$  &$48.867$  &$54.616$  &$20.4$    &$21.474$  &$False$   &$22.8$    &$20$      \\
$8192$    &$12245$   &$0.1\%$   &$19$      &$33$      &$4$       &$4$       &$5$       &$1024$    &$5\%$     &$460.418$ &$484.65$  &$133.319$ &$82.884$  &$87.246$  &$False$   &$177.051$ &$21$      \\
$12288$   &$18389$   &$0.1\%$   &$19$      &$33$      &$6$       &$4$       &$5$       &$1024$    &$11\%$    &$1558.32$ &$1750.921$&$199.979$ &$187.018$ &$210.133$ &$False$   &$594.127$ &$22$      \\
$16384$   &$24533$   &$0.1\%$   &$19$      &$33$      &$3$       &$4$       &$5$       &$1024$    &$24\%$    &$3687.73$ &$4852.276$&$266.638$ &$331.931$ &$436.751$ &$False$   &$1398.629$&$22$      \\
$1024$    &$1493$    &$0.01\%$  &$7$       &$13$      &$4$       &$5$       &$5$       &$512$     &$5\%$     &$0.067$   &$0.071$   &$2.637$   &$0.612$   &$0.644$   &$False$   &$0.402$   &$18$      \\
$2048$    &$3029$    &$0.01\%$  &$7$       &$13$      &$3$       &$5$       &$5$       &$512$     &$21\%$    &$0.541$   &$0.684$   &$5.273$   &$2.461$   &$3.115$   &$False$   &$2.72$    &$19$      \\
$3072$    &$4565$    &$0.01\%$  &$9$       &$15$      &$5$       &$5$       &$5$       &$768$     &$2\%$     &$2.851$   &$2.909$   &$9.12$    &$7.503$   &$7.656$   &$False$   &$8.486$   &$20$      \\
$4096$    &$6101$    &$0.01\%$  &$9$       &$15$      &$5$       &$4$       &$5$       &$512$     &$3\%$     &$6.588$   &$6.792$   &$15.241$  &$10.374$  &$10.695$  &$False$   &$23.559$  &$20$      \\
$8192$    &$12245$   &$0.01\%$  &$9$       &$15$      &$5$       &$4$       &$5$       &$512$     &$9\%$     &$52.92$   &$58.154$  &$30.482$  &$41.666$  &$45.787$  &$False$   &$184.405$ &$21$      \\
$12288$   &$18389$   &$0.01\%$  &$9$       &$15$      &$5$       &$4$       &$5$       &$512$     &$25\%$    &$179.074$ &$238.766$ &$45.724$  &$93.995$  &$125.326$ &$False$   &$618.823$ &$22$      \\
$16384$   &$24533$   &$0.01\%$  &$9$       &$17$      &$4$       &$5$       &$5$       &$1024$    &$3\%$     &$636.452$ &$656.136$ &$56.682$  &$269.484$ &$277.818$ &$False$   &$1136.762$&$23$      \\
    
    """