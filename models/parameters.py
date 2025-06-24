import datetime
import math
import itertools
from typing import NamedTuple, Iterator


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


def parameters_to_attempt(n: int,
                          n_e: int,
                          gate_error_rate: float,
                          opt_win: bool) -> Iterator[Parameters]:
    if n < 1024:
        l1_distances = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        l2_distances = range(3, 30, 2)
        exp_windows = [1, 2, 3, 4, 5]
        mul_windows = [1, 2, 3, 4, 5]
        runway_seps = [32, 64, 128, 256]
        larger_init_lookup = range(1, 20) if opt_win else [0]
        dev_offs = range(2, 10)
    else:
        l1_distances = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        l2_distances = range(9, 51, 2)
        exp_windows = [4, 5, 6]
        mul_windows = [4, 5, 6]
        runway_seps = [512, 768, 1024, 1536, 2048]
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
