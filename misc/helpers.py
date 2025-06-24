import datetime
import itertools

from typing import Tuple, NamedTuple, Iterable, Iterator, Optional

import math
import matplotlib.pyplot as plt

import os.path as pathutils

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
    'tot_err',
    'log qubits'

]

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
        ('deviation_padding', int)
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
        ('total_volume_megaqubitdays', float),
        ('logical_qubits', float)
    ]
)



def significant_bits(n: int) -> int:
    assert n >= 0
    high = n.bit_length()
    low = (n ^ (n - 1)).bit_length()
    return high - low + 1




def plot_datasets(datasets, filename="test.pdf", is_regev=False):
    # Choose bit sizes to plot.
    max_steps = 5
    bits = [1024 * s for s in range(1, max_steps + 1)]
    bits = [e for e in bits if significant_bits(e) <= 3]
    max_y = 1024 * max_steps



    plt.subplots(figsize=(16, 9))  # force 16 x 9 inches layout for the PDF

    for color, name, func, gate_error_rate, marker, radiusFactor in datasets:
        valid_ns = []
        hours = []
        megaqubits = []

        for n in bits:
            cost = func(n, gate_error_rate, radiusFactor)
            if cost is not None:
                expected_hours = cost.total_hours / (1 - cost.total_error)
                hours.append(expected_hours)
                megaqubits.append(cost.total_megaqubits)
                valid_ns.append(n)
        if is_regev==True:
          plt.plot(valid_ns, hours, color=color, label=name + f', C={radiusFactor}, hours', marker=marker)
          plt.plot(valid_ns, megaqubits, color=color, label=name + f' C={radiusFactor}, megaqubits ', linestyle='--', marker=marker)
        else:
          plt.plot(valid_ns, hours, color=color, label=name + f', hours', marker=marker)
          plt.plot(valid_ns, megaqubits, color=color, label=name + f' megaqubits ', linestyle='--', marker=marker)
        
        


    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1024, max_y)
    plt.xticks(bits, [str(e) for e in bits], rotation=90)
    yticks = [(5 if e else 1)*10**k
              for k in range(6)
              for e in range(2)][:-1]
    plt.yticks(yticks, [str(e) for e in yticks])
    plt.minorticks_off()
    plt.grid(True)
    plt.xlabel('modulus length n (bits)')
    plt.ylabel('expected time (hours) and physical qubit count (megaqubits)')
    plt.gcf().subplots_adjust(bottom=0.16)

    plt.legend(loc='upper left', shadow=False)

    plt.tight_layout()  # truncate margins

    # Export the figure to a PDF file.
    path = pathutils.dirname(pathutils.realpath(__file__))
    path = pathutils.normpath(path + f"/{filename}.pdf")
    plt.savefig(path)





def reduce_significant(q: float) -> float:
    """Return only the n most significant digits."""
    if q == 0:
        return 0
    n = math.floor(math.log(q, 10))
    result = math.ceil(q / 10**(n-1)) * 10**(n-1)

    # Handle poor precision in float type.
    if result < 0.1:
        return round(result * 100) / 100
    elif result < 10:
        return round(result * 10) / 10
    else:
        return round(result)


def fips_strength_level(n):
    # From FIPS 140-2 IG CMVP, page 110.
    #
    # This is extrapolated from the asymptotic complexity of the sieving
    # step in the general number field sieve (GNFS).
    ln = math.log
    return (1.923 * (n * ln(2))**(1/3) * ln(n * ln(2))**(2/3) - 4.69) / ln(2)   # hyperparams 


def fips_strength_level_rounded(n):  # NIST-style rounding
    return 8 * round(fips_strength_level(n) / 8)



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







def physical_qubits_per_logical_qubit(code_distance: int) -> int:
    # 2(d+1)^2
    return (code_distance + 1)**2 * 2

