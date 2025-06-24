import math
from models.parameters import CostEstimate
from algorithms.problem_variants import eh_rsa, eh_rsa_orig
from utils.helpers import reduce_significant


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


def tabulate():
    gate_error_rates = [1e-3, 1e-4]
    moduli = [1024, 2048, 3072, 4096, 8192, 12288, 16384]

    datasets = [
        ("Our work with s = 1 in a single run:", eh_rsa),
        ("Ekera-Håstad with s = 1 in a single run:", eh_rsa_orig),
    ]

    for name, func in datasets:
        print()
        print(name)
        print('&'.join(str(e).ljust(10) for e in TABLE_HEADER) + '\\\\')
        print('\hline')
        for e in gate_error_rates:
            for n in moduli:
                tabulate_cost_estimate(func(n, e))
