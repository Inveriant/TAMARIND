import math
from typing import Optional

from models.parameters import CostEstimate
from algorithms.cost_estimator import estimate_best_problem_cost, estimate_all_surviving_problem_cost
from utils.helpers import fips_strength_level_rounded


# RSA with simple improvements
def eh_rsa(n, gate_error_rate) -> Optional[CostEstimate]:
    """Single run."""
    delta = 20  # Required to respect assumptions in the analysis.
    m = math.ceil(n / 2) - 1
    l = m - delta
    n_e = m + 2 * l
    return estimate_best_problem_cost(n, n_e, gate_error_rate, opt_win=True)


def eh_rsa_all_surviving_estimates(n, gate_error_rate) -> list[CostEstimate]:
    """Single run."""
    delta = 20  # Required to respect assumptions in the analysis.
    m = math.ceil(n / 2) - 1
    l = m - delta
    n_e = m + 2 * l
    return estimate_all_surviving_problem_cost(n, n_e, gate_error_rate, opt_win=True)


# Original gidney/ekera implementation of RSA factoring
def eh_rsa_orig(n, gate_error_rate) -> Optional[CostEstimate]:
    """Single run."""
    delta = 20  # Required to respect assumptions in the analysis.
    m = math.ceil(n / 2) - 1
    l = m - delta
    n_e = m + 2 * l
    return estimate_best_problem_cost(n, n_e, gate_error_rate, opt_win=False)


def eh_rsa_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]:
    """With maximum tradeoffs."""
    return estimate_best_problem_cost(n, math.ceil(n / 2), gate_error_rate)


# General DLP
def shor_dlp_general(n, gate_error_rate) -> Optional[CostEstimate]:
    delta = 5  # Required to reach 99% success probability.
    m = n - 1 + delta
    n_e = 2 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_general(n, gate_error_rate) -> Optional[CostEstimate]:
    """Single run."""
    m = n - 1
    n_e = 3 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_general_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]:
    """Multiple runs with maximal tradeoff."""
    m = n - 1
    n_e = m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


# Schnorr DLP
def shor_dlp_schnorr(n, gate_error_rate) -> Optional[CostEstimate]:
    delta = 5  # Required to reach 99% success probability.
    z = fips_strength_level_rounded(n)
    m = 2 * z + delta
    n_e = 2 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_schnorr(n, gate_error_rate) -> Optional[CostEstimate]:
    """Single run."""
    z = fips_strength_level_rounded(n)
    m = 2 * z
    n_e = 3 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_schnorr_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]:
    """Multiple runs with maximal tradeoff."""
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


def eh_dlp_short_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]:
    """Multiple runs with maximal tradeoff."""
    z = fips_strength_level_rounded(n)
    m = 2 * z
    n_e = m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)
