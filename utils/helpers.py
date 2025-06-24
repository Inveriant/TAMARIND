import math


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


def fips_strength_level(n):
    # From FIPS 140-2 IG CMVP, page 110.
    #
    # This is extrapolated from the asymptotic complexity of the sieving
    # step in the general number field sieve (GNFS).
    ln = math.log
    return (1.923 * (n * ln(2))**(1/3) * ln(n * ln(2))**(2/3) - 4.69) / ln(2)


def fips_strength_level_rounded(n):  # NIST-style rounding
    return 8 * round(fips_strength_level(n) / 8)


def significant_bits(n: int) -> int:
    assert n >= 0
    high = n.bit_length()
    low = (n ^ (n - 1)).bit_length()
    return high - low + 1


def reduce_significant(q: float) -> float:
    """Return the value rounded to exactly 3 decimal places."""
    if q == 0:
        return 0.000
    return round(q, 3)
