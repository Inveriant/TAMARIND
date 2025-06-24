import math
from typing import Tuple

from models.parameters import Parameters


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


def physical_qubits_per_logical_qubit(code_distance: int) -> int:
    return (code_distance + 1)**2 * 2
