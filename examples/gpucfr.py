"""Run GPU-accelerated CFR.

The script can take some time during initialization to load the desired
game from OpenSpiel.
"""
from sys import stdout

from orjson import dumps, OPT_SERIALIZE_NUMPY
import noregret as nr

KERNEL = nr.CUDAKernel()
GAME = nr.from_open_spiel(
    KERNEL,
    (
        'battleship('
        'board_height=3,'
        'board_width=2,'
        'ship_sizes=[2;2],'
        'ship_values=[1;1],'
        'num_shots=2)'
    ),
)
PARAMETERS = nr.CFR, True, False


def main():
    R_type, alt, pred = PARAMETERS
    R_row = R_type(KERNEL, GAME.row_sequence_form_polytope)
    R_col = R_type(KERNEL, GAME.column_sequence_form_polytope)
    x_bar, y_bar = nr.regret_minimization(
        GAME,
        R_row,
        R_col,
        alternation=alt,
        prediction=pred,
    )
    data = {
        'x_bar': KERNEL.numpy.asnumpy(x_bar),
        'y_bar': KERNEL.numpy.asnumpy(y_bar),
        'Exploitability': GAME.exploitability(x_bar, y_bar).item(),
        'Expected utility': GAME.expected_row_utility(x_bar, y_bar).item(),
    }

    stdout.buffer.write(dumps(data, option=OPT_SERIALIZE_NUMPY))


if __name__ == '__main__':
    main()
