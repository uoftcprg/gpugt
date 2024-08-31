""":mod:``gpugt.games.tic_tac_toe` defines tic tac toe."""

from functools import partial
from operator import ge
from typing import TypeAlias

from gpugt.collections2 import FrozenOrderedMapping, FrozenOrderedSet
from gpugt.games.finite_extensive_form_game import FiniteExtensiveFormGame
from gpugt.graphs.finite_tree import FiniteTree

_A = int
_V = tuple[_A, ...]
_H = tuple[frozenset[_A], frozenset[_A]]
_I = str
TicTacToe: TypeAlias = FiniteExtensiveFormGame[_V, _H, _A, _I]
"""A type alias for tic-tac-toe."""
_PATTERNS: tuple[set[_A], ...] = (
    {0, 1, 2},
    {3, 4, 5},
    {6, 7, 8},
    {0, 3, 6},
    {1, 4, 7},
    {2, 5, 8},
    {0, 4, 8},
    {2, 4, 6},
)


def create_tic_tac_toe() -> TicTacToe:
    """Create a tic-tac-toe game.

    :return: A tic-tac-toe game.
    """
    vertices = []
    root = ()
    leaves = []
    parents = {}
    information_sets = []
    information_partition = {}
    actions = range(9)
    action_partition = {}
    players = 'XO'
    player_partition = {}
    payoffs = dict[_V, FrozenOrderedMapping[_I, float]]()

    def update(vertex: _V) -> None:
        vertices.append(vertex)

        if vertex:
            parents[vertex] = vertex[:-1]
            action_partition[vertex] = vertex[-1]

        information_set = frozenset(vertex[::2]), frozenset(vertex[1::2])
        nought_pattern_count = sum(
            map(partial(ge, information_set[1]), _PATTERNS),
        )
        cross_pattern_count = sum(
            map(partial(ge, information_set[0]), _PATTERNS),
        )

        if len(vertex) == 9 or nought_pattern_count or cross_pattern_count:
            leaves.append(vertex)

            if nought_pattern_count:
                payoffs[vertex] = FrozenOrderedMapping({'X': -1, 'O': 1})
            elif cross_pattern_count:
                payoffs[vertex] = FrozenOrderedMapping({'X': 1, 'O': -1})
            else:
                payoffs[vertex] = FrozenOrderedMapping({'X': 0, 'O': 0})
        else:
            information_sets.append(information_set)

            information_partition[vertex] = information_set
            player_partition[information_set] = players[len(vertex) % 2]

            for action in set(actions) - set(vertex):
                update(vertex + (action,))

    update(root)

    return FiniteExtensiveFormGame(
        FiniteTree(
            FrozenOrderedSet(vertices),
            root,
            FrozenOrderedSet(leaves),
            FrozenOrderedMapping(parents),
        ),
        FrozenOrderedSet(information_sets),
        FrozenOrderedMapping(information_partition),
        FrozenOrderedSet(actions),
        FrozenOrderedMapping(action_partition),
        FrozenOrderedSet(players),
        None,
        FrozenOrderedMapping(player_partition),
        FrozenOrderedMapping(),
        FrozenOrderedMapping(payoffs),
    )
