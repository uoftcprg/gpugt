""":mod:``gpugt.games.open_spiel` defines games constructed using
OpenSpiel.
"""

from copy import deepcopy
from typing import Any
from warnings import warn

try:
    from pyspiel import Game, SpielError  # type: ignore[import-not-found]
except ImportError:
    raise ImportError('OpenSpiel not installed. Please install through pip.')

from gpugt.collections2 import FrozenOrderedMapping, FrozenOrderedSet
from gpugt.games.finite_extensive_form_game import FiniteExtensiveFormGame
from gpugt.graphs.finite_tree import FiniteTree

_V = Any
_H = Any
_A = int
_I = int


def create_open_spiel(game: Game) -> FiniteExtensiveFormGame[_V, _H, _A, _I]:
    """Create a game from OpenSpiel.

    :param game: The OpenSpiel game.
    :return: A game created from OpenSpiel.
    """
    vertices = []
    root = game.new_initial_state()
    leaves = []
    parents = {}
    information_sets = []
    information_partition = {}
    actions = []
    action_partition = {}
    players = []
    nature = None
    player_partition = {}
    nature_probabilities = dict[_H, FrozenOrderedMapping[_A, float]]()
    payoffs = dict[_V, FrozenOrderedMapping[_I, float]]()

    def update(vertex: _V, parent: _V | None, action: _A | None) -> None:
        nonlocal nature

        vertices.append(vertex)

        if parent is not None:
            parents[vertex] = parent

        if action is not None:
            action_partition[vertex] = action

        if vertex.is_terminal():
            leaves.append(vertex)

            payoffs[vertex] = FrozenOrderedMapping(enumerate(vertex.rewards()))
        else:
            try:
                information_set = vertex.information_state_string()
            except SpielError:
                if not vertex.is_chance_node():
                    warn('Using state as information set for rational player.')

                information_set = vertex

            information_sets.append(information_set)

            information_partition[vertex] = information_set
            sub_actions = vertex.legal_actions()

            actions.extend(sub_actions)

            player = vertex.current_player()

            players.append(player)

            player_partition[information_set] = player

            if vertex.is_chance_node():
                nature = player
                nature_probabilities[information_set] = FrozenOrderedMapping(
                    vertex.chance_outcomes(),
                )

            for action in sub_actions:
                child = deepcopy(vertex)

                child.apply_action(action)
                update(child, vertex, action)

    update(root, None, None)

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
        nature,
        FrozenOrderedMapping(player_partition),
        FrozenOrderedMapping(nature_probabilities),
        FrozenOrderedMapping(payoffs),
    )
