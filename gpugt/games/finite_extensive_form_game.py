""":mod:``gpugt.games.finite_extensive_form_game` defines the finite
extensive-form game.
"""

from collections.abc import Hashable
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar

from gpugt.collections2 import FrozenOrderedMapping, FrozenOrderedSet
from gpugt.graphs.finite_tree import FiniteTree

_V = TypeVar('_V', bound=Hashable)
_H = TypeVar('_H', bound=Hashable)
_A = TypeVar('_A', bound=Hashable)
_I = TypeVar('_I', bound=Hashable)


@dataclass(frozen=True)
class FiniteExtensiveFormGame(Generic[_V, _H, _A, _I]):
    """A class for finite extensive-form games.

    :param game_tree: A finite game tree.
    :param information_sets: A finite set of information_sets.
    :param information_partition: An information partition on a set of
                                  decision nodes.
    :param actions: A finite set of actions.
    :param action_partition: An action partition on a set of non-initial
                             nodes.
    :param players: A finite set of players.
    :param nature: The optional nature.
    :param player_partition: A player partition on an information
                             partition.
    :param chance_probabilities: A family of probabilities of chance
                                 actions.
    :param payoffs: Payoff profiles.
    """

    game_tree: FiniteTree[_V]
    """A finite tree on which the rules of the game are represented."""
    information_sets: FrozenOrderedSet[_H]
    """A finite set of information sets."""
    information_partition: FrozenOrderedMapping[_V, _H]
    """An information (value) partition on a set of decision nodes
    (key).
    """
    actions: FrozenOrderedSet[_A]
    """A finite set of actions."""
    action_partition: FrozenOrderedMapping[_V, _A]
    """An action partition associating each non-initial node (key) to a
    single action.

    At any decision node, the associated actions of the immediate
    successor nodes must be a bijection with the actions available at
    the decision node's information set.
    """
    players: FrozenOrderedSet[_I]
    """A finite set of players."""
    nature: _I | None
    """The optional nature player."""
    player_partition: FrozenOrderedMapping[_H, _I]
    """A player partition on the set of information sets."""
    nature_probabilities: (
        FrozenOrderedMapping[_H, FrozenOrderedMapping[_A, float]]
    )
    """A family of probabilities (value) for the chance actions (key).
    """
    payoffs: FrozenOrderedMapping[_V, FrozenOrderedMapping[_I, float]]
    """The payoff profiles of each player (value) at each terminal node
    (key).
    """

    def __post_init__(self) -> None:
        if self.information_partition.keys() != self.decision_nodes:
            raise ValueError('information partition not on decision nodes')
        elif (
                not set(self.information_partition.values())
                <= self.information_sets
        ):
            raise ValueError('undefined infoset in information partition')
        elif self.action_partition.keys() != self.non_initial_nodes:
            raise ValueError('action partition not on non-initial nodes')
        elif not set(self.action_partition.values()) <= self.actions:
            raise ValueError('undefined action in action partition')
        elif self.nature is not None and self.nature not in self.players:
            raise ValueError('nature not member of players')
        elif self.player_partition.keys() != self.information_sets:
            raise ValueError('player partition not on information sets')
        elif not set(self.player_partition.values()) <= self.players:
            raise ValueError('undefined player in player partition')
        elif self.nature_probabilities.keys() != self.nature_information_sets:
            raise ValueError(
                'nature probabilities undefined for nature information sets',
            )
        elif self.payoffs.keys() != self.terminal_nodes:
            raise ValueError('payoffs undefined for terminal nodes')

        for decision_node in self.decision_nodes:
            information_set = self.information_partition[decision_node]
            actions = Counter(self.available_actions[information_set])
            bijection = Counter(
                map(self.action_partition.get, self.successors[decision_node]),
            )

            if actions != bijection:
                raise ValueError('actions not bijection with successors')

        for information_set in self.nature_information_sets:
            if (
                    self.nature_probabilities[information_set].keys()
                    != self.available_actions[information_set]
            ):
                raise ValueError('probabilities undefined for nature actions')

        for payoffs in self.payoffs.values():
            if payoffs.keys() != self.players - {self.nature}:
                raise ValueError('payoffs not defined for (rational) players')

    @cached_property
    def available_actions(
            self,
    ) -> FrozenOrderedMapping[_H, FrozenOrderedSet[_A]]:
        """Return finite sets of available actions (value) at each
        information set (key).

        :return: The available actions.
        """
        actions = defaultdict(list)

        for node in self.non_initial_nodes:
            predecessor = self.predecessors[node]
            information_set = self.information_partition[predecessor]

            actions[information_set].append(self.action_partition[node])

        return FrozenOrderedMapping(
            zip(actions.keys(), map(FrozenOrderedSet, actions.values())),
        )

    @cached_property
    def nature_information_sets(self) -> FrozenOrderedSet[_H]:
        """Return a finite set of information sets associated with the
        nature.

        :return: The nature information sets.
        """
        nature_information_sets = []

        for information_set in self.information_sets:
            if self.player_partition[information_set] == self.nature:
                nature_information_sets.append(information_set)

        return FrozenOrderedSet(nature_information_sets)

    @cached_property
    def nodes(self) -> FrozenOrderedSet[_V]:
        """Return a finite set of nodes.

        :return: A finite set of nodes.
        """
        return self.game_tree.vertices

    @cached_property
    def initial_node(self) -> _V:
        """Return the unique initial node.

        :return: The unique initial node.
        """
        return self.game_tree.root

    @cached_property
    def terminal_nodes(self) -> FrozenOrderedSet[_V]:
        """Return a finite set of terminal nodes.

        :return: A finite set of terminal nodes.
        """
        return self.game_tree.leaves

    @cached_property
    def predecessors(self) -> FrozenOrderedMapping[_V, _V]:
        """Return the immediate predecessors (value) of each non-roots
        (key).

        :return: An immediate predecessor function.
        """
        return self.game_tree.parents

    @cached_property
    def non_initial_nodes(self) -> FrozenOrderedSet[_V]:
        """Return a finite set of non-initial nodes.

        :return: A finite set of non-initial nodes.
        """
        return self.game_tree.non_roots

    @cached_property
    def decision_nodes(self) -> FrozenOrderedSet[_V]:
        """Return a finite set of decision nodes.

        :return: A finite set of decision nodes.
        """
        return self.game_tree.internal_vertices

    @cached_property
    def successors(self) -> FrozenOrderedMapping[_V, FrozenOrderedSet[_V]]:
        """Return the immediate successors (value) of each internal
        nodes (key).

        :return: An immediate successor function.
        """
        return self.game_tree.children
