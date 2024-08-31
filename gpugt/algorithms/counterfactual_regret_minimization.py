""":mod:``gpugt.games.counterfactual_regret_minimization` defines
counterfactual regret minimization.
"""

from collections.abc import Hashable
from dataclasses import dataclass, field
from functools import cached_property
from itertools import count, repeat
from typing import Any, Generic, TypeVar
from warnings import warn

try:
    from cupyx.scipy.sparse import csr_matrix  # type: ignore[import-untyped]
    import cupy as cp  # type: ignore[import-untyped]
except ImportError:
    from scipy.sparse import (  # type: ignore[import-untyped]
        csr_array as csr_matrix,
    )
    import numpy as cp

    warn('CuPy installation not found. GPU-acceleration disabled...')

from scipy.sparse import lil_array

from gpugt.collections2 import FrozenOrderedSet
from gpugt.games.finite_extensive_form_game import FiniteExtensiveFormGame

_V = TypeVar('_V', bound=Hashable)
_H = TypeVar('_H', bound=Hashable)
_A = TypeVar('_A', bound=Hashable)
_I = TypeVar('_I', bound=Hashable)


@dataclass
class CounterfactualRegretMinimization(Generic[_V, _H, _A, _I]):
    """An implementation of counterfactual regret minimization.

    :param game: The finite extensive-form game.
    """

    game: FiniteExtensiveFormGame[_V, _H, _A, _I]
    """The finite extensive-form game to be solved."""

    def __post_init__(self) -> None:
        self._setup()

    def _setup(self) -> None:
        self._setup_indices()
        self._setup_tensors()
        self._setup_iteration()

    _nodes: dict[_V, int] = field(default_factory=dict, init=False)
    _information_sets: dict[_H, int] = field(default_factory=dict, init=False)
    _actions: dict[tuple[_H, _A], int] = field(
        default_factory=dict,
        init=False,
    )
    _players: dict[_I, int] = field(default_factory=dict, init=False)

    @cached_property
    def nodes(self) -> FrozenOrderedSet[_V]:
        """Return the finite set of nodes.

        :return: The finite set of nodes.
        """
        return FrozenOrderedSet(self._nodes)

    @cached_property
    def information_sets(self) -> FrozenOrderedSet[_H]:
        """Return the finite set of information sets.

        :return: The finite set of information sets.
        """
        return FrozenOrderedSet(self._information_sets)

    @cached_property
    def actions(self) -> FrozenOrderedSet[tuple[_H, _A]]:
        """Return the finite set of actions.

        :return: The finite set of actions.
        """
        return FrozenOrderedSet(self._actions)

    @cached_property
    def players(self) -> FrozenOrderedSet[_I]:
        """Return the finite set of players.

        :return: The finite set of players.
        """
        return FrozenOrderedSet(self._players)

    def _setup_indices(self) -> None:
        self._nodes.update(zip(self.game.nodes, count()))
        self._information_sets.update(
            zip(
                self.game.information_sets - self.game.nature_information_sets,
                count(),
            ),
        )
        indices = count()

        for information_set in self.information_sets:
            actions = self.game.available_actions[information_set]

            self._actions.update(
                zip(zip(repeat(information_set), actions), indices),
            )

        self._players.update(
            zip(self.game.players - {self.game.nature}, count()),
        )

    _graph: Any = field(init=False)
    _level_graphs: list[Any] = field(init=False)
    _action_node_mask: Any = field(init=False)
    _information_set_action_mask: Any = field(init=False)
    _node_player_mask: Any = field(init=False)
    _nature_strategies: Any = field(init=False)
    _strategy_profile: Any = field(init=False)
    _initial_strategy_profile: Any = field(init=False)

    def _setup_tensors(self) -> None:
        self._graph = lil_array((len(self.nodes), len(self.nodes)))
        self._level_graphs = []
        level_nodes = {self.game.initial_node}

        while level_nodes:
            level_graph = lil_array((len(self.nodes), len(self.nodes)))
            next_level_nodes = set[_V]()

            for node in level_nodes:
                successors = self.game.successors.get(node, ())
                v = self._nodes[node]
                vv = list(map(self._nodes.get, successors))
                self._graph[v, vv] = 1
                level_graph[v, vv] = 1

                next_level_nodes.update(successors)

            self._level_graphs.append(csr_matrix(level_graph))

            level_nodes = next_level_nodes

        self._graph = csr_matrix(self._graph)

        assert not self._level_graphs[-1].count_nonzero()

        self._level_graphs.pop()

        self._action_node_mask = lil_array(
            (len(self.actions), len(self.nodes)),
        )
        self._information_set_action_mask = lil_array(
            (len(self.information_sets), len(self.actions)),
        )
        self._node_player_mask = cp.zeros(
            (len(self.nodes), len(self.players)),
            cp.bool_,
        )

        for node in self.nodes:
            if node not in self.game.predecessors:
                continue

            predecessor = self.game.predecessors[node]
            information_set = self.game.information_partition[predecessor]

            if information_set not in self.information_sets:
                continue

            action = self.game.action_partition[node]
            player = self.game.player_partition[information_set]

            v = self._nodes[node]
            h = self._information_sets[information_set]
            a = self._actions[information_set, action]
            i = self._players[player]

            self._action_node_mask[a, v] = 1
            self._information_set_action_mask[h, a] = 1
            self._node_player_mask[v, i] = True

        self._action_node_mask = csr_matrix(self._action_node_mask)
        self._information_set_action_mask = csr_matrix(
            self._information_set_action_mask,
        )
        self._nature_strategies = cp.zeros(len(self.nodes))

        for node in self.nodes:
            if (
                    node in self.game.information_partition
                    and (
                        self.game.information_partition[node]
                        in self.game.nature_information_sets
                    )
            ):
                for successor in self.game.successors[node]:
                    v = self._nodes[successor]
                    information_set = self.game.information_partition[node]
                    action = self.game.action_partition[successor]
                    self._nature_strategies[v] = (
                        self.game.nature_probabilities[information_set][action]
                    )

        self._strategy_profile = cp.reciprocal(
            (
                self._information_set_action_mask.T
                @ self._information_set_action_mask.sum(1)
            ).ravel(),
        )
        self._default_strategy_profile = self._strategy_profile.copy()

    _iteration_count: int = field(default=0, init=False)

    @property
    def iteration_count(self) -> int:
        """Return the number of iterations.

        :return: The number of iterations.
        """
        return self._iteration_count

    def _setup_iteration(self) -> None:
        self._setup_expected_payoffs()
        self._setup_excepted_reach_probabilities()
        self._setup_average_strategy_profile()
        self._setup_next_strategy_profile()

    def iterate(self) -> None:
        self._calculate_strategies()
        self._calculate_expected_payoffs()
        self._calculate_excepted_reach_probabilities()
        self._calculate_counterfactual_reach_probability_terms()
        self._calculate_average_strategy_profile()
        self._calculate_next_strategy_profile()

        self._iteration_count += 1

    _strategies: Any = field(init=False)

    def _calculate_strategies(self) -> None:
        self._strategies = (
            (self._action_node_mask.T @ self._strategy_profile).ravel()
            + self._nature_strategies
        )

    _initial_expected_payoffs: Any = field(init=False)
    _expected_payoffs: Any = field(init=False)

    def _setup_expected_payoffs(self) -> None:
        self._initial_expected_payoffs = cp.zeros(
            (len(self.nodes), len(self.players)),
        )

        for node, payoffs in self.game.payoffs.items():
            for player, payoff in payoffs.items():
                v = self._nodes[node]
                i = self._players[player]
                self._initial_expected_payoffs[v, i] = payoff

    def _calculate_expected_payoffs(self) -> None:
        self._expected_payoffs = self._initial_expected_payoffs.copy()

        for level_graph in reversed(self._level_graphs):
            self._expected_payoffs += (
                level_graph.multiply(self._strategies)
                @ self._expected_payoffs
            )

    _excepted_strategies: Any = field(init=False)
    _initial_excepted_reach_probabilities: Any = field(init=False)
    _excepted_reach_probabilities: Any = field(init=False)

    def _setup_excepted_reach_probabilities(self) -> None:
        self._initial_excepted_reach_probabilities = cp.zeros(
            (len(self.nodes), len(self.players)),
        )
        v = self._nodes[self.game.initial_node]
        self._initial_excepted_reach_probabilities[v] = 1

    def _calculate_excepted_reach_probabilities(self) -> None:
        self._excepted_strategies = cp.broadcast_to(
            self._strategies,
            (len(self.players), len(self.nodes)),
        ).T.copy()
        self._excepted_strategies[self._node_player_mask] = 1
        self._excepted_reach_probabilities = (
            self._initial_excepted_reach_probabilities.copy()
        )

        for level_graph in self._level_graphs:
            self._excepted_reach_probabilities += (
                (level_graph.T @ self._excepted_reach_probabilities)
                * self._excepted_strategies
            )

    _counterfactual_reach_probability_terms: Any = field(init=False)

    def _calculate_counterfactual_reach_probability_terms(self) -> None:
        self._counterfactual_reach_probability_terms = (
            self._node_player_mask
            * self._excepted_reach_probabilities
        ).sum(1)

    _information_set_node_mask: Any = field(init=False)
    _counterfactual_reach_probabilities: Any = field(init=False)
    _counterfactual_reach_probability_sums: Any = field(init=False)
    _average_strategy_profile: Any = field(init=False)

    def get_average_strategy(self, information_set: _H, action: _A) -> Any:
        """Return the average strategy for an action.

        :param information_set: The information set at which an action
                                is queried.
        :param action: The action to query.
        :return: The average strategy.
        """
        a = self._actions[information_set, action]

        return self._average_strategy_profile[a]

    def _setup_average_strategy_profile(self) -> None:
        self._information_set_node_mask = (
            self._information_set_action_mask
            @ self._action_node_mask
        )
        self._information_set_action_counts = (
            self._information_set_action_mask.sum(1).ravel()
        )
        self._counterfactual_reach_probability_sums = cp.zeros(
            len(self.information_sets),
        )
        self._average_strategy_profile = cp.zeros(len(self.actions))

    def _calculate_average_strategy_profile(self) -> None:
        self._counterfactual_reach_probabilities = (
            (
                self._information_set_node_mask
                @ self._counterfactual_reach_probability_terms
            )
            / self._information_set_action_counts
        )
        self._counterfactual_reach_probability_sums += (
            self._counterfactual_reach_probabilities
        )
        self._average_strategy_profile += (
            (
                self._information_set_action_mask.T
                @ (
                    self._counterfactual_reach_probabilities
                    / self._counterfactual_reach_probability_sums
                )
            )
            * (self._strategy_profile - self._average_strategy_profile)
        )

    _regrets: Any = field(init=False)
    _instantaneous_counterfactual_regrets: Any = field(init=False)
    _average_counterfactual_regrets: Any = field(init=False)
    _clipped_average_counterfactual_regrets: Any = field(init=False)

    def _setup_next_strategy_profile(self) -> None:
        self._average_counterfactual_regrets = cp.zeros(len(self.actions))

    def _calculate_next_strategy_profile(self) -> None:
        self._regrets = (
            self._node_player_mask
            * (self._expected_payoffs - self._graph.T @ self._expected_payoffs)
        ).sum(1)
        self._instantaneous_counterfactual_regrets = (
            self._action_node_mask
            @ (self._counterfactual_reach_probability_terms * self._regrets)
        )
        self._average_counterfactual_regrets += (
            (
                self._instantaneous_counterfactual_regrets
                - self._average_counterfactual_regrets
            )
            / (self.iteration_count + 1)
        )
        self._clipped_average_counterfactual_regrets = (
            self._average_counterfactual_regrets.clip(0)
        )
        self._strategy_profile = (
            self._clipped_average_counterfactual_regrets
            / (
                self._information_set_action_mask.T
                @ (
                    self._information_set_action_mask
                    @ self._clipped_average_counterfactual_regrets
                )
            )
        )
        self._strategy_profile = cp.where(
            cp.isnan(self._strategy_profile),
            self._default_strategy_profile,
            self._strategy_profile,
        )
