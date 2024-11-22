""":mod:``gpugt.games.best_response` defines best response."""

from collections.abc import Callable, Hashable
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import starmap
from math import isclose
from operator import mul
from typing import Generic, TypeVar

from gpugt.functools2 import cached_method
from gpugt.games.finite_extensive_form_game import FiniteExtensiveFormGame

_V = TypeVar('_V', bound=Hashable)
_H = TypeVar('_H', bound=Hashable)
_A = TypeVar('_A', bound=Hashable)
_I = TypeVar('_I', bound=Hashable)


@dataclass
class BestResponse(Generic[_V, _H, _A, _I]):
    """An implementation of best response.

    :param game: The finite extensive-form game.
    :param strategy_profile: The strategy profile.
    :param player: The best responder.
    """

    game: FiniteExtensiveFormGame[_V, _H, _A, _I]
    strategy_profile: Callable[[_V, _A], float]
    player: _I
    nodes: defaultdict[_H, list[_V]] = field(
        init=False,
        default_factory=partial(defaultdict, list),
    )
    successors: dict[tuple[_V, _A], _V] = field(
        init=False,
        default_factory=dict,
    )
    counterfactual_reach_probabilities: dict[_V, float] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        for node in self.game.nodes:
            if node in self.game.decision_nodes:
                information_set = self.game.information_partition[node]

                self.nodes[information_set].append(node)

            if node in self.game.non_initial_nodes:
                predecessor = self.game.predecessors[node]
                action = self.game.action_partition[node]
                self.successors[predecessor, action] = node

        self._dfs(self.game.initial_node)

    def _dfs(
            self,
            node: _V,
            counterfactual_reach_probability: float = 1,
    ) -> None:
        self.counterfactual_reach_probabilities[node] = (
            counterfactual_reach_probability
        )

        if node in self.game.terminal_nodes:
            pass
        elif node in self.game.decision_nodes:
            information_set = self.game.information_partition[node]
            player = self.game.player_partition[information_set]
            successors = tuple(self.game.successors[node])

            if player == self.player:
                probabilities = [1] * len(successors)
            else:
                actions = tuple(
                    map(self.game.action_partition.__getitem__, successors),
                )
                probabilities = list(
                    map(
                        partial(  # type: ignore[arg-type]
                            self.strategy_profile,
                            node,
                        ),
                        actions,
                    ),
                )

            for successor, probability in zip(
                    self.game.successors[node],
                    probabilities,
            ):
                self._dfs(
                    successor,
                    counterfactual_reach_probability * probability,
                )
        else:
            raise AssertionError

    def _verify_information_set(self, information_set: _H) -> None:
        player = self.game.player_partition[information_set]

        if player != self.player:
            raise ValueError('actor is not the best responder')

    @cached_method
    def get_best_action(self, information_set: _H) -> _A:
        self._verify_information_set(information_set)

        actions = tuple(self.game.available_actions[information_set])
        expected_payoffs = (
            [0.0]
            * len(self.game.available_actions[information_set])
        )

        for node in self.nodes[information_set]:
            weight = self.counterfactual_reach_probabilities[node]

            for i, action in enumerate(actions):
                expected_payoffs[i] += (
                    weight
                    * self.get_expected_payoff(self.successors[node, action])
                )

        return actions[expected_payoffs.index(max(expected_payoffs))]

    @cached_method
    def get_action_probability(self, information_set: _H, action: _A) -> float:
        self._verify_information_set(information_set)

        return action == self.get_best_action(information_set)

    @cached_method
    def get_expected_payoff(self, node: _V) -> float:
        if node in self.game.terminal_nodes:
            expected_payoff = self.game.payoffs[node][self.player]
        elif node in self.game.decision_nodes:
            successors = tuple(self.game.successors[node])
            expected_payoffs = tuple(map(self.get_expected_payoff, successors))
            information_set = self.game.information_partition[node]
            player = self.game.player_partition[information_set]
            actions = tuple(
                map(self.game.action_partition.__getitem__, successors),
            )

            if player == self.player:
                probabilities = list(
                    map(
                        partial(self.get_action_probability, information_set),
                        actions,
                    ),
                )
            else:
                probabilities = list(
                    map(partial(self.strategy_profile, node), actions),
                )

            assert isclose(sum(probabilities), 1)

            expected_payoff = sum(
                starmap(mul, zip(expected_payoffs, probabilities)),
            )
        else:
            raise AssertionError

        return expected_payoff
