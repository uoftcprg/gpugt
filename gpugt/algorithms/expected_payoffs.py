""":mod:``gpugt.games.expected_payoffs` defines expected payoffs
calculation.
"""

from collections.abc import Callable, Hashable
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from operator import getitem
from typing import Generic, TypeVar

from gpugt.games.finite_extensive_form_game import FiniteExtensiveFormGame

_V = TypeVar('_V', bound=Hashable)
_H = TypeVar('_H', bound=Hashable)
_A = TypeVar('_A', bound=Hashable)
_I = TypeVar('_I', bound=Hashable)


@dataclass
class ExpectedPayoffs(Generic[_V, _H, _A, _I]):
    """An implementation of expected payoffs calculation.

    :param game: The finite extensive-form game.
    :param strategy_profile: The strategy profile.
    """

    game: FiniteExtensiveFormGame[_V, _H, _A, _I]
    strategy_profile: Callable[[_V, _A], float]
    expected_payoffs: defaultdict[_V, defaultdict[_I, float]] = field(
        init=False,
        default_factory=partial(defaultdict, partial(defaultdict, float)),
    )

    def __post_init__(self) -> None:
        self._dfs(self.game.initial_node)

    def _dfs(self, node: _V) -> None:
        if node in self.game.terminal_nodes:
            self.expected_payoffs[node].update(self.game.payoffs[node])
        elif node in self.game.decision_nodes:
            successors = tuple(self.game.successors[node])
            actions = tuple(
                map(partial(getitem, self.game.action_partition), successors),
            )

            for successor in successors:
                self._dfs(successor)

            probabilities = tuple(
                map(partial(self.strategy_profile, node), actions),
            )

            for successor, probability in zip(successors, probabilities):
                for player, expected_payoff in (
                        self.expected_payoffs[successor].items()
                ):
                    self.expected_payoffs[node][player] += (
                        probability
                        * expected_payoff
                    )
        else:
            raise AssertionError

    def __call__(self, node: _V, player: _I) -> float:
        return self.expected_payoffs[node][player]
