from abc import ABC
from dataclasses import dataclass
from functools import partial

from cupyx.scipy.sparse import csr_matrix
from noregret.games import (
    ExtensiveFormGame,
    Game,
    TwoPlayerExtensiveFormGame,
    TwoPlayerZeroSumExtensiveFormGame,
    TwoPlayerZeroSumGame,
)
import cupy as cp

from gpugt.utilities import TreeFormSequentialDecisionProcess


@dataclass
class Game(Game, ABC):
    @property
    def dimensions(self):
        return cp.array(list(map(self.dimension, range(self.player_count))))

    def values(self, *strategies):
        return cp.array(
            [self.value(i, *strategies) for i in range(self.player_count)],
        )

    def correlated_values(self, *strategies):
        return cp.array(
            [
                self.correlated_value(i, *strategies)
                for i in range(self.player_count)
            ],
        )

    def cce_gap(self, *strategies):
        average_strategies = list(map(partial(cp.mean, axis=0), strategies))
        gap = 0

        for i, value in enumerate(self.correlated_values(*strategies)):
            average_opponent_strategies = (
                average_strategies[:i] + average_strategies[i + 1:]
            )
            _, best_response_value = self.best_response(
                i,
                *average_opponent_strategies,
            )
            gap += best_response_value - value

        return gap


@dataclass
class TwoPlayerZeroSumGame(Game, TwoPlayerZeroSumGame, ABC):
    def values(self, row_strategy, column_strategy):
        value = self.row_value(row_strategy, column_strategy)

        return cp.array((value, -value))

    def correlated_values(self, row_strategies, column_strategies):
        value = self.correlated_row_value(row_strategies, column_strategies)

        return cp.array((value, -value))


@dataclass
class ExtensiveFormGame(Game, ExtensiveFormGame):
    @classmethod
    def deserialize(cls, raw_data):
        game = super().deserialize(raw_data)
        game.tree_form_sequential_decision_processes = list(
            map(
                TreeFormSequentialDecisionProcess,
                game.tree_form_sequential_decision_processes,
            ),
        )

        return game


@dataclass
class TwoPlayerExtensiveFormGame(
        ExtensiveFormGame,
        TwoPlayerExtensiveFormGame,
):
    @classmethod
    def deserialize(cls, raw_data):
        game = super().deserialize(raw_data)
        game.utilities = tuple(map(csr_matrix, game.utilities))

        return game


@dataclass
class TwoPlayerZeroSumExtensiveFormGame(
        TwoPlayerExtensiveFormGame,
        TwoPlayerZeroSumGame,
        TwoPlayerZeroSumExtensiveFormGame,
):
    @classmethod
    def deserialize(cls, raw_data):
        game = super(TwoPlayerExtensiveFormGame, cls).deserialize(raw_data)
        game.utilities = csr_matrix(game.utilities)

        return game
