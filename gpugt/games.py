from dataclasses import dataclass

from cupyx.scipy.sparse import csr_matrix
from noregret.games import (
    ExtensiveFormGame,
    TwoPlayerExtensiveFormGame,
    TwoPlayerZeroSumExtensiveFormGame,
)

from gpugt.utilities import TreeFormSequentialDecisionProcess


@dataclass
class ExtensiveFormGame(ExtensiveFormGame):
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
        TwoPlayerZeroSumExtensiveFormGame,
):
    @classmethod
    def deserialize(cls, raw_data):
        game = super(TwoPlayerExtensiveFormGame, cls).deserialize(raw_data)
        game.utilities = csr_matrix(game.utilities)

        return game
