""":mod:``gpugt.games.rock_paper_scissors` defines rock paper scissors
and its variants.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import auto, Enum
from functools import cached_property
from itertools import product, repeat
from typing import TypeAlias

from gpugt.collections2 import FrozenOrderedMapping, FrozenOrderedSet
from gpugt.games.finite_extensive_form_game import FiniteExtensiveFormGame
from gpugt.graphs.finite_tree import FiniteTree


class Hand(Enum):
    """An enum of rock paper scissors hands."""

    ROCK = auto()
    """Rock hand."""
    PAPER = auto()
    """Paper hand."""
    SCISSORS = auto()
    """Scissors hand."""

    @cached_property
    def _index(self) -> int:
        return tuple(type(self)).index(self)

    def __lt__(self, value: Hand) -> bool:
        if not isinstance(value, Hand):
            return NotImplemented

        return (self._index + 1) % 3 == value._index


class Player(Enum):
    """An enum of rock paper scissors players."""

    ROW_PLAYER = auto()
    """The row (i.e. first) player."""
    OPPONENT = auto()
    """The opponent (i.e. second player)."""


RockPaperScissors: TypeAlias = (
    FiniteExtensiveFormGame[tuple[Hand, ...], Player, Hand, Player]
)
"""A type alias for rock paper scissors."""


def create_rock_paper_scissors(
        weights: Mapping[Hand, float] = FrozenOrderedMapping(
            zip(Hand, repeat(1)),
        ),
) -> RockPaperScissors:
    """Create a rock paper scissors game.

    :param weights: Payoff weights for each hand.
    :return: A rock paper scissors game.
    """
    vertices = list[tuple[Hand, ...]]()

    for count in range(3):
        vertices.extend(product(Hand, repeat=count))

    leaves = list(product(Hand, repeat=2))
    parents = {vertex: vertex[:-1] for vertex in vertices if vertex}
    information_sets = Player
    information_partition = {
        vertex: tuple(Player)[len(vertex)]
        for vertex in vertices if len(vertex) < 2
    }
    action_partition = {vertex: vertex[-1] for vertex in vertices if vertex}
    payoffs = {}

    for leaf in leaves:
        weight = weights[leaf[0]] * weights[leaf[1]]

        if leaf[0] < leaf[1]:
            payoffs[leaf] = {
                Player.ROW_PLAYER: -weight,
                Player.OPPONENT: weight,
            }
        elif leaf[1] < leaf[0]:
            payoffs[leaf] = {
                Player.ROW_PLAYER: weight,
                Player.OPPONENT: -weight,
            }
        else:
            payoffs[leaf] = {Player.ROW_PLAYER: 0, Player.OPPONENT: 0}

    return FiniteExtensiveFormGame(
        FiniteTree(
            FrozenOrderedSet(vertices),
            (),
            FrozenOrderedSet(leaves),
            FrozenOrderedMapping(parents),
        ),
        FrozenOrderedSet(information_sets),
        FrozenOrderedMapping(information_partition),
        FrozenOrderedSet(Hand),
        FrozenOrderedMapping(action_partition),
        FrozenOrderedSet(Player),
        None,
        FrozenOrderedMapping(zip(Player, Player)),
        FrozenOrderedMapping({}),
        FrozenOrderedMapping(payoffs),
    )


def create_rock_paper_scissors_plus() -> RockPaperScissors:
    """Create a rock paper scissors+ game.

    In rock paper scissors+, when scissors are played, the payoffs are
    doubled from the standard payoffs.

    :return: A rock paper scissors+ game.
    """
    return create_rock_paper_scissors(
        {Hand.ROCK: 1, Hand.PAPER: 1, Hand.SCISSORS: 2},
    )
