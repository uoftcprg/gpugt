""":mod:``gpugt.games.pokerkit` defines games constructed using
PokerKit.
"""

from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from itertools import chain, combinations, repeat
from typing import Any, TypeAlias

try:
    from pokerkit import (
        Automation,
        BettingStructure,
        Deck,
        HoleDealing,
        KuhnPokerHand,
        Opening,
        Operation,
        State,
        Street,
    )
except ImportError:
    raise ImportError('PokerKit not installed. Please install through pip.')

from gpugt.collections2 import FrozenOrderedMapping, FrozenOrderedSet
from gpugt.games.finite_extensive_form_game import FiniteExtensiveFormGame
from gpugt.graphs.finite_tree import FiniteTree


_V = tuple[Operation, ...]
_H = tuple[Operation | None, ...]
_A = Operation
_I = str
Poker: TypeAlias = FiniteExtensiveFormGame[_V, _H, _A, _I]
"""A type alias for poker games."""


def create_pokerkit(
        initial_state: State,
        players: Sequence[str],
        amount_factory: Callable[[State], Iterable[int | None]],
) -> Poker:
    """Create a poker game from PokerKit.

    :param initial_state: The initial poker state.
    :param players: The player labels.
    :param amount_factory: A function that generates
                           completion/bet/raise amounts.
    :return: a poker game.
    """
    vertices = []
    root = ()
    leaves = []
    parents = {}
    information_sets = []
    information_partition = {}
    actions = list[_A]()
    action_partition = {}
    nature = 'Dealer'
    player_partition = {}
    nature_probabilities = dict[_H, FrozenOrderedMapping[_A, float]]()
    payoffs = dict[_V, FrozenOrderedMapping[_I, float]]()

    def update(state: State, vertex: _V) -> None:
        vertices.append(vertex)

        if vertex:
            parents[vertex] = vertex[:-1]
            action = vertex[-1]

            actions.append(action)

            action_partition[vertex] = action

        information_set: _H | None

        if not state.status:
            leaves.append(vertex)

            payoffs[vertex] = FrozenOrderedMapping(zip(players, state.payoffs))
            player = None
            information_set = None
        elif state.actor_index is None:
            player = nature
            information_set = vertex
        else:
            player = players[state.actor_index]
            information_set = tuple(
                None
                if (
                    isinstance(operation, HoleDealing)
                    and operation.player_index != state.actor_index
                )
                else operation
                for operation in vertex
            )

        if player is not None:
            assert information_set is not None

            player_partition[information_set] = player

        if information_set is not None:
            information_sets.append(information_set)

            information_partition[vertex] = information_set

        if state.can_burn_card():
            state.burn_card('??')

        sub_actions = []

        def act(method: Any, *args: Any, **kwargs: Any) -> None:
            next_state = deepcopy(state)
            sub_action = method(next_state, *args, **kwargs)
            next_vertex = vertex + (sub_action,)

            sub_actions.append(sub_action)
            update(next_state, next_vertex)

        if state.actor_index is not None:
            if state.can_fold():
                act(State.fold)

            if state.can_check_or_call():
                act(State.check_or_call)

            if state.can_complete_bet_or_raise_to():
                for amount in amount_factory(state):
                    act(State.complete_bet_or_raise_to, amount)
        elif state.can_deal_hole():
            assert state.hole_dealee_index is not None

            for cards in combinations(
                    state.deck_cards,
                    len(state.hole_dealing_statuses[state.hole_dealee_index]),
            ):
                act(State.deal_hole, cards)
        elif state.status:
            raise NotImplementedError

        if information_set is not None:
            if player == nature:
                nature_probabilities[information_set] = FrozenOrderedMapping(
                    zip(sub_actions, repeat(1 / len(sub_actions))),
                )

    update(initial_state, root)

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
        FrozenOrderedSet(chain((nature,), players)),
        nature,
        FrozenOrderedMapping(player_partition),
        FrozenOrderedMapping(nature_probabilities),
        FrozenOrderedMapping(payoffs),
    )


def create_kuhn_poker() -> Poker:
    """Create a Kuhn poker game.

    :return: A Kuhn poker game.
    """
    return create_pokerkit(
        State(
            (
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
            ),
            Deck.KUHN_POKER,
            (KuhnPokerHand,),
            (
                Street(
                    False,
                    (False,),
                    0,
                    False,
                    Opening.POSITION,
                    1,
                    None,
                ),
            ),
            BettingStructure.FIXED_LIMIT,
            True,
            (1,) * 2,
            (0,) * 2,
            0,
            (2,) * 2,
            2,
        ),
        ('OOP', 'IP'),
        lambda state: (None,),
    )
