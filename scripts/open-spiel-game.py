from collections import defaultdict
from json import dump
from sys import argv, stdout
from warnings import warn

from ordered_set import OrderedSet
from pyspiel import GameType, load_game, SpielError

GAME_NAME = argv[1]


def main():
    game = load_game(GAME_NAME)
    player_count = game.num_players()

    if player_count != 2:
        raise ValueError('not a 2-player game')

    if game.get_type().utility != GameType.Utility.ZERO_SUM:
        raise ValueError('not a 0-sum game')

    state_count = 0
    children = [{(): OrderedSet()} for _ in range(player_count)]
    utilities = defaultdict(int)

    def dfs(state, chance_probability, sequences):
        nonlocal state_count

        state_count += 1

        if state.is_terminal():
            utilities[tuple(sequences)] += (
                chance_probability * state.rewards()[0]
            )
        elif state.is_chance_node():
            for action, probability in state.chance_outcomes():
                child = state.child(action)

                dfs(child, probability * chance_probability, sequences)
        else:
            try:
                infoset = state.information_state_string()
            except SpielError:
                warn('State as information set for rational player.')

                infoset = str(state)

            player = state.current_player()
            parent_sequence = sequences[player]

            children[player][parent_sequence].add(infoset)

            for action in state.legal_actions():
                child = state.child(action)
                action = state.action_to_string(action)
                sequence = infoset, action
                child_sequences = sequences.copy()
                child_sequences[player] = sequence

                children[player].setdefault(sequence, OrderedSet())
                dfs(child, chance_probability, child_sequences)

    dfs(game.new_initial_state(), 1, [()] * player_count)

    raw_tfsdps = [[] for _ in range(player_count)]
    observation_point_count = 0

    for player, raw_tfsdp in enumerate(raw_tfsdps):
        for sequence, infosets in children[player].items():
            if not infosets:
                raw_tfsdp.append(
                    {
                        'parent_edge': sequence,
                        'node': {
                            'id': '',
                            'type': 'END_OF_THE_DECISION_PROCESS',
                        },
                    },
                )
            elif len(infosets) == 1:
                raw_tfsdp.append(
                    {
                        'parent_edge': sequence,
                        'node': {'id': infosets[0], 'type': 'DECISION_POINT'},
                    },
                )
            else:
                i = observation_point_count
                observation_point_count += 1

                raw_tfsdp.append(
                    {
                        'parent_edge': sequence,
                        'node': {'id': f'o{i}', 'type': 'OBSERVATION_POINT'},
                    },
                )

                for j, infoset in enumerate(infosets):
                    raw_tfsdp.append(
                        {
                            'parent_edge': (f'o{i}', f'e{j}'),
                            'node': {'id': infoset, 'type': 'DECISION_POINT'},
                        },
                    )

    raw_utilities = [
        {'sequences': sequences, 'value': value}
        for sequences, value in utilities.items()
    ]

    raw_game = {
        'game_name': GAME_NAME,
        'player_count': player_count,
        'state_count': state_count,
        'tree_form_sequential_decision_processes': raw_tfsdps,
        'utilities': raw_utilities,
    }

    dump(raw_game, stdout)


if __name__ == '__main__':
    main()
