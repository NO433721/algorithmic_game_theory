import ast
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'templates'))

import week07  # noqa: E402


REFERENCE_PATH = Path(__file__).with_name('ref_fictitious_play_kuhn.txt')
ATOL = 1e-5

CARD_TO_ACTION = {'J': 0, 'Q': 1, 'K': 2}
GAME_ACTION_TO_INDEX = {
    'Bet': 0,
    'Call': 0,
    'Check': 1,
    'Fold': 1,
}

FLOAT = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
UTILITY_RE = re.compile(
    rf'Iter (?P<iteration>\d+): Utility of avg\. strategies: '
    rf'(?P<row>{FLOAT}), (?P<col>{FLOAT})'
)
STRATEGY_RE = re.compile(
    rf'Iter (?P<iteration>\d+): '
    rf'(?P<kind>Avg\. strategy|BR) of P(?P<player>[12])'
    rf"(?: against P[12]'s avg\. strategy)? at "
    rf'(?P<info_set>\(.*\)): '
    rf'(?P<action_0>Bet|Call): (?P<probability_0>{FLOAT}), '
    rf'(?P<action_1>Check|Fold): (?P<probability_1>{FLOAT})'
)


def _empty_iteration() -> dict:
    return {
        'utility': None,
        'average': {0: {}, 1: {}},
        'best_response': {0: {}, 1: {}},
    }


def _convert_info_set(displayed_info_set: tuple[str, ...], player: int) -> tuple:
    """Convert the human-readable reference notation to week07's notation."""
    card = CARD_TO_ACTION[displayed_info_set[player]]
    betting_history = tuple(
        GAME_ACTION_TO_INDEX[action] for action in displayed_info_set[2:]
    )
    return card, betting_history


def _parse_reference() -> dict[int, dict]:
    expected: dict[int, dict] = {}

    for line_number, raw_line in enumerate(
        REFERENCE_PATH.read_text(encoding='utf-8').splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        utility_match = UTILITY_RE.fullmatch(line)
        if utility_match:
            iteration = int(utility_match['iteration'])
            data = expected.setdefault(iteration, _empty_iteration())
            data['utility'] = np.array(
                [float(utility_match['row']), float(utility_match['col'])]
            )
            continue

        strategy_match = STRATEGY_RE.fullmatch(line)
        if strategy_match:
            iteration = int(strategy_match['iteration'])
            player = int(strategy_match['player']) - 1
            displayed_info_set = ast.literal_eval(strategy_match['info_set'])
            info_set = _convert_info_set(displayed_info_set, player)

            strategy = np.zeros(3, dtype=np.float64)
            strategy[GAME_ACTION_TO_INDEX[strategy_match['action_0']]] = float(
                strategy_match['probability_0']
            )
            strategy[GAME_ACTION_TO_INDEX[strategy_match['action_1']]] = float(
                strategy_match['probability_1']
            )

            category = (
                'average'
                if strategy_match['kind'] == 'Avg. strategy'
                else 'best_response'
            )
            data = expected.setdefault(iteration, _empty_iteration())
            data[category][player][info_set] = strategy
            continue

        raise AssertionError(
            f'Unrecognized reference format on line {line_number}: {raw_line!r}'
        )

    return expected


def _uniform_profile(info_sets: dict) -> dict[int, dict]:
    profile: dict[int, dict] = {0: {}, 1: {}}

    for info_set, nodes in info_sets.items():
        node = nodes[0]
        strategy = node.legal_action_mask.astype(np.float64)
        strategy /= strategy.sum()
        profile[node.player][info_set] = strategy

    return profile


def _assert_strategy_matches(
    actual: dict,
    expected: dict,
    *,
    iteration: int,
    player: int,
    category: str,
    atol: float = ATOL,
) -> None:
    assert actual.keys() == expected.keys(), (
        f'Iteration {iteration}, P{player + 1}, {category}: '
        'the information sets differ'
    )

    for info_set, expected_strategy in expected.items():
        np.testing.assert_allclose(
            actual[info_set],
            expected_strategy,
            rtol=0.0,
            atol=atol,
            err_msg=(
                f'Iteration {iteration}, P{player + 1}, {category}, '
                f'information set {info_set}'
            ),
        )


def _best_response_value(root, profile: dict, player: int, best_response: dict) -> float:
    best_response_profile = {0: profile[0], 1: profile[1]}
    best_response_profile[player] = best_response
    return float(week07.evaluate(root, best_response_profile)[player])


def _assert_profile_is_valid(profile: dict, info_sets: dict) -> None:
    for info_set, nodes in info_sets.items():
        node = nodes[0]
        strategy = profile[node.player][info_set]
        legal_actions = node.legal_action_mask.astype(bool)

        assert strategy.shape == node.legal_action_mask.shape
        assert np.all(strategy >= -ATOL)
        np.testing.assert_allclose(strategy.sum(), 1.0, rtol=0.0, atol=ATOL)
        np.testing.assert_allclose(
            strategy[~legal_actions],
            0.0,
            rtol=0.0,
            atol=ATOL,
        )


def test_reference_utilities_and_best_response_values() -> None:
    expected = _parse_reference()
    assert sorted(expected) == list(range(1, 11))

    env = week07.kuhn_poker.KuhnPokerNumpy()
    root, info_sets = week07.traverse_tree(env)

    for iteration, iteration_expected in expected.items():
        profile = iteration_expected['average']
        _assert_profile_is_valid(profile, info_sets)

        np.testing.assert_allclose(
            week07.evaluate(root, profile),
            iteration_expected['utility'],
            rtol=0.0,
            atol=ATOL,
            err_msg=f'Iteration {iteration}: utility',
        )

        for player in (0, 1):
            opponent = 1 - player
            actual_best_response = week07.compute_best_response(
                root,
                player,
                profile[opponent],
                info_sets,
            )

            # Pure best responses need not be unique. Compare their achieved
            # utility instead of requiring the same action at an exact tie.
            actual_value = _best_response_value(
                root, profile, player, actual_best_response
            )
            reference_value = _best_response_value(
                root,
                profile,
                player,
                iteration_expected['best_response'][player],
            )
            np.testing.assert_allclose(
                actual_value,
                reference_value,
                rtol=0.0,
                atol=1e-4,
                err_msg=f'Iteration {iteration}, P{player + 1}: best-response value',
            )


def test_reference_strategy_averaging_transitions() -> None:
    expected = _parse_reference()
    env = week07.kuhn_poker.KuhnPokerNumpy()
    root, _ = week07.traverse_tree(env)

    for iteration in range(1, 10):
        alpha = 1.0 / (iteration + 1)

        for player in (0, 1):
            actual_next_average = week07.compute_average_strategy(
                root,
                expected[iteration]['average'][player],
                expected[iteration]['best_response'][player],
                alpha,
                player,
            )
            _assert_strategy_matches(
                actual_next_average,
                expected[iteration + 1]['average'][player],
                iteration=iteration + 1,
                player=player,
                category='average-strategy transition',
                # The reference inputs and outputs are rounded to five places,
                # so a transition can accumulate slightly more rounding error.
                atol=1e-4,
            )


def test_fictitious_play_matches_reference_until_first_tied_response() -> None:
    expected = _parse_reference()
    env = week07.kuhn_poker.KuhnPokerNumpy()
    root, info_sets = week07.traverse_tree(env)

    initial_profile = _uniform_profile(info_sets)

    # The reference calls the initial uniform profile "Iter 1". The function
    # returns profiles only after an update, so history[0] is reference Iter 2.
    history = week07.fictitious_play(root, info_sets, num_iters=9)
    profiles = [initial_profile, *history]
    assert len(profiles) == 10

    # At reference Iter 8, P2 with Q facing a bet is exactly indifferent
    # between Call and Fold. Different valid tie-breaking choices produce
    # different fictitious-play trajectories from Iter 9 onward.
    for iteration, profile in enumerate(profiles[:8], start=1):
        np.testing.assert_allclose(
            week07.evaluate(root, profile),
            expected[iteration]['utility'],
            rtol=0.0,
            atol=ATOL,
            err_msg=f'Iteration {iteration}: utility',
        )

        for player in (0, 1):
            _assert_strategy_matches(
                profile[player],
                expected[iteration]['average'][player],
                iteration=iteration,
                player=player,
                category='average strategy',
            )

    for profile in profiles:
        _assert_profile_is_valid(profile, info_sets)
