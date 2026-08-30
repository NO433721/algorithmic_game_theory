import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'templates'))

import kuhn_poker  # noqa: E402
import week04  # noqa: E402
import week07  # noqa: E402
import week08  # noqa: E402


@pytest.fixture(scope='module')
def kuhn_game():
    return week07.traverse_tree(kuhn_poker.KuhnPokerNumpy())


def test_convert_kuhn_to_normal_form(kuhn_game) -> None:
    root, _ = kuhn_game
    row_matrix, col_matrix = week08.convert_to_normal_form(root)

    assert row_matrix.shape == (64, 64)
    assert col_matrix.shape == (64, 64)
    assert row_matrix.dtype == np.float64
    assert col_matrix.dtype == np.float64
    np.testing.assert_allclose(col_matrix, -row_matrix)

    row_strategy, col_strategy = week04.find_nash_equilibrium(row_matrix)
    value = row_strategy @ row_matrix @ col_strategy
    np.testing.assert_allclose(value, -1 / 18, atol=1e-8)


def test_convert_kuhn_to_sequence_form(kuhn_game) -> None:
    root, _ = kuhn_game
    row_matrix, col_matrix, row_constraints, row_target, col_constraints, col_target = (
        week08.convert_to_sequence_form(root)
    )

    assert row_matrix.shape == (13, 13)
    assert col_matrix.shape == (13, 13)
    assert row_constraints.shape == (7, 13)
    assert col_constraints.shape == (7, 13)
    assert row_target.shape == (7,)
    assert col_target.shape == (7,)
    np.testing.assert_allclose(col_matrix, -row_matrix)
    np.testing.assert_array_equal(row_target, [1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(col_target, [1, 0, 0, 0, 0, 0, 0])


def test_sequence_form_equilibrium_and_behavioral_conversion(kuhn_game) -> None:
    root, info_sets = kuhn_game
    row_matrix, _, row_constraints, row_target, col_constraints, col_target = (
        week08.convert_to_sequence_form(root)
    )
    row_plan, col_plan = week08.find_nash_equilibrium_sequence_form(root)

    assert row_plan.shape == (13,)
    assert col_plan.shape == (13,)
    assert np.all(row_plan >= -1e-10)
    assert np.all(col_plan >= -1e-10)
    np.testing.assert_allclose(row_constraints @ row_plan, row_target, atol=1e-9)
    np.testing.assert_allclose(col_constraints @ col_plan, col_target, atol=1e-9)
    np.testing.assert_allclose(row_plan @ row_matrix @ col_plan, -1 / 18, atol=1e-8)

    profile = {
        0: week08.convert_realization_plan_to_behavioral_strategy(root, row_plan, 0),
        1: week08.convert_realization_plan_to_behavioral_strategy(root, col_plan, 1),
    }
    utilities = week07.evaluate(root, profile)
    np.testing.assert_allclose(utilities, [-1 / 18, 1 / 18], atol=1e-8)

    exploitability = week07.compute_exploitability(
        root, info_sets, [profile], plot=False
    )[0]
    np.testing.assert_allclose(exploitability, 0.0, atol=1e-8)
