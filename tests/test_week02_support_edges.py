import numpy as np
import pytest

import week02


@pytest.mark.parametrize("swap_players", [False, True])
def test_pure_mixed_equilibrium(swap_players):
    # Row 0 is strictly dominant; the column player can mix between tied actions.
    row_matrix = np.array([[1., 1.], [0., 0.]])
    col_matrix = np.zeros((2, 2))
    expected_row = np.array([1., 0.])
    expected_col = np.array([0.5, 0.5])
    if swap_players:
        row_matrix, col_matrix = col_matrix.T, row_matrix.T
        expected_row, expected_col = expected_col, expected_row

    equilibria = week02.support_enumeration(row_matrix, col_matrix)
    assert any(
        np.allclose(p, expected_row) and np.allclose(q, expected_col)
        for p, q in equilibria
    )
    for p, q in equilibria:
        assert np.max(row_matrix @ q) <= p @ row_matrix @ q + 1e-7
        assert np.max(p @ col_matrix) <= p @ col_matrix @ q + 1e-7


def test_all_tied_pure_equilibria():
    equilibria = week02.support_enumeration(np.zeros((2, 2)), np.zeros((2, 2)))
    for p in np.eye(2):
        for q in np.eye(2):
            assert any(
                np.allclose(actual_p, p) and np.allclose(actual_q, q)
                for actual_p, actual_q in equilibria
            )


def test_pure_mixed_equilibrium_respects_excluded_actions():
    # A boundary column mixture makes an excluded row better than row 0.
    row_matrix = np.array([[0., 0.], [1., -2.], [-2., 1.]])
    col_matrix = np.array([[0., 0.], [1., 0.], [0., 1.]])
    equilibria = week02.support_enumeration(row_matrix, col_matrix)
    candidates = [(p, q) for p, q in equilibria if np.allclose(p, [1., 0., 0.])]
    assert candidates
    for p, q in candidates:
        assert np.all(q > 0)
        assert np.max(row_matrix @ q) <= p @ row_matrix @ q + 1e-7
        assert np.max(p @ col_matrix) <= p @ col_matrix @ q + 1e-7
