#!/usr/bin/env python3

import numpy as np

from typing import Iterable
import numpy as np
from matplotlib import pyplot as plt



def compute_deltas(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute players' incentives to deviate from their strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        Each player's incentive to deviate
    """

    row_expected_payoff=row_strategy@row_matrix@col_strategy
    row_best_response_payoff=np.max(row_matrix@col_strategy)
    row_delta=row_best_response_payoff-row_expected_payoff

    col_expected_payoff=row_strategy@col_matrix@col_strategy
    col_best_response_payoff=np.max(row_strategy@col_matrix)
    col_delta=col_best_response_payoff-col_expected_payoff

    return np.asarray([row_delta, col_delta])
    # raise NotImplementedError


def compute_nash_conv(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.float64:
    """Compute the NashConv value of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The NashConv value of the given strategy profile
    """

    return np.sum(compute_deltas(row_matrix, col_matrix, row_strategy, col_strategy))


def compute_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.float64:
    """Compute the exploitability of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The exploitability value of the given strategy profile
    """
    return compute_nash_conv(row_matrix, col_matrix, row_strategy, col_strategy,)/2

    # raise NotImplementedError


def fictitious_play(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int, naive: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Fictitious Play for a given number of iterations.

    Although any averaging method is valid, the reference solution updates the
    average strategy vectors using a moving average. Therefore, it is recommended
    to use the same averaging method to avoid numerical discrepancies during testing.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for
    naive : bool
        Whether to calculate the best response against the last
        opponent's strategy or the average opponent's strategy

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of average strategy profiles produced by the algorithm
    """

    # raise NotImplementedError
    avg_strategies = []

    m, n = row_matrix.shape
    row_strategy = np.ones(m, dtype=np.float64) / m      
    col_strategy = np.ones(n, dtype=np.float64) / n

    freq_row = np.zeros(m, dtype=np.float64)             
    freq_col = np.zeros(n, dtype=np.float64)

    for epoch in range(num_iters):
       
        if epoch == 0:
            prev_avg_row = row_strategy                   
            prev_avg_col = col_strategy
        else:
            prev_avg_row = freq_row / epoch
            prev_avg_col = freq_col / epoch

        if naive:
            
            br_row = int(np.argmax(row_matrix @ col_strategy))
            br_col = int(np.argmax(row_strategy @ col_matrix))
        else:
            
            br_row = int(np.argmax(row_matrix @ prev_avg_col))
            br_col = int(np.argmax(prev_avg_row @ col_matrix))

        row_strategy = np.zeros(m, dtype=np.float64); row_strategy[br_row] = 1.0
        col_strategy = np.zeros(n, dtype=np.float64); col_strategy[br_col] = 1.0

        freq_row += row_strategy
        freq_col += col_strategy

        avg_row_strategy = (freq_row / (epoch + 1)).astype(np.float64, copy=False)
        avg_col_strategy = (freq_col / (epoch + 1)).astype(np.float64, copy=False)
        avg_strategies.append((avg_row_strategy.copy(), avg_col_strategy.copy()))

    return avg_strategies


def plot_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    strategies: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> list[np.float64]:
    """Compute and plot the exploitability of a sequence of strategy profiles.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    strategies : list[tuple[np.ndarray, np.ndarray]]
        The sequence of strategy profiles
    label : str
        The name of the algorithm that produced `strategies`

    Returns
    -------
    list[np.float64]
        A sequence of exploitability values, one for each strategy profile
    """

    # raise NotImplementedError
    exploitabilities = []

    for row_strategy, col_strategy in strategies:
        exploitabilities.append(compute_exploitability(row_matrix, col_matrix, row_strategy, col_strategy))

    plt.plot(exploitabilities, label=label)

    return exploitabilities


def main() -> None:
    pass


if __name__ == '__main__':
    main()
