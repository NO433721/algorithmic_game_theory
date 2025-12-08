#!/usr/bin/env python3

import numpy as np


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Generate a strategy based on the given cumulative regrets.

    Parameters
    ----------
    regrets : np.ndarray
        The vector containing cumulative regret of each action

    Returns
    -------
    np.ndarray
        The generated strategy
    """

    regrets = np.asarray(regrets, dtype=float)

    positive_regrets = np.maximum(regrets, 0.0)

    sum_positive = positive_regrets.sum()

    if sum_positive > 0.0:
        strategy = positive_regrets / sum_positive
    else:
        num_actions = regrets.size
        strategy = np.ones(num_actions, dtype=float) / num_actions

    return strategy


def regret_minimization(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Regret Minimization for a given number of iterations.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for


    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of `num_iters` average strategy profiles produced by the algorithm
    """
    num_row_actions = row_matrix.shape[0]
    num_col_actions = col_matrix.shape[1]

    
    row_strategy = np.ones(num_row_actions, dtype=float) / num_row_actions
    col_strategy = np.ones(num_col_actions, dtype=float) / num_col_actions

    
    cumulative_regret_row = np.zeros(num_row_actions, dtype=float)
    cumulative_regret_col = np.zeros(num_col_actions, dtype=float)

  
    avg_row_strategy = np.zeros(num_row_actions, dtype=float)
    avg_col_strategy = np.zeros(num_col_actions, dtype=float)

    avg_strategies: list[tuple[np.ndarray, np.ndarray]] = []

    for t in range(1, num_iters + 1):

        row_payoffs = row_matrix @ col_strategy

        col_payoffs = row_strategy @ col_matrix


        expected_payoff_row = float(np.dot(row_strategy, row_payoffs))
        expected_payoff_col = float(np.dot(col_strategy, col_payoffs))

 
        cumulative_regret_row += row_payoffs - expected_payoff_row
        cumulative_regret_col += col_payoffs - expected_payoff_col


        avg_row_strategy = ((t - 1) * avg_row_strategy + row_strategy) / t
        avg_col_strategy = ((t - 1) * avg_col_strategy + col_strategy) / t

 
        avg_strategies.append((avg_row_strategy.copy(), avg_col_strategy.copy()))


        row_strategy = regret_matching(cumulative_regret_row)
        col_strategy = regret_matching(cumulative_regret_col)

    return avg_strategies


def main() -> None:
    pass


if __name__ == '__main__':
    main()
