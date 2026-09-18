#!/usr/bin/env python3

import numpy as np

from itertools import combinations
from typing import Iterable
from matplotlib import pyplot as plt
from scipy.optimize import linprog


def plot_best_response_value_function(row_matrix: np.ndarray, step_size: float) -> None:
    """Plot the best response value function for the row player in a 2xN zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    step_size : float
        The step size for the probability of the first action of the row player
    """

    probability = np.linspace(0, 1, int(1 / step_size) + 1)
    best_response_values = []
    for prob in probability:
        mixed_strategy = np.array([prob, 1 - prob])

        expected_payoffs = mixed_strategy @ row_matrix

        best_response = np.min(expected_payoffs)
        best_response_values.append(best_response)

    plt.figure(figsize=(8, 5))
    plt.scatter(probability, best_response_values, label="Best Response Value", color="cyan", edgecolor="k", s=50)
    plt.xlabel("Probability of First Action of Row player", fontsize=12)
    plt.ylabel("Utility of the Row Player", fontsize=12)
    plt.title("Best Response Values for Row Player", fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()


def verify_support(
    matrix: np.ndarray, row_support: np.ndarray, col_support: np.ndarray, check_best_response: bool = False
) -> np.ndarray | None:
    """Construct a system of linear equations to check whether there
    exists a candidate for a Nash equilibrium for the given supports.

    The reference implementation uses `scipy.optimize.linprog`
    with the default solver -- 'highs'. You can find more information at
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players
    row_support : np.ndarray
        The row player's support
    col_support : np.ndarray
        The column player's support

    Returns
    -------
    np.ndarray | None
        The opponent's strategy, if it exists, otherwise `None`
    """
   
    row_support=np.asarray(row_support)
    col_support=np.asarray(col_support)

    sub_matrix = matrix[np.ix_(row_support, col_support)]

    sub_matrix=np.hstack((sub_matrix, -np.ones(sub_matrix.shape[0]).reshape(-1,1)))
    sub_matrix=np.vstack((sub_matrix, np.ones(sub_matrix.shape[1])))

    sub_matrix[-1, -1]=0


    A_eq=sub_matrix
    b_eq=np.zeros(sub_matrix.shape[0])
    b_eq[-1]=1

    c = np.zeros(A_eq.shape[1])

    bounds = [(0, None) for _ in range(A_eq.shape[1])]
    bounds[-1]=(None, None)

    A_ub = None
    b_ub = None

    if check_best_response:
        num_probs = len(col_support)
        num_actions = matrix.shape[0]

        A_eq = np.hstack((A_eq, np.zeros((A_eq.shape[0], 1)),))

        c = np.zeros(num_probs + 2)
        c[-1] = -1.0

        bounds.append((0, None))

        A_ub = np.hstack((
            matrix[:, col_support],
            -np.ones((num_actions, 1)),
            np.zeros((num_actions, 1)),
        ))

        prob_constraints = np.zeros((num_probs, num_probs + 2))

        for j in range(num_probs):
            prob_constraints[j, j] = -1.0
            prob_constraints[j, -1] = 1.0

        A_ub = np.vstack((A_ub, prob_constraints))
        b_ub = np.zeros(num_actions + num_probs)
        
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                     bounds=bounds, method="highs")

    if result.success and result.x.size == A_eq.shape[1]:
        if check_best_response and result.x[-1] <= 1e-9:
            return None
        q = result.x[:len(col_support)].astype(np.float64)                 

        return q
    else:
        return None

def support_enumeration(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run the Support Enumeration algorithm and return a list of all Nash equilibria

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of strategy profiles corresponding to found Nash equilibria
    """

    num_row_strategies, num_col_strategies = row_matrix.shape
    nash_equilibria = []

    for row_support_size in range(1, num_row_strategies + 1):
        for col_support_size in range(1, num_col_strategies + 1):
            for row_support in combinations(range(num_row_strategies), row_support_size):
                for col_support in combinations(range(num_col_strategies), col_support_size):

                    
                    if len(row_support) == 1 and len(col_support) == 1:

                        i = row_support[0]  
                        j = col_support[0]  

                        if col_matrix[i, j] < np.max(col_matrix[i, :]):
                            continue

                        if row_matrix[i, j] < np.max(row_matrix[:, j]):
                            continue

                        
                        row_strategy = np.zeros(row_matrix.shape[0])
                        row_strategy[i] = 1.0

                        col_strategy = np.zeros(col_matrix.shape[1])
                        col_strategy[j] = 1.0

                        nash_equilibria.append((row_strategy, col_strategy))

                    else:
                       
                        row_probs_on_sup = verify_support(col_matrix.T, col_support, row_support, True)  # row player's probs on row_support
                        col_probs_on_sup = verify_support(row_matrix,    row_support, col_support, True)  # col player's probs on col_support

                        if row_probs_on_sup is not None and col_probs_on_sup is not None:
                           
                            row_strategy = np.zeros(row_matrix.shape[0], dtype=np.float64)
                            row_strategy[np.asarray(row_support, dtype=int)] = row_probs_on_sup

                            col_strategy = np.zeros(col_matrix.shape[1], dtype=np.float64)
                            col_strategy[np.asarray(col_support, dtype=int)] = col_probs_on_sup

                            
                            payoff_row = row_strategy @ row_matrix @ col_strategy
                            payoff_col = row_strategy @ col_matrix @ col_strategy

                        
                            row_dev_payoff = np.max(row_matrix @ col_strategy)

                            
                            col_dev_payoff = np.max(row_strategy @ col_matrix)

                            epsilon = 1e-7
                            if (row_dev_payoff <= payoff_row + epsilon) and (col_dev_payoff <= payoff_col + epsilon):
                            
                                nash_equilibria.append((row_strategy, col_strategy))
                            
    return nash_equilibria


def main() -> None:
    pass


if __name__ == '__main__':
    main()
