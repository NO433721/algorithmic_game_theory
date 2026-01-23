#!/usr/bin/env python3

import numpy as np
from scipy.optimize import linprog

def find_nash_equilibrium(row_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find a Nash equilibrium in a zero-sum normal-form game using linear programming.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A strategy profile that forms a Nash equilibrium
    """
    def solve(row_matrix: np.ndarray):
        A = np.asarray(row_matrix)

        m, n = row_matrix.shape
        c = np.zeros(m + 1)
        c[-1] = -1.0

        A_ub = np.hstack([-A.T, np.ones((n, 1))])
        b_ub = np.zeros(n)

        A_eq = np.zeros((1, m + 1))
        A_eq[0, :m] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, None)] * m + [(None, None)]


        row_result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
        )
        
        if not row_result.success:
                        raise RuntimeError(f"LP failed: {row_result.message}")
        return row_result.x[:m]

    row_strategy=solve(row_matrix)
    col_strategy=solve(-row_matrix.T)

    return row_strategy, col_strategy




def find_correlated_equilibrium(row_matrix: np.ndarray, col_matrix: np.ndarray) -> np.ndarray:
    """Find a correlated equilibrium in a normal-form game using linear programming.

    While the cost vector could be selected to optimize a particular objective, such as
    maximizing the sum of players’ utilities, the reference solution sets it to the zero
    vector to ensure reproducibility during testing.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    np.ndarray
        A distribution over joint actions that forms a correlated equilibrium
    """

    num_row=row_matrix.shape[0]
    num_col=col_matrix.shape[1]

    num_var=num_row*num_col
    A_ub=np.zeros((num_row*(num_row-1)+num_col*(num_col-1), num_var))
    A_ub=[]

    for a_r in range(num_row):
        for a_r_p in range(num_row):
            if a_r_p == a_r:
                continue

            constraint=[]
            constraint.extend([0]*a_r*num_col)

            for a_c in range(num_col):
                constraint.append(row_matrix[a_r,a_c]-row_matrix[a_r_p,a_c])
            constraint.extend([0]*(num_var-(a_r+1)*num_col))

            A_ub.append(constraint)

    for a_c in range(num_col):
        for a_c_p in range(num_col):
            if a_c_p == a_c:
                continue

            constraint=[0]*num_var
            for a_r in range(num_row):

                constraint[a_c + a_r * num_col]= col_matrix.T[a_c, a_r] - col_matrix.T[a_c_p,a_r]
            A_ub.append(constraint)

    A_ub=np.asarray(A_ub)
    #print(A_ub)

    c=np.zeros(num_var)

    A_ub=-A_ub
    b_ub=np.zeros((num_row*(num_row-1)+num_col*(num_col-1)))

    A_eq=np.ones((1,num_var))
    b_eq=1

    result = linprog(c, A_ub=A_ub, A_eq=A_eq, b_ub=b_ub, b_eq=b_eq, method="highs")

    if result.success:
        prob_matrix=np.asarray(result.x).reshape(num_row, num_col)
        return prob_matrix


def main() -> None:
    pass


if __name__ == '__main__':
    main()
