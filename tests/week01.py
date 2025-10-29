#!/usr/bin/env python3

import numpy as np


def evaluate_general_sum(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute the expected utility of each player in a general-sum game.

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
        A vector of expected utilities of the players
    """

    # Compute the expected utility for the row player
    row_expected_utility = row_strategy @ row_matrix @ col_strategy

    # Compute the expected utility for the column player
    col_expected_utility = row_strategy @ col_matrix @ col_strategy

    return np.array([row_expected_utility, col_expected_utility], dtype=float)


    # raise NotImplementedError

    


def evaluate_zero_sum(
    row_matrix: np.ndarray, row_strategy: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute the expected utility of each player in a zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """

    # raise NotImplementedError
    row_expected_utility = row_strategy @ row_matrix @ col_strategy
    col_expected_utility = -row_expected_utility

    return np.array([row_expected_utility, col_expected_utility], dtype=float)

def calculate_best_response_against_row(
    col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the column player against the row player.

    Parameters
    ----------
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.ndarray
        The column player's best response
    """
    
    j = int(np.argmax(row_strategy@col_matrix))
    br = np.zeros(col_matrix.shape[1])
    br[j] = 1.0
    return br
    


def calculate_best_response_against_col(
    row_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the row player against the column player.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        The row player's best response
    """

    # raise NotImplementedError
    i = int(np.argmax(row_matrix@col_strategy))          
    br = np.zeros(row_matrix.shape[0])    
    br[i] = 1.0
    return br


def evaluate_row_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.float64:
    """Compute the utility of the row player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.float64
        The expected utility of the row player
    """

    # raise NotImplementedError
    col_expected=row_strategy@col_matrix
    col_br=np.argmax(col_expected)


    row_utils=row_strategy@row_matrix[:,col_br]
    

    return np.float64(row_utils)


def evaluate_col_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.float64:
    """Compute the utility of the column player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The expected utility of the column player
    """

    # raise NotImplementedError
    row_expected=row_matrix@col_strategy
    row_br=np.argmax(row_expected)




    col_utils=col_matrix[row_br, :]@col_strategy
    

    return np.float64(col_utils)



def find_strictly_dominated_actions(matrix: np.ndarray) -> np.ndarray:
    """Find strictly dominated actions for the given normal-form game.

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players

    Returns
    -------
    np.ndarray
        Indices of strictly dominated actions
    """

    # raise NotImplementedError

    dominated = []
    for i in range(matrix.shape[0]):              
        for j in range(matrix.shape[0]):          
            if i == j:
                continue
            if np.all(matrix[j, :] > matrix[i, :]):   
                dominated.append(i)
                break

    return np.array(dominated, dtype=int)


def iterated_removal_of_dominated_strategies(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Iterated Removal of Dominated Strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Four-tuple of reduced row and column payoff matrices, and remaining row and column actions
    """

    reduced_matrix1 = row_matrix
    reduced_matrix2 = col_matrix

    actions1 = [i for i in range(row_matrix.shape[0])]
    actions2 = [i for i in range(col_matrix.shape[1])]

    while True:
        row_remove = []
        col_remove = []

        for i in range(reduced_matrix1.shape[0]):
            gt = reduced_matrix1 > reduced_matrix1[i, :][None, :]   
            result = np.all(gt, axis=1)
            result[i] = False  
            if result.any():
                row_remove.append(i)
                
                
        if row_remove:
            i = min(row_remove)
            reduced_matrix1 = np.delete(reduced_matrix1, i, axis=0)
            reduced_matrix2 = np.delete(reduced_matrix2, i, axis=0)
            actions1.pop(i)
            continue


        for j in range(reduced_matrix2.shape[1]):
            gt = reduced_matrix2 > reduced_matrix2[:, j][:, None]   
            result = np.all(gt, axis=0)
            result[j] = False
            if result.any():
                col_remove.append(j)
                
        if col_remove:
            j = min(col_remove)
            reduced_matrix1 = np.delete(reduced_matrix1, j, axis=1)
            reduced_matrix2 = np.delete(reduced_matrix2, j, axis=1)
            actions2.pop(j)
            continue


        break

    return (np.asarray(reduced_matrix1),
            np.asarray(reduced_matrix2),
            np.asarray(actions1),
            np.asarray(actions2))


def main() -> None:
    pass


if __name__ == '__main__':
    main()
