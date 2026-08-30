#!/usr/bin/env python3

import itertools
import numpy as np
from week07 import *
from week04 import find_nash_equilibrium

def _collect_sequences(node:Node, pid, sequences: set[tuple], current_sequence: tuple, parent_sequences: dict, actions_at_info_set: dict, action_counts):
        
        if node.is_terminal:
            return

        if node.is_chance:
            for _, child in node.children.items():
                _collect_sequences(child, pid, sequences, current_sequence, parent_sequences, actions_at_info_set, action_counts)
    
        elif node.player == pid:
            parent_sequences[node.info_set] = current_sequence
            actions_at_info_set[node.info_set] = tuple(int(action) for action in node.children)
            action_counts[node.info_set] = len(node.actions)

            for action, child in node.children.items():
                next_sequence = current_sequence + ((node.info_set, int(action)),)
                sequences.add(next_sequence)
                _collect_sequences(child, pid, sequences, next_sequence, parent_sequences, actions_at_info_set, action_counts)
    
        else:
            for _, child in node.children.items():
                _collect_sequences(child, pid, sequences, current_sequence, parent_sequences, actions_at_info_set, action_counts)
    

def convert_to_normal_form(root: Node) -> tuple[np.ndarray, np.ndarray]:
    """Convert an extensive-form game into an equivalent normal-form representation.

    Feel free to conceptually split this function into smaller functions that compute
        - the set of pure strategies for a player, e.g. `_collect_available_actions`
        - the expected utility of a pure strategy profile, e.g. `_compute_expected_utility`

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A pair of payoff matrices for the two players in the resulting normal-form game.
    """

    def _collect_available_actions(node: Node, player_id: int, available_actions: dict):

        if node.is_terminal:
            return available_actions

        if node.is_chance:

            for action, child in node.children.items():
                _collect_available_actions(child, player_id, available_actions)


        elif node.player == player_id:
            if node.info_set not in available_actions:
                
                available_actions[node.info_set] = []

            actions = tuple(int(action) for action in node.children)

            available_actions[node.info_set] = (actions, len(node.actions))

            for action, child in node.children.items():
                _collect_available_actions(child, player_id, available_actions)
        else:
            for action, child in node.children.items():
                _collect_available_actions(child, player_id, available_actions)

        return available_actions

    def _available_actions_to_pure_strategies(available_actions: Dict):

        pure_strategies = []

        info_sets = list(available_actions.keys())
        list_of_action_lists = [available_actions[info_set][0] for info_set in info_sets]

        combinations = itertools.product(*list_of_action_lists)

        for combination in combinations:

            pure_strategy = {info_set: np.zeros(len(available_actions[info_set])) for info_set in info_sets}

            for i, info_set in enumerate(info_sets):
                pure_strategy[info_set][combination[i]] = 1.0

            pure_strategies.append(pure_strategy)

        return pure_strategies

    def _compute_expected_utility(player_0_strategy: dict, player_1_strategy: dict) -> np.ndarray:
        strategies = {0: player_0_strategy, 1: player_1_strategy}
        return evaluate(root, strategies)

    player_0_actions = _collect_available_actions(root, 0, {})
    player_1_actions = _collect_available_actions(root, 1, {})

    player_0_pure_strategies = _available_actions_to_pure_strategies(player_0_actions)
    player_1_pure_strategies = _available_actions_to_pure_strategies(player_1_actions)

    player_0_payoffs = np.zeros((len(player_0_pure_strategies), len(player_1_pure_strategies)))
    player_1_payoffs = np.zeros((len(player_0_pure_strategies), len(player_1_pure_strategies)))

    for i, player_0_strategy in enumerate(player_0_pure_strategies):
        for j, player_1_strategy in enumerate(player_1_pure_strategies):
            expected_utility = _compute_expected_utility(player_0_strategy, player_1_strategy)
            player_0_payoffs[i, j] = expected_utility[0]
            player_1_payoffs[i, j] = expected_utility[1]

    return player_0_payoffs, player_1_payoffs


def convert_to_sequence_form(root) -> tuple[np.ndarray, ...]:
    """Convert an extensive-form game into its sequence-form representation.

    The sequence-form representation consists of:
        - The sequence-form payoff matrices for both players
        - The realization-plan constraint matrices and vectors for both players

    Feel free to conceptually split this function into smaller functions that compute
        - the sequences for a player, e.g. `_collect_sequences`
        - the sequence-form payoff matrix of a player, e.g. `_compute_sequence_form_payoff_matrix`
        - the realization-plan constraints of a player, e.g. `_compute_realization_plan_constraints`

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the sequence-form payoff matrices and realization-plan constraints.
    """
    player_1_sequences = set()
    player_2_sequences = set()

    player_1_sequences.add(())
    player_2_sequences.add(())

    player_1_parent_sequences = {}
    player_1_actions_at_info_set = {}

    player_2_parent_sequences = {}
    player_2_actions_at_info_set = {}

    _collect_sequences(root, 0, player_1_sequences, (), player_1_parent_sequences, player_1_actions_at_info_set)
    _collect_sequences(root, 1, player_2_sequences, (), player_2_parent_sequences, player_2_actions_at_info_set)
   
    player_1_sequences = sorted(player_1_sequences, key=lambda s: (len(s), s))
    player_1_sequence_to_index = {sequence: i for i, sequence in enumerate(player_1_sequences)}

    player_2_sequences = sorted(player_2_sequences, key=lambda s: (len(s), s))
    player_2_sequence_to_index = {sequence: i for i, sequence in enumerate(player_2_sequences)}

    player_1_payoff_matrix = np.zeros((len(player_1_sequences), len(player_2_sequences)))
    player_2_payoff_matrix = np.zeros((len(player_1_sequences), len(player_2_sequences)))

    def _compute_sequence_form_payoff_matrix(node, player_1_sequence, player_2_sequence, current_reach_probability):
        if node.is_terminal:
            i = player_1_sequence_to_index[player_1_sequence]
            j = player_2_sequence_to_index[player_2_sequence]

            player_1_payoff_matrix[i, j] += current_reach_probability * node.payoffs[0]
            player_2_payoff_matrix[i, j] += current_reach_probability * node.payoffs[1]
            return

        if node.is_chance:
            for action, child in node.children.items():
                prob = node.chance_strategy[action]
                _compute_sequence_form_payoff_matrix(
                    child,
                    player_1_sequence,
                    player_2_sequence,
                    current_reach_probability * prob,
                )
            return
        
        for action, child in node.children.items():
            if node.player == 0:
                next_p1 = player_1_sequence + ((node.info_set, action),)
                _compute_sequence_form_payoff_matrix(
                    child,
                    next_p1,
                    player_2_sequence,
                    current_reach_probability,
                )
            else:
                next_p2 = player_2_sequence + ((node.info_set, action),)
                _compute_sequence_form_payoff_matrix(
                    child,
                    player_1_sequence,
                    next_p2,
                    current_reach_probability,
                )
    
    def _compute_realization_plan_constraints(sequences: list, sequence_to_index: dict, parent_sequences: dict, actions_at_info_set: dict):
        number_of_constraints = 1 + len(actions_at_info_set)

        constraint_matrix = np.zeros((number_of_constraints, len(sequences)))
        constraint_vector = np.zeros(number_of_constraints)

        empty_sequence_index = sequence_to_index[()]

        constraint_matrix[0, empty_sequence_index] = 1
        constraint_vector[0] = 1

        for row, (info_set, actions) in enumerate(
            actions_at_info_set.items(),
            start=1,
        ):
            parent_sequence = parent_sequences[info_set]
            parent_index = sequence_to_index[parent_sequence]

            constraint_matrix[row, parent_index] = 1

            for action in actions:
                child_sequence = parent_sequence + (
                    (info_set, action),
                )
                child_index = sequence_to_index[child_sequence]

                constraint_matrix[row, child_index] = -1
        
        return constraint_matrix, constraint_vector

    _compute_sequence_form_payoff_matrix(root, (), (), 1.0)

    E, e = _compute_realization_plan_constraints(
        player_1_sequences,
        player_1_sequence_to_index,
        player_1_parent_sequences,
        player_1_actions_at_info_set,
    )

    F, f = _compute_realization_plan_constraints(
        player_2_sequences,
        player_2_sequence_to_index,
        player_2_parent_sequences,
        player_2_actions_at_info_set,
    )
    return player_1_payoff_matrix, player_2_payoff_matrix, E, e, F, f   

def find_nash_equilibrium_sequence_form(root: Node) -> tuple[np.ndarray, np.ndarray]:
    """Find a Nash equilibrium in a zero-sum extensive-form game using Sequence-form LP.

    This function is expected to received an extensive-form game as input
    and convert it to its sequence-form using `convert_to_sequence_form`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A pair of realization plans for the two players for representing a Nash equilibrium.
    """
    payoff_matrix, _ = convert_to_sequence_form(root)
    return find_nash_equilibrium(payoff_matrix)
    


def convert_realization_plan_to_behavioral_strategy(root, realization_plan, player):
    """Convert a realization plan to a behavioral strategy."""
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")

    sequence_set = {()}
    parent_sequences = {}
    actions_at_info_set = {}
    action_counts = {}

    _collect_sequences(
        root,
        player,
        sequence_set,
        (),
        parent_sequences,
        actions_at_info_set,
        action_counts,
    )

    sequences = sorted(sequence_set, key=lambda sequence: (len(sequence), sequence))

    sequence_to_index = {sequence: index for index, sequence in enumerate(sequences)}

    realization_plan = np.asarray(realization_plan, dtype=np.float64)

    behavioral_strategy = {}

    for info_set, actions in actions_at_info_set.items():
        parent_sequence = parent_sequences[info_set]
        parent_index = sequence_to_index[parent_sequence]
        parent_weight = realization_plan[parent_index]

        local_strategy = np.zeros(action_counts[info_set], dtype=np.float64)

        if parent_weight > 1e-12:
            for action in actions:
                child_sequence = parent_sequence + ((info_set, action))
                child_index = sequence_to_index[child_sequence]

                local_strategy[action] = max(0.0, realization_plan[child_index]) / parent_weight

            local_strategy /= local_strategy.sum()
        else:
            local_strategy[list(actions)] = (1.0 / len(actions))

        behavioral_strategy[info_set] = local_strategy

    return behavioral_strategy



def main() -> None:
    pass


if __name__ == '__main__':
    main()
