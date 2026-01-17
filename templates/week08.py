#!/usr/bin/env python3

import itertools
import numpy as np
from week07 import *
from typing import Dict, List, Any, Tuple



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

    def _collect_available_actions(node: Node|TerminalNode|PlayerNode|ChanceNode, player_id: int, available_actions: Dict):
        
        available_actions = {}


        if node.is_terminal:
            return available_actions

        if isinstance(node, ChanceNode):

            for action, (child, prob) in node.children.items():
                _collect_available_actions(child, player_id, available_actions)

        elif isinstance(node, PlayerNode):

            if node.player == player_id:
                if node.info_set not in available_actions:
                    
                    available_actions[node.info_set] = []

                    for action, child in node.children.items():
                        available_actions[node.info_set].append(action)
                
                        _collect_available_actions(child, player_id, available_actions)
            else:
                for action, child in node.children.items():
                    _collect_available_actions(child, player_id, available_actions)

        return available_actions

    def _available_actions_to_pure_strategies(available_actions: Dict):

        pure_strategies = []

        info_sets = list(available_actions.keys())
        list_of_action_lists = [available_actions[info_set] for info_set in info_sets]

        combinations = itertools.product(*list_of_action_lists)

        for combination in combinations:

            pure_strategy = {info_set: np.zeros(len(available_actions[info_set])) for info_set in info_sets}

            for i, info_set in enumerate(info_sets):
                pure_strategy[info_set][combination[i]] = 1.0

            pure_strategies.append(pure_strategy)

        return pure_strategies

    def _compute_expected_utility():
        


   
        

    raise NotImplementedError


def convert_to_sequence_form(*args, **kwargs) -> tuple[np.ndarray, ...]:
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

    raise NotImplementedError


def find_nash_equilibrium_sequence_form(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Find a Nash equilibrium in a zero-sum extensive-form game using Sequence-form LP.

    This function is expected to received an extensive-form game as input
    and convert it to its sequence-form using `convert_to_sequence_form`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A pair of realization plans for the two players for representing a Nash equilibrium.
    """

    raise NotImplementedError


def convert_realization_plan_to_behavioural_strategy(*args, **kwargs):
    """Convert a realization plan to a behavioural strategy."""

    raise NotImplementedError


def main() -> None:
    pass


if __name__ == '__main__':
    main()
