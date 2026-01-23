#!/usr/bin/env python3

"""
Extensive-form games assignments.

Starting this week, the templates will no longer contain exact function
signatures and there will not be any automated tests like we had for the
normal-form games assignments. Instead, we will provide sample outputs
produced by the reference implementations which you can use to verify
your solutions. The reason for this change is that there are many valid
ways to represent game trees (e.g. flat array-based vs. pointer-based),
information sets and strategies in extensive-form games. Figuring out
the most suitable representations is an important part of assignments
in this block. Unfortunately, this freedom makes automated testing
pretty much impossible.
"""



from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import kuhn_poker


class Node:
    def __init__(self, state: kuhn_poker.State, history):
        self.state = state
        self.history = tuple(int(h) for h in history)
        self.player = int(state.current_player)
        self.children = {}

        self.is_terminal = True if state.terminated or state.truncated else False
        self.is_chance = state.is_chance_node
        
        self.legal_action_mask = state.legal_action_mask
        self.actions = np.arange(len(self.legal_action_mask))


        self.chance_strategy = state.chance_strategy if self.is_chance else None
        self.payoffs = state.rewards if self.is_terminal else None
        
        self.info_set = self._get_info_set()


    def _get_info_set(self):
        if self.is_terminal or self.is_chance:
            return None

        my_card = self.history[self.player]
        betting_history = tuple(self.history[2:])
        return (int(my_card), betting_history)
        

def traverse_tree(env:kuhn_poker.KuhnPokerNumpy):
    """Build the full game tree and collect information sets."""
    state = env.init(0)
    history = ()

    def _traverse(state: kuhn_poker.State, history: Tuple):
        node = Node(state, history)
        if node.is_terminal:
            return node

        else:
            for action in node.actions:
                if node.legal_action_mask[action]:
                    next_state = env.step(state, int(action))
                    child = _traverse(next_state, history + (int(action),))
                    node.children[action] = child

        return node

    
    def _collect_info_sets(node:Node, info_sets:dict):
        if not node.is_chance and not node.is_terminal and node.info_set is not None:
            if node.info_set not in info_sets:
                info_sets[node.info_set] = []
            info_sets[node.info_set].append(node)

        for child in node.children.values():
            _collect_info_sets(child, info_sets)

    
    root = _traverse(state, history)

    info_sets = {}
    _collect_info_sets(root, info_sets)
    return root, info_sets



def evaluate(node:Node, strategies: Dict):
    """Compute the expected utility of each player in an extensive-form game."""
    if node.is_terminal:
        return node.payoffs

    elif node.is_chance:
        expected_utility = np.zeros(2) # kuhn_poker specific
        for action in node.actions:
            if node.legal_action_mask[action]:
                expected_utility += node.chance_strategy[action] * evaluate(node.children[action], strategies)
        return expected_utility

    else:
        expected_utility = np.zeros(2) # kuhn_poker specific
        for action in node.actions:
            if node.legal_action_mask[action]:
                expected_utility += strategies[node.player][node.info_set][action] * evaluate(node.children[action], strategies)
        return expected_utility


def compute_best_response(root: Node, player: int, opponent_strategy: dict, info_sets: dict):
    """Compute a best response strategy for a given player against a fixed opponent's strategy."""
    counterfactual_reach_probabilities = {}

    def _annotate_counterfactual_reach_probabilities(node, p_counterfactual_reach):
        counterfactual_reach_probabilities[node] = p_counterfactual_reach

        if node.is_terminal:
            return

        if node.is_chance:
            for action in node.actions:
                if node.legal_action_mask[action]:
                    _annotate_counterfactual_reach_probabilities(node.children[action], p_counterfactual_reach * node.chance_strategy[action])
        else:
            if node.player == player:
                for action in node.actions:
                    if node.legal_action_mask[action]:
                        _annotate_counterfactual_reach_probabilities(node.children[action], p_counterfactual_reach)
            else:
                local_opponent_strategy = opponent_strategy[node.info_set]
                             
                for action in node.actions:
                    if node.legal_action_mask[action]:
                        _annotate_counterfactual_reach_probabilities(node.children[action], p_counterfactual_reach*local_opponent_strategy[action])
    
    _annotate_counterfactual_reach_probabilities(root, 1.0)

    best_response_strategy = {}
    
    
    def _traverse(node):

        if node.is_terminal:
            return node.payoffs[player]
        
        if node.is_chance:

            expected_utility = 0
            
            for action in node.actions:
                if node.legal_action_mask[action]:
                    expected_utility += node.chance_strategy[action] * _traverse(node.children[action])
            return expected_utility
        
        if node.player != player:

            expected_utility = 0

            local_opponent_strategy = opponent_strategy[node.info_set]
            
            for action in node.actions:
                if node.legal_action_mask[action]:
                    expected_utility += local_opponent_strategy[action]*_traverse(node.children[action])
           
            return expected_utility

        else:
            if node.info_set not in best_response_strategy:
                nodes_in_info_set = info_sets[node.info_set]
            
                num_actions = len(node.actions)
                    
                expected_utility_for_actions = np.zeros(num_actions)

                for h_node in nodes_in_info_set:
                    counterfactual_reach_probability = counterfactual_reach_probabilities[h_node]
                    
                    for action in h_node.actions:
                        if node.legal_action_mask[action]:
                            expected_utility_for_actions[action] += counterfactual_reach_probability*_traverse(h_node.children[action])
                
                illegal = ~node.legal_action_mask.astype(bool)   
                expected_utility_for_actions[illegal] = -np.inf

                best_action = np.argmax(expected_utility_for_actions)
                
                local_best_response_strategy = np.zeros(num_actions)
                local_best_response_strategy[best_action] = 1.0
                best_response_strategy[node.info_set] = local_best_response_strategy
                           
            chosen_strategy = best_response_strategy[node.info_set]
            chosen_action = np.argmax(chosen_strategy)
            return _traverse(node.children[chosen_action])

    _traverse(root)
        
    return best_response_strategy




def compute_average_strategy(root, strategy_a, strategy_b, alpha, player):
    """Compute a weighted average of a pair of behavioral strategies for a given player."""
    average_strategies = {}

    def _traverse(node, reach_a, reach_b):
        if node.is_terminal:
            return

        if node.is_chance:
            for action in node.actions:
                if node.legal_action_mask[action]:
                    _traverse(node.children[action], reach_a, reach_b)

        else:
            if node.player == player:
                num_actions = len(node.actions)
                
                if node.info_set in strategy_a:
                    local_strategy_a = strategy_a[node.info_set]
                else:
                    if sum(node.legal_action_mask) > 0:
                        local_strategy_a = node.legal_action_mask / sum(node.legal_action_mask)
                    else:
                        local_strategy_a = np.zeros(num_actions)
                
                if node.info_set in strategy_b:
                    local_strategy_b = strategy_b[node.info_set]
                else:
                    if sum(node.legal_action_mask) > 0:

                        local_strategy_b = node.legal_action_mask / sum(node.legal_action_mask)

                    else:
                        local_strategy_b = np.zeros(num_actions)
                

                if (1 - alpha) * reach_a + alpha * reach_b > 0:

                    new_probability = ((1 - alpha) * reach_a * local_strategy_a + 
                                 alpha * reach_b * local_strategy_b) / ((1 - alpha) * reach_a + alpha * reach_b)
                else:
                    if sum(node.legal_action_mask) > 0:

                        new_probability = node.legal_action_mask/ sum(node.legal_action_mask)
                    else:
                        new_probability = np.zeros(num_actions)

                average_strategies[node.info_set] = new_probability

                for action in node.actions:
                    if node.legal_action_mask[action]:
                        _traverse(
                            node.children[action], 
                            reach_a * local_strategy_a[action], 
                            reach_b * local_strategy_b[action]
                        )
            else:
                for action in node.actions:
                    if node.legal_action_mask[action]:
                        _traverse(node.children[action], reach_a, reach_b)

    _traverse(root, 1.0, 1.0)
    
    return average_strategies


def fictitious_play(root, info_sets, num_iters=10):
    """Run Extensive-form Fictitious Play for a given number of iterations."""

    average_strategies = {} 
    
    for nodes in info_sets.values():
        node = nodes[0]
        player = node.player
        
        if player not in average_strategies:
            average_strategies[player] = {}
            
        if node.info_set not in average_strategies[player]:

            average_strategies[player][node.info_set] = node.legal_action_mask / sum(node.legal_action_mask)
     

    players = sorted(average_strategies.keys())
    
    history = []

    for t in range(1, num_iters + 1):
        alpha = 1.0 / (t + 1)
        
        next_average_strategies = {player: {} for player in players}
        
        for player in players:
            
            opponent_strategy = {}
            for other_player in players:
                if other_player != player:
                    opponent_strategy.update(average_strategies[other_player])
            
            br_strategy = compute_best_response(root, player, opponent_strategy, info_sets)
            
            current_player_average = average_strategies[player]
            
            new_player_average = compute_average_strategy(
                root, 
                current_player_average, 
                br_strategy, 
                alpha, 
                player
            )
            
            next_average_strategies[player] = new_player_average

        average_strategies = next_average_strategies
        
        history.append(average_strategies)

    return history
            
def compute_exploitability(root, info_sets, strategy_sequence, plot=True):
    """Compute and plot the exploitability of a sequence of strategy profiles."""
    exploitability_values = []

    for strategies in strategy_sequence:
        
        current_utilities = evaluate(root, strategies)
        num_players = len(current_utilities)
        
        total_nash_conv = 0.0
        
        for player_id in range(num_players):
            
            opponent_strategy = {}
            for pid, strat in strategies.items():
                if pid != player_id:
                    opponent_strategy.update(strat)
            
            br_strategy = compute_best_response(root, player_id, opponent_strategy, info_sets)
            
            br_profile = strategies.copy()
            br_profile[player_id] = br_strategy
            
            br_utilities = evaluate(root, br_profile)
            br_value = br_utilities[player_id]
            
            incentive = br_value - current_utilities[player_id]
            total_nash_conv += incentive

        exploitability = total_nash_conv / num_players
        exploitability_values.append(exploitability)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(exploitability_values)), exploitability_values, label='Exploitability')
        plt.xlabel('Iteration')
        plt.ylabel('Exploitability (Avg NashConv)')
        plt.title('Convergence of Exploitability over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

    return exploitability_values


def main() -> None:
    from kuhn_poker import KuhnPokerNumpy as KuhnPoker

    # The implementation of the game is a part of a JAX library called `pgx`.
    # You can find more information about it here: https://www.sotets.uk/pgx/kuhn_poker/
    # We wrap the original implementation to add an explicit chance player and convert
    # everything from JAX arrays to Numpy arrays. There's also a JAX version which you
    # can import using `from kuhn_poker import KuhnPoker` if interested ;)
    env = KuhnPoker()

    # Initialize the environment with a random seed
    state = env.init(0)

    while not (state.terminated or state.truncated):
        if state.is_chance_node:
            uniform_strategy = state.legal_action_mask / np.sum(state.legal_action_mask)
            assert np.allclose(state.chance_strategy, uniform_strategy), (
                'The chance strategy is not uniform!'
            )

        # Pick the first legal action
        action = np.argmax(state.legal_action_mask)

        # Take a step in the environment
        state = env.step(state, action)

    assert np.sum(state.rewards) == 0, 'The game is not zero-sum!'
    assert state.terminated or state.truncated, 'The game is not over!'


if __name__ == '__main__':
    main()
