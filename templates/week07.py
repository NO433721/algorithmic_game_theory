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
import matplotlib.pyplot as plt
import numpy as np
import kuhn_poker

class Node:
    def __init__(self, state: kuhn_poker.State, history):
        self.state = state
        self.history = tuple(int(h) for h in history)
        self.children = {}
        self.is_terminal = False

class TerminalNode(Node):
    def __init__(self, state: kuhn_poker.State, history):
        super().__init__(state, history)
        self.payoffs = np.array(state.rewards)
        self.is_terminal = True

class ChanceNode(Node):
    def __init__(self, state: kuhn_poker.State, history):
        super().__init__(state, history)

class PlayerNode(Node):
    def __init__(self, state: kuhn_poker.State, history):
        super().__init__(state, history)
        self.player = int(state.current_player)
        self.info_set = self._get_info_set()

    def _get_info_set(self):
        my_card = self.history[self.player]
        betting_history = self.history[2:]
        return (my_card, betting_history)

def traverse_tree(env, state:kuhn_poker.State|None = None, history = None):
    
    info_sets = {}

    def _traverse(curr_state:kuhn_poker.State, curr_history):
        if curr_state.terminated or curr_state.truncated:
            return TerminalNode(curr_state, curr_history)

        if curr_state.is_chance_node:

            node = ChanceNode(curr_state, curr_history)
            probs = curr_state.chance_strategy
            for action, is_legal in enumerate(curr_state.legal_action_mask):
                if is_legal:
                    next_state = env.step(curr_state, action)
                    child = _traverse(next_state, curr_history + [action])

                    node.children[action] = (child, probs[action])

            return node
        
        else: 
            node = PlayerNode(curr_state, curr_history)
            
            
            if node.info_set not in info_sets:
                info_sets[node.info_set] = []
            info_sets[node.info_set].append(node)
            

            for action, is_legal in enumerate(curr_state.legal_action_mask):
                if is_legal:
                    next_state = env.step(curr_state, action)
                    child = _traverse(next_state, curr_history + [action])
                    
                    node.children[action] = child
            return node

    if state is None:
        state = env.init(0)
        history = []
        
    root_node = _traverse(state, history)
    
    return root_node, info_sets


def evaluate(node: Node, strategies):
    """Compute the expected utility of each player in an extensive-form game."""
    if node.is_terminal:
        return node.payoffs

    expected_utility = np.zeros(2)

    if isinstance(node, ChanceNode):
        
        for action, (child_node, prob) in node.children.items():
            child_value = evaluate(child_node, strategies)
            expected_utility += prob * child_value

    elif isinstance(node, PlayerNode):
        
        if node.player in strategies and node.info_set in strategies[node.player]:
            action_probs = strategies[node.player][node.info_set]

        else:
            num_actions = len(node.children)
            action_probs = np.ones(num_actions) / num_actions
        
        for action, child_node in node.children.items():
            prob = action_probs[action]
            

            child_value = evaluate(child_node, strategies)
            

            expected_utility += prob * child_value

    return expected_utility

def compute_best_response(root, player_id, opponent_strategy, info_sets):
    
    cf_reach = {}
    def annotate_cf_reach(node, p_opp_chance):
        cf_reach[node] = p_opp_chance

        if node.is_terminal:
            return

        if isinstance(node, ChanceNode):
            for action, (child, prob) in node.children.items():
                annotate_cf_reach(child, p_opp_chance * prob)
                
        elif isinstance(node, PlayerNode):
            if node.player == player_id:
                for action, child in node.children.items():
                    annotate_cf_reach(child, p_opp_chance)
            else:

                if node.info_set in opponent_strategy:
                    strat = opponent_strategy[node.info_set]
                else:
                    strat = np.ones(len(node.children)) / len(node.children)
                
                for action, child in node.children.items():
                    annotate_cf_reach(child, p_opp_chance * strat[action])

    annotate_cf_reach(root, 1.0)


    br_strategy = {}
    
    
    def get_value(node):
        if node.is_terminal:
            return node.payoffs[player_id]
        
        if isinstance(node, ChanceNode):
            ev = 0.0
            for action, (child, prob) in node.children.items():
                ev += prob * get_value(child)
            return ev
            
        if node.player != player_id:
            ev = 0.0
            if node.info_set in opponent_strategy:
                strat = opponent_strategy[node.info_set]
            else:
                strat = np.ones(len(node.children)) / len(node.children)
                
            for action, child in node.children.items():
                ev += strat[action] * get_value(child)
            return ev

        if node.info_set not in br_strategy:
            nodes_in_set = info_sets[node.info_set]
            num_actions = len(node.children)
            
            action_values = np.zeros(num_actions)
            
            for h_node in nodes_in_set:
                reach_prob = cf_reach[h_node]
                
                for action, child in h_node.children.items():
                    val_child = get_value(child)
                    action_values[action] += reach_prob * val_child            
            
            best_action = np.argmax(action_values)
            
            
            strat_vec = np.zeros(num_actions)
            strat_vec[best_action] = 1.0
            br_strategy[node.info_set] = strat_vec
            
        chosen_strat = br_strategy[node.info_set]
        chosen_action = np.argmax(chosen_strat)
        return get_value(node.children[chosen_action])

    get_value(root)
    
    return br_strategy


def compute_average_strategy(root_node, strategy_a, strategy_b, alpha, player_id):
    """Compute a weighted average of a pair of behavioural strategies for a given player."""
    avg_strategy = {}

    def _traverse(node, reach_a, reach_b):
        if node.is_terminal:
            return

        if isinstance(node, ChanceNode):
            for action, (child, prob) in node.children.items():
                _traverse(child, reach_a, reach_b)

        elif isinstance(node, PlayerNode):
            if node.player == player_id:
                num_actions = len(node.children)
                
                if node.info_set in strategy_a:
                    probs_a = strategy_a[node.info_set]
                else:
                    probs_a = np.ones(num_actions) / num_actions
                    
                if node.info_set in strategy_b:
                    probs_b = strategy_b[node.info_set]
                else:
                    probs_b = np.ones(num_actions) / num_actions

                denom = (1 - alpha) * reach_a + alpha * reach_b
                
                if denom > 0:
                    new_probs = ((1 - alpha) * reach_a * probs_a + 
                                 alpha * reach_b * probs_b) / denom
                else:
                    new_probs = np.ones(num_actions) / num_actions

                avg_strategy[node.info_set] = new_probs

                for action, child in node.children.items():
                    _traverse(
                        child, 
                        reach_a * probs_a[action], 
                        reach_b * probs_b[action]
                    )
            else:
                for action, child in node.children.items():
                    _traverse(child, reach_a, reach_b)

    _traverse(root_node, 1.0, 1.0)
    
    return avg_strategy


def fictitious_play(root, info_sets, num_iters=10):
    """Run Extensive-form Fictitious Play for a given number of iterations."""

    avg_strategies = {} 
    
    for nodes in info_sets.values():
        node = nodes[0]
        pid = node.player
        
        if pid not in avg_strategies:
            avg_strategies[pid] = {}
            
        if node.info_set not in avg_strategies[pid]:
            num_actions = len(node.children)
            avg_strategies[pid][node.info_set] = np.ones(num_actions) / num_actions

    players = sorted(avg_strategies.keys())
    
    history = []

    for t in range(1, num_iters + 1):
        alpha = 1.0 / (t + 1)
        
        next_avg_strategies = {pid: {} for pid in players}
        
        for player_id in players:
            
            opponent_strategy = {}
            for other_pid in players:
                if other_pid != player_id:
                    opponent_strategy.update(avg_strategies[other_pid])
            
            br_strat = compute_best_response(root, player_id, opponent_strategy, info_sets)
            
            current_player_avg = avg_strategies[player_id]
            
            new_player_avg = compute_average_strategy(
                root, 
                current_player_avg, 
                br_strat, 
                alpha, 
                player_id
            )
            
            next_avg_strategies[player_id] = new_player_avg

        avg_strategies = next_avg_strategies
        
        history.append(avg_strategies)

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