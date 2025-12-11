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

import numpy as np
import kuhn_poker

class Node:
    def __init__(self, state: kuhn_poker.State, history):
        self.state = state
        self.history = tuple(int(h) for h in history)``````````

class TerminalNode(Node):
    def __init__(self, state: kuhn_poker.State, history):
        super().__init__(state, history)
        self.payoffs = np.array(state.rewards)
        self.is_terminal = True

class ChanceNode(Node):
    def __init__(self, state: kuhn_poker.State, history):
        super().__init__(state, history)
        self.children = {} 
        self.is_terminal = False

class PlayerNode(Node):
    def __init__(self, state: kuhn_poker.State, history):
        super().__init__(state, history)
        self.player = int(state.current_player)
        self.children = {}
        self.is_terminal = False
        self.info_set = self._get_info_set()

    def _get_info_set(self):
        my_card = self.history[self.player]
        betting_history = self.history[2:]
        return (my_card, betting_history)



def traverse_tree(env, state:kuhn_poker.State|None = None, history = None):
    """Builds the full game tree via BFS/DFS."""
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
   


def evaluate(*args, **kwargs):
    """Compute the expected utility of each player in an extensive-form game."""

    raise NotImplementedError


def compute_best_response(*args, **kwargs):
    """Compute a best response strategy for a given player against a fixed opponent's strategy."""

    raise NotImplementedError


def compute_average_strategy(*args, **kwargs):
    """Compute a weighted average of a pair of behavioural strategies for a given player."""

    raise NotImplementedError


def compute_exploitability(*args, **kwargs):
    """Compute and plot the exploitability of a sequence of strategy profiles."""

    raise NotImplementedError


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
