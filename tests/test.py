import numpy as np
import sys
import os

# Ensure we can import your templates
sys.path.append(os.getcwd())
from templates.kuhn_poker import KuhnPokerNumpy

# --- COPY & PASTE YOUR PREVIOUS CLASS DEFINITIONS HERE ---
# (Node, TerminalNode, ChanceNode, PlayerNode)
# For brevity, assuming they are defined or imported as before.

class Node:
    def __init__(self, state, history):
        self.state = state
        self.history = tuple(int(h) for h in history)
        self.children = {}

class TerminalNode(Node):
    def __init__(self, state, history):
        super().__init__(state, history)
        self.payoffs = np.array(state.rewards)
        self.is_terminal = True

class ChanceNode(Node):
    def __init__(self, state, history):
        super().__init__(state, history)
        self.is_terminal = False

class PlayerNode(Node):
    def __init__(self, state, history):
        super().__init__(state, history)
        self.player = int(state.current_player)
        self.is_terminal = False
        self.info_set = self._get_info_set()

    def _get_info_set(self):
        my_card = self.history[self.player]
        betting_history = self.history[2:] 
        return (my_card, tuple(betting_history))

# --- INSERT YOUR FUNCTIONS HERE ---
# Paste your implementations of traverse_tree, evaluate, and compute_best_response

def traverse_tree(env, state=None, history=None):
    # ... (Your implementation from previous turn) ...
    info_sets = {}
    def _traverse(curr_state, curr_history):
        if curr_state.terminated:
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

    if state is None: state = env.init(0); history = []
    root = _traverse(state, history)
    return root, info_sets

def evaluate(node, strategies):
    # ... (Your implementation from previous turn) ...
    if node.is_terminal: return node.payoffs
    expected_utility = np.zeros(2)
    if isinstance(node, ChanceNode):
        for action, (child_node, prob) in node.children.items():
            expected_utility += prob * evaluate(child_node, strategies)
    elif isinstance(node, PlayerNode):
        action_probs = strategies.get(node.info_set, np.ones(3)/3)
        for action, child_node in node.children.items():
            prob = action_probs[action]
            expected_utility += prob * evaluate(child_node, strategies)
    return expected_utility
    
    
def compute_best_response(root, player_id, opponent_strategy, info_sets):
    """
    Compute a best response strategy for a given player against a fixed opponent's strategy.
    Corrected to handle negative payoff values correctly.
    """
    
    # --- Phase 1: Annotate Counterfactual Reach Probabilities (Top-Down) ---
    def annotate_cf_reach(node, p_opp_chance):
        node.cf_reach = p_opp_chance
        
        if node.is_terminal:
            return

        if isinstance(node, ChanceNode):
            for action, (child, prob) in node.children.items():
                annotate_cf_reach(child, p_opp_chance * prob)
                
        elif isinstance(node, PlayerNode):
            if node.player == player_id:
                # Hero's actions do NOT reduce P_{-i}
                for action, child in node.children.items():
                    annotate_cf_reach(child, p_opp_chance)
            else:
                # Opponent actions reduce P_{-i}
                if node.info_set in opponent_strategy:
                    strat = opponent_strategy[node.info_set]
                else:
                    strat = np.ones(3) / 3 # Default if missing
                
                for action, child in node.children.items():
                    annotate_cf_reach(child, p_opp_chance * strat[action])

    annotate_cf_reach(root, 1.0)


    # --- Phase 2: Compute Best Response Values (Bottom-Up) ---
    
    br_strategy = {}
    
    def get_value(node):
        # 1. Terminal Node
        if node.is_terminal:
            return node.payoffs[player_id]
        
        # 2. Chance Node
        if isinstance(node, ChanceNode):
            ev = 0.0
            for action, (child, prob) in node.children.items():
                ev += prob * get_value(child)
            return ev
            
        # 3. Opponent Node
        if node.player != player_id:
            ev = 0.0
            if node.info_set in opponent_strategy:
                strat = opponent_strategy[node.info_set]
            else:
                strat = np.ones(3) / 3
                
            for action, child in node.children.items():
                ev += strat[action] * get_value(child)
            return ev
            
        # 4. Hero Node: MAXIMIZE over the Information Set
        if node.info_set not in br_strategy:
            nodes_in_set = info_sets[node.info_set]
            
            # Use a dictionary to accumulate values only for valid actions
            # This avoids the issue where "0.0" for an invalid action > negative EV
            accumulator = {} 
            
            for h_node in nodes_in_set:
                for action, child in h_node.children.items():
                    if action not in accumulator:
                        accumulator[action] = 0.0
                    
                    # Recursively get value
                    val_child = get_value(child)
                    accumulator[action] += h_node.cf_reach * val_child
            
            # Select Best Action from the valid ones we found
            if not accumulator:
                # Should not happen in a valid tree
                best_action = 0
            else:
                best_action = max(accumulator, key=accumulator.get)
            
            # Create strategy vector
            strat_vec = np.zeros(3)
            strat_vec[best_action] = 1.0
            br_strategy[node.info_set] = strat_vec
            
        # Return the value of THIS specific node assuming the chosen BR strategy
        chosen_strat = br_strategy[node.info_set]
        chosen_action = np.argmax(chosen_strat)
        
        # Safety check: if for some reason the chosen action isn't a child of this specific node
        # (Rare, but possible if the tree structure is asymmetric across an infoset)
        if chosen_action in node.children:
            return get_value(node.children[chosen_action])
        else:
            # Fallback (should ideally not be reached in Kuhn Poker)
            return -np.inf 

    get_value(root)
    
    return br_strategy

# --- MAIN TESTING LOGIC ---

if __name__ == "__main__":
    print("1. Building Game Tree...")
    env = KuhnPokerNumpy()
    root, info_sets = traverse_tree(env)

    # --- Step 1: Define Opponent Strategy (Uniform Random) ---
    # Opponent plays 0 (Bet/Call) and 1 (Check/Fold) with 50% probability
    opponent_strategy = {}
    for info_set in info_sets:
        opponent_strategy[info_set] = np.array([0.5, 0.5, 0.0])

    # --- Step 2: Test P1 Best Response vs P2 Uniform ---
    print("\n2. Computing P1 (Hero) Best Response against P2 (Uniform)...")
    br_p1 = compute_best_response(root, player_id=0, opponent_strategy=opponent_strategy, info_sets=info_sets)
    
    # Merge strategies for evaluation:
    # Use BR for P1, Opponent (Uniform) for P2
    profile_p1_vs_uni = {**opponent_strategy, **br_p1} 
    
    payoffs_1 = evaluate(root, profile_p1_vs_uni)
    print(f"   Payoffs: {payoffs_1}")
    print(f"   P1 Exploitability of Uniform Random: {payoffs_1[0]:.4f}")
    
    if payoffs_1[0] > 0:
        print("   [SUCCESS] P1 has a positive expected value, exploiting P2's randomness.")
    else:
        print("   [FAILURE] P1 failed to exploit the opponent.")


    # --- Step 3: Test P2 Best Response vs P1 Uniform ---
    print("\n3. Computing P2 (Hero) Best Response against P1 (Uniform)...")
    br_p2 = compute_best_response(root, player_id=1, opponent_strategy=opponent_strategy, info_sets=info_sets)
    
    # Merge strategies:
    # Use Opponent (Uniform) for P1, BR for P2
    profile_uni_vs_p2 = {**opponent_strategy, **br_p2}
    
    payoffs_2 = evaluate(root, profile_uni_vs_p2)
    print(f"   Payoffs: {payoffs_2}")
    print(f"   P2 Exploitability of Uniform Random: {payoffs_2[1]:.4f}")

    if payoffs_2[1] > 0:
         print("   [SUCCESS] P2 has a positive expected value, exploiting P1's randomness.")
    else:
         print("   [FAILURE] P2 failed to exploit the opponent.")