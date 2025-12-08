import numpy as np
from collections import deque

import kuhn_poker  # Ensure kuhn_poker.py is in the same directory

# Import your solution here
# from solution import traverse_tree, Node, ChanceNode, TerminalNode, PlayerNode
# Or simply paste this test code at the bottom of your solution file.
from week07 import traverse_tree, Node, ChanceNode, TerminalNode, PlayerNode
# --- Helper to map integers to readable names ---
CARD_MAP = {0: 'J', 1: 'Q', 2: 'K'}
ACTION_MAP = {1: 'Check', 0: 'Bet'}

def pretty_print_history(history_tuple):
    """Converts a history tuple (e.g., (0, 2, 1)) into readable text."""
    if len(history_tuple) < 2:
        return str(history_tuple)
    
    # First two elements are always dealt cards
    cards = [CARD_MAP.get(c, str(c)) for c in history_tuple[:2]]
    
    # Subsequent elements are actions
    actions = [ACTION_MAP.get(a, str(a)) for a in history_tuple[2:]]
    
    return f"Cards({cards[0]}, {cards[1]}) -> " + " -> ".join(actions)

def test_game_tree_structure():
    print("=" * 60)
    print("TESTING KUHN POKER GAME TREE")
    print("=" * 60)

    # 1. Generate the Tree
    env = kuhn_poker.KuhnPokerNumpy()
    root_node, info_sets = traverse_tree(env)

    # 2. Collect Statistics via BFS
    all_nodes = []
    terminal_nodes = []
    queue = deque([root_node])

    while queue:
        node = queue.popleft()
        all_nodes.append(node)

        if node.is_terminal:
            terminal_nodes.append(node)
        
        # Add children to queue
        # Handle dict structure differences between Chance and Player nodes if necessary
        if hasattr(node, 'children'):
            for action, child in node.children.items():
                if isinstance(child, tuple): # Chance node stores (node, prob)
                    queue.append(child[0])
                else:
                    queue.append(child)

    # 3. Verify Counts (The "Hard" Checks)
    n_total = len(all_nodes)
    n_terminal = len(terminal_nodes)
    
    print(f"[Check 1] Total Nodes: {n_total}")
    # Expected: 58 (1 Root + 3 Deal1 + 6 Deal2 + 12 P1_Act + 24 P2_Act + 12 P1_ReAct)
    if n_total == 58:
        print("   >>> PASS: Total node count is 58.")
    else:
        print(f"   >>> FAIL: Expected 58 nodes, found {n_total}.")

    print(f"\n[Check 2] Terminal Histories: {n_terminal}")
    # Expected: 30 (See Slide 7 of Lecture)
    if n_terminal == 30:
        print("   >>> PASS: Terminal node count is 30.")
    else:
        print(f"   >>> FAIL: Expected 30 terminal nodes, found {n_terminal}.")

    # 4. Verify Information Sets (The "Content" Check)
    print(f"\n[Check 3] Information Sets Inspection")
    print("-" * 60)
    
    # Sort info sets by player for readability
    sorted_info_sets = sorted(info_sets.items(), key=lambda x: (x[1][0].player, x[0]))

    for (my_card, bet_history), nodes in sorted_info_sets:
        player_id = nodes[0].player
        card_str = CARD_MAP.get(my_card, str(my_card))
        hist_str = " -> ".join([ACTION_MAP.get(a, str(a)) for a in bet_history])
        if not hist_str: hist_str = "Start of Round"

        print(f"Player {player_id+1} Info Set | Card: {card_str} | History: {hist_str}")
        print(f"   Contains {len(nodes)} indistinguishable node(s):")
        
        for node in nodes:
            # Print the FULL history (including opponent's card) to prove they are different
            print(f"   - Full Reality: {pretty_print_history(node.history)}")
        
        # Verification: Opponent card must vary, but My Card and History must match
        if len(nodes) != 2:
            print("   >>> WARNING: In Kuhn poker, expected 2 nodes per info set (opponent holds one of the other 2 cards).")
        print("-" * 60)

    # 5. Final assertion summary
    assert n_total == 58, "Incorrect Total Node Count"
    assert n_terminal == 30, "Incorrect Terminal Node Count"
    assert len(info_sets) == 12, f"Expected 12 total info sets (6 per player), found {len(info_sets)}"
    print("\nAll structural assertions passed!")

if __name__ == '__main__':
    test_game_tree_structure()