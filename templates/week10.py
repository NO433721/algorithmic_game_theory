#!/usr/bin/env python3
from week07 import *
from week09 import regret_matching, make_strategy, cfr_traverse
import numpy as np

def mc_cfr_traverse(node, pid, strategy, strategy_sums, sampled_actions, average_visited, local_regrets, rng):
        if node.is_terminal:
            return node.payoffs[pid]

        legal_actions = np.array(list(node.children.keys()), dtype=int)
        
        if node.is_chance:
        
                probs = np.asarray(node.chance_strategy[legal_actions], dtype=float)
                probs = probs / probs.sum()

                action = int(rng.choice(legal_actions, p=probs))

                return mc_cfr_traverse(node.children[action], pid, strategy, strategy_sums, sampled_actions, average_visited, local_regrets, rng)

        player = node.player
        info_set = node.info_set
        local_strategy = strategy[player][info_set]

        if player != pid:
            key = (player, info_set)

            if key not in average_visited:
                strategy_sums[player][info_set] += local_strategy
                average_visited.add(key)

            if key not in sampled_actions:
                probs = np.asarray(local_strategy[legal_actions], dtype=float)
                probs = probs / probs.sum()
                sampled_actions[key] = int(rng.choice(legal_actions, p=probs))

            action = sampled_actions[key]

            return mc_cfr_traverse(node.children[action], pid, strategy, strategy_sums, sampled_actions, average_visited, local_regrets, rng)

        action_values = np.zeros(len(node.actions))

        for action, child in node.children.items():

            action_values[action] = mc_cfr_traverse(child, pid, strategy, strategy_sums, sampled_actions, average_visited, local_regrets, rng)
            
        node_value = float(np.dot(local_strategy, action_values))

        legal = node.legal_action_mask.astype(bool)

        local_regrets[info_set][legal] += (action_values[legal] - node_value)

        return node_value

def discounted_cfr(root: Node, info_sets, num_iter: int, alpha, beta, gamma):
    """Run the Discounted CFR algorithm for a given number of iterations."""
    players = sorted({int(nodes[0].player) for nodes in info_sets.values()})

    player_info_sets = {player: [] for player in players}

    for info_set, nodes in info_sets.items():
        player = int(nodes[0].player)
        player_info_sets[player].append(info_set)

    cumulative_regrets = {player: {} for player in players}
    strategy_sums = {player: {} for player in players}

    for player in players:
        for info_set in player_info_sets[player]:
            node = info_sets[info_set][0]
            num_actions = len(node.actions)

            cumulative_regrets[player][info_set] = np.zeros(num_actions)
            strategy_sums[player][info_set] = np.zeros(num_actions)
    
    history = []

    for i in range(1, num_iter + 1):
        for player in players:
            strategy = make_strategy(players, info_sets, player_info_sets, cumulative_regrets)
            average_visited = set()
            
            cfr_traverse(root, player, strategy, np.ones(len(players)), 1.0, strategy_sums, i**gamma, average_visited, cumulative_regrets[player])

            for info_set in player_info_sets[player]:
                regrets = cumulative_regrets[player][info_set]

                positive = regrets >= 0

                positive_factor = (i**alpha / (i**alpha + 1))

                negative_factor = (i**beta / (i**beta + 1))

                regrets[positive] *= positive_factor
                regrets[~positive] *= negative_factor
                
        average_strategy = {player: {} for player in players}

        for player in players:
            for info_set in player_info_sets[player]:
                total = strategy_sums[player][info_set].sum()

                if total > 0:
                    average_strategy[player][info_set] = (strategy_sums[player][info_set] / total)
                else:
                    node = info_sets[info_set][0]
                    average_strategy[player][info_set] = (regret_matching(np.zeros(len(node.actions)), node.legal_action_mask.astype(bool)))

        history.append(average_strategy)

    return history

def monte_carlo_cfr(root: Node, info_sets, num_iter, seed):
    """Run the Monte Carlo CFR algorithm for a given number of iterations."""
    rng = np.random.default_rng(seed)

    players = sorted({int(nodes[0].player) for nodes in info_sets.values()})

    player_info_sets = {player: [] for player in players}

    for info_set, nodes in info_sets.items():
        player = int(nodes[0].player)
        player_info_sets[player].append(info_set)

    cumulative_regrets = {player: {} for player in players}
    strategy_sums = {player: {} for player in players}

    for player in players:
        for info_set in player_info_sets[player]:
            node = info_sets[info_set][0]
            num_actions = len(node.actions)

            cumulative_regrets[player][info_set] = np.zeros(num_actions)
            strategy_sums[player][info_set] = np.zeros(num_actions)
    
    history = []

    for _ in range(num_iter):
        for player in players:
            strategy = make_strategy(players, info_sets, player_info_sets, cumulative_regrets)

            mc_cfr_traverse(root, player, strategy, strategy_sums, {}, set(), cumulative_regrets[player], rng)
        average_strategy = {player: {} for player in players}

        for player in players:
            for info_set in player_info_sets[player]:
                strategy_sum = strategy_sums[player][info_set]
                total = strategy_sum.sum()

                if total > 0:
                    average_strategy[player][info_set] = (strategy_sum / total)
                else:
                    node = info_sets[info_set][0]

                    average_strategy[player][info_set] = regret_matching(np.zeros(len(node.actions)), node.legal_action_mask.astype(bool))

        history.append(average_strategy)

    return history



def main() -> None:
    pass


if __name__ == '__main__':
    main()
