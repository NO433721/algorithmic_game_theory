#!/usr/bin/env python3
from week07 import *
import numpy as np

def regret_matching(regrets, legal_mask):
    positive = np.where(legal_mask, np.maximum(regrets, 0.0), 0.0)

    if positive.sum() > 0:
        return positive / positive.sum()
    
    strategy = legal_mask.astype(float)

    return strategy / strategy.sum()

def make_strategy(players, info_sets, player_info_sets, regrets):
        strategy = {player: {} for player in players}

        for player in players:
            for info_set in player_info_sets[player]:
                node = info_sets[info_set][0]
                legal_mask = node.legal_action_mask.astype(bool)

                strategy[player][info_set] = regret_matching(regrets[player][info_set], legal_mask)

        return strategy

def cfr_traverse(node, pid, strategy, reach, chance_reach, strategy_sums, averaging_weight, average_visited, local_regrets):
        if node.is_terminal:
            return node.payoffs[pid]
        
        if node.is_chance:
            value = 0.0
            for action, child in node.children.items():
                prob = node.chance_strategy[action]
                value += prob * cfr_traverse(child, pid, strategy, reach, chance_reach * prob, strategy_sums, averaging_weight, average_visited, local_regrets)
            
            return value

        player = node.player
        info_set = node.info_set
        local_strategy = strategy[player][info_set]

        action_values = np.zeros(len(node.actions))

        for action, child in node.children.items():
            next_reach = reach.copy()
            next_reach[player] *= local_strategy[action]

            action_values[action] = cfr_traverse(child, pid, strategy, next_reach, chance_reach, strategy_sums, averaging_weight, average_visited, local_regrets)

        node_value = np.dot(local_strategy, action_values)

        if player == pid:
            key = (player, info_set)

            if key not in average_visited:
                strategy_sums[player][info_set] += (averaging_weight * reach[player] * local_strategy)
            average_visited.add(key)

            counterfactual_reach = chance_reach

            for other_player in range(len(reach)):
                if other_player != player:
                    counterfactual_reach *= reach[other_player]

            legal = node.legal_action_mask.astype(bool)

            local_regrets[info_set][legal] += (counterfactual_reach * (action_values[legal] - node_value))

        return node_value

def cfr(root: Node, info_sets, num_iter: int):
    """Run the CFR algorithm for a given number of iterations."""
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
        strategy = make_strategy(players, info_sets, player_info_sets, cumulative_regrets)

        for player in players:
            cfr_traverse(root, player, strategy, np.ones(len(players)), 1.0, strategy_sums, 1.0, set(), cumulative_regrets[player])

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
        
def cfr_plus(root: Node, info_sets, num_iter: int):
    """Run the CFR+ algorithm for a given number of iterations."""
    
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

    for i in range(1, num_iter+1):

        for player in players:
            strategy = make_strategy(players, info_sets, player_info_sets, cumulative_regrets)

            local_regrets = {
            info_set: np.zeros_like(cumulative_regrets[player][info_set])
            for info_set in player_info_sets[player]
            }

            cfr_traverse(root, player, strategy, np.ones(len(players)), 1.0, strategy_sums, i, set(), local_regrets)

            for info_set in player_info_sets[player]:
                cumulative_regrets[player][info_set] = np.maximum(
                    cumulative_regrets[player][info_set]
                    + local_regrets[info_set],
                    0.0
                )

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


def main() -> None:
    pass


if __name__ == '__main__':
    main()
