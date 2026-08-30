#!/usr/bin/env python3

import itertools
from collections.abc import Hashable

import numpy as np
from scipy.optimize import linprog

from week07 import Node, evaluate, traverse_tree


Sequence = tuple[tuple[Hashable, int], ...]
InfoSetData = tuple[Sequence, tuple[int, ...], int]


def _collect_sequence_structure(
    root: Node,
) -> tuple[
    list[list[Sequence]],
    list[dict[Sequence, int]],
    list[dict[Hashable, InfoSetData]],
]:
    """Collect sequences and the parent sequence of every information set."""
    sequences: list[list[Sequence]] = [[()], [()]]
    sequence_indices: list[dict[Sequence, int]] = [{(): 0}, {(): 0}]
    information_sets: list[dict[Hashable, InfoSetData]] = [{}, {}]

    def traverse(node: Node, current_sequences: tuple[Sequence, Sequence]) -> None:
        if node.is_terminal:
            return

        if node.is_chance:
            for child in node.children.values():
                traverse(child, current_sequences)
            return

        player = int(node.player)
        if player not in (0, 1):
            raise ValueError(f'Unsupported player id {player}; expected 0 or 1')

        info_set = node.info_set
        actions = tuple(int(action) for action in node.children)
        action_count = len(node.actions)
        parent_sequence = current_sequences[player]

        previous = information_sets[player].get(info_set)
        current_data = (parent_sequence, actions, action_count)
        if previous is None:
            information_sets[player][info_set] = current_data
        elif previous != current_data:
            raise ValueError(
                f'Information set {info_set!r} violates perfect recall or has '
                'inconsistent legal actions'
            )

        for raw_action, child in node.children.items():
            action = int(raw_action)
            next_sequence = parent_sequence + ((info_set, action),)
            if next_sequence not in sequence_indices[player]:
                sequence_indices[player][next_sequence] = len(sequences[player])
                sequences[player].append(next_sequence)

            child_sequences = list(current_sequences)
            child_sequences[player] = next_sequence
            traverse(child, (child_sequences[0], child_sequences[1]))

    traverse(root, ((), ()))
    return sequences, sequence_indices, information_sets


def _pure_strategies(
    information_sets: dict[Hashable, InfoSetData],
) -> list[dict[Hashable, np.ndarray]]:
    items = list(information_sets.items())
    action_sets = [data[1] for _, data in items]
    strategies: list[dict[Hashable, np.ndarray]] = []

    for selected_actions in itertools.product(*action_sets):
        strategy: dict[Hashable, np.ndarray] = {}
        for (info_set, (_, _, action_count)), action in zip(
            items, selected_actions, strict=True
        ):
            local_strategy = np.zeros(action_count, dtype=np.float64)
            local_strategy[action] = 1.0
            strategy[info_set] = local_strategy
        strategies.append(strategy)

    return strategies


def convert_to_normal_form(root: Node) -> tuple[np.ndarray, np.ndarray]:
    """Convert a two-player extensive-form game to normal form.

    A pure normal-form strategy specifies an action at every information set,
    including information sets that are unreachable under that strategy.
    """
    _, _, information_sets = _collect_sequence_structure(root)
    row_strategies = _pure_strategies(information_sets[0])
    col_strategies = _pure_strategies(information_sets[1])

    row_payoffs = np.empty(
        (len(row_strategies), len(col_strategies)), dtype=np.float64
    )
    col_payoffs = np.empty_like(row_payoffs)

    for row_index, row_strategy in enumerate(row_strategies):
        for col_index, col_strategy in enumerate(col_strategies):
            utility = evaluate(root, {0: row_strategy, 1: col_strategy})
            row_payoffs[row_index, col_index] = utility[0]
            col_payoffs[row_index, col_index] = utility[1]

    return row_payoffs, col_payoffs


def _realization_constraints(
    sequences: list[Sequence],
    sequence_indices: dict[Sequence, int],
    information_sets: dict[Hashable, InfoSetData],
) -> tuple[np.ndarray, np.ndarray]:
    constraints = np.zeros(
        (1 + len(information_sets), len(sequences)), dtype=np.float64
    )
    target = np.zeros(1 + len(information_sets), dtype=np.float64)

    constraints[0, sequence_indices[()]] = 1.0
    target[0] = 1.0

    for row, (info_set, (parent_sequence, actions, _)) in enumerate(
        information_sets.items(), start=1
    ):
        constraints[row, sequence_indices[parent_sequence]] = 1.0
        for action in actions:
            child_sequence = parent_sequence + ((info_set, action),)
            constraints[row, sequence_indices[child_sequence]] = -1.0

    return constraints, target


def _sequence_form_data(
    root: Node,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[list[Sequence]],
    list[dict[Sequence, int]],
    list[dict[Hashable, InfoSetData]],
]:
    sequences, sequence_indices, information_sets = _collect_sequence_structure(root)

    row_payoffs = np.zeros(
        (len(sequences[0]), len(sequences[1])), dtype=np.float64
    )
    col_payoffs = np.zeros_like(row_payoffs)

    def traverse_payoffs(
        node: Node,
        row_sequence: Sequence,
        col_sequence: Sequence,
        chance_reach: float,
    ) -> None:
        if node.is_terminal:
            row_index = sequence_indices[0][row_sequence]
            col_index = sequence_indices[1][col_sequence]
            row_payoffs[row_index, col_index] += chance_reach * float(node.payoffs[0])
            col_payoffs[row_index, col_index] += chance_reach * float(node.payoffs[1])
            return

        if node.is_chance:
            for raw_action, child in node.children.items():
                action = int(raw_action)
                traverse_payoffs(
                    child,
                    row_sequence,
                    col_sequence,
                    chance_reach * float(node.chance_strategy[action]),
                )
            return

        info_set = node.info_set
        for raw_action, child in node.children.items():
            action = int(raw_action)
            if node.player == 0:
                traverse_payoffs(
                    child,
                    row_sequence + ((info_set, action),),
                    col_sequence,
                    chance_reach,
                )
            else:
                traverse_payoffs(
                    child,
                    row_sequence,
                    col_sequence + ((info_set, action),),
                    chance_reach,
                )

    traverse_payoffs(root, (), (), 1.0)

    row_constraints, row_target = _realization_constraints(
        sequences[0], sequence_indices[0], information_sets[0]
    )
    col_constraints, col_target = _realization_constraints(
        sequences[1], sequence_indices[1], information_sets[1]
    )

    return (
        row_payoffs,
        col_payoffs,
        row_constraints,
        row_target,
        col_constraints,
        col_target,
        sequences,
        sequence_indices,
        information_sets,
    )


def convert_to_sequence_form(root: Node) -> tuple[np.ndarray, ...]:
    """Convert a two-player perfect-recall game to sequence form.

    Returns the two sequence payoff matrices ``A`` and ``B``, followed by the
    flow systems ``E @ x = e`` and ``F @ y = f`` for the two realization plans.
    """
    return _sequence_form_data(root)[:6]


def find_nash_equilibrium_sequence_form(root: Node) -> tuple[np.ndarray, np.ndarray]:
    """Find a zero-sum equilibrium as a pair of realization plans."""
    (
        payoff_matrix,
        col_payoff_matrix,
        row_constraints,
        row_target,
        col_constraints,
        col_target,
    ) = convert_to_sequence_form(root)

    if not np.allclose(col_payoff_matrix, -payoff_matrix):
        raise ValueError('Sequence-form equilibrium LP requires a zero-sum game')

    row_sequence_count, col_sequence_count = payoff_matrix.shape
    row_constraint_count = row_constraints.shape[0]
    col_constraint_count = col_constraints.shape[0]

    # max f^T q subject to F^T q <= A^T x and E x = e.
    row_objective = np.concatenate(
        [np.zeros(row_sequence_count), -col_target]
    )
    row_inequalities = np.hstack((-payoff_matrix.T, col_constraints.T))
    row_equalities = np.hstack(
        (
            row_constraints,
            np.zeros((row_constraint_count, col_constraint_count)),
        )
    )
    row_bounds = [(0.0, None)] * row_sequence_count + [
        (None, None)
    ] * col_constraint_count

    row_result = linprog(
        row_objective,
        A_ub=row_inequalities,
        b_ub=np.zeros(col_sequence_count),
        A_eq=row_equalities,
        b_eq=row_target,
        bounds=row_bounds,
        method='highs',
    )
    if not row_result.success:
        raise RuntimeError(f'Row sequence-form LP failed: {row_result.message}')

    # min e^T p subject to A y <= E^T p and F y = f.
    col_objective = np.concatenate(
        [np.zeros(col_sequence_count), row_target]
    )
    col_inequalities = np.hstack((payoff_matrix, -row_constraints.T))
    col_equalities = np.hstack(
        (
            col_constraints,
            np.zeros((col_constraint_count, row_constraint_count)),
        )
    )
    col_bounds = [(0.0, None)] * col_sequence_count + [
        (None, None)
    ] * row_constraint_count

    col_result = linprog(
        col_objective,
        A_ub=col_inequalities,
        b_ub=np.zeros(row_sequence_count),
        A_eq=col_equalities,
        b_eq=col_target,
        bounds=col_bounds,
        method='highs',
    )
    if not col_result.success:
        raise RuntimeError(f'Column sequence-form LP failed: {col_result.message}')

    row_plan = row_result.x[:row_sequence_count]
    col_plan = col_result.x[:col_sequence_count]
    row_plan[np.abs(row_plan) < 1e-12] = 0.0
    col_plan[np.abs(col_plan) < 1e-12] = 0.0
    return row_plan, col_plan


def convert_realization_plan_to_behavioral_strategy(
    root: Node,
    realization_plan: np.ndarray,
    player: int,
) -> dict[Hashable, np.ndarray]:
    """Convert one player's realization plan to a behavioral strategy.

    At information sets that have zero realization probability, the function
    returns the uniform distribution over legal actions.
    """
    if player not in (0, 1):
        raise ValueError(f'Unsupported player id {player}; expected 0 or 1')

    sequences, sequence_indices, information_sets = _collect_sequence_structure(root)
    plan = np.asarray(realization_plan, dtype=np.float64)
    if plan.shape != (len(sequences[player]),):
        raise ValueError(
            f'Expected realization plan shape {(len(sequences[player]),)}, '
            f'got {plan.shape}'
        )

    constraints, target = _realization_constraints(
        sequences[player], sequence_indices[player], information_sets[player]
    )
    if np.any(plan < -1e-8) or not np.allclose(
        constraints @ plan, target, rtol=0.0, atol=1e-7
    ):
        raise ValueError('The supplied vector is not a valid realization plan')

    strategy: dict[Hashable, np.ndarray] = {}
    for info_set, (parent_sequence, actions, action_count) in information_sets[
        player
    ].items():
        local_strategy = np.zeros(action_count, dtype=np.float64)
        parent_probability = plan[sequence_indices[player][parent_sequence]]

        if parent_probability > 1e-12:
            for action in actions:
                child_sequence = parent_sequence + ((info_set, action),)
                child_probability = plan[sequence_indices[player][child_sequence]]
                local_strategy[action] = max(0.0, child_probability) / parent_probability

            # Remove harmless LP roundoff while preserving a distribution.
            local_strategy /= local_strategy.sum()
        else:
            local_strategy[list(actions)] = 1.0 / len(actions)

        strategy[info_set] = local_strategy

    return strategy


def main() -> None:
    from kuhn_poker import KuhnPokerNumpy

    root, _ = traverse_tree(KuhnPokerNumpy())
    row_plan, col_plan = find_nash_equilibrium_sequence_form(root)
    strategies = {
        0: convert_realization_plan_to_behavioral_strategy(root, row_plan, 0),
        1: convert_realization_plan_to_behavioral_strategy(root, col_plan, 1),
    }
    print(evaluate(root, strategies))


if __name__ == '__main__':
    main()
