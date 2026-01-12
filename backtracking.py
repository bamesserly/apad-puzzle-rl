"""Backtracking solver for A-Puzzle-A-Day using exact cover / constraint satisfaction.

Uses the APADEnv environment with backtracking search and heuristics:
- MRV (Minimum Remaining Values): try pieces with fewest valid placements first
- Forward checking: prune branches that create unsolvable island configurations
- Lightweight state management for fast backtracking
"""

import time

import numpy as np

from apad_env import APADEnv, has_islands


def solve(month=None, day=None, verbose=False):
    """Solve the puzzle using backtracking search.

    Args:
        month: Month (1-12) or None for random
        day: Day (1-31) or None for random
        verbose: Print search progress

    Returns:
        List of actions (integers) that solve the puzzle, or None if unsolvable
    """
    result = solve_with_stats(month, day, verbose)
    return result["actions"] if result["solved"] else None


def solve_with_stats(month=None, day=None, verbose=False):
    """Solve with detailed statistics.

    Returns:
        dict with keys: 'solved' (bool), 'actions' (list or None),
                       'nodes_explored' (int), 'time_ms' (float),
                       'backtracks' (int)
    """
    env = APADEnv(month, day)
    env.reset()

    stats = {"nodes_explored": 0, "backtracks": 0, "start_time": time.time()}

    actions = []
    solved = _backtrack(env, actions, stats, verbose)

    stats["time_ms"] = (time.time() - stats["start_time"]) * 1000
    del stats["start_time"]

    return {
        "solved": solved,
        "actions": actions if solved else None,
        "nodes_explored": stats["nodes_explored"],
        "backtracks": stats["backtracks"],
        "time_ms": stats["time_ms"],
    }


def _backtrack(env, actions, stats, verbose):
    """Recursive backtracking search.

    Args:
        env: Current environment state
        actions: List of actions taken so far (modified in place)
        stats: Dict to track statistics
        verbose: Print progress

    Returns:
        True if solved, False otherwise
    """
    stats["nodes_explored"] += 1

    # Base case: all pieces placed
    if np.sum(env.remaining_pieces) == 0:
        return True

    # Get available pieces
    available_pieces = np.where(env.remaining_pieces)[0]

    # MRV heuristic: try piece with fewest valid placements first
    piece_counts = []
    for piece_id in available_pieces:
        count = _count_valid_actions_for_piece(env, piece_id)
        if count == 0:
            # No valid moves for this piece = dead end
            return False
        piece_counts.append((count, piece_id))

    # Sort by count (ascending) - try most constrained piece first
    piece_counts.sort()

    # Try pieces in MRV order
    for count, piece_id in piece_counts:
        valid_actions = _get_valid_actions_for_piece(env, piece_id)

        if verbose:
            remaining = np.sum(env.remaining_pieces)
            print(
                f"Depth {8-remaining}: trying piece {env.piece_names[piece_id]} "
                f"({len(valid_actions)} placements)"
            )

        # Try each valid action for this piece
        for action in valid_actions:
            # Save state
            saved_state = env.save_state()

            # Make move
            env.step(action)
            actions.append(action)

            # Prune if creates unsolvable islands
            if has_islands(env.grid):
                # Undo move
                env.load_state(saved_state)
                actions.pop()
                stats["backtracks"] += 1
                continue

            # Recurse
            if _backtrack(env, actions, stats, verbose):
                return True

            # Backtrack
            env.__dict__.update(saved_state.__dict__)
            actions.pop()
            stats["backtracks"] += 1

    return False


def _count_valid_actions_for_piece(env, piece_id):
    """Count valid placements for a specific piece."""
    mask = env.action_masks()
    count = 0

    # Calculate action range for this piece
    # Action encoding: piece_id * (2 * 4 * 43) + ...
    start_action = piece_id * (2 * 4 * 43)
    end_action = start_action + (2 * 4 * 43)

    for action in range(start_action, end_action):
        if mask[action]:
            count += 1

    return count


def _get_valid_actions_for_piece(env, piece_id):
    """Get list of valid actions for a specific piece."""
    mask = env.action_masks()
    valid_actions = []

    start_action = piece_id * (2 * 4 * 43)
    end_action = start_action + (2 * 4 * 43)

    for action in range(start_action, end_action):
        if mask[action]:
            valid_actions.append(action)

    return valid_actions


def benchmark_dates(n_samples=10, verbose=False):
    """Benchmark solver on random dates.

    Args:
        n_samples: Number of random dates to test
        verbose: Print individual solve details

    Returns:
        dict with statistics: mean_time_ms, median_time_ms, max_time_ms,
                             success_rate, mean_nodes, mean_backtracks
    """
    import random

    times = []
    nodes = []
    backtracks = []
    solved_count = 0

    for i in range(n_samples):
        month = random.randint(1, 12)
        day = random.randint(1, 31)

        if verbose:
            print(f"\n[{i+1}/{n_samples}] Solving {month}/{day}...")

        result = solve_with_stats(month, day, verbose=False)

        if result["solved"]:
            solved_count += 1
            times.append(result["time_ms"])
            nodes.append(result["nodes_explored"])
            backtracks.append(result["backtracks"])

            if verbose:
                print(
                    f"  ✓ Solved in {result['time_ms']:.1f}ms "
                    f"({result['nodes_explored']} nodes, {result['backtracks']} backtracks)"
                )
        else:
            if verbose:
                print("  ✗ No solution found")

    if not times:
        return {"success_rate": 0.0, "error": "No puzzles solved"}

    return {
        "success_rate": solved_count / n_samples,
        "mean_time_ms": np.mean(times),
        "median_time_ms": np.median(times),
        "max_time_ms": np.max(times),
        "min_time_ms": np.min(times),
        "mean_nodes": np.mean(nodes),
        "mean_backtracks": np.mean(backtracks),
    }
