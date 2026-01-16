"""Core backtracking search algorithm."""

import numpy as np

from apad_env import APADEnv

from .observers import (
    FindAllSolutionsObserver,
    ManualSteppingObserver,
    TreeVisualizationObserver,
    VerboseObserver,
)
from .state import SearchConfig, SearchState

try:
    from IPython.display import display
    from ipywidgets import Output, VBox

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


def solve(month=None, day=None, verbose=False, manual_mode=False, find_all=False, verbose_depth=3):
    """Solve the puzzle using backtracking search.

    Args:
        month: Month (1-12) or None for random
        day: Day (1-31) or None for random
        verbose: Print search progress
        manual_mode: Pause at each step for manual stepping (press Enter)
        find_all: If True, find all solutions instead of stopping at first
        verbose_depth: Max depth to show verbose output (default 3, use 999 for all)

    Returns:
        If find_all=False: List of actions or None
        If find_all=True: List of solution lists (empty if no solutions)
    """
    result = solve_with_stats(month, day, verbose, manual_mode, find_all, verbose_depth)
    return result["solutions"] if find_all else (result["actions"] if result["solved"] else None)


def solve_with_stats(
    month=None, day=None, verbose=False, manual_mode=False, find_all=False, verbose_depth=3
):
    """Solve with detailed statistics.

    Returns:
        dict with keys: 'solved' (bool), 'actions' (list or None),
                       'solutions' (list of lists if find_all=True),
                       'num_solutions' (int if find_all=True),
                       'nodes_explored' (int), 'time_ms' (float),
                       'backtracks' (int), 'transposition_hits' (int)
    """
    env = APADEnv(month, day)
    env.reset()

    config = SearchConfig(find_all=find_all, verbose_depth=verbose_depth)
    state = SearchState(config)

    # Set up observers
    observers = []
    tree_obs = TreeVisualizationObserver()
    observers.append(tree_obs)

    # Create display outputs if needed
    display_outputs = None
    if HAS_IPYTHON and (verbose or manual_mode or find_all):
        solutions_output = Output()
        tree_output = Output()
        with solutions_output:
            print("=== SOLUTIONS ===" if find_all else "")
        display(VBox([solutions_output, tree_output]))
        display_outputs = {"solutions": solutions_output, "tree": tree_output}

    # Add appropriate observers based on mode
    if find_all:
        observers.append(
            FindAllSolutionsObserver(env, display_outputs["solutions"] if display_outputs else None)
        )

    if verbose:
        observers.append(
            VerboseObserver(
                env,
                tree_obs,
                state.get_stats,
                verbose_depth,
                display_outputs["tree"] if display_outputs else None,
            )
        )

    if manual_mode:
        observers.append(
            ManualSteppingObserver(
                env, tree_obs, state.get_stats, display_outputs["tree"] if display_outputs else None
            )
        )

    # Run search
    solved = _backtrack(env, state, observers)

    # Build result
    stats = state.get_stats()
    result = {
        "solved": solved if not find_all else len(state.solutions) > 0,
        "actions": state.actions if (solved and not find_all) else None,
        "nodes_explored": stats["nodes_explored"],
        "backtracks": stats["backtracks"],
        "transposition_hits": stats["transposition_hits"],
        "time_ms": stats["time_ms"],
    }

    if find_all:
        result["solutions"] = state.solutions
        result["num_solutions"] = len(state.solutions)

    # Keep widgets alive
    if display_outputs is not None:
        result["_display_outputs"] = display_outputs

    return result


def _backtrack(env, state, observers=None, depth=0):
    """Core backtracking search algorithm.

    Args:
        env: Environment (mutated during search)
        state: SearchState (tracks stats, actions, solutions, visited_states)
        observers: List of SearchObserver instances
        depth: Current search depth

    Returns:
        True if should stop searching (found solution in find-one mode)
    """
    observers = observers or []
    state.record_node()

    # Base case: solution found
    if np.sum(env.remaining_pieces) == 0:
        if state.config.find_all:
            for obs in observers:
                obs.on_solution_found(state.actions, len(state.solutions) + 1)
        return state.handle_solution()

    # Get pieces by MRV heuristic
    available = np.where(env.remaining_pieces)[0]
    piece_order = _get_mrv_piece_order(env, available)

    if isinstance(piece_order, tuple):  # Dead end
        for obs in observers:
            obs.on_backtrack(depth, "dead_end")
        return False

    pieces_to_try = [piece_order[0]] if (depth == 0 and state.config.find_all) else piece_order

    for piece_id, placements in pieces_to_try:
        # Notify tree observers about piece start
        tree_obs = _get_tree_observer(observers)
        if tree_obs:
            tree_obs.on_node_visited(depth, piece_id, 0, len(placements))

        for idx, (chirality, rotation, position) in enumerate(placements):
            for obs in observers:
                obs.on_node_visited(depth, piece_id, idx, len(placements))

            # Try placement
            action = env.encode_action(piece_id, chirality, rotation, position)
            checkpoint = env.save_state()
            env.step(action)
            state.push_action(action)

            # Check if should prune
            should_prune, reason = state.check_prune(env.grid)
            if should_prune:
                _undo_move(state, env, checkpoint)
                for obs in observers:
                    obs.on_backtrack(depth, reason)
                if tree_obs:
                    tree_obs.clear_deeper_subtrees(depth)
                continue

            # Clear stale tree data before recursing
            if tree_obs:
                tree_obs.clear_deeper_subtrees(depth)

            # Recurse
            if _backtrack(env, state, observers, depth + 1):
                return True

            # Backtrack from failed subtree
            _undo_move(state, env, checkpoint)
            for obs in observers:
                obs.on_backtrack(depth, "subtree_failed")
            if tree_obs:
                tree_obs.clear_deeper_subtrees(depth)

        for obs in observers:
            obs.on_piece_exhausted(depth, piece_id)

    return False


def _undo_move(state, env, checkpoint):
    """Undo a move and record backtrack."""
    state.pop_action()
    env.load_state(checkpoint)
    state.record_backtrack()


def _get_mrv_piece_order(env, available_pieces):
    """Get pieces ordered by MRV heuristic (fewest valid placements first).

    Returns:
        list of (piece_id, placements) tuples in MRV order, or (None, piece_name) if dead end.
        placements is a list of (chirality, rotation, position) tuples.
    """
    piece_data = []
    for piece_id in available_pieces:
        placements = env.get_valid_placements(piece_id)
        if not placements:
            return (None, env.piece_names[piece_id])  # Dead end
        piece_data.append((len(placements), piece_id, placements))

    # Sort by count (ascending) - try most constrained piece first
    piece_data.sort()
    return [(pid, placements) for (_, pid, placements) in piece_data]


def _get_tree_observer(observers):
    """Find TreeVisualizationObserver in observers list."""
    for obs in observers:
        if isinstance(obs, TreeVisualizationObserver):
            return obs
    return None


def solve_from_env(env, find_all=False):
    """Solve puzzle starting from current env state.

    Args:
        env: APADEnv instance with some pieces already placed
        find_all: Whether to find all solutions

    Returns:
        dict with 'solved', 'num_solutions' (if find_all), 'time_ms', 'nodes_explored'
    """
    config = SearchConfig(find_all=find_all, verbose_depth=0)
    state = SearchState(config)

    solved = _backtrack(env, state, observers=[], depth=0)

    stats = state.get_stats()
    result = {
        "solved": solved if not find_all else len(state.solutions) > 0,
        "time_ms": stats["time_ms"],
        "nodes_explored": stats["nodes_explored"],
    }

    if find_all:
        result["num_solutions"] = len(state.solutions)

    return result


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
