"""Backtracking solver for A-Puzzle-A-Day using exact cover / constraint satisfaction.

Uses the APADEnv environment with backtracking search and heuristics:
- MRV (Minimum Remaining Values): try pieces with fewest valid placements first
- Forward checking: prune branches that create unsolvable island configurations
- Lightweight state management for fast backtracking
"""

import time

import numpy as np

from apad_env import APADEnv, has_islands

try:
    from IPython.display import clear_output

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


def _wait_or_auto(auto_state, delay=0.1):
    """Wait for user input or auto-continue with delay."""
    if auto_state["enabled"]:
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            auto_state["enabled"] = False
            print("\n[Auto-play stopped]")
    else:
        response = input("[Enter to step, 'a' to auto] ")
        if response.strip().lower() == "a":
            auto_state["enabled"] = True
            print("[Auto-play enabled, use notebook 'Stop' button to pause]")


def solve(month=None, day=None, verbose=False, manual_mode=False):
    """Solve the puzzle using backtracking search.

    Args:
        month: Month (1-12) or None for random
        day: Day (1-31) or None for random
        verbose: Print search progress
        manual_mode: Pause at each step for manual stepping (press Enter)

    Returns:
        List of actions (integers) that solve the puzzle, or None if unsolvable
    """
    result = solve_with_stats(month, day, verbose, manual_mode)
    return result["actions"] if result["solved"] else None


def solve_with_stats(month=None, day=None, verbose=False, manual_mode=False):
    """Solve with detailed statistics.

    Returns:
        dict with keys: 'solved' (bool), 'actions' (list or None),
                       'nodes_explored' (int), 'time_ms' (float),
                       'backtracks' (int)
    """
    env = APADEnv(month, day)
    env.reset()

    stats = {"nodes_explored": 0, "backtracks": 0, "start_time": time.time()}
    tree_state = {
        "path": [],  # Current search path: [(piece_id, action_idx, total_actions)]
        "subtree_nodes": {},  # (depth, piece_id) -> nodes explored in this subtree
        "exhausted": {},  # (depth, piece_id) -> True if fully explored
    }

    actions = []
    auto_state = {"enabled": False} if manual_mode else None
    solved = _backtrack(env, actions, stats, verbose, manual_mode, auto_state, tree_state, depth=0)

    stats["time_ms"] = (time.time() - stats["start_time"]) * 1000
    del stats["start_time"]

    return {
        "solved": solved,
        "actions": actions if solved else None,
        "nodes_explored": stats["nodes_explored"],
        "backtracks": stats["backtracks"],
        "time_ms": stats["time_ms"],
    }


def _clear_deeper_subtrees(tree_state, depth):
    """Remove tree tracking data for depths > depth."""
    keys_to_remove = [k for k in tree_state["subtree_nodes"].keys() if k[0] > depth]
    for k in keys_to_remove:
        del tree_state["subtree_nodes"][k]
    keys_to_remove = [k for k in tree_state["exhausted"].keys() if k[0] > depth]
    for k in keys_to_remove:
        del tree_state["exhausted"][k]


def _display_search_state(
    env, tree_state, stats, status_msg, manual_mode, auto_state, verbose=False
):
    """Handle display logic: clear output, print status, show tree, optionally visualize."""
    # Only display if manual_mode or (verbose and have message)
    if not manual_mode and not verbose:
        return  # Silent mode

    if not status_msg and not manual_mode:
        return  # Nothing to display in verbose mode

    if HAS_IPYTHON and (manual_mode or status_msg):
        clear_output(wait=True)

    if status_msg:
        print(status_msg)

    print("\n" + _render_tree(env, tree_state, stats))
    print()

    if manual_mode:
        env.visualize()
        _wait_or_auto(auto_state)


def _update_tree_path(tree_state, depth, piece_id, action_idx, total_actions):
    """Update the current search path in tree state."""
    if len(tree_state["path"]) > depth:
        tree_state["path"][depth] = (piece_id, action_idx, total_actions)
    else:
        tree_state["path"].append((piece_id, action_idx, total_actions))


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
            return (None, env.piece_names[piece_id])  # Dead end: return which piece caused it
        piece_data.append((len(placements), piece_id, placements))

    # Sort by count (ascending) - try most constrained piece first
    piece_data.sort()
    return [(pid, placements) for (_, pid, placements) in piece_data]


def _render_tree(env, tree_state, stats):
    """Render the search tree visualization."""
    lines = [
        f"Search Progress ({stats['nodes_explored']} nodes, {stats['backtracks']} backtracks):"
    ]

    # Group subtree data by depth
    depth_data = {}  # depth -> {piece_id: (nodes, exhausted)}
    for (d, pid), nodes in tree_state["subtree_nodes"].items():
        if d not in depth_data:
            depth_data[d] = {}
        exhausted = tree_state["exhausted"].get((d, pid), False)
        depth_data[d][pid] = (nodes, exhausted)

    # Render tree recursively
    def render_depth(depth, prefix, is_last_child):
        if depth not in depth_data:
            return

        pieces_at_depth = list(depth_data[depth].items())
        for i, (piece_id, (nodes, exhausted)) in enumerate(pieces_at_depth):
            is_last = i == len(pieces_at_depth) - 1

            # Determine tree characters
            if depth == 0:
                connector = "└─" if is_last else "├─"
            else:
                connector = "└─" if is_last else "├─"

            # Build the line
            piece_name = env.piece_names[piece_id]
            status = (
                "✗"
                if exhausted
                else "(ACTIVE)"
                if depth < len(tree_state["path"]) and tree_state["path"][depth][0] == piece_id
                else "(current)"
            )

            # Check if this piece is in current path
            in_path = depth < len(tree_state["path"]) and tree_state["path"][depth][0] == piece_id

            # Add current action info if at leaf of current path
            extra = ""
            if in_path and depth == len(tree_state["path"]) - 1 and depth < len(tree_state["path"]):
                action_idx, total_actions = (
                    tree_state["path"][depth][1],
                    tree_state["path"][depth][2],
                )
                extra = f" (trying placement {action_idx+1}/{total_actions})"

            line = f"{prefix}{connector} {piece_name}"
            if depth == 0:
                line += f" [root #{piece_id+1}/8]"
            line += f": {nodes} paths {status}{extra}"
            lines.append(line)

            # Recurse to children if this piece is in current path
            if in_path and depth + 1 in depth_data:
                new_prefix = prefix + ("   " if is_last else "│  ")
                render_depth(depth + 1, new_prefix, is_last)

    render_depth(0, "", False)

    # Show untried root pieces
    tried_roots = set(depth_data.get(0, {}).keys())
    untried_count = 8 - len(tried_roots)
    if untried_count > 0:
        lines.append(f"└─ [{untried_count} more root pieces untried]")

    return "\n".join(lines)


def _backtrack(env, actions, stats, verbose, manual_mode, auto_state, tree_state, depth):
    """Recursive backtracking search.

    Args:
        env: Current environment state
        actions: List of actions taken so far (modified in place)
        stats: Dict to track statistics
        verbose: Print progress
        manual_mode: Show visualizations (starts in auto-play, Ctrl+C to pause)
        auto_state: Shared dict for auto-play toggle state
        tree_state: Dict tracking search tree structure
        depth: Current depth in search tree

    Returns:
        True if solved, False otherwise
    """
    stats["nodes_explored"] += 1

    # Base case: all pieces placed
    if np.sum(env.remaining_pieces) == 0:
        # Always show SOLVED (or if manual/verbose at shallow depth)
        if manual_mode or verbose:
            _display_search_state(
                env,
                tree_state,
                stats,
                "SOLVED! All pieces placed.",
                manual_mode,
                auto_state,
                verbose,
            )
        return True

    # Get available pieces and order by MRV heuristic
    available_pieces = np.where(env.remaining_pieces)[0]
    piece_order = _get_mrv_piece_order(env, available_pieces)

    if isinstance(piece_order, tuple):
        # Dead end: at least one piece has no valid placements
        _, piece_name = piece_order
        # Only display if manual mode or (verbose and depth <= 3)
        if manual_mode or (verbose and depth <= 3):
            _display_search_state(
                env,
                tree_state,
                stats,
                f"DEAD END: Piece {piece_name} has no valid placements",
                manual_mode,
                auto_state,
                verbose,
            )
        return False

    # Try pieces in MRV order
    for piece_id, placements in piece_order:
        # Initialize tracking for this subtree
        key = (depth, piece_id)
        if key not in tree_state["subtree_nodes"]:
            tree_state["subtree_nodes"][key] = 0

        # Clear stale data from previous siblings at this depth
        _clear_deeper_subtrees(tree_state, depth)

        # Display progress in verbose mode (only at depth <=3 to avoid output spam)
        if verbose and not manual_mode and depth <= 3:
            remaining = np.sum(env.remaining_pieces)
            msg = f"Depth {8-remaining}: trying piece {env.piece_names[piece_id]} ({len(placements)} placements)"
            _display_search_state(
                env, tree_state, stats, msg, manual_mode=False, auto_state=None, verbose=True
            )

        # Try each valid placement for this piece
        for placement_idx, (chirality, rotation, position) in enumerate(placements):
            # Update tree path and clear stale deeper subtrees
            _update_tree_path(tree_state, depth, piece_id, placement_idx, len(placements))
            _clear_deeper_subtrees(tree_state, depth)

            # Make move (encode action for tracking)
            action = env.encode_action(piece_id, chirality, rotation, position)
            saved_state = env.save_state()
            env.step(action)
            actions.append(action)

            # Display placement in manual mode
            if manual_mode:
                decoded = env.decode_action_verbose(action)
                remaining = np.sum(env.remaining_pieces)
                msg = f"Depth {8-remaining}: Placed {decoded['piece_name']} (action {action})"
                _display_search_state(env, tree_state, stats, msg, manual_mode, auto_state, verbose)

            # Prune if creates unsolvable islands
            if has_islands(env.grid):
                _clear_deeper_subtrees(tree_state, depth)
                # Only display if manual mode or (verbose and depth <= 3)
                if manual_mode or (verbose and depth <= 3):
                    _display_search_state(
                        env,
                        tree_state,
                        stats,
                        f"ISLANDS DETECTED: Backtracking from {env.piece_names[piece_id]}",
                        manual_mode,
                        auto_state,
                        verbose,
                    )
                env.load_state(saved_state)
                actions.pop()
                stats["backtracks"] += 1
                tree_state["subtree_nodes"][key] += 1
                continue

            # Recurse
            if _backtrack(
                env, actions, stats, verbose, manual_mode, auto_state, tree_state, depth + 1
            ):
                return True

            # Backtrack - clean up and undo move
            tree_state["subtree_nodes"][key] += 1
            tree_state["path"] = tree_state["path"][: depth + 1]
            _clear_deeper_subtrees(tree_state, depth)
            # Only display if manual mode or (verbose and depth <= 3)
            if manual_mode or (verbose and depth <= 3):
                _display_search_state(
                    env,
                    tree_state,
                    stats,
                    f"BACKTRACK: Child search failed, undoing {env.piece_names[piece_id]}",
                    manual_mode,
                    auto_state,
                    verbose,
                )
            env.load_state(saved_state)
            actions.pop()
            stats["backtracks"] += 1

        # Mark this piece as exhausted and remove from path
        tree_state["exhausted"][key] = True
        tree_state["path"] = tree_state["path"][:depth]

    return False


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
