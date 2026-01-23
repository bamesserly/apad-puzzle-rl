"""Observer pattern for search event handling."""

import time

from .display import render_tree

try:
    from IPython.display import clear_output

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


class SearchObserver:
    """Base class for observing search events."""

    def on_node_visited(self, depth, piece_id, placement_idx, total_placements):
        """Called when trying a new placement."""
        pass

    def on_solution_found(self, actions, solution_num):
        """Called when solution found."""
        pass

    def on_backtrack(self, depth, reason):
        """Called when backtracking (reason: 'islands', 'dead_end', 'visited', 'subtree_failed')."""
        pass

    def on_piece_exhausted(self, depth, piece_id):
        """Called when all placements of a piece tried."""
        pass


class VerboseObserver(SearchObserver):
    """Prints search progress up to verbose_depth."""

    def __init__(self, env, tree_obs, stats_provider, verbose_depth=3, display_output=None):
        self.env = env
        self.tree_obs = tree_obs
        self.stats_provider = stats_provider
        self.verbose_depth = verbose_depth
        self.display_output = display_output

    def on_node_visited(self, depth, piece_id, placement_idx, total_placements):
        if depth > self.verbose_depth:
            return

        remaining = sum(self.env.remaining_pieces)
        piece_name = self.env.piece_names[piece_id]
        msg = f"Depth {8-remaining}: trying piece {piece_name} ({total_placements} placements)"

        tree_content = render_tree(
            self.env, self.tree_obs.get_state(), self.stats_provider(), max_depth=self.verbose_depth
        )
        output = f"{msg}\n\n{tree_content}"

        if self.display_output is not None:
            with self.display_output:
                clear_output(wait=True)
                print(output)
        else:
            if HAS_IPYTHON:
                clear_output(wait=True)
            print(output)

    def on_backtrack(self, depth, reason):
        if depth > self.verbose_depth and reason != "dead_end":
            return

        msg_map = {
            "islands": "ISLANDS DETECTED",
            "dead_end": "DEAD END: No valid placements",
            "visited": "Already visited state",
            "subtree_failed": "BACKTRACK: Child search failed",
        }
        msg = msg_map.get(reason, f"Backtrack: {reason}")

        tree_content = render_tree(
            self.env, self.tree_obs.get_state(), self.stats_provider(), max_depth=self.verbose_depth
        )
        output = f"{msg}\n\n{tree_content}"

        if self.display_output is not None:
            with self.display_output:
                clear_output(wait=True)
                print(output)
        else:
            if HAS_IPYTHON:
                clear_output(wait=True)
            print(output)


class TreeVisualizationObserver(SearchObserver):
    """Maintains search tree state for visualization."""

    def __init__(self):
        self.path = []  # Current path: [(piece_id, action_idx, total_actions)]
        self.subtree_nodes = {}  # (depth, piece_id) -> nodes explored
        self.exhausted = {}  # (depth, piece_id) -> True

    def on_node_visited(self, depth, piece_id, placement_idx, total_placements):
        # Update path
        if len(self.path) > depth:
            self.path[depth] = (piece_id, placement_idx, total_placements)
        else:
            self.path.append((piece_id, placement_idx, total_placements))

        # Initialize subtree tracking
        key = (depth, piece_id)
        if key not in self.subtree_nodes:
            self.subtree_nodes[key] = 0

    def on_backtrack(self, depth, reason):
        if len(self.path) > depth:
            key = (depth, self.path[depth][0])
            self.subtree_nodes[key] = self.subtree_nodes.get(key, 0) + 1

    def on_piece_exhausted(self, depth, piece_id):
        self.exhausted[(depth, piece_id)] = True
        self.path = self.path[:depth]

    def clear_deeper_subtrees(self, depth):
        """Remove tree tracking data for depths > depth."""
        keys_to_remove = [k for k in self.subtree_nodes.keys() if k[0] > depth]
        for k in keys_to_remove:
            del self.subtree_nodes[k]
        keys_to_remove = [k for k in self.exhausted.keys() if k[0] > depth]
        for k in keys_to_remove:
            del self.exhausted[k]

    def get_state(self):
        """Return tree state dict for rendering."""
        return {
            "path": self.path,
            "subtree_nodes": self.subtree_nodes,
            "exhausted": self.exhausted,
        }


class ManualSteppingObserver(SearchObserver):
    """Interactive stepping with auto-play toggle."""

    def __init__(self, env, tree_obs, stats_provider, display_output=None):
        self.env = env
        self.tree_obs = tree_obs
        self.stats_provider = stats_provider
        self.display_output = display_output
        self.auto_enabled = True  # Start in auto mode

    def on_node_visited(self, depth, piece_id, placement_idx, total_placements):
        # Display current state
        decoded_action = {
            "piece_name": self.env.piece_names[piece_id],
            "placement": f"{placement_idx+1}/{total_placements}",
        }
        remaining = sum(self.env.remaining_pieces)
        msg = f"Depth {8-remaining}: Placed {decoded_action['piece_name']} ({decoded_action['placement']})"

        tree_viz = render_tree(self.env, self.tree_obs.get_state(), self.stats_provider())
        output = f"{msg}\n\n{tree_viz}"

        if self.display_output is not None:
            with self.display_output:
                clear_output(wait=True)
                print(output)
        else:
            if HAS_IPYTHON:
                clear_output(wait=True)
            print(output)

        self.env.visualize()

        # Wait for user input or auto-continue
        self._wait_or_auto()

    def _wait_or_auto(self, delay=0.1):
        if self.auto_enabled:
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                self.auto_enabled = False
                print("\n[Auto-play stopped]")
        else:
            response = input("[Enter to step, 'a' to auto] ")
            if response.strip().lower() == "a":
                self.auto_enabled = True
                print("[Auto-play enabled, use notebook 'Stop' button to pause]")


class FindAllSolutionsObserver(SearchObserver):
    """Displays solutions as they're found in find_all mode."""

    def __init__(self, env, display_output=None):
        self.env = env
        self.display_output = display_output
        self.start_time = time.time()

    def on_solution_found(self, actions, solution_num):
        elapsed = (time.time() - self.start_time) * 1000
        # Format actions as tuples
        pieces = []
        for action in actions:
            decoded = self.env.decode_action(action)
            pid = decoded["piece_id"]
            c, r, p = decoded["chirality"], decoded["rotation"], decoded["position"]
            pieces.append(f"({pid},{c},{r},{p})")
        action_str = ", ".join(pieces)
        solution_msg = f"SOLUTION #{solution_num} at {elapsed:.1f}ms: {action_str}"

        if self.display_output is not None:
            with self.display_output:
                print(solution_msg)
        else:
            print(solution_msg)
