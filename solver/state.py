"""Search state management for backtracking solver."""

import time
from dataclasses import dataclass

from apad_env import has_islands


@dataclass
class SearchConfig:
    """Immutable search configuration."""

    find_all: bool = False
    verbose_depth: int = 3


class SearchState:
    """Mutable state tracked during backtracking search."""

    def __init__(self, config: SearchConfig, use_transposition_table: bool = True):
        self.config = config
        self.actions = []
        self.solutions = [] if config.find_all else None
        self.visited_states = set() if use_transposition_table else None

        # Statistics
        self.nodes_explored = 0
        self.backtracks = 0
        self.transposition_hits = 0
        self.start_time = time.time()

    def record_node(self):
        self.nodes_explored += 1

    def record_backtrack(self):
        self.backtracks += 1

    def push_action(self, action):
        self.actions.append(action)

    def pop_action(self):
        self.actions.pop()

    def check_prune(self, grid):
        """Check if state should be pruned. Returns (should_prune, reason)."""
        if self.visited_states is not None:
            state_hash = grid.tobytes()
            if state_hash in self.visited_states:
                self.transposition_hits += 1
                return True, "visited"
            self.visited_states.add(state_hash)

        if has_islands(grid):
            return True, "islands"

        return False, None

    def handle_solution(self):
        """Handle finding a solution. Returns True if should stop search."""
        if self.config.find_all:
            self.solutions.append(self.actions.copy())
            return False  # Continue searching
        return True  # Stop at first solution

    def get_stats(self):
        """Return statistics dict."""
        elapsed_ms = (time.time() - self.start_time) * 1000
        return {
            "nodes_explored": self.nodes_explored,
            "backtracks": self.backtracks,
            "transposition_hits": self.transposition_hits,
            "time_ms": elapsed_ms,
        }
