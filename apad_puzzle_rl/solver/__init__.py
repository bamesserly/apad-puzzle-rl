"""Backtracking solver for A-Puzzle-A-Day using exact cover / constraint satisfaction.

Uses the APADEnv environment with backtracking search and heuristics:
- MRV (Minimum Remaining Values): try pieces with fewest valid placements first
- Forward checking: prune branches that create unsolvable island configurations
- Transposition table: cache visited board states to avoid redundant work
- Lightweight state management for fast backtracking
"""

from .core import benchmark_dates, solve, solve_with_stats

__all__ = ["solve", "solve_with_stats", "benchmark_dates"]
