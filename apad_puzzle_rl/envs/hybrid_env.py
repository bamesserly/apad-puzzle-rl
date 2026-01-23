"""Hybrid RL-Solver environment for early-game training."""

import numpy as np

from apad_puzzle_rl.envs.apad_env import APADEnv, has_islands
from apad_puzzle_rl.solver.core import solve_from_env


class HybridAPADEnv(APADEnv):
    """Agent places N pieces, solver checks if remaining puzzle is solvable."""

    def __init__(
        self,
        mon: int | None = None,
        day: int | None = None,
        agent_pieces: int = 5,
        mask_islands: bool = False,
    ):
        super().__init__(mon, day, mask_islands=mask_islands)
        self.agent_pieces = agent_pieces

    def step(self, action):
        info = {}

        if not self._place_piece(action):
            return self._get_obs(), -20, False, True, info

        self._cached_action_masks = None
        self._cached_action_masks = self.action_masks()

        n_remaining = np.sum(self.remaining_pieces)
        pieces_placed = 8 - n_remaining

        # Win: all pieces placed
        if n_remaining == 0:
            reward, terminated, truncated = 1.0, True, False

        # Agent finished placement - check solvability
        elif pieces_placed >= self.agent_pieces:
            checkpoint = self.save_state()
            result = solve_from_env(self, find_all=False)
            self.load_state(checkpoint)

            reward = 1.0 if result["solved"] else 0.0
            terminated, truncated = True, False
            info["solver_time_ms"] = result["time_ms"]
            info["solver_nodes"] = result["nodes_explored"]

        # Lost: no valid moves
        elif not np.any(self._cached_action_masks):
            reward, terminated, truncated = 0.0, False, True

        # Created islands - penalize and end
        elif has_islands(self.grid):
            reward = -0.1 if n_remaining > 5 else 0.0
            terminated, truncated = False, True

        # Valid move
        else:
            reward, terminated, truncated = 0.01, False, False

        info["action_mask"] = self._cached_action_masks
        obs = self._get_obs()

        if terminated:
            info["terminal_observation"] = obs

        self.episode_reward += reward

        if terminated or truncated:
            info["r"] = self.episode_reward
            info["l"] = pieces_placed

        return obs, reward, terminated, truncated, info
