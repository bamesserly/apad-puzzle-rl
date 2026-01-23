"""Curriculum learning environment using known solution paths."""

import random

from apad_puzzle_rl.envs.apad_env import APADEnv


class CurriculumAPADEnv(APADEnv):
    """Train agent on partial board states from known solutions.

    Agent sees board state with M pieces already placed (from a valid solution path),
    then must choose which of the remaining N pieces to place next. Reward +1 if the
    chosen action is one that continues along any valid solution path, 0 otherwise.
    """

    # Load solutions once at class level
    _solutions = None

    @classmethod
    def _load_solutions(cls):
        if cls._solutions is None:
            from apad_puzzle_rl.resources.solutions_4_14 import SOLUTIONS

            cls._solutions = SOLUTIONS

    def __init__(self, mon=4, day=14, pieces_remaining=2, replay_prob=0.0):
        """
        Args:
            mon: Month (only 4 supported currently)
            day: Day (only 14 supported currently)
            pieces_remaining: Curriculum level - how many pieces agent places (2-7)
            replay_prob: Probability of replaying easier levels (0.0 = no replay)
        """
        super().__init__(mon, day, mask_islands=False)
        self.pieces_remaining = pieces_remaining
        self.current_level = pieces_remaining
        self.replay_prob = replay_prob
        self.valid_actions = set()
        self._load_solutions()

    def set_curriculum_level(self, pieces_remaining):
        """Update curriculum difficulty."""
        self.pieces_remaining = pieces_remaining
        self.current_level = pieces_remaining

    def reset(self, seed=None, options=None):
        """Reset to random partial board state from a valid solution."""
        super().reset(seed=seed)

        # Optionally replay from easier levels
        if self.replay_prob > 0 and random.random() < self.replay_prob:
            self.pieces_remaining = random.choice(range(2, self.current_level))
        else:
            self.pieces_remaining = self.current_level

        # Choose random solution
        solution = random.choice(self._solutions)

        # Determine how many pieces to pre-place
        pieces_to_place = 8 - self.pieces_remaining  # e.g., if 2 remaining, place 6

        # Randomly select which pieces to place (not always the first N)
        all_indices = list(range(8))
        random.shuffle(all_indices)
        indices_to_place = sorted(all_indices[:pieces_to_place])
        indices_remaining = all_indices[pieces_to_place:]

        # Place selected pieces in order
        for idx in indices_to_place:
            self._place_piece(solution[idx])

        # Store valid next actions (any remaining piece from this solution)
        self.valid_actions = set(solution[idx] for idx in indices_remaining)

        # Update cached masks
        self._cached_action_masks = None
        self._cached_action_masks = self.action_masks()

        info = {"action_mask": self._cached_action_masks}
        return self._get_obs(), info

    def step(self, action):
        """Agent places one piece. Episode ends after single action."""
        info = {}

        # Attempt to place piece
        if not self._place_piece(action):
            # Invalid placement (shouldn't happen with masking)
            return self._get_obs(), -20, False, True, info

        # Check if action was valid (in the solution set)
        action_int = int(action) if hasattr(action, "__iter__") else action
        reward = 1.0 if action_int in self.valid_actions else 0.0

        # Episode always terminates after one action
        terminated, truncated = True, False

        # Update masks (though episode is ending)
        self._cached_action_masks = None
        self._cached_action_masks = self.action_masks()
        info["action_mask"] = self._cached_action_masks

        obs = self._get_obs()
        info["terminal_observation"] = obs

        self.episode_reward += reward
        info["r"] = self.episode_reward
        info["l"] = 1  # Always 1 piece placed per episode

        return obs, reward, terminated, truncated, info
