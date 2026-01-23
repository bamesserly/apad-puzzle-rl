import time
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from apad_puzzle_rl.envs.apad_env import APADEnv


class TimerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def _on_step(self):
        if self.num_timesteps % 1000 == 0:
            elapsed = time.time() - self.start_time
            rate = self.num_timesteps / elapsed
            remaining = (self.locals["total_timesteps"] - self.num_timesteps) / rate
            print(f"Step {self.num_timesteps}, {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
        return True


class GradNormCallback(BaseCallback):
    def _on_step(self):
        if hasattr(self.model.policy, "parameters"):
            total_norm = 0
            for p in self.model.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)

            self.logger.record("train/grad_norm", total_norm)
        return True


class CurriculumCallback(BaseCallback):
    """Up the difficulty upon reaching certain (rolling) mean episode length thresholds"""

    def __init__(self, env, verbose=1):
        super().__init__(verbose=verbose)
        self.env = env
        self.verbose = verbose
        self.thresholds = [7.5, 5]  # episode length required before upping difficulty
        self.current_stage = 0
        self.ep_lengths = deque(maxlen=30)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue

            ep_len = info["episode"]["l"]
            self.ep_lengths.append(ep_len)
            avg_len = sum(self.ep_lengths) / len(self.ep_lengths)
            if (
                self.current_stage < len(self.thresholds)
                and avg_len > self.thresholds[self.current_stage]
            ):
                self.current_stage += 1
                self.env.set_difficulty(self.current_stage)
                if self.verbose:
                    print(
                        f"Average episode length: {avg_len:.2f} — Switched to difficulty: {self.current_stage}"
                    )
        return True


def make_env(mo=None, day=None):
    """Create monitored APAD environment

    Args:
        mo: Month (1-12) or None for random board
        day: Day (1-31) or None for random board

    Returns:
        Monitored APADEnv instance
    """
    if mo is None or day is None:
        env = APADEnv()  # random board
    else:
        env = APADEnv(mo, day)  # fixed board
    return Monitor(env)


def make_hybrid_env(mo=None, day=None, agent_pieces=5, mask_islands=False):
    """Create monitored Hybrid APAD environment

    Args:
        mo: Month (1-12) or None for random board
        day: Day (1-31) or None for random board
        agent_pieces: How many pieces agent places before solver check
        mask_islands: Mask island-creating moves from action space

    Returns:
        Monitored HybridAPADEnv instance
    """
    from apad_puzzle_rl.envs.hybrid_env import HybridAPADEnv

    if mo is None or day is None:
        env = HybridAPADEnv(agent_pieces=agent_pieces, mask_islands=mask_islands)
    else:
        env = HybridAPADEnv(mo, day, agent_pieces=agent_pieces, mask_islands=mask_islands)
    return Monitor(env)


def make_curriculum_env(mo=4, day=14, pieces_remaining=2, replay_prob=0.0):
    """Create monitored Curriculum APAD environment

    Args:
        mo: Month (only 4 supported currently)
        day: Day (only 14 supported currently)
        pieces_remaining: Curriculum level - how many pieces agent places (2-7)
        replay_prob: Probability of replaying easier levels (0.0 = no replay)

    Returns:
        Monitored CurriculumAPADEnv instance
    """
    from apad_puzzle_rl.envs.curriculum_env import CurriculumAPADEnv

    env = CurriculumAPADEnv(mo, day, pieces_remaining=pieces_remaining, replay_prob=replay_prob)
    return Monitor(env)


class CurriculumProgressionCallback(BaseCallback):
    """Automatically progress curriculum when agent masters current level."""

    def __init__(
        self, env, success_threshold=0.85, min_episodes=100, verbose=1, level_thresholds=None
    ):
        super().__init__(verbose=verbose)
        self.env = env
        self.success_threshold = success_threshold
        self.level_thresholds = level_thresholds  # Dict: {pieces_remaining: threshold}
        self.min_episodes = min_episodes
        self.episode_results = deque(maxlen=1000)  # Rolling window

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue

            # Track if episode was successful (reward > 0.5)
            success = info.get("r", 0) > 0.5
            self.episode_results.append(success)

            # Check if ready to progress
            if len(self.episode_results) >= self.min_episodes:
                current = self.env.get_attr("pieces_remaining")[0]
                success_rate = sum(self.episode_results) / len(self.episode_results)
                threshold = (
                    self.level_thresholds.get(current, self.success_threshold)
                    if self.level_thresholds
                    else self.success_threshold
                )

                if success_rate >= threshold and current < 7:
                    new_level = current + 1
                    self.env.env_method("set_curriculum_level", new_level)
                    if self.verbose:
                        print(
                            f"\n🎓 Curriculum advanced: {current} → {new_level} pieces remaining "
                            f"(success rate: {success_rate:.1%})"
                        )
                    self.episode_results.clear()  # Reset tracking for new level

        return True
