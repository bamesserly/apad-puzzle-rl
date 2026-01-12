import time
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from apad_env import APADEnv


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
