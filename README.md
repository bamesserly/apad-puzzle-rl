# Solving A-Puzzle-A-Day by DragonFjord with Reinforcement Learning

## Intro

[Here's the puzzle](https://www.dragonfjord.com/product/a-puzzle-a-day/).

<img src="puzzle.jpg" alt="My puzzle, ready for an April 14th solution" width="400"/>

The idea is to arrange the irregular group of pentominos on the grid to reveal only the current month and date.

This is an [exact cover](https://en.wikipedia.org/wiki/Exact_cover) problem, closely related to its [pentominos instance](https://en.wikipedia.org/wiki/Exact_cover#Pentomino_tiling).

[Here's a reddit post](https://www.reddit.com/r/puzzles/comments/t9uejy/analysis_of_a_puzzle_a_day/) about using [Knuth's Algorithm X](https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X) to brute force it.

Let's see what we can do with RL.

I'll use deep Q learning via sb3 with a custom gymnasium environment

## Environment

Environment is defined here: [apad_env.py](apad_env.py).

[apad_env_validation.ipynb](apad_env_validation.ipynb) contains sanity checks (e.g. same board cell is not allowed to be occupied by two pieces).

Sample board state

<img src="board_state.png" alt="" width="400"/>

Each piece has a bool chirality state and 4 rotation states

<img src="single_piece.png" alt="" width="200"/>

## RL Strategy

Place random moves, mask invalid moves from the action space, and train with Masked PPO.

Training is done in [apad_MPPO_train.ipynb](apad_MPPO_train.ipynb).

```
model = MaskablePPO(
    "MlpPolicy",
    env,
    n_steps = 128,
    tensorboard_log="./maskable_logs/",
    verbose=1,
)
```

Rewards: win +3, valid move +1, lose/brick game -2.

## Notes

Highlights of the development history:
- Env originally considered *any* two open cells a win.
- A few key improvements enabled a solved model: (1) PPO -> MaskedPPO filtering invalid moves, (2) early exit when the game is bricked (`has-islands()`).
- Apparently a bug in PPO for envs with large action spaces leads to policy probility sum not being normalized, called a Simplex constraint violation. Turning off all validations leads to solving the env in ~15k steps. Fixing just the bug in torch leads to solving the env in 20k steps about 50% of the time, and getting stuck at 7/8 pieces placed the other 50% of the time.
- Env now accepts a month and or day or neither to constrain solutions.
- Not able to reproduce, even with no constraint, our quick success and training before these modifications. Gotta revert to see if I can understand the issue.
- Nowadays, we're at 0.01% success rate in 21 hours and 475,000 random episodes. Felt like it was higher in prev commits.

### Environment

 conda create --name rl_2025 python=3.10
 conda activate rl_2025
 conda install pytorch torchvision torchaudio -c pytorch
 pip install stable-baselines3[extra]
 conda install swig
 pip install gymnasium[all]
 pip install tensorboard
 pip install jupyter
 pip install sb3-contrib
