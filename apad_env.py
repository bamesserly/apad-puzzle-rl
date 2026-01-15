from random import randint

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

PIECES = {
    "K": [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1)],  # __|_ knee           0-343
    "A": [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)],  # __|- Asp            344-686
    "C": [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)],  # C                   687-1030
    "L": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],  # |__  L              1031-1374
    "R": [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # [] rectangle  1375-1718
    "P": [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],  # P                   1719-2062
    "I": [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)],  # |_ I                2063-2406
    "Z": [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],  # Z                   2407-2751
}


# islands of 3, 4, 9, and 14 brick the game.
# 3 or more single islands brick the game.
# a single and a double island brick the game.


# todo: check that islands of 6 are rectangles
def has_islands(grid):
    labeled_array, num_features = label(grid == 0)
    if num_features == 0:
        return False
    island_sizes = np.bincount(labeled_array.ravel())[1:]

    has_bad_islands = np.any(np.isin(island_sizes, [1, 2, 3, 4, 7, 8, 9, 12, 13, 14]))

    singles = island_sizes == 1  # numpy version of [x == 1 for x in island_sizes]

    has_too_many_singles = singles.sum() >= 3

    has_single_and_double = (island_sizes == 2).any() and singles.any()

    return has_bad_islands or has_too_many_singles or has_single_and_double


"""
# this method was hopefully a speedup vs the above, but it doesn't have a significant impact. Maybe + 0.1 games/sec.
def has_islands(grid):
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))  # cardinal directions

    for y in range(h):
        for x in range(w):
            # start a flood fill only on unseen 0-cells
            if grid[y, x] == 0 and not visited[y, x]:
                q = deque([(y, x)])
                visited[y, x] = True
                size = 0

                while q:
                    cy, cx = q.pop()
                    size += 1
                    for dy, dx in nbrs:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if grid[ny, nx] == 0 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                q.append((ny, nx))

                if 2 <= size <= 4:
                    return True
    return False
"""


# None -> random date (randomize on reset)
# <0 -> do not use any date (don't randomize on reset)
# other -> selected date (don't randomize on reset)
class APADEnv(gym.Env):
    def __init__(self, mon=None, day=None, handicap=0):
        super().__init__()

        # piece_id is the index of this array
        self.piece_names = ["K", "A", "C", "L", "R", "P", "I", "Z"]
        self.grid_size = 7
        self.valid_spaces = 43
        self.invalid_positions = {(0, 6), (1, 6), (6, 3), (6, 4), (6, 5), (6, 6)}
        self.handicap = handicap
        self.episode_reward = 0

        # 8 pieces × 2 chirality × 4 rotations × 43 positions = 2752
        # In reality, the number is about half of this, accounting for pieces hitting the walls and chiral/rotation symettries.
        self.action_space = gym.spaces.Discrete(8 * 2 * 4 * 43)

        # Observation: 7x7 grid + 8 remaining pieces
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=8,
            shape=(57,),  # 7x7 + 8
            dtype=np.int8,
        )
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for pos in self.invalid_positions:
            self.grid[pos] = -1

        # mark solution date:
        self.mon = mon
        self.day = day
        # mon = random.randint(1, 12)
        # day = random.randint(1, 31)
        valid_positions = [
            (r, c) for r in range(7) for c in range(7) if (r, c) not in self.invalid_positions
        ]
        if self.mon is None:
            self.grid[valid_positions[randint(1, 12) - 1]] = -1
        elif 1 <= self.mon <= 31:
            self.grid[valid_positions[self.mon - 1]] = -1
        else:
            pass

        if self.day is None:
            self.grid[valid_positions[11 + randint(1, 31)]] = -1
        elif 1 <= self.day <= 31:
            self.grid[valid_positions[11 + self.day]] = -1
        else:
            pass

        self.remaining_pieces = np.ones(8, dtype=bool)
        self._cached_action_masks = None

    def _get_obs(self):
        return np.concatenate([self.grid.flatten(), self.remaining_pieces.astype(np.int8)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_reward = 0

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for pos in self.invalid_positions:
            self.grid[pos] = -1
        valid_positions = [
            (r, c) for r in range(7) for c in range(7) if (r, c) not in self.invalid_positions
        ]
        if self.mon is None:
            self.grid[valid_positions[randint(1, 12) - 1]] = -1
        elif 1 <= self.mon <= 31:
            self.grid[valid_positions[self.mon - 1]] = -1
        else:
            pass

        if self.day is None:
            self.grid[valid_positions[11 + randint(1, 31)]] = -1
        elif 1 <= self.day <= 31:
            self.grid[valid_positions[11 + self.day]] = -1
        else:
            pass

        self.remaining_pieces = np.ones(8, dtype=bool)
        self._cached_action_masks = None
        mask = self.action_masks()
        info = {"action_mask": mask}
        return self._get_obs(), info

    def set_date(self, mon, day):
        self.mon = mon
        self.day = day
        self.reset()

    def set_difficulty(self, level):
        if level == 0:
            self.set_date(-1, -1)
        elif level == 1:
            self.set_date(-1, None)
        elif level == 2:
            self.set_date(None, None)
        else:
            raise
        self.reset()
        return True

    def _get_piece_coords(self, piece_id, chirality, rotation):
        base_coords = PIECES[self.piece_names[piece_id]]
        coords = base_coords.copy()

        # Apply chirality (flip x-coordinates)
        if chirality == 1:
            coords = [(-x, y) for x, y in coords]

        # Apply rotation (0, 90, 180, 270 degrees)
        for _ in range(rotation):
            coords = [(-y, x) for x, y in coords]

        # Normalize to ensure anchor at (0,0)
        min_x = min(x for x, y in coords)
        min_y = min(y for x, y in coords)
        coords = [(x - min_x, y - min_y) for x, y in coords]

        return coords

    def _is_valid_placement(self, piece_id, chirality, rotation, position):
        coords = self._get_piece_coords(piece_id, chirality, rotation)
        row, col = divmod(position, self.grid_size)

        # Vectorize coordinate checks
        coords_array = np.array(coords)
        rows = row + coords_array[:, 0]
        cols = col + coords_array[:, 1]

        # Check bounds
        if np.any((rows < 0) | (rows >= self.grid_size) | (cols < 0) | (cols >= self.grid_size)):
            return False

        # Check occupancy
        return np.all(self.grid[rows, cols] == 0)

    def _place_piece_components(self, piece_id, chirality, rotation, position):
        # Return False if piece already used
        if not self.remaining_pieces[piece_id]:
            return False

        # Return False if placement invalid
        if not self._is_valid_placement(piece_id, chirality, rotation, position):
            return False

        coords = self._get_piece_coords(piece_id, chirality, rotation)
        row, col = divmod(position, self.grid_size)

        # Place the piece
        for dr, dc in coords:
            self.grid[row + dr, col + dc] = piece_id + 1

        self.remaining_pieces[piece_id] = False
        # self._cached_action_masks = None  # Invalidate cache
        return True

    def _place_piece(self, action):
        move = self.decode_action(action)
        return self._place_piece_components(
            move["piece_id"], move["chirality"], move["rotation"], move["position"]
        )

    def step(self, action):
        info = {}

        # This path should never occur with action masking active, but check_env likes us to not throw errors
        if not self._place_piece(action):
            print("ERROR")
            return self._get_obs(), -20, False, True, info

        # Cache masks after state change
        self._cached_action_masks = None
        self._cached_action_masks = self.action_masks()

        n_remaining_pieces = np.sum(self.remaining_pieces)

        if n_remaining_pieces == self.handicap:  # win
            reward, terminated, truncated = +1, True, False
        elif not np.any(self._cached_action_masks):  # lose
            reward, terminated, truncated = 0, False, True
        elif has_islands(self.grid):  # bad move! lose eventually
            reward = -0.1 if n_remaining_pieces > 4 else 0
            terminated, truncated = False, True
        else:  # good move
            # reward = (
            #    0.5 if n_remaining_pieces > 3 else 0
            # )  # (8 - n_remaining_pieces) / 8. if n_remaining_pieces > 3 else 0 # small, per-piece reward for early/mid game
            reward, terminated, truncated = 0.01, False, False

        info["action_mask"] = self._cached_action_masks

        obs = self._get_obs()
        if terminated:
            info["terminal_observation"] = obs

        self.episode_reward += reward

        if terminated or truncated:
            info["r"] = self.episode_reward
            info["l"] = 8 - n_remaining_pieces

        return obs, reward, terminated, truncated, info

    # Important to refresh the mask /during/ the step(), but also called by sb3 inbetween step()s, for which we don't need to recalculate.
    def action_masks(self):
        if self._cached_action_masks is not None:
            return self._cached_action_masks

        # zeroes mean every action is invalid
        mask = np.zeros(self.action_space.n, dtype=bool)
        # print(f'initial # of valid actions {np.flatnonzero(mask).size}')

        # Only check positions that are empty
        # valid_positions = np.where((self.grid == 0).flatten())[0]
        # print(f'initial # of unoccupied cells {len(valid_positions)}')

        # Only check available pieces
        available_pieces = np.where(self.remaining_pieces)[0]
        # print(f'initial # of available pieces {len(available_pieces)}')

        total = 0

        for piece_id in available_pieces:
            for chirality in range(2):
                if self.piece_names[piece_id] == "R" and chirality == 1:
                    continue
                for rotation in range(4):
                    if self.piece_names[piece_id] in ["R", "Z", "C"] and rotation >= 2:
                        continue
                    for position in range(43):
                        total += 1
                        # for position in valid_positions:
                        if self._is_valid_placement(piece_id, chirality, rotation, position):
                            action = self.encode_action(piece_id, chirality, rotation, position)
                            mask[action] = True
        # print(f'final # of valid actions {np.flatnonzero(mask).size}')
        # print(f'total # of actions {total}')

        self._cached_action_masks = mask
        return mask

    def decode_action(self, action):
        piece_id = action // (2 * 4 * 43)
        remaining = action % (2 * 4 * 43)
        chirality = remaining // (4 * 43)
        remaining = remaining % (4 * 43)
        rotation = remaining // 43
        position = remaining % 43

        row, col = divmod(position, 7)

        return {
            "piece_id": piece_id,
            "chirality": chirality,
            "rotation": rotation,
            "position": position,
            "grid_pos": (row, col),
        }

    def decode_action_verbose(self, action):
        move = self.decode_action(action)
        move["piece_name"] = self.piece_names[move["piece_id"]]
        return move

    def encode_action(self, piece_id, chirality, rotation, position):
        return piece_id * (2 * 4 * 43) + chirality * (4 * 43) + rotation * 43 + position

    def visualize(self):
        fig, ax = plt.subplots(figsize=(6, 6))

        display = np.ones((*self.grid.shape, 3))
        display[self.grid == -1] = [0, 0, 0]
        display[self.grid > 0] = [0.5, 0.5, 0.5]

        ax.imshow(display)

        # Grid lines
        for i in range(self.grid.shape[0] + 1):
            ax.axhline(i - 0.5, color="gray", linewidth=1)
        for i in range(self.grid.shape[1] + 1):
            ax.axvline(i - 0.5, color="gray", linewidth=1)

        # Piece outlines
        for piece_id in range(1, 9):
            mask = self.grid == piece_id
            if not mask.any():
                continue

            for r in range(self.grid.shape[0]):
                for c in range(self.grid.shape[1]):
                    if mask[r, c]:
                        if r == 0 or not mask[r - 1, c]:
                            ax.axhline(
                                r - 0.5,
                                xmin=c / self.grid.shape[1],
                                xmax=(c + 1) / self.grid.shape[1],
                                color="black",
                                linewidth=3,
                            )
                        if r == self.grid.shape[0] - 1 or not mask[r + 1, c]:
                            ax.axhline(
                                r + 0.5,
                                xmin=c / self.grid.shape[1],
                                xmax=(c + 1) / self.grid.shape[1],
                                color="black",
                                linewidth=3,
                            )
                        if c == 0 or not mask[r, c - 1]:
                            ax.axvline(
                                c - 0.5,
                                ymin=(self.grid.shape[0] - r - 1) / self.grid.shape[0],
                                ymax=(self.grid.shape[0] - r) / self.grid.shape[0],
                                color="black",
                                linewidth=3,
                            )
                        if c == self.grid.shape[1] - 1 or not mask[r, c + 1]:
                            ax.axvline(
                                c + 0.5,
                                ymin=(self.grid.shape[0] - r - 1) / self.grid.shape[0],
                                ymax=(self.grid.shape[0] - r) / self.grid.shape[0],
                                color="black",
                                linewidth=3,
                            )

        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def visualize_ascii(self):
        for row in self.grid:
            print("".join(["O" if x == 0 else "X" if x == -1 else str(x) for x in row]))

    def visualize_piece(self, action):
        move = self.decode_action_verbose(action)
        coords = self._get_piece_coords(move["piece_id"], move["chirality"], move["rotation"])

        # Create 5x5 grid (should fit any piece)
        grid = np.zeros((5, 5))
        for r, c in coords:
            if 0 <= r < 5 and 0 <= c < 5:
                grid[r, c] = 1

        fig, ax = plt.subplots(figsize=(3, 3))
        display = np.ones((*grid.shape, 3))
        display[grid == 1] = [0.5, 0.5, 0.5]

        ax.imshow(display)

        # Grid lines
        for i in range(6):
            ax.axhline(i - 0.5, color="gray", linewidth=1)
            ax.axvline(i - 0.5, color="gray", linewidth=1)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Piece {move['piece_name']} (C:{move['chirality']}, R:{move['rotation']})")
        plt.show()
