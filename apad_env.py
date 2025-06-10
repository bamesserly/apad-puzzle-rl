import gymnasium as gym
import numpy as np
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


def has_islands(grid):
    labeled_array, num_features = label(grid == 0)

    # Count cells in each island
    island_sizes = np.bincount(labeled_array.ravel())[1:]

    # Check if any islands have 4 or fewer cells
    return np.any((island_sizes >= 2) & (island_sizes <= 4))


class APADEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.piece_names = ["K", "A", "C", "L", "R", "P", "I", "Z"]

        # 7x7 grid with 6 invalid spaces (43 valid total)
        self.grid_size = 7
        self.valid_spaces = 43

        # Invalid grid positions
        self.invalid_positions = {(0, 6), (1, 6), (6, 3), (6, 4), (6, 5), (6, 6)}

        # 8 pieces: 5 pentominos + 2 sextominos
        # 8 pieces × 2 chirality × 4 rotations × 43 positions
        self.action_space = gym.spaces.Discrete(8 * 2 * 4 * 43)

        # Observation: 7x7 grid + 8 remaining pieces
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=8,  # -1=invalid, 0=empty, 1-8=piece_ids
            shape=(57,),  # 7x7 + 8
            dtype=np.int8,
        )

        # Initialize grid with invalid positions marked
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for pos in self.invalid_positions:
            self.grid[pos] = -1

        self._cached_action_masks = None

        self.remaining_pieces = np.ones(8, dtype=bool)

    def _get_obs(self):
        return np.concatenate(
            [self.grid.flatten(), self.remaining_pieces.astype(np.int8)]
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        info = {"action_mask": self.action_masks()}
        self._cached_action_masks = None

        for pos in self.invalid_positions:
            self.grid[pos] = -1
        self.remaining_pieces = np.ones(8, dtype=bool)
        return self._get_obs(), info

    def _get_piece_coords(self, piece_id, chirality, rotation):
        # Get base piece coordinates
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

        # Vectorize all coordinate checks
        coords_array = np.array(coords)
        rows = row + coords_array[:, 0]
        cols = col + coords_array[:, 1]

        # Check bounds for all coordinates at once
        if np.any(
            (rows < 0)
            | (rows >= self.grid_size)
            | (cols < 0)
            | (cols >= self.grid_size)
        ):
            return False

        # Check occupancy for all positions at once
        return np.all(self.grid[rows, cols] == 0)

    def _place_piece(self, piece_id, chirality, rotation, position):
        assert self.remaining_pieces[piece_id], f"Piece {piece_id} already used"
        coords = self._get_piece_coords(piece_id, chirality, rotation)
        row, col = divmod(position, self.grid_size)

        try:
            for dr, dc in coords:
                self.grid[row + dr, col + dc] = piece_id + 1
        except IndexError:
            print("Invalid piece placement")
            print(f"row {row}, dr {dr}, col {col}, dc {dc}")
            print(
                f"piece_id, chirality, rotation, position ({piece_id}, {chirality}, {rotation}, {position})"
            )
            self.visualize()
            raise IndexError

        self.remaining_pieces[piece_id] = False

    def _has_valid_moves(self):
        for piece_id in range(8):
            if not self.remaining_pieces[piece_id]:
                continue
            for chirality in range(2):
                for rotation in range(4):
                    for position in range(43):
                        if self._is_valid_placement(
                            piece_id, chirality, rotation, position
                        ):
                            return True
        return False

    # The action passed will be an integer 0 - 2751, spanning the action space.
    # This covers all the possible actions: piece ID (8) x chirality (2) x rotation (4)  x position (43) = 2752
    # latest and greatest signature: observation, reward, terminated, truncated, info = env.step(action)
    def step(self, action):

        # print(f"Action taken: {action}")
        # print(f"Valid actions: {np.where(self.action_masks())[0]}")
        # print(f"Action valid: {self.action_masks()[action]}")

        move = self.decode_action(action)

        # Place piece
        self._place_piece(
            move["piece_id"], move["chirality"], move["rotation"], move["position"]
        )

        # Cache masks after state change
        self._cached_action_masks = None
        self._cached_action_masks = self.action_masks()

        # Check win condition
        done = np.sum(self.remaining_pieces) == 0
        if done:
            return (
                self._get_obs(),
                30,
                True,
                False,
                {
                    "terminal_observation": self._get_obs(),
                    "action_mask": self._cached_action_masks,
                },
            )

        # Check bricked game (i.e there's an empty island of size 2-4 which will ultimately end the game)
        if has_islands(self.grid):
            return (
                self._get_obs(),
                -20,
                False,
                True,
                {
                    "terminal_observation": self._get_obs(),
                    "action_mask": self._cached_action_masks,
                },
            )

        # Check no valid moves (separate from the above island condition -- an island of 5+ remains, but final piece won't fit)
        if not np.any(self.action_masks()):
            return (
                self._get_obs(),
                -5,
                False,
                True,
                {
                    "terminal_observation": self._get_obs(),
                    "action_mask": self._cached_action_masks,
                },
            )

        return (
            self._get_obs(),
            20,
            False,
            False,
            {"action_mask": self._cached_action_masks},
        )
        ## Check piece available
        # if not self.remaining_pieces[move['piece_id']]:
        #   return self._get_obs(), -0.5, False, False, {}

        ## Check valid placement
        # if not self._is_valid_placement(move['piece_id'], move['chirality'], move['rotation'], move['position']):
        #   return self._get_obs(), -0.1, False, False, {}

        ## Check if stuck (no more valid moves)
        # if not self._has_valid_moves():
        #    return self._get_obs(), -5, False, True, {"terminal_observation": self._get_obs()}

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

    def visualize_piece(self, action):
        move = self.decode_action_verbose(action)
        coords = self._get_piece_coords(
            move["piece_id"], move["chirality"], move["rotation"]
        )

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
        ax.set_title(
            f"Piece {move['piece_name']} (C:{move['chirality']}, R:{move['rotation']})"
        )
        plt.show()

    def action_masks(self):
        if self._cached_action_masks is not None:
            return self._cached_action_masks

        mask = np.zeros(self.action_space.n, dtype=bool)

        valid_positions = np.where((self.grid == 0).flatten())[0]

        available_pieces = np.where(self.remaining_pieces)[0]
        for piece_id in available_pieces:
            for chirality in range(2):
                for rotation in range(4):
                    for position in valid_positions:
                        action = self.encode_action(
                            piece_id, chirality, rotation, position
                        )
                        if self._is_valid_placement(
                            piece_id, chirality, rotation, position
                        ):
                            mask[action] = True

        return mask
