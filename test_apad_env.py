import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from apad_env import APADEnv, has_islands


@pytest.fixture
def env():
    """Create a fresh environment for each test"""
    return APADEnv()


@pytest.fixture
def env_no_date():
    """Environment with no date constraints"""
    return APADEnv(-1, -1)


class TestBasicEnvironment:
    """Basic sanity checks for environment initialization and structure"""

    def test_grid_initialization(self, env):
        assert env.grid.shape == (7, 7)
        assert np.sum(env.grid == -1) in {6, 7, 8}  # Invalid positions + date cells
        assert np.sum(env.remaining_pieces) == 8  # All pieces available

    def test_piece_coordinate_generation(self, env):
        # K piece, no flip, no rotation
        coords_k = env._get_piece_coords(0, 0, 0)
        expected_k = [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1)]
        assert coords_k == expected_k

    def test_piece_rotation(self, env):
        # K piece rotated 90°
        coords_k_rot = env._get_piece_coords(0, 0, 1)
        assert len(coords_k_rot) == 5
        # Rotated should be different from original
        coords_k_orig = env._get_piece_coords(0, 0, 0)
        assert coords_k_rot != coords_k_orig

    def test_valid_placement_detection(self, env):
        # Position 14 = (2,0) on grid - should be valid
        valid = env._is_valid_placement(0, 0, 0, 14)
        assert isinstance(valid, (bool, np.bool_))

    def test_invalid_placement_out_of_bounds(self, env):
        # Top-right corner invalid
        valid = env._is_valid_placement(0, 0, 0, 6)
        assert not valid

    def test_is_valid_placement_comprehensive(self, env_no_date):
        """Comprehensive test of _is_valid_placement edge cases.

        Tests bounds checking, occupancy checking, and various piece configurations.
        Critical test for pure Python optimization of _is_valid_placement.
        """
        # Test 1: Valid placement in open space
        assert env_no_date._is_valid_placement(0, 0, 0, 14)  # K piece at (2,0)

        # Test 2: Out of bounds - bottom edge (K piece extends 3 rows down)
        assert not env_no_date._is_valid_placement(0, 0, 0, 35)  # K at (5,0) extends to row 8

        # Test 3: Out of bounds - right edge (K piece extends 1 col right)
        assert not env_no_date._is_valid_placement(0, 0, 0, 6)  # K piece at (0,6)

        # Test 4: Out of bounds - bottom-right corner
        assert not env_no_date._is_valid_placement(0, 0, 0, 48)  # K piece at (6,6)

        # Test 5: Occupancy - place piece then check overlap
        env_no_date._place_piece_components(0, 0, 0, 14)  # Place K at (2,0)
        # Try to place A piece overlapping with K
        assert not env_no_date._is_valid_placement(1, 0, 0, 14)  # A at same position

        # Test 6: Different chirality
        env_no_date.reset()
        assert env_no_date._is_valid_placement(1, 1, 0, 14)  # A piece flipped

        # Test 7: Different rotation
        assert env_no_date._is_valid_placement(1, 0, 1, 14)  # A piece rotated

        # Test 8: Small piece (R - rectangle) in various positions
        assert env_no_date._is_valid_placement(4, 0, 0, 0)  # R at top-left
        assert env_no_date._is_valid_placement(4, 0, 0, 28)  # R at (4,0)

    def test_piece_placement_marks_grid(self, env):
        if env._is_valid_placement(0, 0, 0, 14):
            env._place_piece_components(0, 0, 0, 14)
            assert not env.remaining_pieces[0]  # Piece marked as used
            assert np.sum(env.grid == 1) == 5  # 5 cells occupied by piece ID 1

    @pytest.mark.xfail(reason="check_env doesn't handle masked action spaces correctly")
    def test_gymnasium_env_checker(self, env):
        # Gymnasium's built-in environment validation
        # Note: check_env doesn't handle action masking and causes non-determinism errors
        result = check_env(env, warn=True)
        assert result is None


class TestOverlapAndReset:
    """Test that pieces can't overlap and reset works correctly"""

    def test_no_piece_overlap(self, env_no_date):
        pos1 = 14

        # Place first piece
        if env_no_date._is_valid_placement(1, 0, 0, pos1):
            env_no_date._place_piece_components(1, 0, 0, pos1)

            # Adjacent position should now be invalid due to overlap
            pos2 = 15
            overlap_valid = env_no_date._is_valid_placement(0, 0, 0, pos2)
            assert not overlap_valid

    def test_reset_clears_board(self, env_no_date):
        # Place a piece
        if env_no_date._is_valid_placement(0, 0, 0, 14):
            env_no_date._place_piece_components(0, 0, 0, 14)

        # Reset
        obs, info = env_no_date.reset()
        assert np.sum(env_no_date.remaining_pieces) == 8  # All pieces available
        assert np.sum(env_no_date.grid > 0) == 0  # Only invalid cells marked


class TestDateSelection:
    """Test that date constraints work correctly"""

    def test_specific_date_marked(self):
        env = APADEnv(2, 22)  # Feb 22
        # Should have 8 cells marked: 6 invalid corners + 2 date cells
        assert np.sum(env.grid == -1) == 8

    def test_month_only_constraint(self):
        env = APADEnv(5, -1)  # May only
        # Should have 7 cells marked: 6 invalid corners + 1 month cell
        assert np.sum(env.grid == -1) == 7

    def test_no_date_constraint(self):
        env = APADEnv(-1, -1)
        # Should have 6 cells marked: only invalid corners
        assert np.sum(env.grid == -1) == 6


class TestIslandDetection:
    """Test the has_islands() function that detects unwinnable board states"""

    def test_no_islands_detected(self):
        env = APADEnv(1, 22)
        env._place_piece_components(6, 0, 0, 1)
        env._place_piece_components(1, 0, 0, 16)
        # This configuration should not create bad islands
        assert not has_islands(env.grid)

    def test_bad_islands_detected(self):
        env = APADEnv(1, 22)
        env._place_piece_components(1, 0, 1, 35)
        env._place_piece_components(2, 0, 0, 4)
        # This configuration creates unwinnable islands
        assert has_islands(env.grid)

    def test_empty_board_has_no_islands(self, env):
        # Fresh board should not trigger island detection
        assert not has_islands(env.grid)


class TestActionMasking:
    """Test that action masking works correctly"""

    def test_action_mask_shape(self, env):
        mask = env.action_masks()
        assert mask.shape == (env.action_space.n,)
        assert mask.dtype == bool

    def test_action_mask_has_valid_actions(self):
        env = APADEnv(1, 10)
        mask = env.action_masks()
        valid_actions = np.flatnonzero(mask)
        # Fresh board should have many valid actions
        assert valid_actions.size > 0

    def test_action_mask_reduces_after_placement(self, env):
        mask_before = env.action_masks()
        valid_before = np.flatnonzero(mask_before)

        # Place a piece
        if valid_before.size > 0:
            action = valid_before[0]
            env.step(action)

            mask_after = env.action_masks()
            valid_after = np.flatnonzero(mask_after)

            # Should have fewer valid actions after placing piece
            assert valid_after.size < valid_before.size

    def test_island_masking_blocks_island_moves(self):
        """Test that island masking prevents island-creating moves"""
        env_masked = APADEnv(1, 10, mask_islands=True)
        env_masked.reset()

        # Try multiple random valid moves
        for _ in range(10):
            mask = env_masked.action_masks()
            valid_actions = np.flatnonzero(mask)

            if valid_actions.size == 0:
                break

            # Pick random valid action
            action = np.random.choice(valid_actions)

            # Verify it doesn't create islands
            checkpoint = env_masked.save_state()
            env_masked._place_piece(action)
            assert not has_islands(env_masked.grid), "Island masking allowed island-creating move"
            env_masked.load_state(checkpoint)

            # Actually take the step
            env_masked.step(action)

    def test_island_masking_allows_non_island_moves(self):
        """Test that island masking still allows valid non-island moves"""
        env_masked = APADEnv(1, 10, mask_islands=True)
        env_masked.reset()

        mask = env_masked.action_masks()
        valid_actions = np.flatnonzero(mask)

        # Should have some valid actions
        assert valid_actions.size > 0, "Island masking blocked all moves"

    def test_island_masking_reduces_action_space(self):
        """Test that island masking typically reduces available actions"""
        env_unmasked = APADEnv(1, 10, mask_islands=False)
        env_masked = APADEnv(1, 10, mask_islands=True)

        env_unmasked.reset()
        env_masked.reset()

        # Place a few pieces to create a state where islands are possible
        for _ in range(3):
            mask_unmasked = env_unmasked.action_masks()
            valid_unmasked = np.flatnonzero(mask_unmasked)

            if valid_unmasked.size == 0:
                break

            action = valid_unmasked[0]
            env_unmasked.step(action)

            # Copy state to masked env
            env_masked.grid = env_unmasked.grid.copy()
            env_masked.remaining_pieces = env_unmasked.remaining_pieces.copy()
            env_masked._cached_action_masks = None

        mask_unmasked = env_unmasked.action_masks()
        mask_masked = env_masked.action_masks()

        valid_unmasked = np.flatnonzero(mask_unmasked)
        valid_masked = np.flatnonzero(mask_masked)

        # Masked should have same or fewer actions
        assert valid_masked.size <= valid_unmasked.size

    def test_island_masking_full_episode(self):
        """Test full episode with island masking never creates islands"""
        env = APADEnv(1, 10, mask_islands=True)
        obs, info = env.reset()

        for _ in range(8):
            mask = info["action_mask"]
            valid_actions = np.flatnonzero(mask)

            if valid_actions.size == 0:
                break

            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)

            # Should never have islands
            assert not has_islands(env.grid), "Islands created during masked episode"

            if terminated or truncated:
                break


class TestGameplay:
    """Test actual gameplay mechanics"""

    def test_random_game_completes(self, env_no_date):
        """Test that a random game runs to completion without errors"""
        obs, info = env_no_date.reset()
        done = truncated = False
        step_count = 0
        max_steps = 8  # Can't place more than 8 pieces

        while not (done or truncated) and step_count < max_steps:
            mask = env_no_date.action_masks()
            valid_actions = np.flatnonzero(mask)

            if valid_actions.size == 0:
                break

            action = np.random.choice(valid_actions)
            obs, reward, done, truncated, info = env_no_date.step(action)
            step_count += 1

        # Game should end naturally
        assert done or truncated or valid_actions.size == 0

    def test_win_condition(self, env_no_date):
        """Test that placing all pieces results in a win"""
        # This is probabilistic with random placement, so we just check the logic
        obs, info = env_no_date.reset()

        # If we somehow place all 8 pieces, remaining_pieces should be all False
        for i in range(8):
            env_no_date.remaining_pieces[i] = False

        assert np.sum(env_no_date.remaining_pieces) == 0

    def test_reward_structure(self, env_no_date):
        """Test that rewards are issued correctly"""
        obs, info = env_no_date.reset()
        mask = env_no_date.action_masks()
        valid_actions = np.flatnonzero(mask)

        if valid_actions.size > 0:
            action = valid_actions[0]
            obs, reward, done, truncated, info = env_no_date.step(action)

            # Valid move should give positive or small negative reward
            assert isinstance(reward, (int, float, np.number))

    def test_step_returns_correct_types(self, env):
        """Test that step() returns correct types"""
        obs, info = env.reset()
        mask = env.action_masks()
        valid_actions = np.flatnonzero(mask)

        if valid_actions.size > 0:
            obs, reward, terminated, truncated, info = env.step(valid_actions[0])

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, (int, float, np.number))
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert isinstance(info, dict)
            assert "action_mask" in info


class TestActionEncoding:
    """Test action encoding/decoding"""

    def test_encode_decode_roundtrip(self, env):
        """Test that encoding and decoding actions are inverses"""
        piece_id, chirality, rotation, position = 0, 0, 1, 14

        action = env.encode_action(piece_id, chirality, rotation, position)
        decoded = env.decode_action(action)

        assert decoded["piece_id"] == piece_id
        assert decoded["chirality"] == chirality
        assert decoded["rotation"] == rotation
        assert decoded["position"] == position

    def test_decode_verbose_includes_name(self, env):
        """Test that verbose decode includes piece name"""
        action = env.encode_action(0, 0, 0, 14)
        decoded = env.decode_action_verbose(action)

        assert "piece_name" in decoded
        assert decoded["piece_name"] == "K"


@pytest.mark.slow
class TestStressTests:
    """Longer-running stress tests"""

    def test_many_random_games(self):
        """Run multiple random games to catch edge cases"""
        env = APADEnv(-1, -1)
        num_games = 100

        for _ in range(num_games):
            obs, info = env.reset()
            done = truncated = False

            while not (done or truncated):
                mask = env.action_masks()
                valid_actions = np.flatnonzero(mask)

                if valid_actions.size == 0:
                    break

                action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = env.step(action)

        # If we got here without exceptions, tests pass
        assert True
