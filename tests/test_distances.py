import numpy as np
import pytest

from utils.distances import pixel_distance, px2deg
import constants as cnst


class TestPixelDistance:

    def test_3_4_5_triangle(self):
        assert pixel_distance(0, 0, 3, 4) == pytest.approx(5.0)

    def test_zero_distance(self):
        assert pixel_distance(7.5, 3.2, 7.5, 3.2) == pytest.approx(0.0)

    def test_symmetric(self):
        d1 = pixel_distance(1, 2, 4, 6)
        d2 = pixel_distance(4, 6, 1, 2)
        assert d1 == pytest.approx(d2)

    def test_vectorized_one_point_many_targets(self):
        xs = np.array([3.0, 0.0, 6.0])
        ys = np.array([4.0, 0.0, 8.0])
        dists = pixel_distance(0, 0, xs, ys)
        expected = np.array([5.0, 0.0, 10.0])
        np.testing.assert_allclose(dists, expected)

    def test_vectorized_many_points_many_targets(self):
        # broadcasting: 3 fixations x 2 targets
        fix_x = np.array([0, 0, 3])[:, None]
        fix_y = np.array([0, 0, 4])[:, None]
        tgt_x = np.array([3, 6])[None, :]
        tgt_y = np.array([4, 8])[None, :]
        dists = pixel_distance(fix_x, fix_y, tgt_x, tgt_y)
        assert dists.shape == (3, 2)
        assert dists[0, 0] == pytest.approx(5.0)
        assert dists[2, 0] == pytest.approx(0.0)


class TestPx2deg:

    def test_matches_subject_formula(self):
        screen_dist = 61.4
        expected = 2 * np.degrees(np.arctan2(cnst.PIXEL_SIZE_CM / 2, screen_dist))
        assert px2deg(screen_dist) == pytest.approx(expected)

    def test_closer_means_larger_angle(self):
        assert px2deg(50.0) > px2deg(70.0)

    def test_positive(self):
        assert px2deg(60.0) > 0
