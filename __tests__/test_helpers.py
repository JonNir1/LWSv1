import unittest
import numpy as np
from helpers import distance, convert_units, change_pixel_origin, flatten_or_raise

class TestHelpers(unittest.TestCase):

    def test_distance(self):
        p1, p2 = (0, 0), (3, 4)
        self.assertAlmostEqual(distance(p1, p2), 5.0)
        self.assertAlmostEqual(distance(p1, p2, unit='cm', pixel_size_cm=0.1), 0.5)
        self.assertTrue(0.47 < distance(p1, p2, unit='deg', pixel_size_cm=0.1, screen_distance_cm=60) < 0.48)
        with self.assertRaises(ValueError):
            distance(p1, p1, unit='cm')
            distance((0, 0), (1, 1), unit='deg', pixel_size_cm=0.1)
            distance((0, 0), (1, 1), unit='inch')

    def test_convert_units(self):
        # Identity conversions
        self.assertEqual(convert_units(5, 'px', 'px'), 5)
        self.assertEqual(convert_units(2.5, 'cm', 'cm'), 2.5)
        self.assertEqual(convert_units(1.2, 'deg', 'deg'), 1.2)
        self.assertTrue(np.array_equal(convert_units(np.nan, 'deg', 'deg'), np.nan, equal_nan=True))

        # px ↔ cm
        self.assertAlmostEqual(convert_units(100, 'px', 'cm', pixel_size_cm=0.05), 5.0)
        self.assertAlmostEqual(convert_units(5, 'cm', 'px', pixel_size_cm=0.05), 100)

        # cm ↔ deg
        cm = 1.5
        sd = 60
        deg = convert_units(cm, 'cm', 'deg', screen_distance_cm=sd)
        self.assertAlmostEqual(convert_units(deg, 'deg', 'cm', screen_distance_cm=sd), cm, places=5)

        # px ↔ deg (via cm)
        px = 60
        pixel_size = 0.05
        angle = convert_units(px, 'px', 'deg', pixel_size_cm=pixel_size, screen_distance_cm=sd)
        px_back = convert_units(angle, 'deg', 'px', pixel_size_cm=pixel_size, screen_distance_cm=sd)
        self.assertAlmostEqual(px, px_back, places=5)

        # missing params and invalid conversions
        with self.assertRaises(ValueError):
            convert_units(100, 'px', 'cm')
            convert_units(5, 'cm', 'deg')
            convert_units(20, 'px', 'deg', pixel_size_cm=0.05)
            convert_units(1, 'px', 'inch')  # invalid conversion

    def test_change_pixel_origin(self):
        x, y = np.array([0]), np.array([0])
        new_x, new_y = change_pixel_origin(x, y, (100, 100), 'center', 'top-left')
        np.testing.assert_array_equal(new_x, [50])
        np.testing.assert_array_equal(new_y, [50])

        x, y = np.array([5]), np.array([10])
        new_x, new_y = change_pixel_origin(x, y, (200, 200), 'top-left', 'top-left')
        np.testing.assert_array_equal(new_x, x)
        np.testing.assert_array_equal(new_y, y)

        with self.assertRaises(ValueError):
            change_pixel_origin(np.array([1]), np.array([1]), (100,), 'top-left', 'center')
            change_pixel_origin(np.array([1]), np.array([1]), (100, 100), 'foo', 'center')


    def test_flatten_or_raise(self):
        self.assertTrue(np.array_equal(flatten_or_raise(np.array([1, 2, 3])), [1, 2, 3]))
        self.assertTrue(np.array_equal(flatten_or_raise(np.array([[1, 2, 3]])), [1, 2, 3]))
        self.assertTrue(np.array_equal(flatten_or_raise(np.array([[1], [2], [3]])), [1, 2, 3]))
        with self.assertRaises(ValueError):
            flatten_or_raise(np.array([[1, 2], [3, 4]]))

