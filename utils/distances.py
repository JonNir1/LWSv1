import numpy as np

import constants as cnst


def pixel_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two pixel coordinates.

    Works with scalars, arrays, or any mix: numpy broadcasting applies,
    so ``pixel_distance(fixation_x, fixation_y, target_xs, target_ys)``
    returns one distance per target.
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def px2deg(screen_distance_cm: float) -> float:
    """Angle in degrees subtended by one pixel at *screen_distance_cm*.

    Multiply a pixel distance by this value to convert to DVA.
    Equivalent to ``Subject.px2deg``, extracted so stage-2 code can
    compute DVA without a Subject object.
    """
    return 2 * np.degrees(np.arctan2(cnst.PIXEL_SIZE_CM / 2, screen_distance_cm))
