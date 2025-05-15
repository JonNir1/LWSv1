from typing import Tuple, Literal, Optional

import numpy as np


def distance(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        unit: Literal['px', 'cm', 'deg'] = 'px',
        pixel_size_cm: Optional[float] = None,
        screen_distance_cm: Optional[float] = None,
) -> float:
    pixel_distance = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if unit == 'px':
        return pixel_distance
    if pixel_size_cm is None or not np.isfinite(pixel_size_cm):
        raise ValueError("pixel size must be provided for 'cm' or 'deg' units.")
    cm_distance = pixel_distance * pixel_size_cm
    if unit == 'cm':
        return cm_distance
    if screen_distance_cm is None or not np.isfinite(screen_distance_cm):
        raise ValueError("screen distance must be provided for 'deg' unit.")
    deg_dist = 2 * np.degrees(np.arctan2(cm_distance / 2, screen_distance_cm))
    if unit == 'deg':
        return deg_dist
    raise ValueError(f"Invalid unit '{unit}'. Valid options are: 'px', 'cm', 'deg'.")


def change_pixel_origin(
        x: np.ndarray, y: np.ndarray, resolution: Tuple[int, int], old_origin: str, new_origin: str
) -> (np.ndarray, np.ndarray):
    # validate pixels
    x = flatten_or_raise(x.copy())
    y = flatten_or_raise(y.copy())
    if x.shape != y.shape:
        raise ValueError(f"X and Y arrays have different shapes ({x.shape}, {y.shape}).")

    # validate origins
    if len(resolution) != 2:
        raise ValueError(f"Resolution must be a tuple of two integers (width, height), got {resolution}.")
    width, height = resolution
    origins = {
        "top-left": (0, 0), "bottom-left": (0, height), "center": (width / 2, height / 2),
        "top-right": (width, 0), "bottom-right": (width, height),
    }
    old_origin, new_origin = old_origin.lower(), new_origin.lower()
    if old_origin not in origins:
        raise ValueError(f"Invalid old origin '{old_origin}'. Valid options are: {list(origins.keys())}.")
    if new_origin not in origins:
        raise ValueError(f"Invalid new origin '{new_origin}'. Valid options are: {list(origins.keys())}.")
    if old_origin == new_origin:
        return x, y

    # convert to top-left origin
    curr_offset_x, curr_offset_y = origins[old_origin]
    x = x + curr_offset_x
    y = y + curr_offset_y

    # convert to new origin
    new_offset_x, new_offset_y = origins[new_origin]
    x = x - new_offset_x
    y = y - new_offset_y
    return x, y


def flatten_or_raise(arr: np.ndarray) -> np.ndarray:
    if arr.ndim <= 1:
        return arr
    dims, counts = np.unique(arr.shape, return_counts=True)
    if 1 not in dims:
        raise ValueError(f"Array is not flat ({arr.shape}).")
    count_1 = counts[dims == 1][0]
    if count_1 != arr.ndim - 1:
        raise ValueError(f"Array is not flat ({arr.shape}).")
    return arr.flatten()