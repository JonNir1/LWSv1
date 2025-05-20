from typing import Tuple, Literal, Optional

import numpy as np
import pandas as pd


def closest_indices(s: pd.Series, vals: pd.Series, threshold: float) -> pd.Series:
    """ Finds indices in `s` whose values are closest to the values in `vals`, within a given threshold. """
    assert threshold >= 0, "Threshold must be non-negative"
    s_values = s.values
    val_values = vals.values

    # compute absolute differences between each val and all of s
    diffs = np.abs(s_values[None, :] - val_values[:, None], dtype=float)  # shape: (len(vals), len(s))
    diffs[diffs > threshold] = np.inf           # mask differences that exceed threshold
    closest_idx_in_s = diffs.argmin(axis=1)     # get index (in s) of the closest value for each val

    # Handle cases where all diffs were > threshold
    no_match = np.isinf(diffs.min(axis=1))
    result = pd.Series(s.index[closest_idx_in_s], index=vals.index)
    result[no_match] = np.nan
    return result


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
    if unit == 'cm':
        return convert_units(pixel_distance, 'px', 'cm', pixel_size_cm, screen_distance_cm)
    if unit == 'deg':
        return convert_units(pixel_distance, 'px', 'deg', pixel_size_cm, screen_distance_cm)
    raise ValueError(f"Invalid unit '{unit}'. Valid options are: 'px', 'cm', 'deg'.")


def convert_units(
        length: float,
        orig_unit: Literal['px', 'cm', 'deg'], new_unit: Literal['px', 'cm', 'deg'],
        pixel_size_cm: Optional[float] = None, screen_distance_cm: Optional[float] = None
) -> float:
    assert length >= 0 or np.isnan(length), "Length must be non-negative"
    if orig_unit == new_unit:
        return length
    def validate_argument(name: str, value: Optional[float]):
        if value is None or not np.isfinite(value):
            raise ValueError(f"`{name}` must be provided and finite to convert from {orig_unit} to {new_unit}.")

    if {orig_unit, new_unit} == {'px', 'cm'}:
        validate_argument("pixel_size_cm", pixel_size_cm)
        return length * pixel_size_cm if orig_unit == 'px' else length / pixel_size_cm
    if {orig_unit, new_unit} == {'cm', 'deg'}:
        validate_argument("screen_distance_cm", screen_distance_cm)
        if orig_unit == 'cm':
            return 2 * np.degrees(np.arctan2(length / 2, screen_distance_cm))   # cm to deg
        else:
            return 2 * screen_distance_cm * np.tan(np.radians(length / 2))      # deg to cm
    if {orig_unit, new_unit} == {'px', 'deg'}:
        validate_argument("pixel_size_cm", pixel_size_cm)
        validate_argument("screen_distance_cm", screen_distance_cm)
        cm_length = convert_units(length, orig_unit, 'cm', pixel_size_cm, screen_distance_cm)
        return convert_units(cm_length, 'cm', new_unit, pixel_size_cm, screen_distance_cm)
    raise ValueError(
        f"Invalid unit conversion from {orig_unit} to {new_unit}. Valid options are: 'px', 'cm', 'deg'."
    )


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