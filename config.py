import os
from typing import Union, Literal

import plotly.express.colors as _colors

from constants import *

# TODO: drop this re-export once all consumers import from pipeline.config directly
from pipeline.config import *  # noqa: F401,F403 - backward compat re-export of pipeline thresholds

STIMULI_VERSION = 1

## PATHS ##
IMAGE_DIR_PATH = r"S:\Lab-Shared\Experiments\N170 free scan\ClutteredObjects_scan\Origional_Objects_Pics\organized"

_BASE_PATH = r"S:\Lab-Shared\Experiments\LWS\Tobii Demo"
_BASE_PATH = r"C:\Users\nirjo\Desktop\HCNL\LWS"                         # TODO: remove me!

RAW_DATA_PATH = os.path.join(_BASE_PATH, "RawData")
SEARCH_ARRAY_PATH = os.path.join(_BASE_PATH, "Stimuli")
OUTPUT_PATH = os.path.join(_BASE_PATH, "Results")
SUBJECT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, f"{SUBJECT_STR}s")
PUBLICATIONS_PATH = os.path.join(_BASE_PATH, "Publications")


## VISUALIZATION CONFIGURATION ##
_DISCRETE_COLORMAP = _colors.qualitative.Dark24
_CONTINUOUS_COLORMAP = _colors.sequential.Viridis
_GENERIC_COLOR = "#808080"  # gray color for generic cases

FONT_FAMILY, FONT_COLOR = "Calibri", "black"
TITLE_FONT = dict(family=FONT_FAMILY, size=26, color=FONT_COLOR)
SUBTITLE_FONT = dict(family=FONT_FAMILY, size=22, color=FONT_COLOR)
COMMENT_FONT = dict(family=FONT_FAMILY, size=14, color=FONT_COLOR)
AXIS_LABEL_FONT = dict(family=FONT_FAMILY, size=20, color=FONT_COLOR)
AXIS_TICK_FONT = dict(family=FONT_FAMILY, size=16, color=FONT_COLOR)
AXIS_LABEL_STANDOFF = 2

GRID_LINE_COLOR, GRID_LINE_WIDTH = "lightgray", 1
ZERO_LINE_WIDTH = 2 * GRID_LINE_WIDTH


def get_discrete_color(value: Union[Literal["all"], int], loop: bool = False) -> str:
    """
    Get a discrete color for a given value, either a specific integer or the string "all".
    If `loop` is False, raises an error if the value is not in the expected range.
    """
    if isinstance(value, str) and value.lower() == ALL_STR:
        return _GENERIC_COLOR
    try:
        new_value = float(value)
    except ValueError:
        raise TypeError(f"Value must be an integer or 'all', got `{value}` of type {type(value)}.")
    if new_value != int(new_value):
        raise TypeError(f"Value must be an integer or 'all', got `{value}` of type {type(value)}.")
    value = int(new_value)
    if loop:
        value = value % len(_DISCRETE_COLORMAP)
    assert 0 <= value < len(_DISCRETE_COLORMAP), f"Value {value} out of range for discrete colormap (0-{len(_DISCRETE_COLORMAP)-1})."
    return _DISCRETE_COLORMAP[value]
