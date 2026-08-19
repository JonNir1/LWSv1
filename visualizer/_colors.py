from typing import Tuple

import matplotlib.cm as cm
import numpy as np

RGBA = Tuple[int, int, int, int]
RGB = Tuple[int, int, int]

# Wong colorblind-safe palette (https://www.nature.com/articles/nmeth.1618)
DEFAULT_TARGET_COLORS: dict[str, RGBA] = {
    "HIT":          (0, 158, 115, 160),
    "MISS":         (213, 94, 0, 160),
    "FALSE_ALARM":  (204, 121, 167, 160),
    "REPEATED_HIT": (0, 114, 178, 160),
    "UNKNOWN":      (128, 128, 128, 160),
}

DEFAULT_SCANPATH_COLORSCALE = "viridis"
DEFAULT_HEATMAP_COLORSCALE = "hot"

DEFAULT_GAZE_DOT_COLOR: RGBA = (0, 114, 178, 230)
DEFAULT_GAZE_TRAIL_COLOR: RGBA = (86, 180, 233, 100)
DEFAULT_FIXATION_HIGHLIGHT_COLOR: RGBA = (0, 158, 115, 80)


def colorscale_to_rgba(cmap_name: str, value: float, alpha: int = 255) -> RGBA:
    """Map a 0-1 float to an RGBA tuple using a matplotlib colormap."""
    cmap = cm.get_cmap(cmap_name)
    r, g, b, _ = cmap(np.clip(value, 0.0, 1.0))
    return (int(r * 255), int(g * 255), int(b * 255), alpha)
