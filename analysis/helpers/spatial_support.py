"""Grid-support masking, shared by the Cartesian and polar spatial GAM notebooks."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def compute_grid_support(grid: pd.DataFrame, observed: pd.DataFrame, min_neighbours: int = 5, radius_px: float = 100) -> pd.Series:
    """
    A prediction grid spans the bounding box of observed gaze, which includes screen regions with little
    or no real data - predictions there are extrapolation and must not be plotted as if they were
    estimates. Returns a boolean Series (aligned to `grid`'s index): True where a grid cell has at least
    `min_neighbours` observed (x, y) points within `radius_px` pixels.
    """
    tree = cKDTree(observed[["x", "y"]].dropna().to_numpy())
    counts = tree.query_ball_point(grid[["x", "y"]].to_numpy(), r=radius_px, return_length=True)
    return pd.Series(counts >= min_neighbours, index=grid.index, name="is_supported")
