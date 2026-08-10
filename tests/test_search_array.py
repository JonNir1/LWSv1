"""Search-array geometry and stimulus file paths.

Covers CODE_REVIEW findings L4 (hardcoded geometry validated against the stimulus-generation config) and M3
(the path builder must agree with the on-disk layout).
"""

import os

import numpy as np
import pytest

import config as cnfg
from data_models.LWSEnums import SearchArrayCategoryEnum
from data_models.SearchArray import SearchArray


def icon_centres(array_info: dict) -> np.ndarray:
    """Icon centre coordinates as an (n_rows, n_cols, 2) array of (x, y) in pixels.

    `StimuliLocsCenter` stores MATLAB (row, col) pairs, i.e. (y, x) - the same swap `SearchArray.from_mat` applies.
    """
    return np.array(
        [[(cell[1], cell[0]) for cell in row] for row in array_info["StimuliLocsCenter"]], dtype=float
    )


class TestGeometryAgainstStimulusConfig:
    """L4: the hardcoded constants must match what the stimuli were actually generated with."""

    def test_grid_shape_matches_the_config(self, array_info):
        rows, cols = (int(v) for v in array_info["numImages"])
        assert (SearchArray._NUM_ROWS, SearchArray._NUM_COLS) == (rows, cols)

    def test_icon_grid_has_the_expected_shape(self, array_info):
        centres = icon_centres(array_info)
        assert centres.shape == (SearchArray._NUM_ROWS, SearchArray._NUM_COLS, 2)

    def test_screen_resolution_matches_the_monitor(self):
        import constants as cnst

        assert SearchArray._RESOLUTION == (cnst.TOBII_MONITOR.width, cnst.TOBII_MONITOR.height)

    def test_bottom_strip_lies_within_the_screen(self):
        (left, top), (right, bottom) = (
            SearchArray._BOTTOM_STRIP_TOP_LEFT, SearchArray._BOTTOM_STRIP_BOTTOM_RIGHT
        )
        width, height = SearchArray._RESOLUTION
        assert 0 <= left < right <= width
        assert 0 <= top < bottom <= height

    def test_no_icon_reaches_the_exemplar_strip(self, array_info):
        """The critical property: `is_in_bottom_strip` must never fire on a search-array icon.

        If an icon overlapped the strip, a fixation on it would be counted as a glance at the exemplars and would
        suppress the `not_before_exemplar_visit` LWS criterion for the preceding fixations.
        """
        centres = icon_centres(array_info)
        half = float(array_info["objectsizePix"]) / 2
        strip_top = SearchArray._BOTTOM_STRIP_TOP_LEFT[1]
        lowest_edge = centres[..., 1].max() + half
        assert lowest_edge < strip_top, (
            f"icons reach y={lowest_edge:.0f}, overlapping the strip that starts at y={strip_top}"
        )

    def test_no_icon_centre_is_inside_the_strip(self, array_info):
        centres = icon_centres(array_info).reshape(-1, 2)
        inside = [tuple(p) for p in centres if SearchArray.is_in_bottom_strip((p[0], p[1]))]
        assert not inside, f"{len(inside)} icon centre(s) fall inside the exemplar strip: {inside[:3]}"


class TestStripMembership:
    @pytest.mark.parametrize(
        "point, expected",
        [
            ((960, 1000), True),    # middle of the strip
            ((720, 910), True),     # top-left corner, inclusive
            ((1200, 1080), True),   # bottom-right corner, inclusive
            ((719, 1000), False),   # just left
            ((1201, 1000), False),  # just right
            ((960, 909), False),    # just above
            ((300, 300), False),    # search-array region
        ],
    )
    def test_is_in_bottom_strip(self, point, expected):
        assert SearchArray.is_in_bottom_strip(point) is expected


class TestStimulusPaths:
    """M3: `get_path` must produce paths that exist, and `from_mat` must parse them back."""

    @pytest.mark.parametrize("category", [SearchArrayCategoryEnum.COLOR, SearchArrayCategoryEnum.BW,
                                          SearchArrayCategoryEnum.NOISE])
    def test_get_path_resolves_to_a_real_file(self, stimuli_dir, category):
        path = SearchArray.get_path(cnfg.STIMULI_VERSION, category, 1, "mat")
        assert os.path.isfile(path), f"{path} does not exist - path builder disagrees with the on-disk layout"

    def test_round_trip_through_from_mat(self, stimuli_dir):
        path = SearchArray.get_path(cnfg.STIMULI_VERSION, SearchArrayCategoryEnum.COLOR, 1, "mat")
        array = SearchArray.from_mat(path)
        assert array.image_num == 1
        assert array.version == cnfg.STIMULI_VERSION
        assert array.array_category == SearchArrayCategoryEnum.COLOR
        assert array.num_icons == SearchArray._NUM_ROWS * SearchArray._NUM_COLS
        assert array.num_targets > 0

    def test_mat_path_property_round_trips(self, stimuli_dir):
        """The property that M3 found broken: it built `array_color/` where the loader reads `color/`."""
        path = SearchArray.get_path(cnfg.STIMULI_VERSION, SearchArrayCategoryEnum.COLOR, 1, "mat")
        array = SearchArray.from_mat(path)
        assert os.path.isfile(array.mat_path)
        assert os.path.samefile(array.mat_path, path)

    def test_image_path_property_resolves(self, stimuli_dir):
        path = SearchArray.get_path(cnfg.STIMULI_VERSION, SearchArrayCategoryEnum.COLOR, 1, "mat")
        array = SearchArray.from_mat(path)
        assert os.path.isfile(array.image_path), f"{array.image_path} does not exist"
