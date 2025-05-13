import os
from typing import NamedTuple

import numpy as np
import pandas as pd
import numpy.typing as npt
from pymatreader import read_mat

import config as cnfg
from data_models.LWSEnums import SearchArrayTypeEnum, ImageCategoryEnum


class _SearchArrayImage(NamedTuple):
    """A class to represent a single image within a search array."""

    x: int              # X coordinate of the image center
    y: int              # Y coordinate of the image center
    angle: float        # jitter angle
    sub_path: str       # sub-path to the image file

    @property
    def path(self) -> str:
        return os.path.join(cnfg.IMAGE_DIR_PATH, self.sub_path)

    @property
    def category(self) -> ImageCategoryEnum:
        return self._parse_image_category(self.sub_path, safe=True)

    @staticmethod
    def _parse_image_category(image_path: str, safe: bool) -> ImageCategoryEnum:
        basename = os.path.basename(image_path)
        if basename.startswith("animate_animals_others"):
            return ImageCategoryEnum.ANIMAL_OTHER       # 1
        if basename.startswith("animate_humans_others"):
            return ImageCategoryEnum.HUMAN_OTHER        # 2
        if basename.startswith("animate_animals_faces"):
            return ImageCategoryEnum.ANIMAL_FACE        # 3
        if basename.startswith("animate_humans_faces"):
            return ImageCategoryEnum.HUMAN_FACE         # 4
        if basename.startswith("inanimate_handmade"):
            return ImageCategoryEnum.OBJECT_HANDMADE    # 5
        if basename.startswith("inanimate_natural"):
            return ImageCategoryEnum.OBJECT_NATURAL     # 6
        if safe:
            return ImageCategoryEnum.UNKNOWN            # 0
        raise ValueError(f"Unknown image category for {basename}.")


_NDArrayImageType = npt.NDArray[_SearchArrayImage]
_NDArrayBoolType = npt.NDArray[np.bool_]
_NDArrayGrayScaleType = npt.NDArray[np.float_]


class SearchArray:
    _NUM_ROWS, _NUM_COLS = 10, 18
    _RESOLUTION = (1920, 1080)

    def __init__(
            self,
            num: int,
            version: int,
            array_type: SearchArrayTypeEnum,
            images: _NDArrayImageType,
            is_targets: _NDArrayBoolType,
            grayscale: _NDArrayGrayScaleType,
    ):
        assert num >= 1 and version >= 1
        assert images.shape == (SearchArray._NUM_ROWS, SearchArray._NUM_COLS), \
            f"Images shape {images.shape} doesn't match expected shape ({SearchArray._NUM_ROWS}, {SearchArray._NUM_COLS})."
        assert images.shape == is_targets.shape, \
            f"Images ({images.shape}) and targets ({is_targets.shape}) shapes don't match.)"
        assert grayscale.T.shape == SearchArray._RESOLUTION, \
            f"Grayscale resolution {grayscale.T.shape} doesn't match expected resolution {SearchArray._RESOLUTION}."
        self._num = num
        self._version = version
        self._array_type = array_type
        self._images = images
        self._is_targets = is_targets
        self._grayscale = grayscale

    @staticmethod
    def from_mat(path: str) -> "SearchArray":
        if not (os.path.isfile(path) and path.endswith(".mat")):
            raise FileNotFoundError(f"Not a valid `.mat` file: {path}")
        matfile = read_mat(path)['imageInfo']

        # extract metadata from the path
        base_dir, filename = os.path.split(path)
        array_number = int(filename.split('.')[0].split('_')[-1])       # expected format: `image_111.mat`
        base_dir, array_type_name = os.path.split(base_dir)
        array_type = SearchArrayTypeEnum[array_type_name.upper()]
        base_dir, array_version_dir = os.path.split(base_dir)
        array_version = int(array_version_dir.replace("generated_stim", ""))    # expected format: `generated_stim1`

        # generate the array of images
        stim_paths = pd.DataFrame(matfile['stimInArray']).map(lambda p: p.replace(cnfg.IMAGE_DIR_PATH + "\\", ""))
        stim_centers = pd.DataFrame(matfile['stimCenters']).map(lambda point: {cnfg.X: point[1], cnfg.Y: point[0]})
        stim_angles = pd.DataFrame(matfile['angleJitter'])
        image_array = np.empty((SearchArray._NUM_ROWS, SearchArray._NUM_COLS), dtype=_SearchArrayImage)
        for i in range(SearchArray._NUM_ROWS):
            for j in range(SearchArray._NUM_COLS):
                img = _SearchArrayImage(
                    stim_centers.iloc[i, j][cnfg.X],
                    stim_centers.iloc[i, j][cnfg.Y],
                    stim_angles.iloc[i, j],
                    stim_paths.iloc[i, j],
                )
                image_array[i, j] = img

        # generate the SearchArray object
        search_array = SearchArray(
            num=array_number,
            version=array_version,
            array_type=array_type,
            images=image_array,
            is_targets=matfile['targetsInArray'] != 0,
            grayscale=matfile['ScreenTemplate'],
        )
        assert search_array.num_targets == matfile['numTargets']
        assert np.array_equal(search_array.get_categories(), matfile['categoryInArray'] + matfile['targetsInArray'], equal_nan=True)
        return search_array

    @property
    def version(self) -> int:
        return self._version

    @property
    def array_type(self) -> SearchArrayTypeEnum:
        return self._array_type

    @property
    def num_targets(self) -> int:
        return int(np.sum(self._is_targets))

    @property
    def mat_path(self) -> str:
        return self._get_path(self._version, self._array_type, self._num, "mat")

    @property
    def image_path(self) -> str:
        return self._get_path(self._version, self._array_type, self._num, "bmp")

    def get_categories(self) -> npt.NDArray[ImageCategoryEnum]:
        """
        Get the categories of the images in the search array.
        :return: A 2D array of ImageCategoryEnum values.
        """
        return np.array([[img.category for img in row] for row in self._images])

    @staticmethod
    def _get_path(
            arr_version: int, arr_type: SearchArrayTypeEnum, arr_num: int, file_type: str
    ) -> str:
        return os.path.join(
            cnfg.SEARCH_ARRAY_PATH,
            f"generated_stim{arr_version}",
            f"array_{arr_type.name.lower()}",
            f"image_{arr_num}.{file_type}",
        )
