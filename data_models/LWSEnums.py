from enum import Enum, IntEnum, StrEnum

class SexEnum(StrEnum):
    Male = 'male'
    Female = 'female'
    Other = 'other'


class DominantHandEnum(StrEnum):
    Right = 'right'
    Left = 'left'


class DominantEyeEnum(StrEnum):
    Right = 'right'
    Left = 'left'


class ImageCategoryEnum(IntEnum):
    UNKNOWN = 0
    ANIMAL_OTHER = 1
    HUMAN_OTHER = 2
    ANIMAL_FACE = 3
    HUMAN_FACE = 4
    OBJECT_HANDMADE = 5
    OBJECT_NATURAL = 6


class SearchArrayTypeEnum(IntEnum):
    UNKNOWN = 0
    COLOR = 1
    BW = 2
    NOISE = 3
