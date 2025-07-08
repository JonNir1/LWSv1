from enum import Enum, IntEnum, StrEnum
import config as cnfg

class SexEnum(StrEnum):
    MALE = 'male'
    FEMALE = 'female'
    OTHER = 'other'


class DominantHandEnum(StrEnum):
    RIGHT = cnfg.RIGHT_STR
    LEFT = cnfg.LEFT_STR


class DominantEyeEnum(StrEnum):
    RIGHT = cnfg.RIGHT_STR
    LEFT = cnfg.LEFT_STR


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


class SubjectActionTypesEnum(IntEnum):
    NO_ACTION = 0
    MARK_AND_CONFIRM = 1
    MARK_ONLY = 2
    ATTEMPTED_MARK = 3
    MARK_AND_REJECT = 4


class TargetVisitTypeEnum(StrEnum):
    OTHER = 'other'
    BEFORE = 'before'
    MARKING = 'marking'
    AFTER = 'after'
