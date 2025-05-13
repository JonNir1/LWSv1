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
