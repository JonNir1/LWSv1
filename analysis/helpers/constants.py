TRIAL_SYMBOLS: dict[str, str] = {
    "all": "circle",
    "BW": "x",
    "COLOR": "diamond",
    "NOISE": "square",
}

TARGET_SYMBOLS: dict[str, str] = {
    "OBJECT_HANDMADE": "cross",
    "OBJECT_NATURAL": "x",
    "ANIMAL_OTHER": "square",
    "ANIMAL_FACE": "diamond",
    "HUMAN_OTHER": "star",
    "HUMAN_FACE": "hexagram",
}

TARGET_CATEGORY_CONTRASTS: dict[str, list[float]] = {
    "ANIMACY_EFFECT": [1/4, 1/4, 1/4, 1/4, -1/2, -1/2],
    "HUMAN_FACE_EFFECT": [-1/3, -1/3, -1/3, 1, 0, 0],
    "OBJECT_NATURALITY_EFFECT": [0, 0, 0, 0, 1, -1],
}
