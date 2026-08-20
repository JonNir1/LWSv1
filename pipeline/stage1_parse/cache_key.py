"""Cache invalidation for the per-subject pickles.

`Subject.pkl` and `eye_movements_df.pkl` are written per subject and preferred over re-parsing the raw data. Neither
recorded what produced them, so a change to any preprocessing code left the caches silently stale: results became a
mixture of old and new code, with nothing in the output to say so (CODE_REVIEW H3).

Each cache now carries a sidecar `<name>.cache.json` holding a key. A cache is reused only when its key matches the
current one; otherwise it is ignored and rebuilt.

The key covers the two things that change a cached subject's contents:

- **code**: a hash of every source file that feeds stage 1 preprocessing.
- **parameters**: the stage-1 hyperparameters passed in by the caller.

It deliberately does *not* use git state - the working tree is usually dirty during development, which is exactly
when stale caches are most misleading.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional

# Source files whose behaviour is baked into a cached Subject / fixation table. Paths are relative to the repo root.
_STAGE1_SOURCES = (
    os.path.join("config.py"),
    os.path.join("constants.py"),
    os.path.join("data_models", "Subject.py"),
    os.path.join("data_models", "Trial.py"),
    os.path.join("data_models", "SearchArray.py"),
    os.path.join("data_models", "LWSEnums.py"),
    os.path.join("data_models", "parse", "eye_movements.py"),
    os.path.join("data_models", "parse", "subject_info.py"),
    os.path.join("data_models", "parse", "triggers_and_gaze.py"),
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def stage1_code_hash() -> str:
    """Hash the contents of every stage-1 source file. Missing files hash as empty, so a rename invalidates."""
    digest = hashlib.sha256()
    for rel_path in _STAGE1_SOURCES:
        digest.update(rel_path.encode("utf-8"))
        try:
            with open(os.path.join(_REPO_ROOT, rel_path), "rb") as f:
                digest.update(f.read())
        except FileNotFoundError:
            digest.update(b"<missing>")
    return digest.hexdigest()[:16]


def build_cache_key(**parameters: Any) -> Dict[str, Any]:
    """Build the key stored alongside a cached artifact."""
    return {
        "code_hash": stage1_code_hash(),
        "parameters": {k: _as_jsonable(v) for k, v in sorted(parameters.items())},
    }


def sidecar_path(cache_path: str) -> str:
    return f"{os.path.splitext(cache_path)[0]}.cache.json"


def is_cache_valid(cache_path: str, key: Dict[str, Any]) -> bool:
    """True iff `cache_path` exists and its sidecar records exactly `key`."""
    if not os.path.exists(cache_path):
        return False
    try:
        with open(sidecar_path(cache_path), "r", encoding="utf-8") as f:
            stored = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return False    # unkeyed or corrupt sidecar -> treat as stale
    return stored == key


def write_cache_key(cache_path: str, key: Dict[str, Any]) -> None:
    with open(sidecar_path(cache_path), "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2, sort_keys=True)


def describe_staleness(cache_path: str, key: Dict[str, Any]) -> Optional[str]:
    """Human-readable reason the cache at `cache_path` cannot be reused, or None if it can."""
    if not os.path.exists(cache_path):
        return None     # nothing cached; not stale, just absent
    try:
        with open(sidecar_path(cache_path), "r", encoding="utf-8") as f:
            stored = json.load(f)
    except FileNotFoundError:
        return "cache predates cache-keying (no sidecar)"
    except json.JSONDecodeError:
        return "cache key file is corrupt"
    if stored.get("code_hash") != key["code_hash"]:
        return f"stage-1 code changed ({stored.get('code_hash')} -> {key['code_hash']})"
    if stored.get("parameters") != key["parameters"]:
        return f"parameters changed ({stored.get('parameters')} -> {key['parameters']})"
    return None


def _as_jsonable(value: Any) -> Any:
    """Render hyperparameters stably: enums by name, sequences as sorted lists of the same."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return sorted(_as_jsonable(v) for v in value)
    if hasattr(value, "name") and hasattr(value, "value"):      # Enum
        return value.name
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
