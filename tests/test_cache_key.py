"""Per-subject cache invalidation.

Covers CODE_REVIEW finding H3 - the caches were unkeyed, so editing preprocessing code left them silently stale.
"""

import json
import pathlib

import pytest

from pipeline.cache_key import (
    build_cache_key,
    describe_staleness,
    is_cache_valid,
    sidecar_path,
    stage1_code_hash,
    write_cache_key,
)


@pytest.fixture
def cached(tmp_path) -> pathlib.Path:
    """A stand-in cache file with no sidecar yet."""
    path = tmp_path / "Subject.pkl"
    path.write_bytes(b"pickled subject")
    return path


class TestCodeHash:
    def test_is_stable_across_calls(self):
        assert stage1_code_hash() == stage1_code_hash()

    def test_changes_when_a_stage1_source_changes(self, monkeypatch, tmp_path):
        """Editing preprocessing code must invalidate - the whole point of H3."""
        import pipeline.cache_key as ck

        before = ck.stage1_code_hash()
        fake_root = tmp_path / "repo"
        (fake_root / "data_models" / "preprocess").mkdir(parents=True)
        (fake_root / "data_models" / "preprocess" / "events.py").write_text("# edited", encoding="utf-8")
        monkeypatch.setattr(ck, "_REPO_ROOT", str(fake_root))
        assert ck.stage1_code_hash() != before


class TestCacheKey:
    def test_parameters_are_order_independent(self):
        assert build_cache_key(a=1, b=2) == build_cache_key(b=2, a=1)

    def test_differing_parameters_produce_different_keys(self):
        assert build_cache_key(on_target_threshold_dva=1.75) != build_cache_key(on_target_threshold_dva=2.0)

    def test_enums_render_by_name(self):
        from data_models.LWSEnums import SubjectActionCategoryEnum as Act

        key = build_cache_key(identification_actions=[Act.MARK_AND_CONFIRM])
        assert key["parameters"]["identification_actions"] == ["MARK_AND_CONFIRM"]

    def test_key_is_json_serialisable(self):
        json.dumps(build_cache_key(threshold=1.75, actions=("a", "b")))


class TestValidity:
    def test_absent_cache_is_not_valid(self, tmp_path):
        assert not is_cache_valid(str(tmp_path / "nope.pkl"), build_cache_key())

    def test_unkeyed_cache_is_not_valid(self, cached):
        """The pre-H3 state: a pickle with no sidecar must never be trusted."""
        key = build_cache_key()
        assert not is_cache_valid(str(cached), key)
        assert describe_staleness(str(cached), key) == "cache predates cache-keying (no sidecar)"

    def test_matching_key_is_valid(self, cached):
        key = build_cache_key(threshold=1.75)
        write_cache_key(str(cached), key)
        assert is_cache_valid(str(cached), key)
        assert describe_staleness(str(cached), key) is None

    def test_changed_parameters_invalidate(self, cached):
        write_cache_key(str(cached), build_cache_key(threshold=1.75))
        reason = describe_staleness(str(cached), build_cache_key(threshold=2.0))
        assert reason is not None and "parameters changed" in reason

    def test_changed_code_invalidates(self, cached):
        stored = build_cache_key()
        stored["code_hash"] = "0000000000000000"
        write_cache_key(str(cached), stored)
        reason = describe_staleness(str(cached), build_cache_key())
        assert reason is not None and "stage-1 code changed" in reason

    def test_corrupt_sidecar_invalidates(self, cached):
        sidecar_path(str(cached))
        pathlib.Path(sidecar_path(str(cached))).write_text("{not json", encoding="utf-8")
        key = build_cache_key()
        assert not is_cache_valid(str(cached), key)
        assert describe_staleness(str(cached), key) == "cache key file is corrupt"
