"""
Regression coverage for `analysis/helpers/r_bridge.py`.

The pure-Python tests always run. The rpy2/R-dependent tests use the `r_session` fixture (see
`conftest.py`) and skip on any machine without a working R + Rtools install, rather than failing the whole
suite - mirroring the `output_dir`/`stimuli_dir` skip-when-unavailable convention already used here.
"""
import inspect
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from analysis.helpers.r_bridge import (
    cached_fit,
    gam_metrics,
    get_r_object,
    glmer_metrics,
    predict_gam,
    source_r,
    to_r_dataframe,
)

_R_DIR = pathlib.Path(__file__).resolve().parents[1] / "analysis" / "R"


def _balanced_calls(text: str, fn_name: str) -> list[str]:
    """Return the argument-list text of every `fn_name(...)` call in `text`, using paren balancing (a
    naive regex breaks on the nested parens in e.g. `s(trial, bs="re")`)."""
    marker = fn_name + "("
    calls = []
    i = 0
    while True:
        start = text.find(marker, i)
        if start == -1:
            break
        depth = 1
        j = start + len(marker)
        while j < len(text) and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        calls.append(text[start + len(marker):j - 1])
        i = j
    return calls


class TestCachedFit:
    def test_caches_without_refit(self, tmp_path):
        calls = []

        def fit_fn():
            calls.append(1)
            return {"n_calls": len(calls)}

        cache_path = str(tmp_path / "result.pkl")
        first = cached_fit(cache_path, fit_fn)
        second = cached_fit(cache_path, fit_fn)
        assert first == {"n_calls": 1}
        assert second == {"n_calls": 1}
        assert len(calls) == 1

    def test_force_refit_recomputes(self, tmp_path):
        calls = []

        def fit_fn():
            calls.append(1)
            return {"n_calls": len(calls)}

        cache_path = str(tmp_path / "result.pkl")
        cached_fit(cache_path, fit_fn)
        third = cached_fit(cache_path, fit_fn, force_refit=True)
        assert third == {"n_calls": 2}
        assert len(calls) == 2


class TestDiscreteTrueSegfaultGuard:
    """Pins r_bridge.py's own documented workaround: `bam(..., discrete = TRUE)` must always also pass
    `nthreads = 1`, or its internal OpenMP threading segfaults the whole process when embedded via rpy2."""

    def test_every_discrete_true_bam_call_sets_nthreads_one(self):
        r_files = sorted(_R_DIR.glob("*.R"))
        assert r_files, f"expected at least one .R file under {_R_DIR}"
        checked_any = False
        for path in r_files:
            text = path.read_text(encoding="utf-8")
            for call_body in _balanced_calls(text, "bam"):
                if re.search(r"discrete\s*=\s*TRUE", call_body):
                    checked_any = True
                    assert re.search(r"nthreads\s*=\s*1\b", call_body), (
                        f"{path.name}: a bam(..., discrete = TRUE) call is missing nthreads = 1 "
                        f"(segfault risk under rpy2 - see r_bridge.py's module docstring)"
                    )
        assert checked_any, "expected to find at least one bam(..., discrete = TRUE) call to check"


class TestSourceRImplementation:
    """Pins the second documented segfault source: source_r() must read the file's text and ro.r(text) it,
    not call R's own source() - confirmed by direct reproduction that source() segfaults under memory
    pressure while text-evaluation does not. The failure mode is a hard SIGSEGV, not something a fast unit
    test can safely reproduce, so this checks the implementation shape instead."""

    def test_reads_file_text_instead_of_calling_r_source(self):
        code = inspect.getsource(source_r)
        code_body = code.replace(source_r.__doc__ or "", "")  # docstring discusses source() in prose
        assert ".read()" in code_body
        assert "source(" not in code_body


@pytest.fixture
def synthetic_binary_data() -> pd.DataFrame:
    """Small, fast-to-fit synthetic dataset: enough subjects/rows for gam(s(subject, bs="re")) and
    glmer((1|subject)) to fit without warnings, small enough to fit in well under a second."""
    rng = np.random.default_rng(42)
    n_subjects, n_per_subject = 8, 15
    subject = np.repeat(np.arange(n_subjects), n_per_subject).astype(str)
    x = rng.uniform(-2, 2, size=n_subjects * n_per_subject)
    p = 1 / (1 + np.exp(-0.7 * x))
    y = rng.binomial(1, p)
    return pd.DataFrame({"subject": subject, "x": x, "y": y})


class TestRpy2Integration:
    def test_setup_rpy2_initializes(self, r_session):
        import rpy2.robjects as ro

        assert ro.r("1 + 1")[0] == 2

    def test_mgcv_and_lme4_loadable(self, r_session):
        import rpy2.robjects as ro

        ro.r("library(mgcv)")
        ro.r("library(lme4)")

    def test_to_r_dataframe_roundtrip(self, r_session, synthetic_binary_data):
        import rpy2.robjects as ro

        to_r_dataframe(synthetic_binary_data, "dat")
        assert int(ro.r("nrow(dat)")[0]) == len(synthetic_binary_data)
        # M10-adjacent regression guard: subject must be an *unordered* factor, else R silently uses
        # polynomial contrasts instead of treatment contrasts for any categorical predictor.
        assert bool(ro.r("is.factor(dat$subject)")[0])
        assert not bool(ro.r("is.ordered(dat$subject)")[0])
        assert "trial_uid" not in list(ro.r("names(dat)"))  # no "trial" column here -> not built

    def test_gam_metrics_and_predict_gam(self, r_session, synthetic_binary_data, tmp_path):
        r_script = tmp_path / "test_gam.R"
        r_script.write_text(
            'library(mgcv)\n'
            'model <- gam(y ~ s(x, k = 3) + s(subject, bs = "re"), data = dat, '
            'family = binomial(), method = "REML")\n',
            encoding="utf-8",
        )
        to_r_dataframe(synthetic_binary_data, "dat", build_trial_uid=False)
        source_r(str(r_script))
        model = get_r_object("model")

        metrics = gam_metrics(model, label="test")
        for key in ("label", "aic", "bic", "r_sq_adj", "dev_expl", "n", "smooth_terms"):
            assert key in metrics
        assert metrics["label"] == "test"
        assert metrics["n"] == len(synthetic_binary_data)
        assert isinstance(metrics["smooth_terms"], pd.DataFrame)
        assert "term" in metrics["smooth_terms"].columns

        grid = pd.DataFrame({"x": [-1.0, 0.0, 1.0], "subject": ["0", "0", "0"]})
        preds = predict_gam(model, grid)
        assert preds.shape == (3,)
        assert np.all((preds >= 0) & (preds <= 1))

    def test_glmer_metrics(self, r_session, synthetic_binary_data, tmp_path):
        r_script = tmp_path / "test_glmer.R"
        r_script.write_text(
            'library(lme4)\n'
            'model <- glmer(y ~ x + (1 | subject), data = dat, family = binomial())\n',
            encoding="utf-8",
        )
        to_r_dataframe(synthetic_binary_data, "dat", build_trial_uid=False)
        source_r(str(r_script))
        model = get_r_object("model")

        metrics = glmer_metrics(model, label="test")
        for key in (
                "label", "aic", "bic", "log_lik", "r2_marginal", "r2_conditional",
                "n", "coefficients", "is_singular", "converged",
        ):
            assert key in metrics
        assert metrics["n"] == len(synthetic_binary_data)
        assert isinstance(metrics["coefficients"], pd.DataFrame)
        assert "term" in metrics["coefficients"].columns
