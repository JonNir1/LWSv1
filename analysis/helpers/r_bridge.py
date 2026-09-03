"""
Shared helpers for fitting R (mgcv/lme4) models via rpy2, in-process - no subprocess/CSV round-trip.

`setup_rpy2()` works around a real rpy2 bug on this machine's Windows R installation (no Rtools): see its
docstring. Everything else here is a thin, pandas-in/pandas-out layer over rpy2 so notebooks can fit GAMs
and GLMMs without hand-writing rpy2 conversion boilerplate at every call site.
"""
import os
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd

_RPY2_READY = False


def setup_rpy2(
        r_home: str = r"C:\Program Files\R\R-4.5.2",
        r_libs_user: str = "C:/Users/nirjo/R_library/4.5",
) -> None:
    """
    Initialize rpy2 on this machine. Must run before any `import rpy2...` / `from pymer4...` import
    anywhere in the process (rpy2 does its DLL/config setup at import time) - call this first, at the top
    of the notebook/script, before any other rpy2-touching import.

    Works around a real rpy2 bug on Windows installations without Rtools (TODO CLAUDE.md: installing
    Rtools45 would let this fallback be retired in favor of rpy2's normal init path): rpy2 initializes by
    calling `R CMD config --ldflags` to find directories to register via `os.add_dll_directory()`, so
    Windows can later `LoadLibrary` R's own compiled package DLLs (e.g. stats.dll, needed by nlme, needed
    by mgcv/lme4). `R CMD config` internally needs `make` (via config.sh) - absent here - so config.sh
    fails silently but exits 0 with empty stdout rather than a non-zero code. rpy2's own code anticipates
    "R CMD config unavailable" and has a graceful fallback for exactly that (which correctly locates the R
    DLL directories itself) - but the fallback only triggers on subprocess.CalledProcessError (non-zero
    exit). Exit 0 + empty output instead crashes one line earlier as an unrelated-looking
    `IndexError: list index out of range`, so the intended fallback is never reached. Two fixes, both
    needed: (1) prepend R_HOME/bin/x64 to PATH so the DLL loader has a chance even before rpy2's own
    directory-registration runs, (2) monkeypatch the IndexError into the CalledProcessError rpy2 already
    knows how to handle, so its own fallback actually fires.
    """
    global _RPY2_READY
    if _RPY2_READY:
        return

    os.environ["PATH"] = os.path.join(r_home, "bin", "x64") + os.pathsep + os.environ.get("PATH", "")
    os.environ["R_LIBS_USER"] = r_libs_user

    import subprocess
    import rpy2.situation
    _orig_get_r_cmd_config = rpy2.situation._get_r_cmd_config

    def _patched_get_r_cmd_config(r_home_arg, about, allow_empty=False):
        try:
            return _orig_get_r_cmd_config(r_home_arg, about, allow_empty=allow_empty)
        except IndexError:
            raise subprocess.CalledProcessError(1, "R CMD config")

    rpy2.situation._get_r_cmd_config = _patched_get_r_cmd_config

    import rpy2.robjects as ro
    ro.r(f'.libPaths(c("{r_libs_user}", .libPaths()))')
    _RPY2_READY = True


def to_r_dataframe(df: pd.DataFrame, name: str = "dat", build_trial_uid: bool = True) -> None:
    """
    Convert a pandas DataFrame to an R data.frame and assign it to `name` in the R global environment,
    casting the usual grouping columns to factors. If `build_trial_uid` and both `subject`/`trial` columns
    are present, also builds `trial_uid` (subject:trial interaction) for the M10 nested-RE fix - see
    CODE_REVIEW.md M10.
    """
    setup_rpy2()
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    with (ro.default_converter + pandas2ri.converter).context():
        ro.globalenv[name] = df

    for col in ("subject", "trial_category", "target_category"):
        if col in df.columns:
            ro.r(f'{name}${col} <- as.factor({name}${col})')

    if build_trial_uid and "subject" in df.columns and "trial" in df.columns:
        ro.r(f'{name}$trial_uid <- interaction({name}$subject, {name}$trial, drop = TRUE)')


def source_r(path: str) -> None:
    """Source an R file (formula/fit-call only - reads `dat` already in the R global env, assigns fitted
    model objects back into it) into the R global environment via rpy2."""
    setup_rpy2()
    import rpy2.robjects as ro
    ro.r(f'source("{path}")')


def get_r_object(name: str) -> Any:
    """Fetch an object (e.g. a fitted model) from the R global environment by name."""
    setup_rpy2()
    import rpy2.robjects as ro
    return ro.globalenv[name]


def gam_metrics(model: Any, label: str = "") -> dict:
    """
    Extract goodness-of-fit metrics from a fitted mgcv gam/bam object: AIC, BIC, adjusted R-sq (caveated -
    not a true R^2 for binomial), deviance explained, the (f)REML/GCV score, and a per-smooth-term table
    (edf, ref.df, statistic, p-value).
    """
    setup_rpy2()
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    r_summary = ro.r["summary"](model)
    aic = float(ro.r["AIC"](model)[0])
    bic = float(ro.r["BIC"](model)[0])
    r_sq = float(r_summary.rx2("r.sq")[0])
    dev_expl = float(r_summary.rx2("dev.expl")[0])

    s_table = r_summary.rx2("s.table")
    with (ro.default_converter + pandas2ri.converter).context():
        smooth_df = ro.conversion.get_conversion().rpy2py(ro.r["as.data.frame"](s_table))
    smooth_df.index.name = "term"
    smooth_df = smooth_df.reset_index()
    smooth_df.columns = [c.replace(".", "_").replace(">", "").replace("(", "").replace(")", "").lower() for c in smooth_df.columns]

    return {
        "label": label,
        "aic": aic,
        "bic": bic,
        "r_sq_adj": r_sq,
        "dev_expl": dev_expl,
        "n": int(ro.r["nobs"](model)[0]),
        "smooth_terms": smooth_df,
    }


def glmer_metrics(model: Any, label: str = "") -> dict:
    """
    Extract goodness-of-fit metrics from a fitted lme4 glmer object: AIC, BIC, log-likelihood, marginal +
    conditional R^2 (via performance::r2_nakagawa), the fixed-effect coefficient table, random-effect
    variance components, and convergence status.
    """
    setup_rpy2()
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    performance = importr("performance")
    r_summary = ro.r["summary"](model)

    aic_tab = r_summary.rx2("AICtab")
    aic = float(aic_tab.rx2("AIC")[0])
    bic = float(aic_tab.rx2("BIC")[0])
    log_lik = float(aic_tab.rx2("logLik")[0])

    r2 = performance.r2_nakagawa(model)
    r2_marginal = float(r2.rx2("R2_marginal")[0])
    r2_conditional = float(r2.rx2("R2_conditional")[0])

    coef_table = r_summary.rx2("coefficients")
    with (ro.default_converter + pandas2ri.converter).context():
        coef_df = ro.conversion.get_conversion().rpy2py(ro.r["as.data.frame"](coef_table))
    coef_df.index.name = "term"
    coef_df = coef_df.reset_index()
    coef_df.columns = [c.replace(" ", "_").replace(">", "").replace("(", "").replace(")", "").replace("|", "").lower() for c in coef_df.columns]

    is_singular = bool(ro.r["isSingular"](model)[0])
    # lme4 attaches messages to optinfo$conv$lme4$messages when something went wrong; length 0 (R NULL)
    # means clean convergence.
    ro.globalenv["rbridge_tmp_model"] = model
    conv_messages = ro.r("rbridge_tmp_model@optinfo$conv$lme4$messages")
    converged = conv_messages is ro.NULL or len(conv_messages) == 0

    return {
        "label": label,
        "aic": aic,
        "bic": bic,
        "log_lik": log_lik,
        "r2_marginal": r2_marginal,
        "r2_conditional": r2_conditional,
        "n": int(ro.r["nobs"](model)[0]),
        "coefficients": coef_df,
        "is_singular": is_singular,
        "converged": converged,
    }


def predict_gam(model: Any, grid: pd.DataFrame, exclude: Optional[list] = None) -> np.ndarray:
    """Predict response-scale probabilities from a fitted gam/bam model over `grid` (a pandas DataFrame of
    predictor combinations), optionally excluding named smooth terms (e.g. ["s(trial_uid)"])."""
    setup_rpy2()
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    with (ro.default_converter + pandas2ri.converter).context():
        r_grid = ro.conversion.get_conversion().py2rpy(grid)
        r_exclude = ro.StrVector(exclude) if exclude else ro.NULL
        preds = ro.r["predict"](model, newdata=r_grid, type="response", exclude=r_exclude)
        return np.asarray(preds)


def cached_fit(cache_path: str, fit_fn, force_refit: bool = False):
    """
    Fit-or-load: if `cache_path` exists and `force_refit` is False, unpickle and return its contents;
    otherwise call `fit_fn()` (which must return a plain-Python-picklable object - metrics dicts and
    DataFrames, not live rpy2 model handles, which can't survive a pickle round-trip), pickle the result
    to `cache_path`, and return it.
    """
    if not force_refit and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    result = fit_fn()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result
