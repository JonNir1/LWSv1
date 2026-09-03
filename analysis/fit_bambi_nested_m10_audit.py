"""
M10 severity audit: bambi nested random-effects fit.

_publications_/_modeling.py's cached model (`is_lws ~ trial_category * target_category *
centered_abs_target_angle + (1 | subject)`, visit-level, used by both `_publications_/2026_mind-il.ipynb`
and `stimulus_features.ipynb`'s Bayesian section - identical formula and data, so this session's plan
treats them as one shared comparison rather than fitting the nested version twice) has a flat `(1|subject)`
random effect only - the same M10 concern documented throughout this repo's frequentist analyses. This
script fits the nested `(1|subject/trial)` companion at the same spec as the cached flat fit (2000 draws /
1000 tune / 4 chains / 2 cores), so the two are directly comparable.

Run as a script, not a notebook cell, to keep this multi-*-hour fit out of interactive execution:
    C:\\Users\\nirjo\\Documents\\University\\PhD\\Projects\\LWSv1\\.venv\\Scripts\\python.exe analysis/fit_bambi_nested_m10_audit.py

Writes alongside the existing cache (never overwrites it):
    visit_idata_nested.nc, visit_data_nested.pkl (same model_data, kept alongside for traceability)
and a small `bambi_m10_metrics.pkl` with R^2/WAIC/LOO for both the flat and nested fits, consumed by the
M10 severity HTML report.
"""
import os
import pickle
import time

import pytensor
# PyTensor's C-compiler backend fails to compile the graph compute_log_likelihood() builds for a fitted
# model of this size (confirmed reproduction: g++ chokes trying to embed the full posterior array,
# shape (4 chains, 2000 draws, 5605 obs), as inline C source). Force the pure-Python/numpy fallback
# instead - slower per-call, but this only runs a handful of times on already-drawn samples, not during
# the expensive MCMC sampling itself, so the cost is negligible here.
pytensor.config.cxx = ""

import arviz as az
import bambi as bmb
import pandas as pd

import config as cnfg
from analysis.helpers.read_data import load_data
from pipeline.stage3_classify.build_funnels import build_event_classification_funnel

FLAT_FORMULA = "is_lws ~ trial_category * target_category * centered_abs_target_angle + (1 | subject)"
NESTED_FORMULA = "is_lws ~ trial_category * target_category * centered_abs_target_angle + (1 | subject/trial)"
SEED = 42

OUTPUT_PATH = os.path.join(cnfg.PUBLICATIONS_PATH, "2026_10_Mind-IL")
FLAT_DATA_PATH = os.path.join(OUTPUT_PATH, "visit_data.pkl")
FLAT_IDATA_PATH = os.path.join(OUTPUT_PATH, "visit_idata.nc")
NESTED_IDATA_PATH = os.path.join(OUTPUT_PATH, "visit_idata_nested.nc")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "R", "_cache", "bambi_m10_metrics.pkl")


def _model_metrics(model: bmb.Model, idata: az.InferenceData, model_data: pd.DataFrame, label: str) -> dict:
    """R^2, WAIC, LOO for a fitted bambi model. compute_log_likelihood()/predict() are needed first -
    bambi doesn't populate posterior_predictive/log_likelihood on `idata` during .fit() by default."""
    model.predict(idata, data=model_data, inplace=True, kind="response")
    model.compute_log_likelihood(idata, data=model_data, inplace=True)
    r2 = model.r2_score(idata, summary=True)
    waic = az.waic(idata)
    loo = az.loo(idata)
    return {
        "label": label,
        "r2_mean": float(r2["r2"]),
        "r2_std": float(r2["r2_std"]),
        "waic": float(waic.elpd_waic),
        "waic_se": float(waic.se),
        "loo": float(loo.elpd_loo),
        "loo_se": float(loo.se),
        "n_divergences": int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else None,
    }


def _build_model_data_with_trial() -> pd.DataFrame:
    """Same filtering/columns as `_publications_/_modeling.py::_load_and_prepare_model_data()`, but keeping
    `trial` (dropped there), needed here for the `(1 | subject/trial)` nested random effect. Deterministic
    given the same pickles (no randomness in the filter/query chain), so this reproduces the exact same
    rows, in the same order, as the cached `visit_data.pkl` used to fit the flat model - verified below
    before it's used for anything - so the extra `trial` column is the only difference."""
    data = load_data(cnfg.OUTPUT_PATH)
    funnel_data = build_event_classification_funnel(data, "lws", "visit")
    valid_trials = data.trial_funnel.loc[data.trial_funnel["is_valid_trial"], ["subject", "trial"]]
    funnel_data = funnel_data.merge(valid_trials, on=["subject", "trial"], how="inner")
    return (
        funnel_data
        .query("upto_on_target == 1")
        .loc[:, ["subject", "trial", "trial_category", "target_category", "target_angle", "is_lws"]]
        .assign(
            abs_target_angle=lambda df: df["target_angle"].abs(),
            centered_abs_target_angle=lambda df: df["abs_target_angle"] - df["abs_target_angle"].mean(),
        )
        .reset_index(drop=True)
        .copy()
    )


def main():
    if not (os.path.exists(FLAT_DATA_PATH) and os.path.exists(FLAT_IDATA_PATH)):
        raise FileNotFoundError(
            f"Expected cached flat model at {FLAT_DATA_PATH} / {FLAT_IDATA_PATH} - "
            "run _publications_/2026_mind-il.ipynb (or stimulus_features.ipynb's Bayesian section) first."
        )

    cached_model_data = pd.read_pickle(FLAT_DATA_PATH)
    model_data = _build_model_data_with_trial()
    print(f"Rebuilt model_data (with trial): {len(model_data)} rows (cached flat model_data: {len(cached_model_data)} rows)")
    shared_cols = ["subject", "trial_category", "target_category", "centered_abs_target_angle", "is_lws"]
    if len(model_data) != len(cached_model_data) or not model_data[shared_cols].equals(cached_model_data[shared_cols]):
        raise ValueError(
            "Rebuilt model_data does not match the cached flat model_data on shared columns - "
            "the flat idata's fit and this script's nested fit would not be on the same rows."
        )
    print("Rebuilt model_data matches the cached flat model_data on all shared columns - safe to reuse for both fits.")

    flat_idata = az.from_netcdf(FLAT_IDATA_PATH)
    flat_model = bmb.Model(FLAT_FORMULA, model_data, family="bernoulli")
    print("Computing flat model metrics (R^2/WAIC/LOO)...")
    flat_metrics = _model_metrics(flat_model, flat_idata, model_data, "flat (1|subject)")
    print(flat_metrics)

    if os.path.exists(NESTED_IDATA_PATH):
        print(f"Nested idata already cached at {NESTED_IDATA_PATH}, loading instead of refitting.")
        nested_idata = az.from_netcdf(NESTED_IDATA_PATH)
        nested_model = bmb.Model(NESTED_FORMULA, model_data, family="bernoulli")
    else:
        nested_model = bmb.Model(NESTED_FORMULA, model_data, family="bernoulli")
        print(f"Fitting nested model ({NESTED_FORMULA})...")
        start = time.time()
        nested_idata = nested_model.fit(
            draws=2000, tune=1000, chains=4, cores=2, target_accept=0.95, progressbar=False, random_seed=SEED,
        )
        elapsed = time.time() - start
        print(f"Nested fit completed in {int(elapsed // 3600)}:{int((elapsed % 3600) // 60)}:{elapsed % 60:.2f} (hh:mm:ss)")
        az.to_netcdf(nested_idata, NESTED_IDATA_PATH)
        print(f"Saved nested idata to {NESTED_IDATA_PATH}")

    print("Computing nested model metrics (R^2/WAIC/LOO)...")
    nested_metrics = _model_metrics(nested_model, nested_idata, model_data, "nested (1|subject/trial)")
    print(nested_metrics)

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "wb") as f:
        pickle.dump({"flat": flat_metrics, "nested": nested_metrics}, f)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
