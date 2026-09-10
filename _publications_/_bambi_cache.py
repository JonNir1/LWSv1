"""Generic fit-or-load-from-cache pipeline for a Bambi model, independent of any specific
data shape or formula. Two symmetric pairs: load_or_fit (data + formula -> model, idata) and
load_or_predict (model, idata + predictors -> preds), so a caller that needs to fit several
candidate models before choosing one to predict from (e.g. model selection via az.compare())
isn't forced through a single fit-then-predict step.

load_or_fit never persists the fitted Model object itself, only idata: bambi's Model.predict()
only needs the formula/data to (re)build its design-matrix machinery, not the fitted PyMC
backend, so a cache hit rebuilds a fresh, unfit Model from the caller-supplied model_data and
pairs it with the cached idata. This also sidesteps a real failure mode of pickling a fitted
Model from inside a Jupyter kernel: ipykernel monkeypatches the `input` builtin, and something
in bambi/PyMC's object graph holds a stale reference to it, which cloudpickle refuses to
pickle ("it's not the same object as builtins.input")."""
import contextlib
import logging
import os
import time
import warnings
from typing import Optional, Tuple

import pandas as pd
import bambi as bmb
import arviz as az

_QUIET_LOGGER_NAMES = ["pymc", "bambi"]


@contextlib.contextmanager
def _quiet_fit(show_warnings: bool):
    if show_warnings:
        yield
        return
    loggers = [logging.getLogger(name) for name in _QUIET_LOGGER_NAMES]
    previous_levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            for logger, level in zip(loggers, previous_levels):
                logger.setLevel(level)


def _idata_path(name: str, dir_path: str) -> str:
    return os.path.join(dir_path, f"{name}_idata.nc")


def _fit_model(
        model_data: pd.DataFrame,
        formula: str,
        name: str,
        dir_path: Optional[str],
        seed: int = 42,
        idata_kwargs: Optional[dict] = None,
        verbose: bool = False,
        show_warnings: bool = False,
) -> Tuple[bmb.Model, az.InferenceData]:
    if verbose:
        print(f"Fitting model with formula {formula}")
    start = time.time()
    with _quiet_fit(show_warnings):
        model = bmb.Model(formula, model_data, family="bernoulli")
        idata = model.fit(
            draws=2000, tune=1000, chains=4, cores=2, target_accept=0.95,
            progressbar=False, random_seed=seed, idata_kwargs=idata_kwargs or {},
        )
    elapsed = time.time() - start
    if verbose:
        print(f"Model fitted in {int(elapsed // 60)}:{elapsed % 60:05.2f} mm:ss")

    if dir_path is not None:
        os.makedirs(dir_path, exist_ok=True)
        idata_path = _idata_path(name, dir_path)
        az.to_netcdf(idata, idata_path)
        if verbose:
            print(f"Idata saved to {idata_path}")

    return model, idata


def load_or_fit(
        model_data: pd.DataFrame,
        formula: str,
        name: str,
        dir_path: Optional[str],
        seed: int = 42,
        idata_kwargs: Optional[dict] = None,
        force_fit: bool = False,
        verbose: bool = False,
        show_warnings: bool = False,
) -> Tuple[bmb.Model, az.InferenceData]:
    if dir_path is not None and not force_fit:
        idata_path = _idata_path(name, dir_path)
        if os.path.isfile(idata_path):
            model = bmb.Model(formula, model_data, family="bernoulli")
            with az.rc_context({"data.load": "eager"}):
                idata = az.from_netcdf(idata_path)
            if verbose:
                print(f"Loaded cached idata from {idata_path}")
            return model, idata

    return _fit_model(
        model_data, formula, name, dir_path,
        seed=seed, idata_kwargs=idata_kwargs, verbose=verbose, show_warnings=show_warnings,
    )


def load_or_predict(
        model: bmb.Model,
        idata: az.InferenceData,
        predictors: pd.DataFrame,
        name: str,
        dir_path: Optional[str],
        force_predict: bool = False,
        verbose: bool = False,
) -> az.InferenceData:
    preds_path = os.path.join(dir_path, f"{name}_preds.nc") if dir_path is not None else None

    if preds_path is not None and not force_predict and os.path.isfile(preds_path):
        if verbose:
            print(f"Loaded cached predictions from {preds_path}")
        with az.rc_context({"data.load": "eager"}):
            return az.from_netcdf(preds_path)

    preds = model.predict(idata, data=predictors, inplace=False, kind="response")
    if preds_path is not None:
        az.to_netcdf(preds, preds_path)
        if verbose:
            print(f"Predictions saved to {preds_path}")
    return preds
