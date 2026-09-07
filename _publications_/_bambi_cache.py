"""Generic fit-or-load-from-cache pipeline for a Bambi model, independent of any specific
data shape or formula. Callers supply how to turn a DataStore into a model-ready DataFrame
via `prepare_data`; this module only knows about fitting, predicting, and caching to disk."""
import time
from typing import Any, Callable, Tuple

import pandas as pd
import bambi as bmb
import arviz as az

import config as cnfg
from analysis.helpers.read_data import load_data


def load_cached_data(
        model_data_path: str, idata_path: str, preds_path: str
) -> Tuple[pd.DataFrame, az.InferenceData, az.InferenceData]:
    model_data = pd.read_pickle(model_data_path)
    idata = az.from_netcdf(idata_path)
    preds = az.from_netcdf(preds_path)
    return model_data, idata, preds


def execute_pipeline(
        prepare_data: Callable[[Any], pd.DataFrame],
        model_data_path: str, idata_path: str, preds_path: str,
        formula: str,
        make_predictors: Callable[[pd.DataFrame], pd.DataFrame],
        seed: int = 42,
) -> Tuple[pd.DataFrame, az.InferenceData, az.InferenceData]:
    print("====\tRunning Inference Pipeline\t====")
    start = time.time()

    print("Preparing data for a new model...")
    data = load_data(cnfg.OUTPUT_PATH)
    model_data = prepare_data(data)
    model_data.to_pickle(model_data_path)
    print(f"MODEL_DATA saved to {model_data_path}")

    print(f"Creating a model with formula:\n\t{formula}")
    model = bmb.Model(formula, model_data, family="bernoulli")

    print("Fitting the model to get new idata...")
    fit_start = time.time()
    idata = model.fit(
        draws=2000, tune=1000, chains=4, cores=2, target_accept=0.95, progressbar=False, random_seed=seed,
    )
    fit_elapsed = time.time() - fit_start
    print(f"Model fitting completed in {int(fit_elapsed // 3600)}:{int((fit_elapsed % 3600) // 60)}:{fit_elapsed % 60:.2f} (hh:mm:ss)")
    az.to_netcdf(idata, idata_path)
    print(f"IDATA saved to {idata_path}")

    print("Generating predictions from the fitted model...")
    predictors = make_predictors(model_data)
    preds = model.predict(idata, data=predictors, inplace=False, kind="response")
    az.to_netcdf(preds, preds_path)
    print(f"PREDS saved to {preds_path}")

    elapsed = time.time() - start
    print(f"====\tInference Pipeline Completed in {int(elapsed // 3600)}:{int((elapsed % 3600) // 60)}:{elapsed % 60:.2f} (hh:mm:ss)\t====")
    return model_data, idata, preds


def load_or_fit_model(
        prepare_data: Callable[[Any], pd.DataFrame],
        model_data_path: str, idata_path: str, preds_path: str,
        formula: str,
        make_predictors: Callable[[pd.DataFrame], pd.DataFrame],
        seed: int = 42,
) -> Tuple[pd.DataFrame, az.InferenceData, az.InferenceData]:
    try:
        model_data, idata, preds = load_cached_data(model_data_path, idata_path, preds_path)
        print("Data loaded successfully from disk.")
    except FileNotFoundError:
        print("Data files not found. Executing the full pipeline to generate them...")
        model_data, idata, preds = execute_pipeline(
            prepare_data, model_data_path, idata_path, preds_path,
            formula=formula, make_predictors=make_predictors, seed=seed,
        )
    return model_data, idata, preds
