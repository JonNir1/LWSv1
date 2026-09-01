"""Shared data-loading, Bayesian model-fitting, and contrast-analysis helpers for the poster notebooks."""
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
import xarray as xr

import config as cnfg
from analysis.helpers.read_data import load_data
from pipeline.stage3_classify.build_funnels import build_event_classification_funnel

DEFAULT_FORMULA = "is_lws ~ trial_category * target_category * centered_abs_target_angle + (1 | subject)"


def load_cached_data(
        model_data_path: str, idata_path: str, preds_path: str
) -> Tuple[pd.DataFrame, az.InferenceData, az.InferenceData]:
    model_data = pd.read_pickle(model_data_path)
    idata = az.from_netcdf(idata_path)
    preds = az.from_netcdf(preds_path)
    return model_data, idata, preds


def execute_pipeline(
        event_type: Literal["fixation", "visit"],
        model_data_path: str, idata_path: str, preds_path: str,
        subjects: Optional[List[int]] = None,
        seed: int = 42,
        formula: str = DEFAULT_FORMULA,
) -> Tuple[pd.DataFrame, az.InferenceData, az.InferenceData]:
    print("====\tRunning Inference Pipeline\t====")
    start = time.time()

    print("Preparing data for a new model...")
    data = load_data(cnfg.OUTPUT_PATH)
    model_data = _load_and_prepare_model_data(data, event_type=event_type, subjects=subjects, verbose=True)
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
    predictors = _make_predictors_df(model_data)
    preds = model.predict(idata, data=predictors, inplace=False, kind="response")
    az.to_netcdf(preds, preds_path)
    print(f"PREDS saved to {preds_path}")

    elapsed = time.time() - start
    print(f"====\tInference Pipeline Completed in {int(elapsed // 3600)}:{int((elapsed % 3600) // 60)}:{elapsed % 60:.2f} (hh:mm:ss)\t====")
    return model_data, idata, preds


def load_or_fit_model(
        event_type: Literal["fixation", "visit"],
        model_data_path: str, idata_path: str, preds_path: str,
        subjects: Optional[List[int]] = None,
        seed: int = 42,
        formula: str = DEFAULT_FORMULA,
) -> Tuple[pd.DataFrame, az.InferenceData, az.InferenceData]:
    try:
        model_data, idata, preds = load_cached_data(model_data_path, idata_path, preds_path)
        print("Data loaded successfully from disk.")
    except FileNotFoundError:
        print("Data files not found. Executing the full pipeline to generate them...")
        model_data, idata, preds = execute_pipeline(
            event_type, model_data_path, idata_path, preds_path, subjects=subjects, seed=seed, formula=formula,
        )
    return model_data, idata, preds


def _load_and_prepare_model_data(
        data,
        event_type: Literal["fixation", "visit"],
        subjects: Optional[List[int]] = None,
        verbose: bool = True,
) -> pd.DataFrame:
    """ Loads the funnel data and filters out the irrelevant rows and columns for fitting the Bayesian model """
    funnel_data = build_event_classification_funnel(data, "lws", event_type)
    valid_trials = data.trial_funnel.loc[data.trial_funnel["is_valid_trial"], ["subject", "trial"]]
    funnel_data = funnel_data.merge(valid_trials, on=["subject", "trial"], how="inner")
    if subjects is not None:
        funnel_data = funnel_data.query("subject in @subjects")
    model_data = (
        funnel_data
        .query("upto_on_target == 1")
        .loc[:, ["subject", "trial_category", "target_category", "target_angle", "is_lws"]]
        .assign(
            abs_target_angle=lambda df: df["target_angle"].abs(),
            centered_abs_target_angle=lambda df: df["abs_target_angle"] - df["abs_target_angle"].mean()
        )
        .copy()
    )
    if verbose:
        print(f"Full dataset size: {len(funnel_data)} rows")
        print(f"Model dataset size: {len(model_data)} rows ({100 * len(model_data) / len(funnel_data) :.1f}%)")
        print(f"Overall average P[miss | on-target]: {100 * model_data['is_lws'].mean() :.1f}%")
        print(f"Subjects included ({model_data['subject'].nunique()}): {sorted(model_data['subject'].unique().tolist())}")
    return model_data


def _make_predictors_df(model_data: pd.DataFrame) -> pd.DataFrame:
    """ Creates a DataFrame with all combinations of the predictor levels for generating predictions from the model """
    pred_cols = ["subject", "trial_category", "target_category", "centered_abs_target_angle"]
    predictors = pd.DataFrame(product(*[sorted(model_data[col].unique()) for col in pred_cols]), columns=pred_cols)
    return predictors


@dataclass(frozen=True)
class AnalysisContext:
    """ Bundles the model data, predictors, and posterior predictions that ContrastGroupType/ContrastType operate on. """
    model_data: pd.DataFrame
    predictors: pd.DataFrame
    posterior_probs: xr.DataArray


@dataclass(frozen=True)
class ContrastGroupSelectorType:
    predictor: str
    levels: Tuple[Any, ...]

    def __hash__(self):
        return hash((self.predictor, self.levels))


@dataclass(frozen=True)
class ContrastGroupType:
    name: str
    selectors: Tuple[ContrastGroupSelectorType, ...]

    def __post_init__(self):
        # validate no intersecting selectors (selectors on the same predictor variable)
        predictor_counts = Counter(sel.predictor for sel in self.selectors)
        duplicate_predictors = [p for p, c in predictor_counts.items() if c > 1]
        if duplicate_predictors:
            raise ValueError(
                f"Contrast Group '{self.name}' has overlapping selectors for predictor(s): {duplicate_predictors}"
            )

    def extract_empirical_probabilities_per_subject(self, context: AnalysisContext) -> pd.DataFrame:
        idxs = self._get_indices_for_selectors(context.model_data)
        subset = context.model_data.iloc[idxs]
        aggregates = (
            subset
            .groupby("subject", observed=True)["is_lws"]
            .agg(["count", "mean", "sem"])
            .fillna({"count": 0, "sem": 0})
        )
        return aggregates

    def extract_posteriors_probabilities_per_subject(self, context: AnalysisContext) -> xr.DataArray:
        selector_idxs = self._get_indices_for_selectors(context.predictors)
        subj_ids, subj_probs = [], []
        for subj in context.predictors["subject"].unique():
            subject_idxs = np.where(context.predictors["subject"] == subj)[0]
            idxs = np.intersect1d(subject_idxs, selector_idxs)
            probs = context.posterior_probs.isel(__obs__=idxs).mean(dim="__obs__")      # shape (num_chains, num_draws)
            subj_ids.append(subj)
            subj_probs.append(probs)
        out = xr.concat(subj_probs, dim="subject").assign_coords(subject=subj_ids)
        return out

    def aggregate_posterior(self, context: AnalysisContext) -> Tuple[float, float]:
        posteriors = self.extract_posteriors_probabilities_per_subject(context)
        mean_per_subject = posteriors.mean(dim=["chain", "draw"]).to_pandas()
        overall_mean = mean_per_subject.mean()
        overall_sem = mean_per_subject.sem()
        return overall_mean, overall_sem

    @staticmethod
    def intersect_groups(groups: List["ContrastGroupType"]) -> "ContrastGroupType":
        if not groups:
            raise ValueError("Cannot intersect an empty list of groups.")

        by_predictor = defaultdict(list)
        for group in groups:
            for sel in group.selectors:
                by_predictor[sel.predictor].append(sel)

        intersected_selectors = []
        for predictor, sels in by_predictor.items():
            shared_levels = set(sels[0].levels)
            for sel in sels[1:]:
                shared_levels &= set(sel.levels)

            if not shared_levels:
                group_names = [g.name for g in groups]
                raise ValueError(
                    f"No shared levels found for predictor '{predictor}' "
                    f"when intersecting groups {group_names}."
                )

            # preserve the order from the first selector
            ordered_shared_levels = tuple(
                lvl for lvl in sels[0].levels if lvl in shared_levels
            )

            intersected_selectors.append(
                ContrastGroupSelectorType(
                    predictor=predictor,
                    levels=ordered_shared_levels,
                )
            )

        name = " ∩ ".join(group.name for group in groups)
        return ContrastGroupType(
            name=name,
            selectors=tuple(intersected_selectors),
        )

    def _get_indices_for_selectors(self, df: pd.DataFrame) -> np.ndarray:
        missing_predictors = [sel.predictor for sel in self.selectors if sel.predictor not in df.columns]
        if missing_predictors:
            raise KeyError(f"Predictors {missing_predictors} not found in DataFrame.")
        mask = pd.Series(True, index=df.index)
        for selector in self.selectors:
            mask &= df[selector.predictor].isin(selector.levels)
        idxs = np.where(mask)[0]
        return idxs

    def __hash__(self):
        return hash((self.name, self.selectors))


@dataclass(frozen=True)
class ContrastType:
    name: str
    group1: ContrastGroupType
    group2: ContrastGroupType

    def calculate_contrast(self, context: AnalysisContext, verbose: bool = True) -> dict:
        out = dict()
        mean1, sem1 = self.group1.aggregate_posterior(context)
        out[self.group1.name] = {"mean": mean1, "sem": sem1}
        mean2, sem2 = self.group2.aggregate_posterior(context)
        out[self.group2.name] = {"mean": mean2, "sem": sem2}
        p_positive = self.calculate_positive_contrast_probability(context)
        out["p_positive"] = p_positive
        if verbose:
            print(f"Posterior P[LWS | {self.group1.name}] = {100 * mean1 :.2f} ± {100 * sem1 :.1f} %")
            print(f"Posterior P[LWS | {self.group2.name}] = {100 * mean2 :.2f} ± {100 * sem2 :.1f} %")
            print(f"Probability that P[LWS | {self.group1.name}] > P[LWS | {self.group2.name}]\t::\t{p_positive:.4f}")
        return out

    def calculate_positive_contrast_probability(self, context: AnalysisContext) -> float:
        posteriors1 = self.group1.extract_posteriors_probabilities_per_subject(context)
        posteriors2 = self.group2.extract_posteriors_probabilities_per_subject(context)
        aligned1, aligned2 = xr.align(posteriors1, posteriors2, join="exact")   # safety: make sure both arrays have the same subjects
        contrast_per_subject = aligned1 - aligned2
        averaged_diff = contrast_per_subject.mean(dim="subject")
        p_over_zero = (averaged_diff > 0).mean()
        return p_over_zero

    def plot_posteriors(self, context: AnalysisContext) -> None:
        from _publications_._plotting import plot_posterior_distributions
        plot_posterior_distributions([self.group1, self.group2], context, title=f"Posterior Distributions: {self.name}")
