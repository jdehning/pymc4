import collections
import numpy as np
import arviz as az
from typing import Optional, Tuple, List, Dict, Any

KERNEL_KWARGS_SET = collections.namedtuple(
    "KERNEL_ARGS_SET", ["kernel", "adaptive_kernel", "kernel_kwargs", "adaptive_kwargs"]
)

from pymc4 import Model, flow
from pymc4.distributions import distribution


def initialize_sampling_state(
    model: Model, observed: Optional[dict] = None, state: Optional[flow.SamplingState] = None,
) -> Tuple[flow.SamplingState, List[str]]:
    """
    Initialize the model provided state and/or observed variables.
    Parameters
    ----------
    model : pymc4.Model
    observed : Optional[dict]
    state : Optional[flow.SamplingState]
    Returns
    -------
    state: pymc4.flow.SamplingState
        The model's sampling state
    deterministic_names: List[str]
        The list of names of the model's deterministics_values
    """
    _, state = flow.evaluate_model_transformed(model, observed=observed, state=state)
    deterministic_names = list(state.deterministics_values)
    state, transformed_names = state.as_sampling_state()
    return state, deterministic_names + transformed_names


def initialize_state(
    model: Model, observed: Optional[dict] = None, state: Optional[flow.SamplingState] = None,
) -> Tuple[
    flow.SamplingState,
    flow.SamplingState,
    List[str],
    List[str],
    Dict[str, distribution.Distribution],
    Dict[str, distribution.Distribution],
]:
    """
    Get list of discrete/continuous distributions
    Parameters
    ----------
    model : pymc4.Model
    observed : Optional[dict]
    state : Optional[flow.SamplingState]
    Returns
    -------
    state: Model
        Unsampled version of sample object
    sampling_state:
        The model's sampling state
    free_discrete_names: List[str]
        The list of free discrete variables
    free_continuous_names: List[str]
        The list of free continuous variables
    cont_distr: List[distribution.Distribution]
        The list of all continous distributions
    disc_distr: List[distribution.Distribution]
        The list of all discrete distributions
    """
    _, state = flow.evaluate_model_transformed(model)
    free_discrete_names, free_continuous_names = (
        list(state.discrete_distributions),
        list(state.continuous_distributions),
    )
    observed_rvs = list(state.observed_values.keys())
    free_discrete_names = list(filter(lambda x: x not in observed_rvs, free_discrete_names))
    free_continuous_names = list(filter(lambda x: x not in observed_rvs, free_continuous_names))
    sampling_state = None
    cont_distrs = state.continuous_distributions
    disc_distrs = state.discrete_distributions
    sampling_state, _ = state.as_sampling_state()
    return (
        state,
        sampling_state,
        free_discrete_names,
        free_continuous_names,
        cont_distrs,
        disc_distrs,
    )


def trace_to_arviz(
    trace=None,
    sample_stats=None,
    observed_data=None,
    prior_predictive=None,
    posterior_predictive=None,
    inplace=True,
):
    """
    Tensorflow to Arviz trace convertor.
    Creates an ArviZ's InferenceData object with inference, prediction and/or sampling data
    generated by PyMC4
    Parameters
    ----------
    trace : dict or InferenceData
    sample_stats : dict
    observed_data : dict
    prior_predictive : dict
    posterior_predictive : dict
    inplace : bool
    Returns
    -------
    ArviZ's InferenceData object
    """

    # Replace all / with | because arviz and xarray are confusion / with folder structure in
    # special cases!

    if trace is not None and isinstance(trace, dict):
        trace = {
            k.replace("/", "|"): np.swapaxes(v.numpy(), 1, 0) for k, v in trace.items() if "/" in k
        }
    if sample_stats is not None and isinstance(sample_stats, dict):
        sample_stats = {k: np.swapaxes(v.numpy(), 1, 0) for k, v in sample_stats.items()}
    if prior_predictive is not None and isinstance(prior_predictive, dict):
        prior_predictive = {k: v[np.newaxis] for k, v in prior_predictive.items()}
    if posterior_predictive is not None and isinstance(posterior_predictive, dict):
        if isinstance(trace, az.InferenceData) and inplace == True:
            return trace + az.from_dict(posterior_predictive=posterior_predictive)
        else:
            trace = None

    return az.from_dict(
        posterior=trace,
        sample_stats=sample_stats,
        prior_predictive=prior_predictive,
        posterior_predictive=posterior_predictive,
        observed_data=observed_data,
    )


def scope_remove_transformed_part_if_required(name: str, transformed_values: Dict[str, Any]):
    name_split = name.split("/")
    if transformed_values and name in transformed_values:
        name_split[-1] = name_split[-1][2:][name_split[-1][2:].find("_") + 1 :]
    return "/".join(name_split), name_split[-1]
