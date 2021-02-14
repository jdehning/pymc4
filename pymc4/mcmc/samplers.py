import abc
import itertools
import inspect
from typing import Optional, List, Union, Any, Dict
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import mcmc

import logging
from pymc4.mcmc.utils import (
    initialize_sampling_state,
    trace_to_arviz,
    initialize_state,
    scope_remove_transformed_part_if_required,
    KERNEL_KWARGS_SET,
)
import tqdm

from functools import partial

from pymc4.coroutine_model import Model
from pymc4.utils import NameParts
from pymc4 import flow
from pymc4.mcmc.tf_support import _CompoundStepTF
import logging
import numpy as np

log = logging.getLogger(__name__)


MYPY = False

__all__ = ["HMC", "NUTS", "RandomWalkM", "CompoundStep", "NUTSSimple", "HMCSimple"]
reg_samplers = {}

# TODO: better design for logging
console = logging.StreamHandler()
_log = logging.getLogger("pymc4.sampling")
if not MYPY:
    logging._warn_preinit_stderr = 0
    _log.root.handlers = []  # remove tf absl logging handlers
_log.setLevel(logging.INFO)
_log.addHandler(console)


def register_sampler(cls):
    reg_samplers[cls._name] = cls
    return cls


class _BaseSampler(metaclass=abc.ABCMeta):
    _grad = False

    def __init__(
        self,
        model: Model,
        num_chains=2,
        num_samples_binning=10,
        init=None,
        init_state=None,
        step_size=1e-4,
        state=None,
        observed=None,
        use_auto_batching=True,
        xla=False,
        bijector=None,
        seed: Optional[int] = None,
        is_compound=False,
        step_size_adaption_per_chain=False,
        **kwargs,
    ):
        if not isinstance(model, Model):
            raise TypeError(
                "`sample` function only supports `pymc4.Model` objects, but \
                    you've passed `{}`".format(
                    type(model)
                )
            )

        _, _, disc_names, cont_names, _, _ = initialize_state(model)
        # if sampler has the gradient calculation during `one_step`
        # and the model contains discrete distributions then we throw the
        # error.
        if self._grad is True and disc_names:
            raise ValueError(
                "Discrete distributions can't be used with \
                    gradient-based sampler"
            )

        self.model = model
        self.xla = xla
        self.num_chains = np.array(num_chains, dtype="int32")
        self.num_samples_binning = num_samples_binning
        self.is_compound = False
        self.stat_names: List[str] = []
        self.parent_inds: List[int] = []
        # assign arguments from **kwargs to distinct kwargs for
        # `kernel`, `adaptation_kernel`, `chain_sampler`
        self._assign_arguments(kwargs)
        self._bound_kwargs()
        # check arguments for correctness
        self._check_arguments()
        # TODO: problem with tf.function when passing as argument to self._run_chains
        self.seed = seed
        self.step_size_adaption_per_chain = step_size_adaption_per_chain
        self._bijector = bijector

        if state is not None and observed is not None:
            raise ValueError("Can't use both `state` and `observed` arguments")
        (
            logpfn,
            init_random,
            _deterministics_callback,
            deterministic_names,
            state_,
        ) = build_logp_and_deterministic_functions(
            self.model,
            num_chains=self.num_chains,
            state=state,
            observed=observed,
            collect_reduced_log_prob=use_auto_batching,
            parent_inds=self.parent_inds if self.is_compound else None,
        )

        if use_auto_batching:
            self.parallel_logpfn = vectorize_logp_function(logpfn)
            self.deterministics_callback = vectorize_logp_function(_deterministics_callback)
        else:
            self.parallel_logpfn = logpfn
            self.deterministics_callback = _deterministics_callback

        if init_state is None:
            if init is None:
                init_state = list(init_random.values())
                init_keys = list(init_random.keys())
            else:
                tf.print(f"init:\n{list(init.items())[-1]}")
                init_state = [init[key] for key in init_random.keys()]
                init_keys = list(init_random.keys())
            tf.print(f"init state:\n{init_keys[-1]}\n{init_state[-1]} ")
            if self.is_compound:
                init_state = [init[i] for i in self.parent_inds]
                init_keys = [init_keys[i] for i in self.parent_inds]
            init_state = tile_init(init_state, num_chains)
        else:
            init_state = init_state
            init_keys = list(init_random.keys())

        if hasattr(step_size, "__len__"):
            step_size = step_size
        else:
            step_size = tf.convert_to_tensor(step_size, dtype=init_state[0].dtype)
            step_size = [
                step_size * tf.ones(init_part.shape[1:], dtype=init_part.dtype)
                for init_part in init_state
            ]
            # The dimension size of 1 in the leading dimension has as consequence that
            # the step size is averaged across chains. Change to num_chains to get
            # and individual adaption per chain during sampling:
            step_size = tile_init(step_size, num_chains if step_size_adaption_per_chain else 1)

        current_state = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x, name="current_state"), init_state
        )
        kernel_tmp = self._kernel(
            target_log_prob_fn=self.parallel_logpfn, step_size=step_size, **self.kernel_kwargs
        )
        if self._bijector:
            kernel_tmp = mcmc.TransformedTransitionKernel(
                inner_kernel=kernel_tmp,
                bijector=self._bijector
            )

            # trace_fn has to be redefined, as the statistics are now one kernel deeper.
            def trace_fn(self, current_state: flow.SamplingState,
                         pkr: Union[tf.Tensor, Any]):
                return (
                           pkr.inner_results.inner_results.target_log_prob,
                           pkr.inner_results.inner_results.leapfrogs_taken,
                           pkr.inner_results.inner_results.has_divergence,
                           pkr.inner_results.inner_results.energy,
                           pkr.inner_results.log_accept_ratio,
                           tf.tile(
                               tf.stack(tf.nest.map_structure(tf.reduce_mean,
                                                              pkr.new_step_size))[
                                   tf.newaxis],
                               tuple(pkr.inner_results.inner_results.target_log_prob.shape) + (1,),
                           ),
                       ) + tuple(self.deterministics_callback(*current_state))
            self.trace_fn = trace_fn()

        if self._adaptation:
            adapt_kernel_tmp = self._adaptation(
                inner_kernel=kernel_tmp,
                reduce_fn=tfp.math.reduce_log_harmonic_mean_exp,
                num_adaptation_steps=tf.convert_to_tensor(self.num_samples_binning, dtype="int32"),
                shrinkage_target=step_size,
                # Use log_harmonic_mean
                # for robustness as proposed in https://github.com/tensorflow/probability/blob/43a9d6c3f5992a24ddf7aa3fa826a038d70697c5/tensorflow_probability/python/mcmc/dual_averaging_step_size_adaptation.py#L143
                **self.adaptation_kwargs,
            )
        init_kernel_results = adapt_kernel_tmp.bootstrap_results(current_state)

        # Needs to be initialized here, with stable dimensions of the arguments
        # to avoid a retracing/recompilation of the function
        @tf.function(autograph=False, experimental_compile=self.xla, experimental_relax_shapes=False)
        def _run_chains_compiled(init, init_kernel_results, num_samples, num_adaptation_steps):

            kernel = self._kernel(
                target_log_prob_fn=self.parallel_logpfn,
                step_size=init_kernel_results.new_step_size,
                **self.kernel_kwargs,
            )
            if self._bijector:
                kernel = mcmc.TransformedTransitionKernel(
                    inner_kernel=kernel,
                    bijector=self._bijector
                )
            if self._adaptation:
                adapt_kernel = self._adaptation(
                    inner_kernel=kernel,
                    reduce_fn=tfp.math.reduce_log_harmonic_mean_exp,
                    num_adaptation_steps=num_adaptation_steps,
                    shrinkage_target=step_size,
                    # Use log_harmonic_mean
                    # for robustness as proposed in https://github.com/tensorflow/probability/blob/43a9d6c3f5992a24ddf7aa3fa826a038d70697c5/tensorflow_probability/python/mcmc/dual_averaging_step_size_adaptation.py#L143
                    **self.adaptation_kwargs,
                )
            else:
                adapt_kernel = kernel

            results, sample_stats, final_kernel_results = mcmc.sample_chain(
                num_samples,
                current_state=init,
                previous_kernel_results=init_kernel_results,
                kernel=adapt_kernel,
                num_burnin_steps=0,
                trace_fn=self.trace_fn,
                seed=self.seed,
                return_final_kernel_results=True,
                **self.chain_kwargs,
            )
            return results, sample_stats, final_kernel_results

        def run_chains(init_state, init_kernel_results, num_samples, num_adaptation_steps=0):
            # Necessary to be an int, because xla compilation requires a static trace length
            num_samples = int(num_samples)
            num_adaptation_steps = tf.convert_to_tensor(round(num_adaptation_steps), dtype="int32")
            # target_accept_prob = tf.convert_to_tensor(target_accept_prob, dtype = 'float32')
            return _run_chains_compiled(
                init_state, init_kernel_results, num_samples, num_adaptation_steps,
            )

        self._run_chains = run_chains

        self.init_keys = init_keys
        self._deterministics_callback = _deterministics_callback
        self._deterministic_names = deterministic_names
        self._state = state_

        # Run one sample, to already trace/compile the function, is not strictly
        # necessary, but it feels like to be the correct location to do it here already.
        _ = self._run_chains(init_state, init_kernel_results, self.num_samples_binning)
        self.last_results = init_state
        self.last_kernel_results = init_kernel_results

        self.accumulated_results = None
        self.accumulated_sample_stats = None

    def tune(self, n_start=10, n_tune=150, ratio_epochs=1.5):
        n_window = n_start * ratio_epochs ** np.arange(
            round(np.log((n_tune * (ratio_epochs - 1) / n_start) + 1) / np.log(ratio_epochs))
        )
        n_window = np.ceil(n_window / self.num_samples_binning) * self.num_samples_binning
        n_window = n_window.astype("int32")
        _log.info(f"tuning windows: {n_window}")

        pbar = tqdm.tqdm(total=np.sum(n_window))
        for num_samples in n_window:
            for i in range(0, num_samples, self.num_samples_binning):
                results, sample_stats, kernel_results = self._run_chains(
                    self.last_results,
                    self.last_kernel_results,
                    self.num_samples_binning,
                    num_adaptation_steps=np.sum(n_window),
                )
                self.last_results = tf.nest.map_structure(lambda x: x[-1], results)
                self.last_kernel_results = kernel_results
                self._append_results(results, sample_stats)

                pbar.update(n=self.num_samples_binning)
                pbar.set_description(f"log-like: {np.average(sample_stats[0][-1]):.1f}")

            new_step_size = _calc_mass_matrix(
                results,
                tf.nest.map_structure(tf.math.exp, self.last_kernel_results.log_averaging_step),
                self.step_size_adaption_per_chain,
            )
            kernel_results = set_step_dual_averaging_kernel(self.last_kernel_results, new_step_size)
            self.last_kernel_results = kernel_results
        pbar.close()

    def sample(
        self, num_samples: int = 1000, burn_in=None, trace_discrete: Optional[List[str]] = None,
    ):
        if burn_in is None:
            burn_in = num_samples / 4
        # tf.print(f"init state:\n{init_keys[-1]}\n{init_state[-1]} ")
        burn_in = int(burn_in)
        num_adaptation = int(np.ceil(burn_in * 0.8))
        num_samples = int(
            np.ceil(num_samples / self.num_samples_binning) * self.num_samples_binning
        )

        pbar = tqdm.tqdm(total=num_samples)
        self.last_kernel_results._replace(step=tf.convert_to_tensor(0, dtype="int32"))
        for i in range(0, num_samples, self.num_samples_binning):
            results, sample_stats, kernel_results = self._run_chains(
                self.last_results,
                self.last_kernel_results,
                self.num_samples_binning,
                num_adaptation_steps=num_adaptation,
            )

            self.last_results = tf.nest.map_structure(lambda x: x[-1], results)
            self.last_kernel_results = kernel_results

            # results_without_burn_in = tf.nest.map_structure(lambda x: x[burn_in:], results)
            # sample_stats_without_burn_in = tf.nest.map_structure(lambda x: x[burn_in:], sample_stats)

            self._append_results(results, sample_stats)

            init_state = self.last_results
            # new_step_size = self._calc_mass_matrix(results)
            # self.step_size = new_step_size

            pbar.update(n=self.num_samples_binning)
            pbar.set_description(f"log-like: {np.average(sample_stats[0][-1]):.1f}")

        results_without_burn_in = tf.nest.map_structure(
            lambda x: x[burn_in:], self.accumulated_results
        )
        sample_stats_without_burn_in = tf.nest.map_structure(
            lambda x: x[burn_in:], self.accumulated_sample_stats
        )
        self.accumulated_results = results_without_burn_in
        self.accumulated_sample_stats = sample_stats_without_burn_in
        pbar.close()

    def _append_results(self, results, sample_stats):
        if self.accumulated_results is None:
            self.accumulated_results = results
        else:
            self.accumulated_results = tf.nest.map_structure(
                lambda *x: tf.concat(x, axis=0), self.accumulated_results, results
            )
        if self.accumulated_sample_stats is None:
            self.accumulated_sample_stats = sample_stats
        else:
            self.accumulated_sample_stats = tf.nest.map_structure(
                lambda *x: tf.concat(x, axis=0), self.accumulated_sample_stats, sample_stats
            )

    def retrieve_trace_and_reset(self, trace_discrete=False):
        trace = self.make_trace(trace_discrete)
        self.accumulated_results = None
        self.accumulated_sample_stats = None
        return trace

    def make_trace(self, trace_discrete=False):
        posterior = dict(zip(self.init_keys, self.accumulated_results))
        if trace_discrete:
            # TODO: maybe better logic can be written here
            # The workaround to cast variables post-sample.
            # `trace_discrete` is the list of vairables that need to be casted
            # to tf.int32 after the sampling is completed.
            init_keys_ = [
                scope_remove_transformed_part_if_required(_, state_.transformed_values)[1]
                for _ in init_keys
            ]
            discrete_indices = [init_keys_.index(_) for _ in trace_discrete]
            keys_to_cast = [init_keys[_] for _ in discrete_indices]
            for key in keys_to_cast:
                posterior[key] = tf.cast(posterior[key], dtype=tf.int32)

        # Keep in sync with pymc3 naming convention
        if len(self.accumulated_sample_stats) > len(self.stat_names):
            deterministic_values = self.accumulated_sample_stats[len(self.stat_names) :]
            sample_stats = self.accumulated_sample_stats[: len(self.stat_names)]
        sampler_stats = dict(zip(self.stat_names, sample_stats))
        if len(self._deterministic_names) > 0:
            posterior.update(dict(zip(self._deterministic_names, deterministic_values)))
        trace = trace_to_arviz(
            posterior,
            sampler_stats if self.is_compound is False else None,
            observed_data=self._state.observed_values,
        )
        return trace

    def _assign_arguments(self, kwargs):
        kwargs_keys = set(kwargs.keys())
        # fetch adaptation kernel, kernel, and `sample_chain` kwargs keys
        adaptation_keys = (
            set(list(inspect.signature(self._adaptation.__init__).parameters.keys())[1:])
            if self._adaptation
            else set()
        )
        kernel_keys = set(list(inspect.signature(self._kernel.__init__).parameters.keys())[1:])
        chain_keys = set(list(inspect.signature(mcmc.sample_chain).parameters.keys()))

        # intersection of key sets of each object from
        # (`self._adaptation`, `self._kernel`, `sample_chain`)
        # is the kwargs we are trying to find
        self.adaptation_kwargs = {k: kwargs[k] for k in (adaptation_keys & kwargs_keys)}
        self.kernel_kwargs = {k: kwargs[k] for k in (kernel_keys & kwargs_keys)}
        self.chain_kwargs = {k: kwargs[k] for k in (chain_keys & kwargs_keys)}

    def _check_arguments(self):
        # check if there is an ambiguity of the kwargs keys for
        # kernel, adaptation_kernel adn sample_chain method
        if (
            (self.adaptation_kwargs.keys() & self.kernel_kwargs.keys())
            or (self.adaptation_kwargs.keys() & self.chain_kwargs.keys())
            or (self.kernel_kwargs.keys() & self.chain_kwargs.keys())
        ):
            raise ValueError(
                "Ambiguity in setting kwargs for `kernel`, \
                        `adaptation_kernel`, `chain_sampler`"
            )

    def _bound_kwargs(self, *args):
        # set all the default kwargs which are distinct
        # for each type of sampler. If a user has passed
        # the key argument then we don't change the kwargs set
        for k, v in self.default_kernel_kwargs.items():
            self.kernel_kwargs.setdefault(k, v)
        for k, v in self.default_adapter_kwargs.items():
            self.adaptation_kwargs.setdefault(k, v)

    @classmethod
    def _default_kernel_maker(cls):
        # The function is used for compound step support.
        # by supporting collection we could easily instantiate
        # kernel inside the `one_step`
        # TODO: maybe can be done with partial, but not
        # sure how to do it recursively
        kernel_collection = KERNEL_KWARGS_SET(
            kernel=cls._kernel,
            adaptive_kernel=cls._adaptation,
            kernel_kwargs=cls.default_kernel_kwargs,
            adaptive_kwargs=cls.default_adapter_kwargs,
        )
        return kernel_collection

    @abc.abstractmethod
    def trace_fn(self, current_state: flow.SamplingState, pkr: Union[tf.Tensor, Any]):
        """
        Support a tracing for each sampler

        Parameters
        ----------
        current_state : flow.SamplingState
            state for tracing
        pkr : Union[tf.Tensor, Any]
            A `Tensor` or nested collection of `Tensor`s
        """
        pass


def set_step_dual_averaging_kernel(kernel, new_step_size):
    step = tf.constant(0, dtype=tf.int32)
    log_shrinkage_target = tf.nest.map_structure(tf.math.log, new_step_size)
    error_sum = tf.nest.map_structure(tf.zeros_like, kernel.error_sum)
    log_averaging_step = tf.nest.map_structure(tf.zeros_like, new_step_size)
    kernel = kernel._replace(
        step=step,
        log_shrinkage_target=log_shrinkage_target,
        error_sum=error_sum,
        log_averaging_step=log_averaging_step,
        new_step_size=new_step_size,
    )
    return kernel


def _calc_mass_matrix(results, step_size, step_size_adaption_per_chain):
    def calc_norm_global(step_size):
        norm_parts = tf.nest.map_structure(tfp.math.reduce_log_harmonic_mean_exp, step_size)
        return tfp.math.reduce_log_harmonic_mean_exp(tf.stack(norm_parts))

    def calc_norm_per_chain(step_size):
        reduce_func = lambda arr: tf.map_fn(tfp.math.reduce_log_harmonic_mean_exp, arr)
        norm_parts = tf.nest.map_structure(reduce_func, step_size)
        return tfp.math.reduce_log_harmonic_mean_exp(tf.stack(norm_parts), axis=0)

    calc_norm = calc_norm_per_chain if step_size_adaption_per_chain else calc_norm_global

    if not step_size_adaption_per_chain:
        step_size_new = tf.nest.map_structure(
            lambda x: tf.math.reduce_std(x, axis=(0, 1))[tf.newaxis], results
        )
    else:
        step_size_new = tf.nest.map_structure(lambda x: tf.math.reduce_std(x, axis=(0,)), results)

    # scale mass matrix/step sizes to match the norm of the previous step sizes
    norm_old = calc_norm(step_size)
    norm_new = calc_norm(step_size_new)

    scale_norm_global = lambda x: x / norm_new * norm_old / 4

    def scale_norm_per_chain(x):
        shape = x.shape[0] + tf.TensorShape(np.ones(len(x.shape[1:]), dtype="int32"))
        return x / tf.reshape(norm_new, shape) * tf.reshape(norm_old, shape) / 2

    scale_norm = scale_norm_per_chain if step_size_adaption_per_chain else scale_norm_global

    step_size_new = tf.nest.map_structure(scale_norm, step_size_new)

    return step_size_new


@register_sampler
class HMC(_BaseSampler):
    """
    Sampler to run one_step of Hamiltonian Monte Carlo (HMC).
    HMC is a Markov chain Monte Carlo (MCMC) algorithm that takes a
    series of gradient-informed steps to produce a Metropolis proposal.
    Mathematical details and derivations can be found in [Neal (2011)].

    The adaptation scheme for this class is `tfp.mcmc.DualAveragingStepSizeAdaptation`
    which is the dual averaging policy that uses a noisy step size for exploration,
    while averaging over tuning steps to provide a smoothed estimate of an optimal value.
    It is based on [section 3.2 of Hoffman and Gelman (2013)], which modifies the
    [stochastic convex optimization scheme of Nesterov (2009)].

    More about the implementation of the HMC:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/HamiltonianMonteCarlo]

    More about the implementation of the adaptation:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/DualAveragingStepSizeAdaptation]

    Default values for HMC kernel are:
        {
            step_size:0.1,
            num_leapfrog_steps:3
        }

    Default stat_names:
        ["mean_tree_accept"]

    Default trace_fn:
        Leave only `log_accept_ratio` and calculate deterministic values
    """

    # TODO: provide ref links for papers

    _name = "hmc"
    _grad = True
    _adaptation = mcmc.DualAveragingStepSizeAdaptation
    _kernel = mcmc.HamiltonianMonteCarlo

    default_kernel_kwargs: dict = {"num_leapfrog_steps": 3}
    default_adapter_kwargs: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat_names = {"mean_tree_accept"}

    def trace_fn(self, current_state: flow.SamplingState, pkr: Union[tf.Tensor, Any]):
        return (pkr.inner_results.log_accept_ratio,) + tuple(
            self.deterministics_callback(*current_state)
        )


@register_sampler
class HMCSimple(HMC):
    """
    Sampler to run one_step of Hamiltonian Monte Carlo (HMC).
    HMC is a Markov chain Monte Carlo (MCMC) algorithm that takes a
    series of gradient-informed steps to produce a Metropolis proposal.
    Mathematical details and derivations can be found in [Neal (2011)].

    The adaptation scheme for this class is `tfp.mcmc.SimpleStepSizeAdaptation`
    which multiplicatively increases or decreases the step_size of the inner kernel
    based on the value of log_accept_prob. It is based on [equation 19 of Andrieu and Thoms (2008)].


    More about the implementation of the HMC:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/HamiltonianMonteCarlo]

    More about the implementation of the adaptation:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/SimpleStepSizeAdaptation]
    """

    _name = "hmc_simple"
    _adaptation = mcmc.SimpleStepSizeAdaptation


@register_sampler
class NUTS(_BaseSampler):
    """
    Sampler to run one_step of No U-Turn Sampler (NUTS).
    NUTS is an adaptive variant of the Hamiltonian Monte
    Carlo (HMC) method for MCMC. NUTS adapts the distance traveled in response to
    the curvature of the target density. Conceptually, one proposal consists of
    reversibly evolving a trajectory through the sample space, continuing until
    that trajectory turns back on itself (hence the name, 'No U-Turn'). This class
    implements one random NUTS step from a given `current_state`.
    Mathematical details and derivations can be found in
    [Hoffman, Gelman (2011)] and [Betancourt (2018)].

    The adaptation scheme for this class is `tfp.mcmc.DualAveragingStepSizeAdaptation`
    which is the dual averaging policy that uses a noisy step size for exploration,
    while averaging over tuning steps to provide a smoothed estimate of an optimal value.
    It is based on [section 3.2 of Hoffman and Gelman (2013)], which modifies the
    [stochastic convex optimization scheme of Nesterov (2009)].

    More about the implementation of the NUTS:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/NoUTurnSampler]

    More about the implementation of the adaptation:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/DualAveragingStepSizeAdaptation]

    Default arguments for NUTS kernel are:
        {step_size=0.1}

    Default arguments for adaptation scheme:
        {
            num_adaptation_steps: 100,
            step_size_getter_fn: lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn: lambda pkr: pkr.log_accept_ratio,
            step_size_setter_fn: lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
        }

    Default `stat_names` values:
        ["lp", "tree_size", "diverging", "energy", "mean_tree_accept"]

    Default `trace_fn` returns:
        (
            inner_results.target_log_prob,
            inner_results.leapfrogs_taken,
            inner_results.has_divergence,
            inner_results.energy,
            inner_results.log_accept_ratio,
            *deterministic_values,
        )

    """

    _name = "nuts"
    _grad = True
    _adaptation = mcmc.DualAveragingStepSizeAdaptation
    _kernel = mcmc.NoUTurnSampler

    # we set default kwargs to support previous sampling logic
    # optimal values can be modified in future
    default_adapter_kwargs: dict = {
        "decay_rate": 0.75,
        "exploration_shrinkage": 0.05,
        "step_count_smoothing": 10,
    }
    default_kernel_kwargs: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat_names = [
            "lp",
            "tree_size",
            "diverging",
            "energy",
            "mean_tree_accept",
            "step_size",
        ]

    def trace_fn(self, current_state: flow.SamplingState, pkr: Union[tf.Tensor, Any]):
        return (
            pkr.inner_results.target_log_prob,
            pkr.inner_results.leapfrogs_taken,
            pkr.inner_results.has_divergence,
            pkr.inner_results.energy,
            pkr.inner_results.log_accept_ratio,
            tf.tile(
                tf.stack(tf.nest.map_structure(tf.reduce_mean, pkr.new_step_size))[tf.newaxis],
                tuple(pkr.inner_results.target_log_prob.shape) + (1,),
            ),
        ) + tuple(self.deterministics_callback(*current_state))


@register_sampler
class NUTSSimple(NUTS):
    """
    Sampler to run one_step of No U-Turn Sampler (NUTS).
    NUTS is an adaptive variant of the Hamiltonian Monte
    Carlo (HMC) method for MCMC. NUTS adapts the distance traveled in response to
    the curvature of the target density. Conceptually, one proposal consists of
    reversibly evolving a trajectory through the sample space, continuing until
    that trajectory turns back on itself (hence the name, 'No U-Turn'). This class
    implements one random NUTS step from a given `current_state`.
    Mathematical details and derivations can be found in
    [Hoffman, Gelman (2011)] and [Betancourt (2018)].

    The adaptation scheme for this class is `tfp.mcmc.SimpleStepSizeAdaptation`
    which multiplicatively increases or decreases the step_size of the inner kernel
    based on the value of log_accept_prob. It is based on [equation 19 of Andrieu and Thoms (2008)].

    More about the implementation of the HMC:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/NoUTurnSampler]

    More about the implementation of the adaptation:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/SimpleStepSizeAdaptation]
    """

    _name = "nuts_simple"
    _adaptation = mcmc.SimpleStepSizeAdaptation


@register_sampler
class RandomWalkM(_BaseSampler):
    """
    Sampler to run one_step of Random Walk Metropolis (RWM).
    RWM is a gradient-free Markov chain Monte Carlo (MCMC)
    algorithm. The algorithm involves a proposal generating
    step proposal_state = current_state + perturb by a random perturbation,
    followed by Metropolis-Hastings accept/reject step. For more details see
    Section 2.1 of Roberts and Rosenthal (2004)

    More about the implementation of the RWM:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/RandomWalkMetropolis]

    Default `stat_names` values:
        ["mean_accept"]

    Default `trace_fn` returns:
        (
            log_accept_ratio,
            *deterministic_values,
        )

    """

    _name = "rwm"
    _adaptation = None
    _kernel = mcmc.RandomWalkMetropolis
    _grad = False

    default_kernel_kwargs: dict = {}
    default_adapter_kwargs: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat_names = ["mean_accept"]

    def trace_fn(self, current_state: flow.SamplingState, pkr: Union[tf.Tensor, Any]):
        return (pkr.log_accept_ratio,) + tuple(self.deterministics_callback(*current_state))


@register_sampler
class CompoundStep(_BaseSampler):
    """
    The basic implementation of the compound step

    Default `stat_names` values:
        ["compound_results"]

    Default `trace_fn` returns:
        (
            *,
            *deterministic_values,
        )
    """

    _name = "compound"
    _adaptation = None
    _kernel = _CompoundStepTF
    _grad = False

    default_adapter_kwargs: dict = {}
    default_kernel_kwargs: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat_names = ["compound_results"]

    def trace_fn(self, current_state: flow.SamplingState, pkr: Union[tf.Tensor, Any]):
        # TODO: I'm not sure if adding python loop here is the best idea
        # so I'm leaving all the results in trace and discarding it later
        return (pkr,) + tuple(self.deterministics_callback(*current_state))

    @staticmethod
    def _convert_sampler_methods(sampler_methods):
        if not sampler_methods:
            return {}
        sampler_methods_dict = {}
        for sampler_item in sampler_methods:
            # user can pass tuple of lenght=2 or lenght=3
            # we ned to check that.
            if len(sampler_item) not in [2, 3]:
                raise ValueError(
                    "You need to provide `sampler_methods` as the tuple of \
                    the length in [2, 3]. If additional kwargs for kernel \
                    are provided then the lenght of tuple is equal to 3."
                )
            if len(sampler_item) < 3:
                var, sampler = sampler_item
                kwargs = {}
            else:
                var, sampler, kwargs = sampler_item
            if isinstance(var, (list, tuple)):
                for var_ in var:
                    sampler_methods_dict[var_] = (sampler, kwargs)
            else:
                sampler_methods_dict[var] = (sampler, kwargs)
        return sampler_methods_dict

    def _merge_samplers(self, make_kernel_fn, part_kernel_kwargs):
        num_vars = len(make_kernel_fn)
        parents = list(range(num_vars))
        kernels = []

        # Also can be done in O(nlogn) with sorting (maybe more clear solution)
        # With dsu we have explicitly calculated parent indices that are used
        # to sort variables later in logp function

        def cmp_(item1, item2):
            # hack to separetely compare proposal function and
            # the rest of the kwargs of the sampler kernel
            key = "new_state_fn"
            if (key in item1 and key in item2) or (key not in item1 and key not in item2):
                # compare class instances instea of _fn for `new_state_fn`
                if key in item1:
                    if item1[key].__self__ != item2[key].__self__:
                        return False
                    value1, value2 = item1.pop(key), item2.pop(key)
                    flag = item1 == item2
                    item1[key], item2[key] = value1, value2
                else:
                    flag = item1 == item2
                return flag
            else:
                return False

        # DSU ops
        def get_set(p):
            return p if parents[p] == p else get_set(parents[p])

        def union_set(p1, p2):
            p1, p2 = get_set(p1), get_set(p2)
            if p1 != p2:
                parents[max(p1, p2)] = min(p1, p2)

        # merge sets, DSU
        for (i, j) in itertools.combinations(range(num_vars), 2):
            # For the sampler to be the same we are comparing
            # both classes of the chosen sampler and the key
            # arguments.
            if make_kernel_fn[i] == make_kernel_fn[j] and cmp_(
                part_kernel_kwargs[i], part_kernel_kwargs[j]
            ):
                union_set(i, j)
        # assign kernels based on unique sets

        used_p = {}
        for i, p in enumerate(parents):
            if p not in used_p:
                kernels.append((make_kernel_fn[i], part_kernel_kwargs[i]))
                used_p[p] = True

        # calculate independent set lengths
        parent_used, set_lengths = {}, []
        for p in parents:
            if p in parent_used:
                set_lengths[parent_used[p]] += 1
            else:
                parent_used[p] = len(set_lengths)
                set_lengths.append(1)
        self.parent_inds = sorted(range(len(parents)), key=lambda k: parents[k])
        return kernels, set_lengths

    def _assign_default_methods(
        self,
        *,
        sampler_methods: Optional[List] = None,
        state: Optional[flow.SamplingState] = None,
        observed: Optional[dict] = None,
    ):
        converted_sampler_methods: List = CompoundStep._convert_sampler_methods(sampler_methods)

        (_, state, _, _, continuous_distrs, discrete_distrs) = initialize_state(
            self.model, observed=observed, state=state
        )
        init = state.all_unobserved_values
        init_state = list(init.values())
        init_keys = list(init.keys())

        # assignd samplers for free variables
        make_kernel_fn: list = []
        # user passed kwargs for each sampler in `make_kernel_fn`
        part_kernel_kwargs: list = []
        # keep the list for proposal func names
        func_names: list = []

        for i, state_part in enumerate(init_state):
            untrs_var, unscoped_tr_var = scope_remove_transformed_part_if_required(
                init_keys[i], state.transformed_values
            )
            # get the distribution for the random variable name

            distr = continuous_distrs.get(untrs_var, None)
            if distr is None:
                distr = discrete_distrs[untrs_var]

            # get custom `new_state_fn` for the distribution
            func = distr._default_new_state_part

            # simplest way of assigning sampling methods
            # if the sampler_methods was passed and if a var is provided
            # then the var will be assigned to the given sampler
            # but will also be checked if the sampler supports the distr

            # 1. If sampler is provided by the user, we create new sampler
            #    and add to `make_kernel_fn`
            # 2. If the distribution has `new_state_fn` then the new sampler
            #    should be create also. Because sampler is initialized with
            #    the `new_state_fn` argument.
            if unscoped_tr_var in converted_sampler_methods:
                sampler, kwargs = converted_sampler_methods[unscoped_tr_var]

                # check for the sampler able to sampler from the distribution
                if not distr._grad_support and sampler._grad:
                    raise ValueError(
                        "The `{}` doesn't support gradient, please provide an appropriate sampler method".format(
                            unscoped_tr_var
                        )
                    )

                # add sampler to the dict
                make_kernel_fn.append(sampler)
                part_kernel_kwargs.append({})
                # update with user provided kwargs
                part_kernel_kwargs[-1].update(kwargs)
                # if proposal function is provided then replace
                func = part_kernel_kwargs[-1].get("new_state_fn", func)
                # add the default `new_state_fn` for the distr
                # `new_state_fn` is supported for only RandomWalkMetropolis transition
                # kernel.
                if func and sampler._name == "rwm":
                    part_kernel_kwargs[-1]["new_state_fn"] = partial(func)()
            elif callable(func):
                # If distribution has defined `new_state_fn` attribute then we need
                # to assign `RandomWalkMetropolis` transition kernel
                make_kernel_fn.append(RandomWalkM)
                part_kernel_kwargs.append({})
                part_kernel_kwargs[-1]["new_state_fn"] = partial(func)()
            else:
                # by default if user didn't not provide any sampler
                # we choose NUTS for the variable with gradient and
                # RWM for the variable without the gradient
                sampler = NUTS if distr._grad_support else RandomWalkM
                make_kernel_fn.append(sampler)
                part_kernel_kwargs.append({})
                # _log.info("Auto-assigning NUTS sampler...")
            # save proposal func names
            func_names.append(func._name if func else "default")

        # `make_kernel_fn` contains (len(state)) sampler methods, this could lead
        # to more overhed when we are iterating at each call of `one_step` in the
        # compound step kernel. For that we need to merge some of the samplers.
        kernels, set_lengths = self._merge_samplers(make_kernel_fn, part_kernel_kwargs)
        # log variable sampler mapping
        CompoundStep._log_variables(init_keys, kernels, set_lengths, self.parent_inds, func_names)
        # save to use late for compound kernel init
        self.kernel_kwargs["compound_samplers"] = kernels
        self.kernel_kwargs["compound_set_lengths"] = set_lengths

    def __call__(self, *args, **kwargs):
        return self._sample(*args, is_compound=True, **kwargs)

    @staticmethod
    def _log_variables(var_keys, kernel_kwargs, set_lengths, parent_inds, func_names):
        var_keys = [var_keys[i] for i in parent_inds]
        func_names = [func_names[i] for i in parent_inds]
        log_output = ""
        curr_indx = 0
        for i, (kernel_kwargsi, set_leni) in enumerate(zip(kernel_kwargs, set_lengths)):
            kernel, kwargs = kernel_kwargsi
            vars_ = var_keys[curr_indx : curr_indx + set_leni]
            log_output += ("\n" if i > 0 else "") + " -- {}[vars={}, proposal_function={}]".format(
                kernel._name, [item.split("|")[1] for item in vars_], (func_names[curr_indx]),
            )
            curr_indx += set_leni
        _log.info(log_output)


def build_logp_and_deterministic_functions(
    model,
    num_chains: Optional[int] = None,
    observed: Optional[dict] = None,
    state: Optional[flow.SamplingState] = None,
    collect_reduced_log_prob: bool = True,
    parent_inds: Optional[List] = None,
):
    if not isinstance(model, Model):
        raise TypeError(
            "`sample` function only supports `pymc4.Model` objects, but you've passed `{}`".format(
                type(model)
            )
        )
    if state is not None and observed is not None:
        raise ValueError("Can't use both `state` and `observed` arguments")

    state, deterministic_names = initialize_sampling_state(model, observed=observed, state=state)

    if not state.all_unobserved_values:
        raise ValueError(
            f"Can not calculate a log probability: the model {model.name or ''} has no unobserved values."
        )

    observed_var = state.observed_values
    unobserved_keys, unobserved_values = zip(*state.all_unobserved_values.items())

    if parent_inds:
        unobserved_keys = [unobserved_keys[i] for i in parent_inds]
        unobserved_values = [unobserved_values[i] for i in parent_inds]

    if collect_reduced_log_prob:

        @tf.function(autograph=False)
        def logpfn(*values, **kwargs):
            if kwargs and values:
                raise TypeError("Either list state should be passed or a dict one")
            elif values:
                kwargs = dict(zip(unobserved_keys, values))
            st = flow.SamplingState.from_values(kwargs, observed_values=observed)
            _, st = flow.evaluate_model_transformed(model, state=st)
            return st.collect_log_prob()

    else:
        # When we use manual batching, we need to manually tile the chains axis
        # to the left of the observed tensors
        if num_chains is not None:
            obs = state.observed_values
            if observed is not None:
                obs.update(observed)
            else:
                observed = obs
            for k, o in obs.items():
                o = tf.convert_to_tensor(o)
                o = tf.tile(o[None, ...], [num_chains] + [1] * o.ndim)
                observed[k] = o

        @tf.function(autograph=False)
        def logpfn(*values, **kwargs):
            if kwargs and values:
                raise TypeError("Either list state should be passed or a dict one")
            elif values:
                kwargs = dict(zip(unobserved_keys, values))
            st = flow.SamplingState.from_values(kwargs, observed_values=observed)
            _, st = flow.evaluate_model_transformed(model, state=st)
            return st.collect_unreduced_log_prob()

    @tf.function(autograph=False)
    def deterministics_callback(*values, **kwargs):
        if kwargs and values:
            raise TypeError("Either list state should be passed or a dict one")
        elif values:
            kwargs = dict(zip(unobserved_keys, values))
        st = flow.SamplingState.from_values(kwargs, observed_values=observed_var)
        _, st = flow.evaluate_model_transformed(model, state=st)
        for transformed_name in st.transformed_values:
            untransformed_name = NameParts.from_name(transformed_name).full_untransformed_name
            st.deterministics_values[untransformed_name] = st.untransformed_values.pop(
                untransformed_name
            )
        return st.deterministics_values.values()

    return (
        logpfn,
        dict(state.all_unobserved_values),
        deterministics_callback,
        deterministic_names,
        state,
    )


def vectorize_logp_function(logpfn):
    # TODO: vectorize with dict
    def vectorized_logpfn(*state):
        return tf.vectorized_map(lambda mini_state: logpfn(*mini_state), state)

    return vectorized_logpfn


def tile_init(init, num_repeats):
    if num_repeats is not None:
        return [tf.tile(tf.expand_dims(tens, 0), [num_repeats] + [1] * tens.ndim) for tens in init]
    else:
        return init


def calculate_log_likelihood(
    model: Model, posterior: Dict[str, tf.Tensor], sampling_state: flow.SamplingState
) -> Dict[str, tf.Tensor]:
    """Compute log likelihood by vectorizing chains and draws."""

    def extract_log_likelihood(values, observed_rv):
        st = flow.SamplingState.from_values(values, observed_values=sampling_state.observed_values)
        _, st = flow.evaluate_model_transformed(model, state=st)
        try:
            dist = st.continuous_distributions[observed_rv]
        except KeyError:
            dist = st.discrete_distributions[observed_rv]
        return dist.log_prob(dist.model_info["observed"])

    # First vectorizing chains and then draws, but ultimately, draws and chains get
    # swapped for mcmc trace while passing to `trace_to_arviz`.
    log_likelihood_dict = dict()
    for observed_rv in sampling_state.observed_values:
        extract_log_likelihood = partial(extract_log_likelihood, observed_rv=observed_rv)
        vectorized_chains = vectorize_logp_function(extract_log_likelihood)
        vectorized_draws = vectorize_logp_function(vectorized_chains)
        log_likelihood_dict[observed_rv] = vectorized_draws(posterior)
    return log_likelihood_dict
