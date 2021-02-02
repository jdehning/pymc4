This is a fork of the PyMC4 package

Changes that have been made:
   * To the sampler was added a mass matrix tuner, which adapts the diagonal during tuning epochs 
   with increasing length. pm.sample is configured to use this more advanced tuning method.
   It currently only works NUTS and DualStepAveraging.
   * As initial value, a random value from the prior is sampled instead of zeros
   * Pull request [321](https://github.com/pymc-devs/pymc4/pull/321) is merged.
