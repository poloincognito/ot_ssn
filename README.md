# A quick kernel-based OT estimator

Reimplementation of:
https://doi.org/10.48550/arXiv.2310.14087

For the SSIPM, see https://github.com/BorisMuzellec/kernel-sos-ot.git.

## About the EG formulation

Given the minmax problem from (4.1), I use the operator $(\delta_\gamma, -\delta_X)$ to get $f$ that is later reinjected in the EG algorithm:
$$y_{t+1} = proj(x_t-\tau f(x_t))\\
x_{t+1} = proj(x_t - \tau f(y_{t+1}))$$