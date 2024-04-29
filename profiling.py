import jax
import jax.numpy as jnp
import ott
from ssn import KernelOT, get_gaussian_kernel_func

# Samples
rng = jax.random.PRNGKey(42)
rng1, rng2 = jax.random.split(rng)
dim, n_sample = 5, 17
src = ott.tools.gaussian_mixture.gaussian.Gaussian.from_random(
    rng1, dim, stdev_mean=2e-1, stdev_cov=1e-2, ridge=0.5
)
tgt = ott.tools.gaussian_mixture.gaussian.Gaussian.from_random(
    rng2, dim, stdev_mean=2e-1, stdev_cov=1e-2, ridge=0.5
)
X, Y = src.sample(rng1, n_sample), tgt.sample(rng2, n_sample)

# Kernel-based OT
bandwidth = 0.1
gaussian_kernel = get_gaussian_kernel_func(bandwidth)
kot = KernelOT(X, Y, gaussian_kernel)
v0, theta0 = jnp.ones(n_sample + n_sample**2), 0.5

# Call jit funcs to compile aot
to_w = lambda w: (w[: kot.n][:, jnp.newaxis], w[kot.n :].reshape((kot.n, kot.n)))
w0 = to_w(v0)
_R = kot.get_R(w0)
_delta_w = kot.get_update(w0, theta0)

# Profiling
with jax.profiler.trace("./tmp/jax-trace", create_perfetto_link=True):
    kot.run_ssn(v0, theta0, max_iter=40, verbose=True)
