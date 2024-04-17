# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import jit

import ott
from ott.tools import gaussian_mixture as gm

# %%
rng = jax.random.PRNGKey(37)
rng_src, rng_tgt = jax.random.split(rng)

# Define the source and target samplers
n_dim = 1
n_components_src = 3
n_components_tgt = 5

src_sampler = gm.gaussian_mixture.GaussianMixture.from_random(
    rng_src, n_components_src, n_dim, stdev_mean=0.3
)
tgt_sampler = gm.gaussian_mixture.GaussianMixture.from_random(
    rng_tgt, n_components_tgt, n_dim, stdev_mean=0.3
)
# src_sampler.sample(rng,16)

# %%
gm1, gm2 = src_sampler, tgt_sampler
print("gm1.loc.shape: ", gm1.loc.shape)
print("gm1.scale_params.shape: ", gm1.scale_params.shape)


@jit
def gm1_prob(x):
    return jnp.exp(jax.vmap(gm1.log_prob)(x[:, jnp.newaxis]))

@jit
def gm2_prob(x):
    return jnp.exp(jax.vmap(gm2.log_prob)(x[:, jnp.newaxis]))

# plot src
x = np.linspace(-1, 1, 100)
y = gm1_prob(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title(
    "Probability Density Function of the source\n{} components".format(n_components_src)
)
plt.show()

# plot tgt
x = np.linspace(-1, 1, 100)
y = gm2_prob(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title(
    "Probability Density Function of the target\n{} components".format(n_components_tgt)
)
plt.show()

# %%
def get_cdf_from_gm(gm, vectorized=True):
    """Return the cdf of a gaussian mixture."""
    loc, covariance = gm.loc, gm.covariance.squeeze()
    n_components, n_dim = loc.shape
    weights = gm.component_weights

    assert n_dim == 1, "CDF is only available for unidimensional gaussian mixtures"

    @jit
    def _cdf(x):
        return jnp.dot(weights[jnp.newaxis, :], norm.cdf(x, loc.squeeze(), covariance))

    if vectorized:
        cdf = jax.vmap(_cdf)
        return cdf
    else:
        return _cdf


# %%
cdf1 = get_cdf_from_gm(gm1)

# plot src
x = np.linspace(-1, 1, 100)
y1 = gm1_prob(x)
y2 = cdf1(x)
plt.plot(x, y1, label="pdf")
plt.plot(x, y2, label="cdf")
plt.xlabel("x")
plt.title("{} components".format(n_components_src))
plt.legend()
plt.show()

cdf2 = get_cdf_from_gm(gm2)

# plot tgt
x = np.linspace(-1, 1, 100)
y1 = gm2_prob(x)
y2 = cdf2(x)
plt.plot(x, y1, label="pdf")
plt.plot(x, y2, label="cdf")
plt.xlabel("x")
plt.title("{} components".format(n_components_tgt))
plt.legend()
plt.show()


# %%
def get_lim_from_gm(gm, safety_factor=7):
    """Return the limits of the gaussian mixture."""
    loc, covariance = gm.loc, gm.covariance.squeeze()
    n_components, n_dim = loc.shape
    assert n_dim == 1, "limits are only available for unidimensional gaussian mixtures"

    lb = jnp.min(loc - safety_factor * covariance)
    ub = jnp.max(loc + safety_factor * covariance)

    return lb, ub

print("limits of the source: ", get_lim_from_gm(gm1))


# %%
def get_exact_map_from_1d_gm(gm1, gm2):
    """Return the exact OT map given two mixture of gaussians in 1D,
    using their cdf and inverse cdf.
    Not vectorized."""
    cdf1 = get_cdf_from_gm(gm1, vectorized=False)
    cdf2 = get_cdf_from_gm(gm2, vectorized=False)

    # lazy inversion
    lb, ub = get_lim_from_gm(gm2)

    def inv_cdf2(x, inv_precision=1e-5):
        low, high = lb, ub
        mid = (low + high) / 2
        while jnp.abs(cdf2(mid) - x) > inv_precision:
            if cdf2(mid) < x:
                low = mid
            else:
                high = mid
            mid = (low + high) / 2
        return mid

    def exact_map(x):
        return inv_cdf2(cdf1(x))

    return exact_map


# %%

# test cdf inversion
lb, ub = get_lim_from_gm(gm2)
_cdf2 = get_cdf_from_gm(gm2, vectorized=False)


def inv_cdf2(x, inv_precision=1e-5):
    low, high = lb, ub
    mid = (low + high) / 2
    while jnp.abs(_cdf2(mid) - x).squeeze() > inv_precision:
        if _cdf2(mid) < x:
            low = mid
        else:
            high = mid
        mid = (low + high) / 2
    return mid


elem = jnp.array([0.5])
median = inv_cdf2(elem)
print("median: ", median)

# %%

# plot
x = np.linspace(-1, 1, 100)
y1 = gm2_prob(x)
y2 = cdf2(x)
y3 = [inv_cdf2(_y2) for _y2 in y2]
plt.plot(x, y1, label="pdf")
plt.plot(x, y2, label="cdf")
plt.plot(x, y3, label="inv_cdf cdf")
plt.xlabel("x")
plt.title("{} components".format(n_components_tgt))
plt.legend()
plt.show()

# %%
safety_factor = 3
lb, ub = get_lim_from_gm(gm1, safety_factor=safety_factor)
gm1_support = jnp.linspace(float(lb), float(ub), 100)
exact_map = get_exact_map_from_1d_gm(src_sampler, tgt_sampler)
exact_map_arr = [float(exact_map(_x)) for _x in gm1_support]
plt.plot(gm1_support, exact_map_arr)
plt.xlabel("x")
plt.ylabel("y")
plt.title("OT map from cdf")
plt.show()


# %%
def get_approx_map_from_sampler(rng, src_sampler, tgt_sampler, n_sample):
    # Sample
    rng1, rng2 = jax.random.split(rng)
    src_sample = src_sampler.sample(rng1, n_sample)
    tgt_sample = tgt_sampler.sample(rng2, n_sample)

    # Solve via Sinkhorn
    geom = ott.geometry.pointcloud.PointCloud(
        src_sample, tgt_sample
    )  # Define an euclidean geometry
    problem = ott.problems.linear.linear_problem.LinearProblem(
        geom
    )  # Define your problem
    solver = ott.solvers.linear.sinkhorn.Sinkhorn()  # Select the Sinkhorn solver
    out = solver(problem)

    # Define entropic map with the output of the solver
    entropic_map = out.to_dual_potentials()
    return entropic_map.transport


# %%

# parameters
n_sample_ref = 1024
rng1, rng2 = jax.random.split(rng)
src_sample = src_sampler.sample(rng1, n_sample_ref)
tgt_sample = tgt_sampler.sample(rng2, n_sample_ref)

src_lim = (float(e) for e in get_lim_from_gm(src_sampler, safety_factor=safety_factor))
tgt_lim = (float(e) for e in get_lim_from_gm(src_sampler, safety_factor=safety_factor))
data = pd.DataFrame({"source": src_sample.squeeze(), "target": tgt_sample.squeeze()})

# get approx maps
approx_maps = {}
for n_sample in [16, 64, 256, 1024, 4096]:
    print("case n_sample: ", n_sample)
    approx_map = get_approx_map_from_sampler(rng, src_sampler, tgt_sampler, n_sample)
    approx_map_arr = approx_map(gm1_support[:, np.newaxis]).squeeze()
    approx_maps[n_sample] = approx_map_arr

# %%
# plot
sns.jointplot(data=data, x="source", y="target", kind="kde", xlim=src_lim, ylim=tgt_lim)
sns.lineplot(
    data=pd.DataFrame({"x": gm1_support, "y": exact_map_arr}),
    x="x",
    y="y",
    color="red",
    label="OT map",
)
for n_sample, approx_map_arr in approx_maps.items():
    sns.lineplot(
        data=pd.DataFrame({"x": gm1_support, "y": approx_map_arr}),
        x="x",
        y="y",
        label="OT map approx {}".format(n_sample),
        linestyle="--",
    )

# %%
