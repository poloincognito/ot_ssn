# %%
import numpy as np
from scipy.stats import norm
import jax.numpy as jnp

import ot

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

os.chdir("..")

### Quasi-random sequences
from sobol_seq import i4_sobol_generate

### Kernel SoS OT functions
from optim import interior_point
from utils import (
    make_kernels,
    transport_cost,
    transport_1D,
    potential_1D,
    gaussian_kernel,
)
from ssn import KernelOT, get_gaussian_kernel_func, _phi_star_op, get_r_norm

# %%
nfill = 100
nsamples = 100

#
n = 2000

mu1 = [0.7, 0.3]
mu2 = [0.2, 0.5, 0.75]


t1 = [0.4, 0.6]
t2 = [0.2, 0.2, 0.6]

x = np.linspace(0, 1, n)

r_tmp = []
for mode in mu1:
    r_tmp.append(norm.pdf(x, mode, 0.09))

c_tmp = []
for mode in mu2:
    c_tmp.append(norm.pdf(x, mode, 0.075))

mu = np.dot(t1, r_tmp)
nu = np.dot(t2, c_tmp)

#
np.random.seed(123)

u1 = np.random.rand(nsamples)
u2 = np.random.rand(nsamples)

X = np.zeros(nsamples)
Y = np.zeros(nsamples)

for i in range(nsamples):
    if u1[i] < t1[0]:
        X[i] = np.random.randn() * 0.1 + mu1[0]
    else:
        X[i] = np.random.randn() * 0.1 + mu1[1]
    if u2[i] < t2[0]:
        Y[i] = np.random.randn() * 0.075 + mu2[0]
    elif u2[i] < t2[1] + t2[0]:
        Y[i] = np.random.randn() * 0.075 + mu2[1]
    else:
        Y[i] = np.random.randn() * 0.075 + mu2[2]

#
x = np.linspace(0, 1, n)

f, ax = plt.subplots()

ax.plot(x, mu, label="mu density")
ax.plot(x, nu, label="nu density")

ax.scatter(X, mu[(n * X).astype(int)], label="mu samples")
ax.scatter(Y, nu[np.minimum((n * Y).astype(int), n - 1)], label="nu samples")


plt.legend()
plt.show()

# %%
### Sobol quasi-random samples to fill the space X x Y.
sob = i4_sobol_generate(2, nfill, skip=3000)


# ## Add some points in the corners (optional)
# sob = np.insert(sob, 0, np.array([1e-2, 1e-2]))
# sob = np.insert(sob, 0, np.array([1 - 1e-2, 1 - 1e-2]))
# sob = np.insert(sob, 0, np.array([1e-2, 1 - 1e-2]))
# sob = np.insert(sob, 0, np.array([1.0 - 1e-2, 1e-2]))

# sob = sob.reshape(-1, 2)[:-4, :]


X_fill = sob[:, :1]
Y_fill = sob[:, 1:]
plt.scatter(X_fill, Y_fill)

# %%
kernel = "gaussian"
l = 0.1
Phi, M, Kx1, Ky1, Kx2, Ky2, Kx3, Ky3 = make_kernels(
    X[:, None], Y[:, None], X_fill, Y_fill, l=l, kernel=kernel
)

# %%
# Compare the kernels to KernelOT implementation
bandwidth = (0.05) ** 0.5  # l = 2*bandwidth**2
my_gaussian_kernel = get_gaussian_kernel_func(bandwidth)
to_multidim_jnp = lambda x: jnp.array(x)[:, jnp.newaxis]
_X, _Y = to_multidim_jnp(X), to_multidim_jnp(Y)
kot = KernelOT(_X, _Y, my_gaussian_kernel)
assert jnp.allclose(X_fill, kot.X_fillings)
_w_src = jnp.mean(Kx2, axis=0)[:, jnp.newaxis]
assert jnp.allclose(_w_src, kot.w_src)
_q2 = jnp.mean(Kx3) + jnp.mean(Ky3)
assert jnp.isclose(kot.q2, _q2)
XY_fill = np.hstack([X_fill, Y_fill])
assert jnp.allclose(XY_fill, kot.XY_fillings)
_K_XY = gaussian_kernel(XY_fill[:, None], XY_fill, l)
# assert min(jnp.linalg.eigvalsh(kot.K_XY)) > 0.0
assert jnp.allclose(_K_XY, kot.K_XY)
assert jnp.linalg.norm(kot.R_cholesky - Phi) < 1e-2 * jnp.linalg.norm(kot.R_cholesky)

# %%
assert jnp.allclose(Kx2.mean(0), _w_src.flatten())
assert jnp.allclose(Kx3.mean() + Ky3.mean(), _q2)

# %%
## Regularization parameters

lbda_1 = 1 / nfill
lbda_2 = 1 / nsamples  # **0.5  # to be removed


## Optimization problem parameters

eps_start = nfill
eps_end = 1e-8

tau = 1e-8

niter = 1000

# SSIPM
G, eps = interior_point(
    M,
    Phi,
    Kx1,
    Kx2,
    Kx3,
    Ky1,
    Ky2,
    Ky3,
    lbda_1=lbda_1,
    lbda_2=lbda_2,
    eps_start=eps_start,
    eps_end=eps_end,
    eps_div=2,
    tau=tau,
    niter=niter,
    verbose=True,
    report_interval=100,
)

kernel_sos_ot = transport_cost(G, Kx2, Kx3, Ky2, Ky3, lbda_2, product_sampling=False)

# %%
compare = lambda x, y: jnp.linalg.norm(x - y) / jnp.linalg.norm(y)

# Compare SSIPM and EG results
v0 = jnp.ones((nfill + nfill**2)) / (nfill + nfill**2)  # beware, eps = 1e-10
w, r_norms = kot.run_eg(v0)
plt.plot(r_norms)
plt.title("EG convergence")
plt.show()
eg_rdiff = compare(w[0].flatten(), G)
print("Relative difference is {}".format(eg_rdiff))

# # %%
# # Just another strategy, is G a fixed point, find X ?
# f, proj, _ = kot.get_eg_params()
# v = jnp.concat((jnp.array(G).flatten(), jnp.zeros(nfill**2)))
# grad_v = f(v)
# print("grad1 norm: ", jnp.linalg.norm(grad_v[:nfill]))
# print("grad2 norm: ", jnp.linalg.norm(grad_v[nfill:]))

# # Frozen G, update X
# update_norms, num_updates, stepsize = [], 100, 1e-3
# for i in range(num_updates):
#     grad_v = np.array(f(v))
#     grad_v[:nfill] = 0.0
#     new_v = proj(v - stepsize * grad_v)
#     update_norms.append(jnp.linalg.norm(new_v - v))
#     v = new_v

# plt.plot(update_norms)
# plt.title("G frozen, norm of the X updates")
# plt.show()

# # Check if v is now a fixed point
# w = (v[:nfill], v[nfill:].reshape(nfill, nfill))
# print("r_norm is now: ", get_r_norm(kot.get_R(w)))

# %%
# Compare SSIPM and SSN results
theta0 = 1.0
w, r_norms = kot.run_ssn(v0, theta0, 1e-10, 40)
plt.plot(r_norms)
plt.title("SSN convergence")
plt.show()

G_ssn = w[0].flatten()
ssn_rdiff = compare(G_ssn, G)
print("Relative difference is {}".format(ssn_rdiff))

# %%
### Compute OT from samples
x = np.linspace(0.0, 1.0, n)

M_ot = ((x[:, None] - x) ** 2) / 2
P, log = ot.emd(mu / mu.sum(), nu / nu.sum(), M_ot, log=True)
sampled_ot = (P * M_ot).sum()

print(
    f"Plugin estimator (n={n}): {sampled_ot:.3e}\nKernel SoS estimator (n={nsamples}, l={nfill}): {kernel_sos_ot:.3e}"
)

# %%
# Plot 1
import matplotlib.gridspec as gridspec

plt.clf()

fig = plt.figure(figsize=(5, 5))

gs = gridspec.GridSpec(3, 3, wspace=0.0, hspace=0.0)

xp, yp = np.where(P > 0)

na, nb = P.shape

xa = np.arange(na)
xb = np.arange(nb)

Txa = np.argmax(P, 1)


ax1 = plt.subplot(gs[0, 1:])
ax1.plot(xa, mu, "r", label="Source distribution")
ax1.fill_between(xa, mu, color="red", alpha=0.1)
plt.ylim(ymin=0)
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    labelbottom=False,
    right=False,
    left=False,
    labelleft=False,
)
ax1.axis("off")

ax2 = plt.subplot(gs[1:, 0])
ax2.plot((nu), xb, "b", label="Target distribution")
ax2.fill_between((nu)[:], xb[:], color="blue", interpolate=True, alpha=0.1)
ax2.set_xlim(xmin=0)
ax2.invert_xaxis()

ax2.axis("off")


ax3 = plt.subplot(gs[1:, 1:], sharex=ax1, sharey=ax2)
ax3.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    labelbottom=False,
    right=False,
    left=False,
    labelleft=False,
)

ax3.plot(xa, Txa, linewidth=2, color="black", ls="--", label="True map")

x = np.linspace(0, 1, len(xa))
TX = transport_1D(x, G, X, X_fill, lbda_2, kernel=kernel, l=l)
TX_ssn = transport_1D(x, G_ssn, X, X_fill, lbda_2, kernel=kernel, l=l)

ax3.plot(xa, TX * na, color="r", lw=2, label="Inferred map")
ax3.plot(xa, TX_ssn * na, color="purple", lw=2, label="Inferred map (SSN)")

ax1.scatter(
    X * na, mu[(n * X).astype(int)], label="mu samples", marker="x", color="r", s=50
)
ax2.scatter(
    nu[np.minimum((n * Y).astype(int), n - 1)],
    Y * na,
    label="nu samples",
    marker="x",
    color="b",
    s=50,
)


ax3.scatter(
    sob[:, 0] * na, sob[:, 1] * na, color="violet", s=20, label="Filling samples"
)


plt.tight_layout()
plt.legend(fontsize=14)

plt.show()

# %%
# Plot 2
from mpl_toolkits.axes_grid1 import make_axes_locatable

x = np.linspace(0.0, 1.0, n)
y = np.linspace(0.0, 1.0, n)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax1, ax2 = ax

### True constraint function

#### Compute true function
a = log["u"]
b = log["v"]
Z = M_ot - a - b[:, None]


#### Plot
levels = 20
CS = ax1.contour(x, y, Z, levels=levels, colors="black")
CF = ax1.contourf(x, y, Z, levels=levels, cmap="bwr")
ax1.plot(x, Txa / na, color="r", ls="--", lw=3, label="True map")

## nice colorbar
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(CF, cax=cax)

ax1.set_aspect("equal")
ax1.set_xlabel("x", fontsize=16)
ax1.set_ylabel("y", fontsize=16)
ax1.set_title(r"$c(x, y) - u_{*}(x) - v_{*}(y)$", fontsize=18)
ax1.legend(fontsize=16)


### Inferred constraint function

#### Compute model
M_ot = ((x[:, None] - y) ** 2) / 2
u = potential_1D(x[:, None], G, X, X_fill, lbda_2, l, kernel)
v = potential_1D(x[:, None], G, Y, Y_fill, lbda_2, l, kernel)
Z = M_ot - u - v[:, None]

#### Plot
CS = ax2.contour(x, y, Z, levels=levels, colors="black")
CF = ax2.contourf(x, y, Z, levels=levels, cmap="bwr")

ax2.set_aspect("equal")
ax2.plot(xa / na, TX, color="r", lw=3, ls="--", label="Inferred map")
ax2.set_ylim(ymin=0, ymax=1)
ax2.scatter(X_fill, Y_fill, color="lime", s=20, label="Filling samples")

#### nice colorbar
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(CF, cax=cax)

ax2.set_xlabel("x", fontsize=16)
ax2.set_ylabel("y", fontsize=16)
ax2.set_title(r"$c(x,y) - \hat{u}(x) - \hat{v}(y)$", fontsize=18)
ax2.legend(fontsize=16)


plt.show()

# %%
