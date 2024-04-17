# %%
import jax
import jax.numpy as jnp
import jax.random as random

import matplotlib.pyplot as plt

from ssn import (
    get_fillings,
    get_gaussian_kernel_func,
    _get_OT_from_gamma,
    _phi_op,
    _phi_star_op,
    proj_sym_pos,
    _get_R,
    get_r_norm,
    get_mu,
    get_xhi,
    get_psi,
    get_T_op,
)

# %%

# Samples
rng = jax.random.PRNGKey(42)
rng1, rng2 = jax.random.split(rng)
dim, n_sample = 5, 17
X, Y = jax.random.normal(rng1, (n_sample, dim)), jax.random.normal(
    rng2, (n_sample, dim)
)
assert X.shape == (n_sample, dim)
print("X.shape: ", X.shape)

# %%

# Fillings
XY_fillings = get_fillings(2 * dim, n_sample)
n = XY_fillings.shape[0]
assert XY_fillings.shape == (n, 2 * dim)
print("XY_fillings.shape: ", XY_fillings.shape)
X_fillings, Y_fillings = XY_fillings[:, :dim], XY_fillings[:, dim:]

# %%

# Unidimensional fillings
uni_XY_fillings = get_fillings(2, n_sample)
uni_X, uni_Y = uni_XY_fillings[:, 0], uni_XY_fillings[:, 1]
plt.scatter(uni_X, uni_Y)
plt.title("Unidimensional fillings (Sobol sequence)")
plt.show()

# %%

# Kernels
bandwidth = 0.1
gaussian_kernel = get_gaussian_kernel_func(bandwidth)

K_X = gaussian_kernel(X_fillings, Y_fillings)
assert K_X.shape == (n, n)
print("K_X.shape: ", K_X.shape)
K_Y = gaussian_kernel(Y_fillings, Y_fillings)
XY = jnp.concatenate([X_fillings, Y_fillings], axis=1)
K_XY = gaussian_kernel(XY_fillings, XY_fillings)
assert K_Y.shape == (n, n)
print("K_XY.shape: ", K_XY.shape)
Q = K_X + K_Y

# %%

# Other parameters
q2 = jnp.mean(gaussian_kernel(X, X) ** 2) + jnp.mean(gaussian_kernel(Y, Y) ** 2)
unif_sample = jnp.ones(n_sample) / n_sample
w_src = (unif_sample @ gaussian_kernel(X, X_fillings))[:, jnp.newaxis]
assert w_src.shape == (n, 1)
print("w_src.shape: ", w_src.shape)
w_tgt = (unif_sample @ gaussian_kernel(Y, Y_fillings))[:, jnp.newaxis]
assert w_tgt.shape == (n, 1)
print("w_tgt.shape: ", w_tgt.shape)
lambda1, lambda2 = 1 / n, 1 / n_sample**0.5
R = jnp.linalg.cholesky(K_XY).T
assert R.shape == (n, n)
print("R.shape: ", R.shape)
_ = R.T[:, jnp.newaxis, :]
A = jax.vmap(jnp.kron, in_axes=0)(_, _).squeeze()
assert A.shape == (n, n**2)
print("A.shape: ", A.shape)
z = (
    w_src
    + w_tgt
    - lambda2 * jnp.sum((X_fillings - Y_fillings) ** 2, axis=1)[:, jnp.newaxis]
)
assert z.shape == (n, 1)
print("z.shape: ", z.shape)

# %%

# Kernel OT class
from ssn import KernelOT

kot = KernelOT(X, Y, gaussian_kernel)
gamma = jax.random.normal(rng, (kot.n, 1))
_ot_hat = _get_OT_from_gamma(q2, gamma, w_src, w_tgt, lambda2)
ot_hat = kot.get_OT_from_gamma(gamma)
print("ot_hat: ", ot_hat)

# %%
P = jax.random.orthogonal(rng, n)
Sigma = jnp.diag(jax.random.normal(rng, (n,)))
sym = P @ Sigma @ P.T
apply_phi = _phi_op(A, sym)
print("apply_phi.shape: ", apply_phi.shape)
apply_phi_star = _phi_star_op(A, gamma, n)
print("apply_phi_star.shape: ", apply_phi_star.shape)

# %%
print("sym.shape: ", sym.shape)
eigenval, _ = jnp.linalg.eigh(sym)
print("sym eigenvalues: ", eigenval)
sym_pos = proj_sym_pos(sym)
print("sym_pos.shape: ", sym_pos.shape)
eigenval_pos, _ = jnp.linalg.eigh(sym_pos)
print("sym_pos eigenvalues: ", eigenval_pos)
sym_pos, _ = proj_sym_pos(sym, True)

# %%
Id = jnp.eye(n)
w = (gamma, sym)
R = _get_R(w, lambda2, Q, z, A, lambda1, Id, n)
print("R[0].shape: ", R[0].shape)
print("R[1].shape: ", R[1].shape)
R_norm = get_r_norm(R)
print("R_norm: ", R_norm)

# %%
theta = 0.5
mu = get_mu(theta, R)
print("mu.shape: ", mu.shape)  # ambiguity R from K cholseky and R the residual

# %%
xhi, _ = get_xhi(eigenval, mu)
alpha, alpha_bar = _
print("xhi.shape: ", xhi.shape)
print("alpha: ", alpha)

# %%
psi = get_psi(xhi, mu)
print("psi.shape: ", psi.shape)

# %%
T_op = get_T_op(mu, P, -eigenval, n)
print("T_op(sym).shape: ", T_op(sym).shape)

T_op = get_T_op(mu, P, eigenval, n)
alpha = sum((eigenval > 0))
print("|alpha|/n_sample: ", alpha / n_sample)
print("T_op(sym).shape: ", T_op(sym).shape)

# %%
a = kot.apply_C2(mu, T_op, R)
print("a1.shape: ", a[0].shape)
print("a2.shape: ", a[1].shape)

# %%
a_tilde_1, a_tilde_2 = kot.apply_B(a, mu, T_op)
print("a_tilde_1.shape: ", a_tilde_1.shape)
print("a_tilde_2.shape: ", a_tilde_2.shape)

# %%
delta_w = kot.apply_C1((a_tilde_1, a_tilde_2), T_op)
print("delta_w[0].shape: ", delta_w[0].shape)
print("delta_w[1].shape: ", delta_w[1].shape)

# %%
delta_w = kot.get_update(w, theta)
print("delta_w[0].shape: ", delta_w[0].shape)
print("delta_w[1].shape: ", delta_w[1].shape)

# %%
f, proj, L = kot.get_eg_params()
_ = jnp.concatenate((gamma.flatten(), sym.flatten()), axis=0)
print("proj(sym).shape: ", proj(_).shape)
print("f(_).shape: ", f(_).shape)
print("L: ", L)

# %%
rho = kot.get_rho(w, delta_w)
print("rho: ", rho)

# %%
theta = kot.update_theta(theta, delta_w, w)
print("theta: ", theta)

# %%
# Solve via EG
v0 = jnp.zeros(n + n**2)
error, max_iter = 1e-2, 40
w, r_norms = kot.run_eg(v0, error, max_iter)
ot_eg_estim = kot.get_OT_from_gamma(w[0])

# Plot
plt.plot(range(len(r_norms)), r_norms)
plt.xlabel("Iteration")
plt.ylabel("r_norm")
plt.title("Convergence of r_norm")
plt.show()

# %%
# Solve via SSN
error, max_iter = 1e-2, 40
theta0 = 1e2
w, r_norms = kot.run_ssn(v0, theta0, error, max_iter)
ot_cuturi_estim = kot.get_OT_from_gamma(w[0])

# %%
# Plot SSN convergence
plt.plot(range(len(r_norms)), r_norms)
plt.xlabel("Iteration")
plt.ylabel("r_norm")
plt.title("Convergence of r_norm")
plt.show()

# %%
import ott

# Solve via Sinkhorn
geom = ott.geometry.pointcloud.PointCloud(X, Y)  # Define an euclidean geometry
problem = ott.problems.linear.linear_problem.LinearProblem(geom)  # Define your problem
solver = ott.solvers.linear.sinkhorn.Sinkhorn()  # Select the Sinkhorn solver
out = solver(problem)
ot_ott_estim = out.primal_cost

print("Wasserstein distance from OTT: ", ot_ott_estim)
print("Wasserstein distance from EG: ", ot_eg_estim)
print("Wasserstein distance from Cuturi: ", ot_cuturi_estim)

# %%
