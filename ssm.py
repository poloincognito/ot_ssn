# %%
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit

# %%
rng = jax.random.PRNGKey(37)
rng1, rng2 = jax.random.split(rng)
n_sample = 17
dim = 5
X, Y = jax.random.normal(rng1, (n_sample, dim)), jax.random.normal(
    rng2, (n_sample, dim)
)
print("X.shape: ", X.shape)


# %%
bandwidth = 0.05


@jit
def get_K(X):
    diff = X[:, jnp.newaxis, :] - X[jnp.newaxis, :, :]
    return jnp.exp(-jnp.sum(diff**2 / (2 * bandwidth), axis=-1))


K_X = get_K(X)
print("K_X.shape: ", K_X.shape)
K_Y = get_K(Y)
K_XY = get_K(jnp.concatenate([X, Y], axis=1))
print("K_XY.shape: ", K_XY.shape)
Q = K_X + K_Y
unif = jnp.ones(n_sample) / n_sample
q2 = jnp.sum(K_X**2 + K_Y**2) / n_sample**2  # square missing ?

# %%
lambda1, lambda2 = 1, 1
z = Q @ unif - lambda2 * jnp.sum((X - Y) ** 2, axis=-1)
print("z.shape: ", z.shape)


@jit
def get_OT_from_gamma(gamma):
    return (1 / (2 * lambda2) * (q2 - gamma.T @ Q @ unif)).squeeze()


ot_test = get_OT_from_gamma(unif)
print("ot_test: ", ot_test)

# %%
mat = jax.random.normal(rng, (n_sample, n_sample))
R = jnp.linalg.cholesky(K_XY).T
_phi = jnp.einsum("ji,ki->ijk", R, R)

# %%
_ = R.T[:, jnp.newaxis, :]
A = jax.vmap(jnp.kron, in_axes=0)(_, _).squeeze()
print("A.shape: ", A.shape)


# %%
@jit
def _phi_op(sym):
    return jnp.einsum("ijk,jk->i", _phi, sym)


@jit
def phi_op(sym):
    return A @ sym.flatten()


print("phi_op(mat).shape: ", _phi_op(mat).shape)
print("phi_op(mat) == A vec(mat): ", (_phi_op(mat) == A @ mat.flatten()).all())  # True


# %%
@jit
def _phi_star_op(gamma):
    return gamma @ _phi


@jit
def phi_star_op(gamma):
    return (A.T @ gamma).reshape((n_sample, n_sample))


print("phi_star_op(gamma).shape: ", _phi_star_op(unif).shape)
ref = _phi_star_op(unif)
from_A = (A.T @ unif.flatten()).reshape((n_sample, n_sample))
print("phi_star_op(gamma) == A.T @ unif.flatten(): ", jnp.allclose(ref, from_A))  # True


# %%
@partial(jit, static_argnames=["return_decomposition"])
def proj_sym_pos(Z, return_decomposition=False):
    values, vectors = jnp.linalg.eigh(Z)
    pos_values = values @ jnp.diag((values > 0))
    proj = vectors @ jnp.diag(pos_values) @ vectors.T
    if not return_decomposition:
        return proj
    else:
        return proj, (values, vectors)


P = jax.random.orthogonal(rng, n_sample)
Sigma = jnp.diag(jax.random.normal(rng, (n_sample,)))
sym = P @ Sigma @ P.T
print("sym.shape: ", sym.shape)
eigenval, _ = jnp.linalg.eigh(sym)
print("sym eigenvalues: ", eigenval)
sym_pos = proj_sym_pos(sym)
print("sym_pos.shape: ", sym_pos.shape)
eigenval_pos, _ = jnp.linalg.eigh(sym_pos)
print("sym_pos eigenvalues: ", eigenval_pos)
sym_pos, _ = proj_sym_pos(sym, True)


# %%
Id = jnp.eye(n_sample)


@partial(jit, static_argnames=["return_decomposition"])
def get_R(w, return_decomposition=False):
    gamma, sym = w[0].squeeze(), w[1]
    r1 = 1 / (2 * lambda2) * (Q @ gamma - z.squeeze()) - phi_op(sym)
    Z = sym - (phi_star_op(gamma) + lambda1 * Id)
    proj, _ = proj_sym_pos(Z, True)
    r2 = sym - proj

    if return_decomposition:
        return (r1, r2), _
    else:
        return (r1, r2)


w = (unif, mat)
r1, r2 = get_R(w)
print("r1.shape: ", r1.shape)
print("r2.shape: ", r2.shape)


# %%
@jit
def get_r_norm(R):
    r1, r2 = R
    return jnp.linalg.norm(r1) + jnp.linalg.norm(
        r2
    )  # norm 2 + norm Froebenius by default


@jit
def get_mu(theta, R):
    return (theta * get_r_norm(R)).squeeze()


theta = 0.5
mu = get_mu(theta, (r1, r2))
print("mu.shape: ", mu.shape)  # ambiguity R from K cholseky and R the residual


# %%
def get_xhi(eigenval, mu, return_alpha=True):
    """Compute eta from eigenvalues of sym matrix.
    Not jitted."""
    alpha_pos_filter = eigenval > 0
    alpha, alpha_bar = jnp.where(alpha_pos_filter)[0], jnp.where(~alpha_pos_filter)[0]
    alpha_pos = eigenval[alpha]
    alpha_neg = eigenval[alpha_bar]
    diff = alpha_pos[:, jnp.newaxis] - alpha_neg[jnp.newaxis, :]
    eta = alpha_pos[:, jnp.newaxis] / diff
    xhi = eta / (mu + 1 - eta)

    if return_alpha:
        return xhi, (alpha, alpha_bar)
    else:
        return xhi


xhi, _ = get_xhi(eigenval, mu)
alpha, alpha_bar = _
print("xhi: ", xhi)
print("alpha: ", alpha)


# %%
@jit
def get_psi(xhi, mu):
    _alpha, _alpha_bar = xhi.shape
    ones = jnp.ones((_alpha, _alpha))
    zeros = jnp.zeros((_alpha_bar, _alpha_bar))
    return jnp.block([[ones / mu, xhi], [xhi.T, zeros]])


psi = get_psi(xhi, mu)
print("psi.shape: ", psi.shape)


# %%
def get_T_op(mu, P, eigenval):
    """Returns T as a function.
    T_op: n² -> n²
    Not optimized."""
    xhi, _ = get_xhi(eigenval, mu)
    alpha, alpha_bar = _
    alpha_norm = alpha.shape[0]

    if alpha_norm < 0.5 * n_sample:
        P_alpha = P[:, alpha]
        P_alpha_bar = P[:, alpha_bar]

        def T_op(S):
            U = P_alpha.T @ S
            _1 = U @ P_alpha @ P_alpha.T / (2 * mu)
            _2 = U @ P_alpha_bar
            G = P_alpha @ (_1 + jnp.multiply(xhi, _2) @ P_alpha_bar.T)
            return G + G.T

    else:
        psi = get_psi(xhi, mu)

        def T_op(S):
            _ = 1 / mu * jnp.ones((n_sample, n_sample)) - psi
            _ = jnp.multiply(_, P.T @ S @ P)
            _ = P @ _ @ P.T
            return S / mu - _

    return T_op


T_op = get_T_op(mu, P, -eigenval)
print("T_op(mat).shape: ", T_op(mat).shape)

T_op = get_T_op(mu, P, eigenval)
alpha = sum((eigenval > 0))
print("|alpha|/n_sample: ", alpha / n_sample)
print("T_op(mat).shape: ", T_op(mat).shape)


# %%
def apply_C2(mu, T_op, _R):
    r1, r2 = _R
    a1 = -r1 - 1 / mu * (phi_op(r2 + T_op(r2)))
    a2 = -r2
    return a1, a2


a = apply_C2(mu, T_op, (r1, r2))
print("a1.shape: ", a[0].shape)
print("a2.shape: ", a[1].shape)


# %%
def apply_B(a, mu, T_op):
    a1, a2 = a
    _block1 = (
        lambda _a1: (Q / (2 * lambda2) + mu * Id) @ _a1
        + A @ T_op(phi_star_op(_a1)).flatten()
    )
    a_tilde_1, _ = jax.scipy.sparse.linalg.cg(_block1, a1)
    a_tilde_2 = 1 / (1 + mu) * (a2 + T_op(a2))
    return a_tilde_1, a_tilde_2


a_tilde_1, a_tilde_2 = apply_B(a, mu, T_op)
print("a_tilde_1.shape: ", a_tilde_1.shape)
print("a_tilde_2.shape: ", a_tilde_2.shape)


# %%
def apply_C1(a_tilde, T_op):
    a_tilde_1, a_tilde_2 = a_tilde
    return a_tilde_1, a_tilde_2 - T_op(phi_star_op(a_tilde_1))


delta_w = apply_C1((a_tilde_1, a_tilde_2), T_op)
print("delta_w[0].shape: ", delta_w[0].shape)
print("delta_w[1].shape: ", delta_w[1].shape)


# %%
def get_update(w, theta):
    # compute parameters
    gamma, sym = w
    _R, Z_decomp = get_R(w, True)
    r1, r2 = _R
    eigenval, P = Z_decomp
    mu = get_mu(theta, _R)
    T_op = get_T_op(mu, P, eigenval)

    # update
    a = apply_C2(mu, T_op, _R)
    a_tilde = apply_B(a, mu, T_op)
    update = apply_C1(a_tilde, T_op)

    return update


delta_w = get_update(w, theta)
print("delta_w[0].shape: ", delta_w[0].shape)
print("delta_w[1].shape: ", delta_w[1].shape)


# %%
# EG class parameters
@jit
def f(x):
    _gamma, _X = x[:n_sample], x[n_sample:].reshape((n_sample, n_sample))
    f1 = 1 / (2 * lambda2) * (Q @ _gamma - z.flatten()) - _X.flatten() @ A.T
    f2 = -phi_star_op(_gamma) - lambda1 * Id
    return jnp.concatenate((f1, f2.flatten()))


@jit
def proj(x):
    _gamma, _X = x[:n_sample], x[n_sample:].reshape((n_sample, n_sample))
    proj_X = proj_sym_pos(_X)
    return jnp.concatenate((_gamma, proj_X.flatten()))


L = 1e2
v0 = jnp.zeros(n_sample + n_sample**2)
_ = f(v0)
print("f(w0).shape: ", _.shape)

# %%
from eg import EG

eg = EG(f, proj, L)
eg.init(v0)
v1 = eg.step()
print("v1.shape: ", v1.shape)

# %%
import matplotlib.pyplot as plt

N = 100
r_norms = []
to_w = lambda w: (w[:n_sample], w[n_sample:].reshape((n_sample, n_sample)))

eg.init(v0)
for _ in range(N):
    _ = eg.step()
    w = to_w(_)
    _R = get_R(w)
    r_norm = get_r_norm(_R)
    r_norms.append(r_norm)
ot_eg_estim = get_OT_from_gamma(w[0])

plt.plot(range(N), r_norms)
plt.xlabel("Iteration")
plt.ylabel("r_norm")
plt.title("Convergence of r_norm")
plt.show()


# %%
@jit
def get_rho(w_tilde, delta_w):
    _R = get_R(w_tilde)
    r1, r2 = _R
    rho = -delta_w[0] @ r1 - delta_w[1].flatten() @ r2.flatten()[:, jnp.newaxis]
    return rho[0]


rho = get_rho(w, delta_w)
print("rho: ", rho)


# %%
theta_min, theta_max = 1e-7, 1e7
alpha_1, alpha_2 = 1e-6, 1.0
beta_0, beta_1, beta_2 = 0.5, 1.2, 5.0


def update_theta(theta, delta_w, w_tilde):  # not jitted
    rho = get_rho(w_tilde, delta_w)
    delta_w_norm = get_r_norm(delta_w)
    if rho >= alpha_2 * delta_w_norm**2:
        return max(theta_min, beta_0 * theta)
    elif alpha_1 * delta_w_norm**2 <= rho < alpha_2 * delta_w_norm**2:
        return beta_1 * theta
    else:
        return min(theta_max, beta_2 * theta)


theta = update_theta(theta, delta_w, w)
print("theta: ", theta)


# %%
def algo2(v0, theta0, error=1e-2, max_iter=100):
    """Cuturi et al. algorithm 2."""

    # Parameters
    eg = EG(f, proj, 1e-3)
    eg.init(v0)

    # Variables
    _R = get_R(to_w(v0))
    mu = get_mu(theta0, _R)
    r_norms = []
    v, w, theta = v0, to_w(v0), theta0

    # Iteration
    for step in range(max_iter):
        print("Step: ", step)

        # EG
        v = eg.step()

        # SSN
        delta_w = get_update(w, theta)
        gamma, _X = w
        gamma_tilde, _X_tilde = gamma + delta_w[0], _X + delta_w[1]
        w_tilde = gamma_tilde, proj_sym_pos(_X_tilde)

        # Choosing the update
        R_w, R_v = get_R(w_tilde), get_R(to_w(v))
        if get_r_norm(R_w) < get_r_norm(R_v):
            w = w_tilde
            _R = R_w
        else:
            w = to_w(v)
            _R = R_v

        # Update theta
        theta = update_theta(theta, delta_w, w_tilde)
        mu = get_mu(theta, _R)

        # Save
        r_norm = get_r_norm(_R)
        print("r_norm: ", r_norm)
        r_norms.append(r_norm)
        if r_norm < error:
            print("The algorithm converged in {} steps.".format(step))
            break

    return w, r_norms


# %%
error, max_iter = 1e-2, 20
theta0 = 1e2
w, r_norms = algo2(v0, theta0, error, max_iter)
ot_cuturi_estim = get_OT_from_gamma(w[0])

# Plot convergence
plt.plot(range(max_iter), r_norms)
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

# %%
print("Wasserstein distance from OTT: ", ot_ott_estim)
print("Wasserstein distance from EG: ", ot_eg_estim)
print("Wasserstein distance from Cuturi: ", ot_cuturi_estim)

# %%
