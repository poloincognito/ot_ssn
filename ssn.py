# Import
from functools import partial
import numpy as np

# from scipy.stats import qmc
from sobol_seq import i4_sobol_generate
import math

import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax import jit

from eg import EG


def get_fillings(dim, n_fillings):
    """Get fillings for the kernel using the Sobol sequence.
    Not jitted.
    """
    # sampler = qmc.Sobol(d=dim, scramble=False)
    # m = math.ceil(math.log2(n_fillings))
    # sample = sampler.random_base2(m=m)
    # return jnp.array(sample)
    return jnp.array(i4_sobol_generate(dim, n_fillings, skip=3000))


# Kernel OT class
class KernelOT:
    """Kernel Optimal Transport class."""

    # Constants
    theta_min, theta_max = 1e-5, 1e5
    alpha_1, alpha_2 = 1e-6, 1.0
    beta_0, beta_1, beta_2 = 0.5, 1.2, 5.0

    def __init__(self, src, tgt, kernel):
        # Parameters
        dim, n_sample = src.shape[1], src.shape[0]
        self.src = src
        self.tgt = tgt
        self.XY_fillings = get_fillings(2 * dim, n_sample)
        self.n = self.XY_fillings.shape[0]
        self.X_fillings, self.Y_fillings = (
            self.XY_fillings[:, :dim],
            self.XY_fillings[:, dim:],
        )

        # Compute kernel matrices
        self.K_X = kernel(self.X_fillings, self.X_fillings)
        self.K_Y = kernel(self.Y_fillings, self.Y_fillings)
        self.Q = self.K_X + self.K_Y
        self.K_XY = kernel(self.XY_fillings, self.XY_fillings)

        # Other parameters
        self.lambda1, self.lambda2 = 1 / self.n, 1 / n_sample**0.5
        self.q2 = jnp.mean(kernel(self.src, self.src) ** 2) + jnp.mean(
            kernel(self.tgt, self.tgt) ** 2
        )
        self.w_src = jnp.mean(kernel(self.src, self.X_fillings), axis=0)[:, jnp.newaxis]
        self.w_tgt = jnp.mean(kernel(self.tgt, self.Y_fillings), axis=0)[:, jnp.newaxis]

        # Cholesky
        min_eigenvals = min(np.linalg.eigvalsh(self.K_XY))
        if min_eigenvals < 0:
            print("K_XY is not SDP, minimal eigenvalue is {}".format(min_eigenvals))
            if min_eigenvals > -1e-6:  # correction
                self.K_XY = self.K_XY + 2 * (-min_eigenvals) * jnp.eye(
                    self.K_XY.shape[0]
                )
                assert min(np.linalg.eigvalsh(self.K_XY)) > 0.0
            else:
                raise ValueError("K_XY is not SDP")
        self.R_cholesky = np.linalg.cholesky(self.K_XY).T
        _ = self.R_cholesky.T[:, jnp.newaxis, :]
        self.A = jax.vmap(jnp.kron, in_axes=0)(_, _).squeeze()
        self.Id = jnp.eye(self.n)
        self.z = (
            self.w_src
            + self.w_tgt
            - (
                self.lambda2 * jnp.sum((self.X_fillings - self.Y_fillings) ** 2, axis=1)
            )[:, jnp.newaxis]
        )

    def get_OT_from_gamma(self, gamma):
        """Compute the quadratic wasserstein distance estimate from gamma."""
        return (self.q2 - jnp.sum(gamma * (self.w_src + self.w_tgt))) / (
            2 * self.lambda2
        )

    def phi_op(self, sym):
        return _phi_op(self.A, sym)

    def phi_star_op(self, gamma):
        return _phi_star_op(self.A, gamma, self.n)

    def get_R(self, w, return_decomposition=False):
        return _get_R(
            w,
            self.lambda2,
            self.Q,
            self.z,
            self.A,
            self.lambda1,
            self.Id,
            self.n,
            return_decomposition=return_decomposition,
        )

    def apply_C2(self, mu, T_op, R):
        r1, r2 = R
        a1 = -r1 - 1 / mu * (self.phi_op(r2 + T_op(r2)))
        a2 = -r2
        return a1, a2

    def apply_B(self, a, mu, T_op):
        a1, a2 = a
        _block1 = (
            lambda _a1: (self.Q / (2 * self.lambda2) + mu * self.Id) @ _a1
            + self.A @ T_op(self.phi_star_op(_a1)).flatten()
        )
        a_tilde_1, _ = jax.scipy.sparse.linalg.cg(_block1, a1.squeeze())
        a_tilde_2 = 1 / (1 + mu) * (a2 + T_op(a2))
        return a_tilde_1[:, jnp.newaxis], a_tilde_2

    def apply_C1(self, a_tilde, T_op):
        a_tilde_1, a_tilde_2 = a_tilde
        return a_tilde_1, a_tilde_2 - T_op(self.phi_star_op(a_tilde_1))

    def get_update(self, w, theta):
        # compute parameters
        _R, Z_decomp = self.get_R(w, True)
        eigenval, P = Z_decomp
        mu = get_mu(theta, _R)
        T_op = get_T_op(mu, P, eigenval, self.n)

        # update
        a = self.apply_C2(mu, T_op, _R)
        a_tilde = self.apply_B(a, mu, T_op)
        update = self.apply_C1(a_tilde, T_op)

        return update

    def get_rho(self, w_tilde, delta_w):
        _R = self.get_R(w_tilde)
        _R = jnp.concatenate((_R[0].flatten(), _R[1].flatten()))
        _delta_w = jnp.concatenate((delta_w[0].flatten(), delta_w[1].flatten()))
        return -_R @ _delta_w

    # About the extra-gradient method

    def get_eg_params(self):
        # EG class parameters
        n = self.n
        lambda2, Q, z, A, lambda1, Id = (
            self.lambda2,
            self.Q,
            self.z,
            self.A,
            self.lambda1,
            self.Id,
        )

        @jit
        def f(x):
            _gamma, _X = x[:n][:, jnp.newaxis], x[n:]
            f1 = 1 / (2 * lambda2) * (Q @ _gamma - z) - A @ _X[:, jnp.newaxis]
            f2 = -self.phi_star_op(_gamma) - lambda1 * Id
            return jnp.concatenate((f1.flatten(), f2.flatten()))

        @jit
        def proj(x):
            _gamma, _X = x[:n], x[n:].reshape((n, n))
            proj_X = proj_sym_pos(_X)
            return jnp.concatenate((_gamma, proj_X.flatten()))

        # Lipschitz constant
        gamma_bound, X_bound = n**0.5, n
        L1 = (norm(Q) * gamma_bound + norm(z)) / (self.lambda2) + norm(A.T) * X_bound
        L2 = norm(A) * gamma_bound + 1 / self.lambda1
        L = L1 + L2

        return f, proj, L

    def run_eg(self, v0, error=1e-2, max_iter=100):
        # Parameters
        r_norms = []
        n = self.n
        to_w = lambda w: (w[:n][:, jnp.newaxis], w[n:].reshape((n, n)))
        f, proj, L = self.get_eg_params()
        eg = EG(f, proj, L)

        # Run
        eg.init(v0)
        for _ in range(max_iter):
            _ = eg.step()
            w = to_w(_)
            _R = self.get_R(w)
            r_norm = get_r_norm(_R)
            r_norms.append(r_norm)
            if r_norm < error:
                print("The algorithm converged in {} steps.".format(_))
                break

        return w, r_norms

    # About the SSN algorithm

    def update_theta(self, theta, delta_w, w_tilde):  # not jitted
        rho = self.get_rho(w_tilde, delta_w)
        delta_w_norm = get_r_norm(delta_w)
        if rho >= self.alpha_2 * delta_w_norm**2:
            return max(self.theta_min, self.beta_0 * theta)
        elif self.alpha_1 * delta_w_norm**2 <= rho < self.alpha_2 * delta_w_norm**2:
            return self.beta_1 * theta
        else:
            return min(self.theta_max, self.beta_2 * theta)

    def run_ssn(self, v0, theta0, error=1e-2, max_iter=100):
        """Cuturi et al. algorithm 2.
        Termination condition not implemented."""

        # Parameters
        n = self.n
        to_w = lambda w: (w[:n][:, jnp.newaxis], w[n:].reshape((n, n)))
        f, proj, L = self.get_eg_params()
        eg = EG(f, proj, L)
        eg.init(v0)
        _R = self.get_R(to_w(v0))
        r_norms = []
        v, w, theta = v0, to_w(v0), theta0

        # Iteration
        for step in range(max_iter):
            print("Step: ", step)

            # EG
            v = eg.step()

            # SSN
            assert ~jnp.isnan(w[0]).any()  # DEBUG
            assert ~jnp.isnan(w[1]).any()  # DEBUG
            delta_w = self.get_update(w, theta)
            assert ~jnp.isnan(delta_w[0]).any()  # DEBUG
            assert ~jnp.isnan(delta_w[1]).any()  # DEBUG
            print("delta_w norm: ", get_r_norm(delta_w))
            gamma, _X = w
            gamma_tilde, _X_tilde = gamma + delta_w[0], _X + delta_w[1]
            w_tilde = gamma_tilde, _X_tilde

            # Update theta
            theta = self.update_theta(theta, delta_w, w_tilde)

            # Choosing the update
            R_w, R_v = self.get_R(w_tilde), self.get_R(to_w(v))
            if get_r_norm(R_w) < get_r_norm(R_v):
                print("SSN update")
                w = w_tilde
                _R = R_w
            else:
                print("EG update")
                w = to_w(v)
                _R = R_v

            # Save
            r_norm = get_r_norm(_R)
            print("r_norm: ", r_norm)
            r_norms.append(r_norm)
            if r_norm < error:
                print("The algorithm converged in {} steps.".format(step))
                break

        return w, r_norms


# Kernel
def get_gaussian_kernel_func(bandwith):
    @jit
    def gaussian_kernel(a, b):
        # assert a.shape[1] == b.shape[1], "Gaussian kernel defined for same dimension inputs." # No assertion in jit
        diff = a[:, jnp.newaxis, :] - b[jnp.newaxis, :, :]
        return jnp.exp(-jnp.sum(diff**2 / (2 * bandwith**2), axis=-1))

    return gaussian_kernel


@jit
def _get_OT_from_gamma(q2, gamma, w_src, w_tgt, lambda2):
    """Compute the quadratic wasserstein distance estimate from gamma."""
    return (q2 - (w_src + w_tgt).T @ gamma) / (2 * lambda2)


@jit
def _phi_op(A, sym):
    return (A @ sym.flatten())[:, jnp.newaxis]


@partial(jit, static_argnums=2)
def _phi_star_op(A, gamma, n):
    return (A.T @ gamma).reshape((n, n))


@partial(jit, static_argnames=["return_decomposition"])
def proj_sym_pos(Z, return_decomposition=False):
    values, vectors = jnp.linalg.eigh(Z)
    pos_values = values @ jnp.diag((values > 0))
    proj = vectors @ jnp.diag(pos_values) @ vectors.T
    if not return_decomposition:
        return proj
    else:
        return proj, (values, vectors)


@partial(jit, static_argnums=7, static_argnames=["return_decomposition"])
def _get_R(w, lambda2, Q, z, A, lambda1, Id, n, return_decomposition=False):
    gamma, sym = w[0], w[1]
    r1 = 1 / (2 * lambda2) * (Q @ gamma - z) - _phi_op(A, sym)
    Z = sym - (_phi_star_op(A, gamma, n) + lambda1 * Id)
    proj, _ = proj_sym_pos(Z, True)
    r2 = sym - proj

    if return_decomposition:
        return (r1, r2), _
    else:
        return (r1, r2)


@jit
def get_r_norm(R):
    r1, r2 = R
    return jnp.linalg.norm(r1) + jnp.linalg.norm(
        r2
    )  # norm 2 + norm Froebenius by default


@jit
def get_mu(theta, R):
    return (theta * get_r_norm(R)).squeeze()


@jit
def get_xhi(eigenval, mu):
    """Compute xhi from eigenvalues of sym matrix."""
    alpha_filter = jnp.diag((eigenval > 0))
    alpha_bar_filter = jnp.diag(~(eigenval > 0))
    col = jnp.ones((eigenval.shape[0], 1))
    alpha_rows = col @ eigenval[jnp.newaxis, :]
    alpha_cols = alpha_rows.T
    diff = alpha_rows - alpha_cols
    diff = (
        diff
        + alpha_bar_filter @ jnp.ones_like(diff)
        + jnp.ones_like(diff) @ alpha_filter
    )  # no division by zero
    eta = alpha_rows / diff
    eta = alpha_filter @ eta @ alpha_bar_filter
    xhi = eta / (mu + 1 - eta)
    return xhi


@jit
def get_psi(xhi, mu, eigenval):
    alpha_filter = jnp.diag((eigenval > 0))
    return 1 / mu * alpha_filter @ jnp.ones_like(xhi) @ alpha_filter + xhi + xhi.T


def get_T_op(mu, P, eigenval, n):
    """Returns T as a function.
    T_op: n² -> n²
    Not optimized."""
    xhi = get_xhi(eigenval, mu)
    alpha_norm = jnp.sum(eigenval > 0)

    if alpha_norm < 0.5 * n:
        P_alpha = P @ jnp.diag((eigenval > 0))

        @jit
        def T_op(S):
            U = P_alpha.T @ S
            _1 = U @ P_alpha @ P_alpha.T / (2 * mu)
            _2 = U @ P
            G = P_alpha @ (_1 + jnp.multiply(xhi, _2) @ P.T)
            return G + G.T

    else:
        psi = get_psi(xhi, mu, eigenval)

        @jit
        def T_op(S):
            _ = 1 / mu * jnp.ones((n, n)) - psi
            _ = jnp.multiply(_, P.T @ S @ P)
            _ = P @ _ @ P.T
            return S / mu - _

    return T_op
