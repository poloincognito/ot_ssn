import numpy as np
import scipy.linalg
import time

from utils import transport_cost


def log_det_grad_hess(G, Phi, lbda):
    """
    Computes the gradient and the hessian of the barrier function.

    Parameters
    ----------
    G : ndarray
        Parameter being optimized
    Phi : ndarray
        Cholesky factor of the kernel matrix between filling samples (x, y) on X x Y
    lbda : float
        Regularization strength

    Returns
    -------
    grad: ndarray
        Gradient at G of the barrier function
    hess: ndarray
        Hessian at G of the barrier function
    """
    H = Phi.T @ np.linalg.inv((Phi * G) @ Phi.T + lbda * np.eye(Phi.shape[0])) @ Phi
    H = (H + H.T) / 2 ### Ensure symmetry for better stability
    grad = np.diag(H)
    hess = - H**2
    return grad, hess



def interior_point(M, Phi, Kx1, Kx2, Kx3, Ky1, Ky2, Ky3, lbda_1=1e-3, lbda_2=1e-3,
                   tau=1e-10, eps_start=1e-3, eps_end=1e-6, eps_div=2,
                   niter=10000, verbose=True, report_interval=1000,  G_init=None,
                   kernel='gaussian'):
    """
    Optimize the dual problem (5) from Vacher et al. 2021 using a interior point method.

    Parameters
    ----------
    M : ndarray
        Squared distance matrix d(x, y) between filling samples.
    Phi : ndarray
        Cholesky factor of the kernel matrix between filling samples (x, y) on X x Y.
    Kx1 : ndarray
        Kernel matrix between mu filling samples.
    Kx2 : ndarray
        Kernel matrix between mu samples X and filling samples X_fill.
    Kx3 : ndarray
        Kernel matrix between mu samples.
    Ky1 : ndarray
        Kernel matrix between nu filling samples.
    Ky2 : ndarray
        Kernel matrix between nu samples Y and filling samples Y_fill.
    Ky3 : ndarray
        Kernel matrix between nu samples.
    lbda_1 : float, default=1e-3
        Regularization strenght (positive operator).
    lbda_2 : float, default=1e-3
        Regularization strenght (potentials).
    tau : float, default=1e-10
        Stopping criterion (gradient norm).
    eps_start : float, default=1e-3
        Initial value for the barrier function factor.
    eps_end : float, default=1e-6
        Value for the barrier function factor at which to stop.
    eps_div : float, default=2
        Divider for the barrier function factor when precision is reached.
    niter : int, default=10000
        Maximum total number of iterations.
    verbose : bool, default=True
        If true, report loss value periodically.
    report_interval : int, default=1000
        If verbose, period at which the loss is reported.
    G_init : ndarray or None, default=None
        Initial value for G (optional).
    kernel : str, default='gaussian'
        Kernel type.
        Valid values : 'gaussian' or 'sobolev'.

    Returns
    -------
    G: ndarray
        Optimization parameter.
    eps: float
        Barrier function factor at last iteration.
    """

    nfill = len(M)
    n = Kx3.shape[0]
    m = Ky3.shape[0]


    if G_init is None:
        G = np.ones(nfill) / nfill
    else:
        G = G_init.copy()
    

    eps_barrier = eps_start

    ## Variables as in the paper
    Q = Kx1 + Ky1
    z = Kx2.sum(0) / n + Ky2.sum(0) / m - 2 * lbda_2 * M 

    ## Damped Newton

    start_time = time.time()

    for i in range(niter):

        barrier_grad, barrier_hess = log_det_grad_hess(G, Phi, lbda_1)

        ### Form Hessian
        hess = - eps_barrier * barrier_hess + Q / (2 * lbda_2)
        hess = (hess + hess.T) / 2.
        inv_hess = np.linalg.inv(hess)

        ### Symmetrize inverse hessian (numerical stability)
        inv_hess = (inv_hess + inv_hess.T) / 2.

        ### Form gradient
        grad =  (Q @ G - z) / (2  * lbda_2) - eps_barrier * barrier_grad

        v = inv_hess @ grad
        decr = np.sqrt(grad @ v)

        G_ = G - 1. / (1. + decr / np.sqrt(eps_barrier)) * v

        ### Check that nothing has blown up
        if np.isnan(decr):
            print("NaN decrement, stopping")
            return G, eps_barrier
        else:
            G = G_


        ### Check decrement, and decrease epsilon if small than threshold
        if decr < tau:
            eps_barrier /= eps_div
            if eps_barrier < eps_end:
                if verbose:
                    print(
                        f"iter {i + 1}:\ttransport: {transport_cost(G, Kx2, Kx3, Ky2, Ky3, lbda_2):.2e}" \
                        f"\tdecr: {decr:.2e}\teps: {eps_barrier:.2e}")
                    print(f"Precision {eps_end:.2e} reached in {time.time() - start_time:.2e} seconds")

                return G, eps_barrier * eps_div

        if verbose and i % report_interval == 0:
            print(
                f"iter {i + 1}:\ttransport: {transport_cost(G, Kx2, Kx3, Ky2, Ky3, lbda_2):.2e}" \
                f"\tdecr: {decr:.2e}\teps: {eps_barrier:.2e}")

    if verbose:
        print(f"Precision {eps_end:.2e} not reached in {niter} iterations and {time.time() - start_time:.2e} seconds")
    
    return G, eps_barrier * eps_div
