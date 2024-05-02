import numpy as np
from scipy.special import kv, kvp, gamma


def sobolev_kernel(x, y, l=1.0, s=None):
    """
    Computes the sobolev kernel matrix between x and y (vectorized).

    Parameters
    ----------
    x : ndarray
            First kernel argument.
    y : ndarray
            Second kernel argument.
    l : float, default=1.
            Kernel bandwidth.
    s : float, default=None
            Sobolev parameter.
            If None, the default value will be (d + 1) / 2. The corresponds to the exponential kernel.

    Returns
    -------
    ndarray
            Kernel matrix

    """
    if len(x.shape) > 1:
        dim = x.shape[-1]
        arg = np.sqrt(((x - y) ** 2).sum(-1)) / l
    else:
        dim = 1
        arg = np.sqrt(((x - y) ** 2)) / l
    if s is None or s == (dim + 1) / 2:
        return np.exp(-arg)
    ### If s > (d + 1) / 2 we may get warnings due to the hack that handles x = y
    else:
        res = np.ones_like(arg)
        c = 2 ** (1 + dim / 2 - s) / gamma(s - dim / 2)
        res[arg != 0.0] = (c * arg ** (s - dim / 2) * kv(s - dim / 2, arg))[arg != 0.0]
    return res


def gaussian_kernel(x, y, l=1.0):
    """
    Computes the Gaussian kernel matrix between x and y (vectorized).

    Parameters
    ----------
    x : ndarray
            First kernel argument.
    y : ndarray
            Second kernel argument.
    l : float, default=1.
            Kernel bandwidth.

    Returns
    -------
    ndarray
            Kernel matrix

    """
    return np.exp(-((x - y) ** 2).sum(-1) / l)


def make_kernels(
    X, Y, X_fill, Y_fill, l=0.1, kernel="gaussian", product_sampling=False, **kwargs
):
    """
    Build kernel matrices from samples.

    Parameters
    ----------
    X : ndarray
            Samples from mu.
    Y : ndarray
            Samples from nu.
    X_fill : ndarray
            Filling samples for the support of mu.
    Y_fill : ndarray
            Filling samples for the support of nu.
    l : float, default=.1
            Kernel bandwidth.
    kernel : str, default='gaussian'
            Kernel type.
            Valid values : 'gaussian' or 'sobolev'.
    product_sampling : bool, default=False
            If True, the filling samples correspond to the all products of mu and nu samples.
            Should be used jointly with product_sampling=True in optim.interior_point.
            Supersedes X_fill and Y_fill.
    **kwargs
            Optional keyword arguments for kernels.
            Use ths to pass s for the Sobolev kernel.

    Returns
    -------
    Phi : ndarray
            Cholesky factor of the kernel matrix between filling samples (x, y) on X x Y.
    M : ndarray
            Squared distance matrix d(x, y) between filling samples.
    Kx1 : ndarray
            Kernel matrix between mu filling samples.
    Ky1 : ndarray
            Kernel matrix between nu filling samples.
    Kx2 : ndarray
            Kernel matrix between mu samples X and filling samples X_fill.
    Ky2 : ndarray
            Kernel matrix between nu samples Y and filling samples Y_fill.
    Kx3 : ndarray
            Kernel matrix between mu samples.
    Ky3 : ndarray
            Kernel matrix between nu samples.
    """

    n, dim = X.shape
    m = len(Y)
    nfill = len(X_fill)

    XY_fill = np.hstack([X_fill, Y_fill])

    ## Distance matrix
    M = ((X_fill - Y_fill) ** 2).sum(1) / 2

    assert kernel in ["sobolev", "gaussian"]

    if kernel == "sobolev":
        kernel_func = sobolev_kernel
    else:
        kernel_func = gaussian_kernel

    ## Build kernel matrices
    if product_sampling:

        Kx1 = kernel_func(X[:, None], X, l, **kwargs)
        Ky1 = kernel_func(Y[:, None], Y, l, **kwargs)

        Kx2 = Kx1
        Ky2 = Ky1

        Kx3 = Kx1
        Ky3 = Ky1

    else:

        Kx1 = kernel_func(X_fill[:, None], X_fill, l, **kwargs)
        Ky1 = kernel_func(Y_fill[:, None], Y_fill, l, **kwargs)

        Kx2 = kernel_func(X[:, None], X_fill, l, **kwargs)
        Ky2 = kernel_func(Y[:, None], Y_fill, l, **kwargs)

        Kx3 = kernel_func(X[:, None], X, l, **kwargs)
        Ky3 = kernel_func(Y[:, None], Y, l, **kwargs)

    K = kernel_func(XY_fill[:, None], XY_fill, l, **kwargs)
    Phi = np.linalg.cholesky(K).T

    return Phi, M, Kx1, Ky1, Kx2, Ky2, Kx3, Ky3


###### Transport functions #######


def transport_cost(G, Kx2, Kx3, Ky2, Ky3, lbda_2, product_sampling=False):
    """
    Computes the kernel SoS optimal transport estimator (4)

    Parameters
    ----------
    G : ndarray
            Optimization parameter.
    Kx2 : ndarray
            Kernel matrix between mu samples X and filling samples X_fill.
    Kx3 : ndarray
            Kernel matrix between mu samples.
    Ky2 : ndarray
            Kernel matrix between nu samples Y and filling samples Y_fill.
    Ky3 : ndarray
            Kernel matrix between nu samples.
    lbda_2 : float
            Regularization strength (potentials).
    product_sampling : bool, default=False
            If True, the filling samples correspond to the all products of mu and nu samples.
            Should be used jointly with product_sampling=True in optim.interior_point.

    Returns
    -------
    float
            Kernel SoS estimator of the optimal transport metric.
    """
    n = Kx3.shape[0]
    m = Ky3.shape[0]
    if product_sampling:
        Gam = G.reshape(n, m)
        return (
            Kx3.mean() + Ky3.mean() - (Kx2 @ Gam).sum() / n - (Gam @ Ky2).sum() / m
        ) / (2 * lbda_2)
    else:
        return (
            Kx3.mean() + Ky3.mean() - (G * Kx2.mean(0)).sum() - (G * Ky2.mean(0)).sum()
        ) / (2 * lbda_2)


def sobolev_grad(x, y, l=1.0, s=None):
    """
    Computes the gradient of the Sobolev kernel between x and y (vectorized)

    Parameters
    ----------
    x : ndarray
            First kernel argument.
    y : ndarray
            Second kernel argument.
    l : float, default=1.
            Kernel bandwidth.
    s : float, default=None
            Sobolev parameter.
            If None, the default value will be (d + 1) / 2. The corresponds to the exponential kernel.

    Returns
    -------
    ndarray
            Gradients of the Sobolev kernel
    """
    dim = x.shape[-1]
    arg = np.sqrt(((x - y) ** 2).sum(-1))
    idxs = arg != 0
    if len(x.shape) == 1:
        arg = arg[:, None]
    elif len(x.shape) > 1:
        arg = arg[:, :, None]
    res = np.zeros_like(x - y)
    if s is None or s == (dim + 1) / 2:
        res[idxs] = -((x - y) / (l * arg) * np.exp(-arg / l))[idxs]
    ### If s > (d + 1) / 2 we may get warnings due to the hack that handles x = y
    else:
        c = 2 ** (1 + dim / 2 - s) / gamma(
            s - dim / 2
        )  ## If l != 1, is this the right constant?
        res[idxs] = (
            c
            * (
                (s - dim / 2)
                / (l ** (s - dim / 2))
                * (x - y)
                * arg ** (s - dim / 2 - 2)
                * kv(s - dim / 2, arg / l)
                + (x - y)
                / (l ** (s - dim / 2 + 1))
                * arg ** (s - dim / 2 - 1)
                * kvp(s - dim / 2, arg / l)
            )[idxs]
        )
    return res.squeeze()


def gaussian_grad(x, y, l=1.0):
    """
    Computes the gradient of the Gaussian kernel between x and y (vectorized)

    Parameters
    ----------
    x : ndarray
            First kernel argument.
    y : ndarray
            Second kernel argument.
    l : float, default=1.
            Kernel bandwidth.

    Returns
    -------
    ndarray
            Gradients of the Gaussian kernel
    """
    if len(x.shape) == 1:
        return (-2.0 / l * (x - y) * gaussian_kernel(x, y, l)[:, None]).squeeze()
    else:
        return (-2.0 / l * (x - y) * gaussian_kernel(x, y, l)[:, :, None]).squeeze()


def potential_1D(x, G, X, X_fill, lbda, l=1.0, kernel="gaussian", **kwargs):
    """
    Computes the kernel SoS estimator of the potentials (1D setting).

    Parameters
    ----------
    x : ndarray
            Point at which to evaluate the potential function.
    G : ndarray
            Optimization parameter (dual problem).
    X : ndarray
            Mu or nu samples.
    X_fill : ndarray
            Filling samples for mu or nu.
    lbda : float
            Regularization strength (potentials).
    l : float, default = 1.
            Kernel bandwidth.
    kernel : str, default='gaussian'
            Kernel type.
            Valid values : 'gaussian' or 'sobolev'.
    **kwargs
            Optional keyword arguments for kernels.
            Use ths to pass s for the Sobolev kernel.

    Returns
    -------
    float or ndarray
            Kernel SoS potential function at x.
    """
    n = len(X)
    if kernel == "sobolev":
        kernel_func = sobolev_kernel
    else:
        kernel_func = gaussian_kernel
    if len(x.shape) == 1:
        return (
            1
            / (2 * lbda)
            * (
                1 / n * kernel_func(x, X, l=l, **kwargs).sum()
                - (G * kernel_func(x, X_fill, l=l, **kwargs)).sum(0)
            )
        )

    else:
        return (
            1
            / (2 * lbda)
            * (
                (1 / n * kernel_func(x[:, None], X[:, None], l=l, **kwargs)).sum(1)
                - (G * kernel_func(x[:, None], X_fill, l=l, **kwargs)).sum(1)
            )
        )


def transport_1D(x, G, X, X_fill, lbda_2, l=1.0, kernel="gaussian", **kwargs):
    """
    Computes the kernel SoS estimator of the transport map (1D setting).

    Parameters
    ----------
    x : ndarray
            Point at which to evaluate the transport map.
    G : ndarray
            Optimization parameter (dual problem).
    X : ndarray
            Mu or nu samples.
    X_fill : ndarray
            Filling samples for mu or nu.
    lbda_1 : float
            Regularization strength (potentials).
    l : float, default = 1.
            Kernel bandwidth.
    kernel : str, default='gaussian'
            Kernel type.
            Valid values : 'gaussian' or 'sobolev'.
    **kwargs
            Optional keyword arguments for kernels.
            Use ths to pass s for the Sobolev kernel.

    Returns
    -------
    float or ndarray
            Kernel SoS transport map at x.
    """
    n = len(X)

    if kernel == "sobolev":
        grad_func = sobolev_grad
    else:
        grad_func = gaussian_grad

    return x - (
        1.0 / n * grad_func(x[:, None, None], X[:, None], l, **kwargs).sum(1)
        - (G * grad_func(x[:, None, None], X_fill, l, **kwargs)).sum(1)
    ) / (2 * lbda_2)
