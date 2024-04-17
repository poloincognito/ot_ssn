# See "The Subgradient Extragradient Method
# for Solving Variational Inequalities in Hilbert Space"
# for the algorithm

# See https://conferences.mpi-inf.mpg.de/adfocs-22/material/alina/adaptive-L3.pdf
# for the link between VI and min max

# See https://openreview.net/forum?id=snUOkDdJypm for the stepsize
# (optimal stepsize is 1/(2**0.5 * L))


class EG:
    """Extragradient method for solving variational inequalities.
    In the case of a min max problem, with F convex in its first arg
    and concave in its second arg, f should be: (d1 F, -d2 F)"""

    def __init__(self, f, proj, L=1.0):
        self.f = f  # gradient
        self.proj = proj
        self.stepsize = 1 / (2**0.5 * L)  # optimal stepsize is 1/(2**0.5 * L)

    def init(self, x0):
        self.x = x0

    def step(self):
        x = self.x
        y = self.proj(x - self.stepsize * self.f(x))
        self.x = self.proj(x - self.stepsize * self.f(y))
        return self.x
