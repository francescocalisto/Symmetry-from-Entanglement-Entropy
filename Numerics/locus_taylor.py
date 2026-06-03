"""Fast numerical Taylor-mode evaluator for T^(k)(lambda).

Avoids the sympy expansion of (D_beta)^k T, which explodes by k=3.
Instead, integrate lambda(t) along the RG flow as a degree-K truncated
Taylor series in t and read off
    T^(k)(lambda_0) = k! * [t^k] T(lambda(t)).

Strategy
--------
Represent each coupling lam_alpha(t) as an (K+1)-vector
    a = [a_0, a_1, ..., a_K]    meaning  lam(t) = sum_k a_k t^k.
The ODE dlam/dt = beta(lam) gives, after truncating to order K and
matching t-coefficients,
    (k+1) a_{k+1} = [t^k] beta(a(t)),    k = 0, 1, ..., K-1.
beta is a fully-symmetric quartic polynomial in lambda evaluated by
einsum on the symmetric tensor; multiplication / contraction on Taylor
series is just polynomial multiplication coefficient-by-coefficient.

This reduces each call to O(N^8) for the einsums (rho, beta) times K^2
for the truncated multiplications, with NO symbolic expansion -- the
polynomial-degree blow-up that crushes sympy never appears.

The Jacobian needed by Levenberg-Marquardt is obtained by finite
differences across the lambda_0 inputs (cheap: one residual eval per
input coupling).  This converges fine for the polynomial system at
tolerance 1e-6.

Usage
-----
    res, jac, var_keys = make_residual(N=3, k_max=3, pin_norm=True)
    from scipy.optimize import least_squares
    least_squares(res, x0, jac=jac, method='lm', ...)
"""

from __future__ import annotations

from itertools import product, permutations
import numpy as np


# ---------------------------------------------------------------------------
# Coupling-index utilities.
# ---------------------------------------------------------------------------

def make_keys(N: int):
    return sorted({tuple(sorted(t)) for t in product(range(N), repeat=4)})


def inflate(x: np.ndarray, keys, N: int) -> np.ndarray:
    """Inflate independent components x[a] -> full NxNxNxN symmetric tensor.
    Preserves dtype of x so that complex perturbations propagate through.
    """
    x = np.asarray(x)
    lam = np.zeros((N, N, N, N), dtype=x.dtype)
    for v, k in zip(x, keys):
        for p in set(permutations(k)):
            lam[p] = v
    return lam


def deflate(lam: np.ndarray, keys) -> np.ndarray:
    """Read independent components from a fully-symmetric tensor."""
    return np.array([lam[k] for k in keys])


# ---------------------------------------------------------------------------
# Truncated Taylor series in t.
# ---------------------------------------------------------------------------

class Taylor:
    """Truncated power series in t with numpy-tensor coefficients.

    Stored as a list / ndarray with first axis = t-degree, remaining axes
    = the value's shape.  Supports +, -, *, scalar mul, contraction via
    einsum-friendly arithmetic.
    """
    __slots__ = ("c", "K")

    def __init__(self, coeffs, K=None):
        c = np.asarray(coeffs)
        if K is None:
            K = c.shape[0] - 1
        self.c = c
        self.K = K

    @classmethod
    def constant(cls, value, K):
        value = np.asarray(value)
        c = np.zeros((K + 1,) + value.shape, dtype=value.dtype)
        c[0] = value
        return cls(c, K)

    @classmethod
    def variable(cls, value, K):
        """A Taylor series with constant term `value` and zero higher order."""
        return cls.constant(value, K)

    def __add__(self, other):
        if isinstance(other, Taylor):
            return Taylor(self.c + other.c, self.K)
        c = self.c.copy()
        c[0] = c[0] + other
        return Taylor(c, self.K)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Taylor):
            return Taylor(self.c - other.c, self.K)
        c = self.c.copy()
        c[0] = c[0] - other
        return Taylor(c, self.K)

    def __neg__(self):
        return Taylor(-self.c, self.K)

    def __mul__(self, other):
        if isinstance(other, Taylor):
            # Convolution along axis 0, truncate to K+1
            K = self.K
            shape = np.broadcast_shapes(self.c.shape[1:], other.c.shape[1:])
            out = np.zeros((K + 1,) + shape)
            for i in range(K + 1):
                for j in range(K + 1 - i):
                    out[i + j] = out[i + j] + self.c[i] * other.c[j]
            return Taylor(out, K)
        return Taylor(self.c * other, self.K)

    __rmul__ = __mul__


def taylor_einsum(spec: str, *taylors):
    """einsum on Taylor-series objects.  Computes coefficient-by-coefficient
    along the t-axis, contracting on the remaining axes per `spec`.
    Preserves the (possibly complex) dtype of inputs.
    """
    K = taylors[0].K
    shapes = [t.c.shape[1:] for t in taylors]
    dtype = np.result_type(*[t.c.dtype for t in taylors])
    probe = [np.zeros(s, dtype=dtype) for s in shapes]
    out_shape = np.einsum(spec, *probe).shape
    out = np.zeros((K + 1,) + out_shape, dtype=dtype)
    n = len(taylors)
    def gen(remaining, sumleft):
        if remaining == 1:
            for k in range(sumleft + 1):
                yield (k,)
            return
        for k in range(sumleft + 1):
            for tail in gen(remaining - 1, sumleft - k):
                yield (k,) + tail
    for idx in gen(n, K):
        out[sum(idx)] += np.einsum(
            spec, *(taylors[i].c[idx[i]] for i in range(n))
        )
    return Taylor(out, K)


# ---------------------------------------------------------------------------
# beta and T as functions of Taylor series.
# ---------------------------------------------------------------------------

def beta_taylor(lam_t: Taylor, N: int) -> Taylor:
    """beta_{ijkl} = -lam_{ijkl} + sum over 3 channels of sum_{a,b}
    lam_{(p,q,a,b)} lam_{(a,b,r,t)}.  lam_t is a Taylor series of NxNxNxN
    fully symmetric tensors.
    """
    minus_lam = (-1.0) * lam_t
    # channel 1: sum_{ab} lam_{ijab} lam_{abkl}
    ch1 = taylor_einsum("ijab,abkl->ijkl", lam_t, lam_t)
    # channel 2: sum_{ab} lam_{ikab} lam_{abjl}
    ch2 = taylor_einsum("ikab,abjl->ijkl", lam_t, lam_t)
    # channel 3: sum_{ab} lam_{ilab} lam_{abjk}
    ch3 = taylor_einsum("ilab,abjk->ijkl", lam_t, lam_t)
    return minus_lam + ch1 + ch2 + ch3


def T_taylor(lam_t: Taylor, N: int) -> Taylor:
    """T_{ij} = rho_{ij} - (1/N) tr(rho) delta_{ij}
    with rho_{ij} = sum_{klm} lam_{iklm} lam_{jklm}."""
    rho = taylor_einsum("iklm,jklm->ij", lam_t, lam_t)
    tr_rho = taylor_einsum("ii->", rho)
    I = np.eye(N, dtype=lam_t.c.dtype)
    K = lam_t.K
    sub = np.zeros((K + 1, N, N), dtype=lam_t.c.dtype)
    for k in range(K + 1):
        sub[k] = (tr_rho.c[k] / N) * I
    return rho + Taylor(-sub, K)


def evolve_lambda(lam0: np.ndarray, K: int, N: int) -> Taylor:
    """Compute the Taylor-series solution lambda(t) of dlambda/dt = beta(lambda)
    with lambda(0) = lam0, truncated to order K.
    """
    lam0 = np.asarray(lam0)
    c = np.zeros((K + 1, N, N, N, N), dtype=lam0.dtype)
    c[0] = lam0
    lam_t = Taylor(c, K)
    for k in range(K):
        beta_t = beta_taylor(lam_t, N)
        lam_t.c[k + 1] = beta_t.c[k] / (k + 1)
    return lam_t


# ---------------------------------------------------------------------------
# Build residuals and finite-difference Jacobian for LM.
# ---------------------------------------------------------------------------

def make_residual(N: int, k_max: int, pin_norm: bool = True):
    """Return (residual_fn, fd_jacobian_fn, keys) callables.
    residual_fn(x): returns the polynomial residual vector
        [T^(0)_ij, T^(1)_ij, ..., T^(k_max)_ij  for i<=j, !=(N-1,N-1)]
        optionally + ||x||^2 - 1.
    fd_jacobian_fn(x): finite-difference Jacobian.
    """
    keys = make_keys(N)
    n_vars = len(keys)
    K = k_max  # need Taylor degrees up to K to read off T^(K)

    # Indices (i,j) for upper triangle minus (N-1,N-1)
    ij_pairs = [(i, j) for i in range(N) for j in range(i, N)
                if (i, j) != (N - 1, N - 1)]
    n_T_entries = len(ij_pairs)

    def _residual(x):
        """Polymorphic in x.dtype (real or complex)."""
        x = np.asarray(x)
        lam0 = inflate(x, keys, N)
        lam_t = evolve_lambda(lam0, K, N)
        T_t = T_taylor(lam_t, N)
        out = np.zeros(
            n_T_entries * (K + 1) + (1 if pin_norm else 0), dtype=x.dtype
        )
        from math import factorial
        idx = 0
        for k in range(K + 1):
            fac = float(factorial(k))
            for (i, j) in ij_pairs:
                out[idx] = fac * T_t.c[k, i, j]
                idx += 1
        if pin_norm:
            out[-1] = np.dot(x, x) - 1.0
        return out

    def residual(x):
        return _residual(np.asarray(x, dtype=float)).real

    def cs_jacobian(x, h=1e-30):
        """Complex-step Jacobian: each column from one complex residual eval.
        No truncation error (it's literally the derivative), no cancellation
        (we read out the imaginary part scaled by h).
        """
        x = np.asarray(x, dtype=float)
        m = len(_residual(x))
        J = np.zeros((m, n_vars))
        for i in range(n_vars):
            xc = x.astype(np.complex128)
            xc[i] = xc[i] + 1j * h
            J[:, i] = _residual(xc).imag / h
        return J

    return residual, cs_jacobian, keys


# ---------------------------------------------------------------------------
# Demo / sanity check.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from scipy.optimize import least_squares

    for N, k_max in [(2, 1), (2, 3), (3, 1), (3, 3), (4, 1), (4, 2)]:
        res, jac, keys = make_residual(N, k_max, pin_norm=True)
        n_vars = len(keys)
        n_res = len(res(np.zeros(n_vars)))
        method = "lm" if n_res >= n_vars else "trf"
        rng = np.random.default_rng(0)
        x0 = rng.standard_normal(n_vars); x0 /= np.linalg.norm(x0)
        t0 = time.perf_counter()
        r = res(x0)
        t_res = time.perf_counter() - t0
        t0 = time.perf_counter()
        J = jac(x0)
        t_jac = time.perf_counter() - t0
        print(f"N={N}  k_max={k_max}  n_vars={n_vars}  n_res={n_res}  "
              f"method={method}  "
              f"t_res={t_res * 1e3:.1f}ms  t_jac={t_jac * 1e3:.1f}ms")
        t0 = time.perf_counter()
        out = least_squares(res, x0, jac=jac, method=method,
                            xtol=1e-15, ftol=1e-15, gtol=1e-15,
                            max_nfev=200)
        dt = time.perf_counter() - t0
        n_poly = len(out.fun) - 1
        max_poly = np.max(np.abs(out.fun[:n_poly]))
        print(f"   LM:  converged in {dt:.2f}s, "
              f"|| poly res ||_inf = {max_poly:.2e}, "
              f"|| x || = {np.linalg.norm(out.x):.4f}")
