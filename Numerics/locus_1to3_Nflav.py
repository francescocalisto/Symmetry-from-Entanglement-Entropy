"""
Numerical study of the {T = 0, dT/dt = 0} locus for the 1|N partition
of the two-flavour-style real symmetric quartic.

Setup (matching rg_22_intersection_note.pdf and build_Tk_general.wl):
    * Fully symmetric quartic coupling lam_{ijkl}, indices in {1..N}.
    * rho_{1|N}_{ij} = sum_{k,l,m} lam_{i,k,l,m} lam_{j,k,l,m}.
    * T = rho - (1/N) tr(rho) I   (traceless symmetric, spin-2 part).
    * One-loop beta with eps = kappa = 1:
          beta_{ijkl} = -lam_{ijkl} + sum over (s,t,u) channels of
                        sum_{a,b} lam_{(pq)(ab)} lam_{(ab)(rt)}.
    * Lie derivative: dF/dt = sum_alpha beta_alpha (dF/dlam_alpha).

Polynomial system we solve to ||.||_inf < 1e-6:
    residuals = [ T_ij  for i<=j, (i,j) != (N-1,N-1) ]
              + [ (dT/dt)_ij  for the same (i,j) ]
The (N-1,N-1) entry is omitted because of the trace constraint
(T_{NN} = - sum_{i<N} T_{ii}).

For N=2 the analytic locus is (eq. 43 of the note)
    E_- = {(a,b,c,d,e) = (a, b, c, -b, a)}.

Approach
--------
Levenberg-Marquardt on the residual vector with exact symbolic Jacobian
(via sympy.lambdify). LM has quadratic local convergence on smooth
systems and handles rank-deficient Jacobians via its damping term, which
is exactly the regime here: the joint locus is a positive-dim variety,
so the Jacobian at a solution has a non-trivial kernel.

Why this scales: each LM step does one Jacobian solve, O(M^3) where
M = #residuals. The number of independent couplings grows like
binomial(N+3,4), so for N <= 5 (70 couplings) sympy lambdify is still
fast. For larger N, swap build_polynomials for a pure-numpy einsum
evaluator on the full symmetric tensor and use autodiff (JAX) for the
Jacobian -- the LM driver is unchanged.
"""

from __future__ import annotations

from itertools import product
from time import perf_counter
import argparse

import numpy as np
import sympy as sp
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Symbolic build: T(lambda) and dT/dt(lambda) for a given N (flavours).
# ---------------------------------------------------------------------------

def build_polynomials(N: int, k_max: int = 1):
    """Build symbolic T^(0), T^(1), ..., T^(k_max) for N flavours.

    T^(0) = rho_{1|N} - (1/N) tr(rho_{1|N}) I.
    T^(k+1) = Lie_beta T^(k)  (one-loop RG, eps = kappa = 1).
    Returns (var_list, [T0, T1, ..., T_kmax], keys).
    """
    # Independent couplings: sorted 4-tuples with entries in 0..N-1.
    keys = sorted({tuple(sorted(t)) for t in product(range(N), repeat=4)})
    sym = {k: sp.Symbol("l" + "".join(str(i + 1) for i in k)) for k in keys}

    def lam(i, j, k, l):
        return sym[tuple(sorted([i, j, k, l]))]

    # rho_{1|N}_{ij}
    rho = sp.zeros(N, N)
    for i in range(N):
        for j in range(N):
            rho[i, j] = sum(
                lam(i, k, l, m) * lam(j, k, l, m)
                for k in range(N) for l in range(N) for m in range(N)
            )
    rho = sp.expand(rho)

    # T = rho - (1/N) tr(rho) I
    T0 = sp.expand(rho - sp.Rational(1, N) * rho.trace() * sp.eye(N))

    # One-loop beta for each independent coupling.
    beta = {}
    for k in keys:
        i, j, kk, l = k
        b = -lam(i, j, kk, l)
        channels = [(i, j, kk, l), (i, kk, j, l), (i, l, j, kk)]
        for (p, q, r, t) in channels:
            for a in range(N):
                for bb in range(N):
                    b += lam(p, q, a, bb) * lam(a, bb, r, t)
        beta[sym[k]] = sp.expand(b)

    var_list = [sym[k] for k in keys]

    def lie(M):
        out = sp.zeros(*M.shape)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                out[i, j] = sp.expand(
                    sum(beta[v] * sp.diff(M[i, j], v) for v in var_list)
                )
        return out

    T_list = [T0]
    for _ in range(k_max):
        T_list.append(lie(T_list[-1]))

    return var_list, T_list, keys


def build_residuals(N: int, k_max: int = 1, pin_norm: bool = True):
    """Return (residual, jacobian, var_list, keys) as fast numpy callables.

    Residual = concat( T^(0), T^(1), ..., T^(k_max) ) on upper-triangle
    entries (skipping the (N-1,N-1) trace-constrained entry), optionally
    plus ||x||^2 - 1 as the final entry.
    """
    var_list, T_list, keys = build_polynomials(N, k_max=k_max)

    res_sym = []
    for T_k in T_list:
        for i in range(N):
            for j in range(i, N):
                if (i, j) == (N - 1, N - 1):
                    continue
                res_sym.append(T_k[i, j])
    if pin_norm:
        res_sym.append(sum(v * v for v in var_list) - 1)
    R = sp.Matrix(res_sym)
    J = R.jacobian(var_list)

    R_fn = sp.lambdify(var_list, R, modules="numpy")
    J_fn = sp.lambdify(var_list, J, modules="numpy")

    def residual(x):
        return np.asarray(R_fn(*x), dtype=float).reshape(-1)

    def jacobian(x):
        return np.asarray(J_fn(*x), dtype=float)

    return residual, jacobian, var_list, keys


# ---------------------------------------------------------------------------
# Multistart Levenberg-Marquardt search on the unit sphere.
# ---------------------------------------------------------------------------

def find_locus(N: int, n_starts: int = 500, tol: float = 1e-6,
               k_max: int = 1, seed: int = 0, verbose: bool = True):
    residual, jacobian, var_list, keys = build_residuals(
        N, k_max=k_max, pin_norm=True
    )
    n_vars = len(var_list)
    n_res_full = len(residual(np.zeros(n_vars)))
    n_poly_res = n_res_full - 1  # last entry is the norm constraint
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"[build] N = {N}, k_max = {k_max}, "
              f"{n_vars} independent couplings, "
              f"{n_poly_res} polynomial residuals + 1 norm constraint.")

    # MINPACK 'lm' refuses n_res < n_vars; use trust-region reflective in that
    # regime.  'lm' is faster on square systems (N=2 with the norm pin).
    method = "lm" if n_res_full >= n_vars else "trf"
    if verbose:
        print(f"[solve] method = {method}")

    solutions = []
    n_converged = 0
    t0 = perf_counter()

    for _ in range(n_starts):
        x0 = rng.standard_normal(n_vars)
        x0 /= np.linalg.norm(x0)
        try:
            res = least_squares(
                residual, x0, jac=jacobian, method=method,
                xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=400,
            )
        except Exception:
            continue

        poly_res = res.fun[:n_poly_res]
        max_poly = float(np.max(np.abs(poly_res)))
        norm = float(np.linalg.norm(res.x))

        # Accept if polynomial residuals are small and norm ~ 1
        # (the constraint pins it but we tolerate it not being perfect).
        if max_poly < tol and 0.5 < norm < 2.0:
            n_converged += 1
            solutions.append(res.x / norm)

    dt = perf_counter() - t0
    sols = np.asarray(solutions)
    if verbose:
        print(f"[solve] {n_starts} starts in {dt:.2f}s "
              f"({dt / n_starts * 1e3:.1f} ms / start)")
        print(f"        {n_converged} starts converged with "
              f"||poly residual||_inf < {tol:g}.")
    return sols, var_list, keys


# ---------------------------------------------------------------------------
# Analytic check + plot for N = 2.
# ---------------------------------------------------------------------------

def report_N2(sols: np.ndarray):
    """For N=2 the locus is E_- : a=e, d=-b.  Quantify and plot."""
    if sols.size == 0:
        print("No solutions found.")
        return

    # var ordering from build_polynomials at N=2:
    # keys = [(0,0,0,0),(0,0,0,1),(0,0,1,1),(0,1,1,1),(1,1,1,1)]
    # i.e. (a, b, c, d, e).
    a, b, c, d, e = sols.T

    diff_ae = a - e
    diff_bd = b + d
    deviation = np.maximum(np.abs(diff_ae), np.abs(diff_bd))

    print()
    print("---- Analytic check: E_- = {a = e, d = -b} ----")
    print(f"  max |a - e|       = {np.max(np.abs(diff_ae)):.3e}")
    print(f"  max |b + d|       = {np.max(np.abs(diff_bd)):.3e}")
    print(f"  max deviation     = {np.max(deviation):.3e}")
    print(f"  median deviation  = {np.median(deviation):.3e}")
    print(f"  solutions with deviation < 1e-5 : "
          f"{int(np.sum(deviation < 1e-5))} / {len(sols)}")

    # Plot: distance from E_- and the locus itself on the unit sphere.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(13, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(diff_ae, diff_bd, s=14, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax1.axhline(0, color="k", lw=0.5)
    ax1.axvline(0, color="k", lw=0.5)
    ax1.set_xlabel(r"$a - e$")
    ax1.set_ylabel(r"$b + d$")
    ax1.set_title(r"Distance of numerical minima from $\mathcal{E}_-$ "
                  r"($a=e,\, d=-b$)")
    lim = max(1e-8, 1.2 * np.max(np.abs(np.concatenate([diff_ae, diff_bd]))))
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal", "box")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sc = ax2.scatter(a, b, c, c=c, cmap="viridis", s=14, alpha=0.7)
    # Overlay analytic surface 2a^2 + 2b^2 + c^2 = 1 (unit-sphere locus).
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(-1, 1, 60)
    U, V = np.meshgrid(u, v)
    # parameterize:  c = V,  a = sqrt((1-V^2)/2) cos U, b = sqrt((1-V^2)/2) sin U
    R_uv = np.sqrt(np.clip((1 - V ** 2) / 2, 0, None))
    A = R_uv * np.cos(U)
    B = R_uv * np.sin(U)
    C = V
    ax2.plot_surface(A, B, C, alpha=0.15, color="grey",
                     linewidth=0, antialiased=False)
    ax2.set_xlabel("a")
    ax2.set_ylabel("b")
    ax2.set_zlabel("c")
    ax2.set_title(r"Numerical minima on $\mathcal{E}_-$ "
                  r"(unit sphere $\Rightarrow$ $2a^2+2b^2+c^2=1$)")
    fig.colorbar(sc, ax=ax2, shrink=0.55, label="c")

    plt.tight_layout()
    out = "locus_N2.png"
    plt.savefig(out, dpi=150)
    print(f"  plot saved -> {out}")


# ---------------------------------------------------------------------------
# Sigma analytic check + O(N)-invariant orbit check (general N).
# ---------------------------------------------------------------------------

def classify_keys(keys, N):
    """Group the independent-coupling indices by multiset shape."""
    from collections import Counter
    idx_iiii, idx_iijj, idx_iiij, idx_iijk, idx_ijkl = [], [], [], [], []
    for n, k in enumerate(keys):
        shape = tuple(sorted(Counter(k).values(), reverse=True))
        if shape == (4,):           idx_iiii.append(n)
        elif shape == (3, 1):       idx_iiij.append(n)
        elif shape == (2, 2):       idx_iijj.append(n)
        elif shape == (2, 1, 1):    idx_iijk.append(n)
        elif shape == (1, 1, 1, 1): idx_ijkl.append(n)
    return idx_iiii, idx_iijj, idx_iiij, idx_iijk, idx_ijkl


def full_tensor(x, keys, N):
    """Inflate the independent-component vector x into the full symmetric
    NxNxNxN tensor lam_{ijkl}."""
    from itertools import permutations
    lam = np.zeros((N, N, N, N))
    for val, k in zip(x, keys):
        for p in set(permutations(k)):
            lam[p] = val
    return lam


def flattening_matrix(x, keys, N):
    """Build the symmetric M_{(ij),(kl)} = lam_{ijkl} on unordered pairs.
    For N=3 this is a 6x6 symmetric matrix; its spectrum is O(N)-invariant.
    """
    lam = full_tensor(x, keys, N)
    pairs = [(i, j) for i in range(N) for j in range(i, N)]
    n_p = len(pairs)
    M = np.zeros((n_p, n_p))
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            M[a, b] = lam[i, j, k, l]
    return M


def sigma_spectrum(u, w, N):
    """Analytic spectrum of M on Sigma for general N.  M decomposes as
    A (acting on N diagonal pairs (ii)) plus B = w * I (acting on the
    N(N-1)/2 off-diagonal pairs (ij), i!=j).
        A_ii = u,  A_ij = w (i != j),
    so spec(A) = { u + (N-1) w  (mult 1),  u - w  (mult N-1) }
    and  spec(B) = { w  (mult N(N-1)/2) }.
    """
    return np.array(
        [u + (N - 1) * w] + [u - w] * (N - 1) + [w] * (N * (N - 1) // 2)
    )


def report_NgeN(sols, residual_fn, var_list, keys, N):
    """For N >= 3:
       (a) verify Sigma analytically (substitute the ansatz into residuals),
       (b) check whether each numerical solution lies in the O(N) orbit of
           Sigma via the spectrum of the flattening matrix M_{(ij)(kl)},
       (c) estimate the dimension of the solution set by PCA.
    """
    if sols.size == 0:
        print("No solutions found.")
        return

    n_poly = len(residual_fn(sols[0])) - 1  # last residual is the norm pin
    idx_iiii, idx_iijj, idx_iiij, idx_iijk, idx_ijkl = classify_keys(keys, N)

    # ---- (a) analytic Sigma check ----
    print()
    print(f"---- Analytic Sigma check (lam_iiii = u, lam_iijj = w, others = 0) ----")
    print(f"  index counts: iiii={len(idx_iiii)}, iijj={len(idx_iijj)}, "
          f"iiij={len(idx_iiij)}, iijk={len(idx_iijk)}, ijkl={len(idx_ijkl)}")
    rng = np.random.default_rng(123)
    max_r = 0.0
    n_test = 12
    for _ in range(n_test):
        u, w = rng.uniform(-1, 1, 2)
        x = np.zeros(len(var_list))
        x[idx_iiii] = u
        x[idx_iijj] = w
        r = residual_fn(x)[:n_poly]
        max_r = max(max_r, float(np.max(np.abs(r))))
    print(f"  Sigma satisfies T = dT/dt = 0:  max |poly residual| over "
          f"{n_test} random (u,w) = {max_r:.3e}")

    # ---- (b) O(N) orbit check via M-spectrum ----
    print()
    print(f"---- O({N}) orbit check via spectrum of M_(ij)(kl) = lam_ijkl ----")
    n_p = N * (N + 1) // 2  # M is n_p x n_p
    # For Sigma the spec has multiplicities {1, N-1, N(N-1)/2}.
    target_mults = sorted([1, N - 1, N * (N - 1) // 2])
    print(f"  Sigma-orbit spectrum multiplicities: {target_mults} "
          f"(in increasing eigenvalue order may differ)")

    n_match = 0
    fit_errors = []  # how well does the spectrum fit the Sigma form?
    for x in sols:
        M = flattening_matrix(x, keys, N)
        eigs = np.sort(np.linalg.eigvalsh(M))
        # Best-fit (u, w) to the analytic Sigma spectrum, allowing the 3
        # possible orderings of {u+(N-1)w, u-w, w}.  We just sweep the 3
        # multiplicity patterns and pick the smallest residual.
        best = np.inf
        for perm in [(1, N - 1, n_p - N), (N - 1, 1, n_p - N),
                     (1, n_p - N, N - 1), (n_p - N, 1, N - 1),
                     (N - 1, n_p - N, 1), (n_p - N, N - 1, 1)]:
            if sum(perm) != n_p:
                continue
            # Cluster the sorted eigs into consecutive groups of sizes perm.
            edges = np.cumsum([0] + list(perm))
            clusters = [eigs[edges[i]:edges[i + 1]] for i in range(3)]
            # Try identifying which cluster is {u+(N-1)w, mult 1},
            # {u-w, mult N-1}, {w, mult N(N-1)/2}, by mult.
            mults = perm
            try:
                i_single = mults.index(1)
                i_diag = mults.index(N - 1)
                i_off = mults.index(n_p - N)
            except ValueError:
                continue
            # If two mults coincide we skip to avoid ambiguity.
            if i_single == i_diag or i_single == i_off or i_diag == i_off:
                continue
            uw_single = np.mean(clusters[i_single])  # = u + (N-1)w
            uw_diag = np.mean(clusters[i_diag])      # = u - w
            uw_off = np.mean(clusters[i_off])        # = w
            u_est = uw_diag + uw_off
            w_est = uw_off
            # Check consistency with the (mult 1) eigenvalue
            err1 = abs(uw_single - (u_est + (N - 1) * w_est))
            # And residual within each cluster
            err2 = max(np.std(c) for c in clusters)
            total_err = max(err1, err2)
            if total_err < best:
                best = total_err
        fit_errors.append(best)
        if best < 1e-5:
            n_match += 1

    fit_errors = np.array(fit_errors)
    print(f"  solutions consistent with Sigma-orbit spectrum (err < 1e-5): "
          f"{n_match} / {len(sols)}")
    print(f"  max spectrum-fit error: {np.max(fit_errors):.3e}")
    print(f"  median spectrum-fit error: {np.median(fit_errors):.3e}")

    # ---- (c) PCA dimension of solution set ----
    print()
    print(f"---- PCA dimension of solution set ----")
    sols_c = sols - sols.mean(axis=0)
    s = np.linalg.svd(sols_c, compute_uv=False)
    s = s / s[0]
    print(f"  normalized singular values (top 12): " +
          ", ".join(f"{v:.2e}" for v in s[:12]))
    # Count "large" singular values as effective dimension
    eff_dim = int(np.sum(s > 1e-3))
    print(f"  effective dimension (sigma_i / sigma_1 > 1e-3): {eff_dim}")
    # Predicted: dim(L) - 1 (norm pin reduces by 1).
    # For Sigma alone the orbit has dim = dim(Sigma) + dim(O(N)/stab).
    # In N=3, dim(Sigma)=2, dim(O(3))=3, S_3 stab is 0-dim -> orbit dim 5,
    # unit-sphere slice 4.  We expect the locus to be at least 4-dim.
    if N == 3:
        print(f"  (N=3: O(3)-orbit of Sigma has dim 5; unit-sphere slice dim 4)")

    # ---- plot eigenvalue spectra of M for each solution ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # eigenvalue spectrum: each row is a sol, sorted eigs in columns
    specs = np.array([
        np.sort(np.linalg.eigvalsh(flattening_matrix(x, keys, N)))
        for x in sols
    ])
    for i in range(specs.shape[1]):
        axes[0].scatter([i + 1] * len(specs), specs[:, i],
                        s=8, alpha=0.4, color="C0")
    axes[0].set_xlabel("eigenvalue index (sorted)")
    axes[0].set_ylabel("eigenvalue of $M_{(ij)(kl)}$")
    axes[0].set_title(f"M-spectra of {len(sols)} numerical minima "
                      f"(should cluster with mults {target_mults})")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(np.arange(1, len(s) + 1), s, "o-")
    axes[1].axhline(1e-3, color="r", ls="--", lw=0.8,
                    label=r"$10^{-3}$ cutoff")
    axes[1].set_xlabel("singular-value index")
    axes[1].set_ylabel(r"$\sigma_i / \sigma_1$")
    axes[1].set_title(f"PCA of solution set (eff. dim = {eff_dim})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out = f"locus_N{N}.png"
    plt.savefig(out, dpi=150)
    print(f"  plot saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=2, help="number of flavours")
    parser.add_argument("--starts", type=int, default=500,
                        help="random initial conditions")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="||residual||_inf tolerance")
    parser.add_argument("--kmax", type=int, default=1,
                        help="impose T^(0), ..., T^(kmax) = 0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sols, var_list, keys = find_locus(args.N, n_starts=args.starts,
                                      tol=args.tol, k_max=args.kmax,
                                      seed=args.seed)
    print(f"        vars = {var_list}")

    if args.N == 2:
        report_N2(sols)
    else:
        # Need the residual function (with norm pin) for the analytic check
        residual_fn, _, _, _ = build_residuals(args.N, k_max=args.kmax,
                                               pin_norm=True)
        report_NgeN(sols, residual_fn, var_list, keys, args.N)


if __name__ == "__main__":
    main()
