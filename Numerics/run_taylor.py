"""Fast multistart using the Taylor-mode residual evaluator.

Replicates locus_1to3_Nflav.find_locus + report_NgeN but ~100x faster
for high k_max.
"""
import argparse
import time
import numpy as np
from scipy.optimize import least_squares

from locus_taylor import make_residual, make_keys, inflate
from locus_1to3_Nflav import classify_keys, flattening_matrix


def multistart(N, k_max, n_starts=500, tol=1e-6, seed=0,
               save_path=None, max_nfev=200):
    res_fn, jac_fn, keys = make_residual(N, k_max, pin_norm=True)
    n_vars = len(keys)
    n_res = len(res_fn(np.zeros(n_vars)))
    method = "lm" if n_res >= n_vars else "trf"
    print(f"[build] N={N} k_max={k_max}  n_vars={n_vars}  "
          f"n_poly_res={n_res - 1}  method={method}", flush=True)

    rng = np.random.default_rng(seed)
    sols = []
    t0 = time.perf_counter()
    n_converged = 0
    for j in range(n_starts):
        x0 = rng.standard_normal(n_vars)
        x0 /= np.linalg.norm(x0)
        try:
            out = least_squares(
                res_fn, x0, jac=jac_fn, method=method,
                xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=max_nfev,
            )
        except Exception:
            continue
        poly_res = out.fun[:n_res - 1]
        if np.max(np.abs(poly_res)) < tol:
            n_converged += 1
            norm = np.linalg.norm(out.x)
            if 0.5 < norm < 2.0:
                sols.append(out.x / norm)
        if (j + 1) % 25 == 0:
            dt = time.perf_counter() - t0
            print(f"  ... {j+1}/{n_starts} starts, "
                  f"{n_converged} converged, {dt:.0f}s elapsed", flush=True)
            # Periodic checkpoint so a kill / timeout doesn't lose progress.
            if save_path is not None and len(sols) > 0:
                np.savez(save_path, sols=np.asarray(sols),
                         keys=np.array(keys), N=N, k_max=k_max)
    dt = time.perf_counter() - t0
    sols = np.asarray(sols)
    print(f"[solve] {n_starts} starts in {dt:.1f}s "
          f"({dt / n_starts * 1e3:.1f} ms/start)")
    print(f"        {n_converged} converged, {len(sols)} non-trivial.")
    if save_path is not None and len(sols) > 0:
        np.savez(save_path, sols=sols, keys=np.array(keys),
                 N=N, k_max=k_max)
        print(f"        saved solutions -> {save_path}")
    return sols, keys, res_fn, jac_fn


def report(sols, keys, res_fn, jac_fn, N, k_max):
    if sols.size == 0:
        print("No solutions to report.")
        return

    # Sigma analytic check
    idx_iiii, idx_iijj, idx_iiij, idx_iijk, _ = classify_keys(keys, N)
    rng = np.random.default_rng(123)
    n_test = 12
    max_r = 0.0
    for _ in range(n_test):
        u, w = rng.uniform(-1, 1, 2)
        x = np.zeros(len(keys))
        x[idx_iiii] = u
        x[idx_iijj] = w
        r = res_fn(x)[:-1]
        max_r = max(max_r, float(np.max(np.abs(r))))
    print(f"\n[Sigma analytic] T^(0..{k_max}) at Sigma ansatz: "
          f"max |res| = {max_r:.3e}  over {n_test} random (u, w)")

    # O(N) orbit check (only meaningful for N >= 3)
    if N >= 3:
        n_p = N * (N + 1) // 2
        target = sorted([1, N - 1, N * (N - 1) // 2])
        print(f"[O({N}) orbit] Sigma-orbit mults = {target}")
        n_match = 0
        errs = []
        for x in sols:
            M = flattening_matrix(x, keys, N)
            eigs = np.sort(np.linalg.eigvalsh(M))
            best = np.inf
            for perm in [(1, N - 1, n_p - N), (N - 1, 1, n_p - N),
                         (1, n_p - N, N - 1), (n_p - N, 1, N - 1),
                         (N - 1, n_p - N, 1), (n_p - N, N - 1, 1)]:
                if sum(perm) != n_p: continue
                if N - 1 == n_p - N: continue
                edges = np.cumsum([0] + list(perm))
                clusters = [eigs[edges[i]:edges[i + 1]] for i in range(3)]
                err = max(np.std(c) for c in clusters)
                best = min(best, err)
            errs.append(best)
            if best < 1e-5:
                n_match += 1
        errs = np.array(errs)
        print(f"           {n_match} / {len(sols)} solutions consistent "
              f"with Sigma-orbit spectrum (err < 1e-5)")
        print(f"           median fit error: {np.median(errs):.3e}")

    # PCA dimension
    sols_c = sols - sols.mean(axis=0)
    U, s, Vt = np.linalg.svd(sols_c, full_matrices=False)
    s_rel = s / s[0]
    print(f"\n[PCA] sigma_i / sigma_1:")
    for i, v in enumerate(s_rel):
        marker = "  <-- gap" if i > 0 and s_rel[i - 1] / max(v, 1e-30) > 50 else ""
        print(f"  {i + 1:2d}: {v:.3e}{marker}")
    eff_dim = int(np.sum(s_rel > 1e-3))
    print(f"  effective dim (>1e-3): {eff_dim}")
    print(f"  effective dim (>1e-2): {int(np.sum(s_rel > 1e-2))}")

    # Manifold dim via Jacobian rank (using analytic FD jacobian for speed)
    ranks = []
    for x in sols[: min(40, len(sols))]:
        J = jac_fn(x)[:-1]  # poly residuals only (drop norm row)
        sv = np.linalg.svd(J, compute_uv=False)
        rank = int(np.sum(sv > 1e-6 * sv[0]))
        ranks.append(rank)
    print(f"[manifold] Jacobian rank distribution at {len(ranks)} samples: "
          f"{dict(zip(*np.unique(ranks, return_counts=True)))}")
    print(f"           -> local manifold dim = {len(keys) - int(np.median(ranks))}")

    # Near-null PCA directions = (near-)linear constraints on the locus
    null_idx = np.where(s_rel < 1e-2)[0]
    if len(null_idx) > 0:
        print(f"\n[constraints] {len(null_idx)} near-null PCA directions "
              f"(sigma/sigma_1 < 1e-2)")
        labels = [None] * len(keys)
        for i in idx_iiii: labels[i] = "iiii"
        for i in idx_iijj: labels[i] = "iijj"
        for i in idx_iiij: labels[i] = "iiij"
        for i in idx_iijk: labels[i] = "iijk"
        # For N=4, there is also "ijkl" (all distinct)
        for i_, lbl in enumerate(labels):
            if lbl is None:
                labels[i_] = "ijkl"
        key_strs = ["".join(str(k_ + 1) for k_ in k) for k in keys]
        for nidx in null_idx:
            v = Vt[nidx]
            print(f"\n  null dir {nidx}  (sigma/sigma_1 = {s_rel[nidx]:.2e}):")
            ordering = np.argsort(-np.abs(v))
            for k_ in ordering:
                if abs(v[k_]) < 5e-3:
                    continue
                print(f"     {v[k_]:+.3f} * lam_{key_strs[k_]} "
                      f"({labels[k_]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-N", type=int, default=3)
    p.add_argument("--kmax", type=int, default=1)
    p.add_argument("--starts", type=int, default=300)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None,
                   help="path to save solutions (npz)")
    p.add_argument("--max-nfev", type=int, default=200)
    args = p.parse_args()
    sols, keys, res_fn, jac_fn = multistart(
        args.N, args.kmax, n_starts=args.starts, tol=args.tol, seed=args.seed,
        save_path=args.save, max_nfev=args.max_nfev,
    )
    report(sols, keys, res_fn, jac_fn, args.N, args.kmax)


if __name__ == "__main__":
    main()
