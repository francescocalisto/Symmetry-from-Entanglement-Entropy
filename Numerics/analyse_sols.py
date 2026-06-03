"""Analyse a saved npz of solutions: PCA structure, manifold dim, Sigma orbit,
and (named) linear constraints from the PCA kernel.

Usage:
    python3 analyse_sols.py n4k3_sols.npz
"""
import sys
import numpy as np

from locus_taylor import make_residual
from locus_1to3_Nflav import classify_keys, flattening_matrix


def main(path):
    d = np.load(path, allow_pickle=False)
    sols = d["sols"]
    keys = [tuple(k) for k in d["keys"].tolist()]
    N = int(d["N"])
    k_max = int(d["k_max"])
    print(f"Loaded {len(sols)} solutions from {path}")
    print(f"  N = {N},  k_max = {k_max},  n_vars = {len(keys)}")

    res_fn, jac_fn, _ = make_residual(N, k_max, pin_norm=True)

    # --- Sigma analytic check ---
    idx_iiii, idx_iijj, idx_iiij, idx_iijk, idx_ijkl = classify_keys(keys, N)
    print(f"\nCoupling-shape counts: iiii={len(idx_iiii)}, "
          f"iijj={len(idx_iijj)}, iiij={len(idx_iiij)}, "
          f"iijk={len(idx_iijk)}, ijkl={len(idx_ijkl)}")
    rng = np.random.default_rng(123)
    max_r = 0.0
    for _ in range(12):
        u, w = rng.uniform(-1, 1, 2)
        x = np.zeros(len(keys))
        x[idx_iiii] = u
        x[idx_iijj] = w
        r = res_fn(x)[:-1]
        max_r = max(max_r, float(np.max(np.abs(r))))
    print(f"\n[Sigma analytic] T^(0..{k_max}) at Sigma ansatz: "
          f"max |res| = {max_r:.3e}")

    # --- O(N) orbit check via M-spectrum multiplicity ---
    n_p = N * (N + 1) // 2
    target = sorted([1, N - 1, N * (N - 1) // 2])
    print(f"\n[O({N}) orbit check]  Sigma-orbit M-spectrum multiplicities: {target}")
    print(f"  (M is the {n_p}x{n_p} flattening matrix lam_(ij)(kl).)")
    n_match = 0
    errs = []
    for x in sols:
        M = flattening_matrix(x, keys, N)
        eigs = np.sort(np.linalg.eigvalsh(M))
        best = np.inf
        seen = set()
        for perm in [(1, N - 1, n_p - N), (N - 1, 1, n_p - N),
                     (1, n_p - N, N - 1), (n_p - N, 1, N - 1),
                     (N - 1, n_p - N, 1), (n_p - N, N - 1, 1)]:
            if sum(perm) != n_p or perm in seen: continue
            seen.add(perm)
            edges = np.cumsum([0] + list(perm))
            clusters = [eigs[edges[i]:edges[i + 1]] for i in range(3)]
            err = max(np.std(c) for c in clusters)
            best = min(best, err)
        errs.append(best)
        if best < 1e-5:
            n_match += 1
    errs = np.array(errs)
    print(f"  {n_match} / {len(sols)} solutions consistent with Sigma-orbit "
          f"spectrum (err < 1e-5)")
    print(f"  median fit error: {np.median(errs):.3e}")

    # --- PCA ---
    sols_c = sols - sols.mean(axis=0)
    U, s, Vt = np.linalg.svd(sols_c, full_matrices=False)
    s_rel = s / s[0]
    print(f"\n[PCA] normalized singular values:")
    for i, v in enumerate(s_rel):
        marker = ""
        if i > 0 and s_rel[i - 1] / max(v, 1e-30) > 50:
            marker = "  <-- gap"
        print(f"  {i + 1:2d}: {v:.3e}{marker}")
    eff_dim_3 = int(np.sum(s_rel > 1e-3))
    eff_dim_2 = int(np.sum(s_rel > 1e-2))
    print(f"  effective dim (>1e-3): {eff_dim_3}")
    print(f"  effective dim (>1e-2): {eff_dim_2}")

    # --- Manifold dim via Jacobian rank ---
    ranks = []
    sample = sols[: min(30, len(sols))]
    for x in sample:
        J = jac_fn(x)[:-1]
        sv = np.linalg.svd(J, compute_uv=False)
        rank = int(np.sum(sv > 1e-6 * sv[0]))
        ranks.append(rank)
    print(f"\n[manifold] Jacobian rank at {len(ranks)} samples: "
          f"{dict(zip(*np.unique(ranks, return_counts=True)))}")
    print(f"  -> local manifold dim = "
          f"{len(keys) - int(np.median(ranks))}")

    # --- Near-null PCA directions = linear constraints ---
    labels = [None] * len(keys)
    for i in idx_iiii: labels[i] = "iiii"
    for i in idx_iijj: labels[i] = "iijj"
    for i in idx_iiij: labels[i] = "iiij"
    for i in idx_iijk: labels[i] = "iijk"
    for i in idx_ijkl: labels[i] = "ijkl"
    key_strs = ["".join(str(k_ + 1) for k_ in k) for k in keys]

    null_idx = np.where(s_rel < 1e-2)[0]
    print(f"\n[constraints] {len(null_idx)} near-null PCA directions "
          f"(sigma/sigma_1 < 1e-2)")
    for nidx in null_idx:
        v = Vt[nidx]
        print(f"\n  null dir {nidx}  (sigma/sigma_1 = {s_rel[nidx]:.2e}):")
        ordering = np.argsort(-np.abs(v))
        for k_ in ordering:
            if abs(v[k_]) < 5e-3:
                continue
            print(f"     {v[k_]:+.3f} * lam_{key_strs[k_]} "
                  f"({labels[k_]})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "n4k3_sols.npz")
