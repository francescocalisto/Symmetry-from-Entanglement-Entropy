"""Quick analysis of the {T = T' = T'' = T''' = 0} locus at N = 3.

Reuses the build_residuals machinery in the main script (which already
contains a cached lambdify after the previous long run) and:
    (a) measures the local manifold dim via Jacobian rank at a sample;
    (b) extracts the (near-)linear constraints from the PCA kernel.

Only ~30 starts -- enough for diagnostics.  Then dumps the dominant
near-null directions in the basis of (lam_iiii, lam_iijj, lam_iiij, lam_iijk).
"""
import numpy as np
from locus_1to3_Nflav import (
    build_residuals, find_locus, classify_keys, flattening_matrix,
)

N = 3
print("Building residuals for N=3, k_max=3 (this can take a few minutes)...")
residual, jacobian, var_list, keys = build_residuals(N, k_max=3, pin_norm=False)
print(f"  done.  {len(residual(np.zeros(len(var_list))))} polynomial residuals.")

# Find a small batch of solutions to inspect.
sols, _, _ = find_locus(N, n_starts=40, k_max=3, seed=11, verbose=False)
print(f"  {len(sols)} converged solutions.")

# (a) Local manifold dim via Jacobian rank.
n_res = len(residual(sols[0]))
ranks = []
for s in sols:
    J = jacobian(s)
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > 1e-8 * sv[0]))
    ranks.append(rank)
print(f"\nJacobian (poly residuals only) shape: {n_res} x {len(var_list)}")
print(f"  Jacobian-rank distribution at solutions: "
      f"{dict(zip(*np.unique(ranks, return_counts=True)))}")
print(f"  -> local manifold dim = {len(var_list)} - rank = "
      f"{len(var_list) - int(np.median(ranks))}")

# (b) Linear constraints from PCA kernel.
sols_c = sols - sols.mean(axis=0)
U, s, Vt = np.linalg.svd(sols_c, full_matrices=False)
print(f"\nNormalized singular values (sols-set PCA): "
      + ", ".join(f"{v/s[0]:.2e}" for v in s))

print("\nNear-null PCA directions (constraints satisfied on the locus):")
idx_iiii, idx_iijj, idx_iiij, idx_iijk, _ = classify_keys(keys, N)
labels = [None] * len(keys)
for i in idx_iiii: labels[i] = "iiii"
for i in idx_iijj: labels[i] = "iijj"
for i in idx_iiij: labels[i] = "iiij"
for i in idx_iijk: labels[i] = "iijk"
key_strs = ["".join(str(k_+1) for k_ in k) for k in keys]

null_idx = np.where(s / s[0] < 1e-2)[0]
print(f"  ({len(null_idx)} directions with sigma/sigma_1 < 1e-2)")
for i in null_idx:
    v = Vt[i]  # right singular vector = direction in coupling space
    # Round small entries, show all 15
    print(f"\n  null dir {i}  (sigma/sigma_1 = {s[i]/s[0]:.2e}):")
    # Sort by |v| descending and show
    for rank, k_ in enumerate(np.argsort(-np.abs(v))):
        if abs(v[k_]) < 1e-3:
            continue
        print(f"     {v[k_]:+.3f} * lam_{key_strs[k_]} ({labels[k_]})")
