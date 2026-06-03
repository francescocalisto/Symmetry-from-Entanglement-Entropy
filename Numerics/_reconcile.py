"""Reconcile numerics (dim=5) with the note (dim=2 'modulo rotations').

Claims tested
-------------
(A) CONSTRAINT CORRECTNESS
    A1. Taylor-mode T^(k) reproduces T(lambda(t)) from an *independent* RK
        ODE integration of dlambda/dt = beta(lambda).   (validates locus_taylor)
    A2. A random cubic/tetrahedral plane point  lam_iiii=u, lam_iijj=w (else 0),
        rotated by a random O(3), kills T^(0..3)  (residual ~ 0).
    A3. The O(3)-equivariance: residual is invariant in magnitude under rotation.

(B) DIMENSION RECONCILIATION   5 = 2 (moduli) + 3 (rotation orbit)
    B1. dim null(J) at a generic locus point = 5.
    B2. the 3 so(3) infinitesimal-rotation tangents lie in null(J), and are
        rank 3 (generic stabilizer is finite).
    B3. transverse moduli = dim null(J) - rotation rank = 2.
    B4. higher constraints k_max=4,5 do NOT shrink the locus: still dim 5.

(C) THE SOLUTIONS ARE THE NOTE'S LOCUS
    On genuine multistart solutions, verify the note's algebra:
      q_ij = lam_ijkk - (1/3)tr      ~ 0           (no spin-2 part, eq 8/16)
      H = lam - sD is harmonic (H_ijkk ~ 0) and isotropic
      C_ij = H_iklm H_jklm - (1/3)tr ~ 0           (cubic-harmonic, eq 28)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from locus_taylor import (make_keys, inflate, deflate, make_residual,
                          evolve_lambda, T_taylor)

np.set_printoptions(precision=3, suppress=True)
N = 3
keys = make_keys(N)
nv = len(keys)
rng = np.random.default_rng(7)


# ---- independent beta / rho / T on full tensors (no Taylor code) -----------
def beta_full(lam):
    ch1 = np.einsum("ijab,abkl->ijkl", lam, lam)
    ch2 = np.einsum("ikab,abjl->ijkl", lam, lam)
    ch3 = np.einsum("ilab,abjk->ijkl", lam, lam)
    return -lam + ch1 + ch2 + ch3


def T_full(lam):
    rho = np.einsum("iklm,jklm->ij", lam, lam)
    return rho - np.trace(rho) / N * np.eye(N)


# ============================ (A1) ========================================
x = rng.standard_normal(nv)
lam0 = inflate(x, keys, N)
# independent ODE integration of T(t)
def rhs(t, y):
    return beta_full(y.reshape(N, N, N, N)).ravel()
ts = np.array([0.0, 0.01, 0.02, 0.03])
sol = solve_ivp(rhs, (0, ts[-1]), lam0.ravel(), t_eval=ts,
                rtol=1e-12, atol=1e-14, method="DOP853")
T_ode = np.array([T_full(sol.y[:, i].reshape(N, N, N, N)) for i in range(len(ts))])
# Taylor reconstruction: T(t) = sum_k ([t^k]T) t^k,  [t^k]T = T^(k)/k!
K = 6
lam_t = evolve_lambda(lam0, K, N)
T_t = T_taylor(lam_t, N)          # T_t.c[k] = [t^k] T
T_rec = np.array([sum(T_t.c[k] * (tt ** k) for k in range(K + 1)) for tt in ts])
errA1 = np.max(np.abs(T_ode - T_rec))
print(f"[A1] Taylor T(t) vs independent ODE: max|diff| = {errA1:.2e}  "
      f"{'PASS' if errA1 < 1e-7 else 'FAIL'}")

# ============================ (A2,A3) =====================================
res_fn, jac_fn, _ = make_residual(N, 3, pin_norm=False)


def cubic_tetra_point(u, w):
    lam = np.zeros((N, N, N, N))
    for i in range(N):
        lam[i, i, i, i] = u
    for i in range(N):
        for j in range(N):
            if i != j:
                # all index patterns with two i's and two j's
                for p in {(i, i, j, j), (i, j, i, j), (i, j, j, i),
                          (j, i, i, j), (j, i, j, i), (j, j, i, i)}:
                    lam[p] = w
    return lam


def rand_O3():
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    return Q @ np.diag(np.sign(np.diag(R)))


u, w = rng.uniform(-1, 1, 2)
lamP = cubic_tetra_point(u, w)
xP = deflate(lamP, keys)
resP = np.max(np.abs(res_fn(xP)))
R = rand_O3()
lamPR = np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, lamP)
xPR = deflate(lamPR, keys)
resPR = np.max(np.abs(res_fn(xPR)))
print(f"[A2] cubic/tetra plane (u={u:.3f},w={w:.3f}) residual: aligned={resP:.2e}"
      f"  rotated={resPR:.2e}  {'PASS' if max(resP, resPR) < 1e-9 else 'FAIL'}")

# ============================ (B) =========================================
def so3_tangents(lam):
    eps = np.zeros((3, 3, 3))
    for a, b, c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        eps[a, b, c] = 1; eps[a, c, b] = -1
    out = []
    for a in range(3):
        G = eps[a]
        d = (np.einsum("pi,ijkl->pjkl", G, lam)
             + np.einsum("pj,ijkl->ipkl", G, lam)
             + np.einsum("pk,ijkl->ijpl", G, lam)
             + np.einsum("pl,ijkl->ijkp", G, lam))
        out.append(deflate(d, keys))
    return np.array(out)            # (3, nv)


def nullspace_dim(J, rtol=1e-7):
    sv = np.linalg.svd(J, compute_uv=False)
    return int(np.sum(sv < rtol * sv[0])) + (nv - len(sv) if len(sv) < nv else 0), sv


# use the generic rotated plane point as a generic locus point
lamX = lamPR
xX = deflate(lamX, keys)
J = jac_fn(xX)              # pin_norm=False -> only poly residuals
sv = np.linalg.svd(J, compute_uv=False)
rank = int(np.sum(sv > 1e-7 * sv[0]))
nulldim = nv - rank
print(f"[B1] dim null(J) at generic locus point = {nulldim}  "
      f"(rank {rank} of {nv})  {'PASS' if nulldim == 5 else 'FAIL'}")

rot = so3_tangents(lamX)
# how well do rotation tangents lie in null(J)?
Jn = np.linalg.norm(J, 2)
rot_in_null = max(np.linalg.norm(J @ rot[a]) / (Jn * np.linalg.norm(rot[a]))
                  for a in range(3))
rot_rank = int(np.linalg.matrix_rank(rot, tol=1e-7 * np.linalg.norm(rot)))
print(f"[B2] so(3) tangents in null(J): max rel ||J.v|| = {rot_in_null:.2e}; "
      f"rank(rotations) = {rot_rank}  "
      f"{'PASS' if rot_in_null < 1e-6 and rot_rank == 3 else 'FAIL'}")
print(f"[B3] transverse moduli = {nulldim} - {rot_rank} = {nulldim - rot_rank}"
      f"  {'PASS (= note dim 2)' if nulldim - rot_rank == 2 else 'FAIL'}")

# B4: higher k_max
for km in (4, 5):
    _, jk, _ = make_residual(N, km, pin_norm=False)
    svk = np.linalg.svd(jk(xX), compute_uv=False)
    rk = int(np.sum(svk > 1e-7 * svk[0]))
    print(f"[B4] k_max={km}: dim null(J) = {nv - rk}  "
          f"{'PASS' if nv - rk == 5 else 'FAIL'}")

# ============================ (C) =========================================
# genuine multistart solutions
sols = []
res3, jac3, _ = make_residual(N, 3, pin_norm=True)
nres = len(res3(np.zeros(nv)))
while len(sols) < 6:
    x0 = rng.standard_normal(nv); x0 /= np.linalg.norm(x0)
    o = least_squares(res3, x0, jac=jac3, method="lm",
                      xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200)
    if np.max(np.abs(o.fun[:nres - 1])) < 1e-8:
        n = np.linalg.norm(o.x)
        if 0.5 < n < 2.0:
            sols.append(o.x / n)

Dt = (np.einsum("ij,kl->ijkl", np.eye(N), np.eye(N))
      + np.einsum("ik,jl->ijkl", np.eye(N), np.eye(N))
      + np.einsum("il,jk->ijkl", np.eye(N), np.eye(N)))
DD = np.sum(Dt * Dt)
maxq = maxHtr = maxC = 0.0
for xs in sols:
    lam = inflate(xs, keys, N)
    tau = np.einsum("ijkk->ij", lam)
    q = tau - np.trace(tau) / N * np.eye(N)
    maxq = max(maxq, np.max(np.abs(q)))
    s = np.sum(lam * Dt) / DD
    H = lam - s * Dt
    maxHtr = max(maxHtr, np.max(np.abs(np.einsum("ijkk->ij", H))))
    HH = np.einsum("iklm,jklm->ij", H, H)
    C = HH - np.trace(HH) / N * np.eye(N)
    maxC = max(maxC, np.max(np.abs(C)) / max(np.max(np.abs(HH)), 1e-30))
print(f"[C ] {len(sols)} multistart sols:  max|q| = {maxq:.2e}  "
      f"max|H_ijkk| = {maxHtr:.2e}  max rel|C| = {maxC:.2e}")
print(f"     {'PASS: solutions are the cubic-harmonic (note) locus' if maxq<1e-6 and maxHtr<1e-6 and maxC<1e-5 else 'FAIL'}")
