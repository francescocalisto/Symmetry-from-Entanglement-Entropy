# Σ numerics — unified toolkit (`SigmaTools.wl`)

Tools for the RG‑invariant maximal `1|N` entropy locus

```
Σ = { λ ∈ Sym⁴(ℝᴺ) :  T = ℒ_β T = ℒ_β² T = … = 0 }
T_ij = (ρ′_{1|N})_ij − (1/N) tr(ρ′) δ_ij,   (ρ′)_ij = λ_iklm λ_jklm
β_ijkl = −λ_ijkl + (λ_ijab λ_abkl + λ_ikab λ_abjl + λ_ilab λ_abjk)
```

This package merges the strengths of the two existing numeric stacks and adds the
piece both were missing: a way to **read the analytic structure out of a pile of
decimal solutions**.

---

## Why this design

| Need | Best tool | Where it lives |
|---|---|---|
| Generate solutions fast at large N | Python Taylor‑mode + complex‑step Jacobian (`locus_taylor.py`) — ~100× faster, scales | keep as the *generator* |
| Solve a symmetry sector exactly / get closed forms | Mathematica `Solve`, `GroebnerBasis`, `RootApproximant` | `SigmaTools.wl` |
| Decide if two messy solutions are the *same* orbit | O(N)‑invariant fingerprints | `SigmaTools.wl` |
| Recover the ansatz (which components are 0) | gauge‑fix / align, then sort | `SigmaTools.wl` |

The Python solver and the Mathematica interpreter are complementary: **run
multistart in Python (or `multistartSolve` here), then interpret in
Mathematica.** Solutions move between them as JSON (`importSols`).

> Note: the search notebook `n_flavour_rg_invariant_entropy_search.nb` `Get`s a
> helper file `n_flavour_rg_invariant_entropy_search.wl` that is **not in the
> repo**. `SigmaTools.wl` is self‑contained and supersedes it
> (`makeTheory`, `lieChain`, `multistartSolve`).

---

## Your actual question, answered

> *If a system of polynomial equations is invariant under some transformation of
> the variables, is it guaranteed that every solution is also invariant under
> that transformation?*

**No — and rigorously no.** A symmetry of the equations only guarantees that the
symmetry group **permutes the solution set** (maps solutions to solutions). It
does **not** force any individual solution to be fixed by the group.

The one‑line counterexample: `x² = 1` is invariant under `x ↦ −x`, but its
solutions `±1` are not. The group swaps them. This is exactly **spontaneous
symmetry breaking**, and it is the whole point of your project: the equations are
O(N)‑invariant, yet the cubic/tetrahedral solution keeps only the *discrete*
cubic symmetry, and the hypertetrahedral N=4 solution keeps only S₅×ℤ₂. If all
solutions were forced to be fully O(N)‑symmetric there would be only the single
SO(N) singlet `D`‑tensor and no story to tell. **"Symmetry from entanglement"
is precisely this breaking.**

So you cannot shortcut the search by assuming symmetric solutions. But symmetry
is still your most powerful tool, in two rigorous ways:

### (a) The constructive road — equivariant ansätze (this *would* have found the loci)

For a candidate residual symmetry subgroup `H ⊆ O(N)`, restrict the couplings to
the **fixed subspace** `Fix(H)` and solve the (much smaller) system there. This
is rigorous: because the equations are equivariant, `Fix(H)` is an invariant
subspace and the equations map it to itself, so any solution you find in
`Fix(H)` is a genuine solution of the full system (this is the "principle of
symmetric criticality" / equivariant branching flavour). It is exactly the
"S₃ sector", "S₄ sector", "S₅×ℤ₂ invariant", "even/odd block" ansätze in your
notes.

`SigmaTools` implements this: `solvePermSector[N]`, `solveFixedPoints[N]`.
At N=3 it reproduces **all six** fixed points in closed form — Gaussian,
decoupled `(1/3,0,0,0)`, cubic `(2/9,0,1/9,0)`, Heisenberg `(3/11,0,1/11,0)`,
and the two "rogue" points P₂ `(70,−2,19,4)/243`, P₃ `(11,2,8,−4)/81` — without
ever solving the full 15‑variable system. Run `solvePermSector[4]` /
`solveFixedPoints[4]` to get the hypertetrahedral point in closed form the same
way.

You enumerate candidate `H` (the permutation group S_N and its subgroups, the
hypertetrahedral S_{N+1}, the SO(N) singlet, …); each gives a tiny exact solve.

### (b) The classification road — invariant fingerprints

To decide whether two numerical solutions are the same orbit (the thing you
*can't* see by staring at decimals), compute **O(N)‑invariants**: these are equal
across an orbit by construction. The discriminating one is the eigenvalue
spectrum (with multiplicities) of the **Sym²(ℝᴺ) flattening**
`Q_(ij),(kl) = λ_ijkl` — an O(N) invariant, so rotation cannot change it. Two
solutions with different spectra are *provably* different orbits; equal spectra
plus an explicit aligning rotation confirms they are the same.

> Implementation detail worth knowing: the existing Python `flattening_matrix`
> uses the **raw** `λ_(ij)(kl)`, whose eigenvalues are **not** strictly
> O(N)‑invariant. `SigmaTools.flatteningMatrix` uses the correct **√2‑weighted**
> (orthonormal) flattening, whose spectrum *is* invariant (verified by rotating a
> random tensor — the spectrum is unchanged to 1e‑8). Use this one for orbit
> tests.

### Useful partial converses
- **Uniqueness ⇒ symmetry.** If the system has a *unique* solution in a
  G‑invariant region, that solution must be G‑symmetric (G maps it to a solution,
  which must be itself). Isolated invariant pieces tend to be symmetric or come
  in finite orbits.
- **Equivariance block‑diagonalises the linearisation.** At a G‑symmetric
  solution the Jacobian commutes with G, so it splits by irrep — this is the
  `Sym⁴ = V₀ ⊕ V₂ ⊕ V₄` decomposition doing its work, and it is how you organise
  a sector‑by‑sector search.

**Bottom line.** Don't assume solutions are symmetric. *Generate* them by
equivariant ansätze in candidate subgroups (exact, closed form), and *classify*
the numerical ones by invariant fingerprints. That is exactly the workflow
`SigmaTools` automates.

---

## The interpretation pipeline (turning decimals into structure)

`interpretSolutions[sols, keys, N]` runs, for a list of solution vectors:

1. **Fingerprint + cluster** (`clusterOrbits`) — collapses N lists of decimals
   into a handful of distinct O(N) orbits by invariant signature
   (flattening spectrum & multiplicities, singlet coeff `s`, spin‑2 defect `|q|`,
   normalized cubic invariant).
2. **Align** (`alignTensor`) — gauge‑fixes the O(N) rotation by concentrating the
   quartic on the diagonal (maximising `Σ_i λ_iiii²`), then sorts/sign‑fixes
   axes. Same orbit → same aligned vector → the **zero components become
   visible**, revealing the ansatz (e.g. only `λ_iiii` and `λ_iijj` survive ⇒
   the cubic/tetrahedral plane).
3. **Detect residual symmetry** (`residualSymmetry`) — the discrete
   signed‑permutation stabiliser (order identifies cubic/tetra/S_N/…) and the
   continuous so(N) stabiliser dimension (>0 ⇒ enhanced SO symmetry, e.g. the
   `u=3w` Heisenberg line).
4. **Rationalise** (`rationalizeVec`) — `RootApproximant` on the aligned
   components → closed form, **certified** by exact resubstitution into the
   T‑chain. This is the step that yields forms like `(41−3√5)/168`.

---

## Quick start

```wolfram
SetDirectory["/path/to/Numerics/Mathematica"];
Get["SigmaTools.wl"]; Needs["SigmaTools`"];

(* exact symmetry-sector solving — the constructive road *)
solveFixedPoints[3]["solutions"]      (* all six N=3 fixed points, closed form *)
solvePermSector[4, 1]["solutions"]    (* N=4 S4-sector solutions, closed form  *)

(* interpret an existing solution set *)
{sols, keys, n} = importSols["../N4_study/n4_clean.json"];
interpretSolutions[sols, keys, n]     (* cluster -> align -> symmetry -> closed form *)

(* generate from scratch (self-contained; Python Taylor solver is faster at large N) *)
sols = multistartSolve[3, 3, 200];
```

To regenerate the JSON solution files from the saved `.npz`:

```bash
/Users/<you>/miniconda3/bin/python3 N4_study/_npz_to_json.py
```

## Validated against the analytic notes
- `solveFixedPoints[3]` → Gaussian, decoupled `(1/3,0,0,0)`, cubic `(2/9,0,1/9,0)`,
  Heisenberg `(3/11,0,1/11,0)`, and the rotated rogues P₂ `(70,−2,19,4)/243`,
  P₃ `(11,2,8,−4)/81` — matches `rg_22_intersection_note.pdf` §2.
- `solveFixedPoints[4]` → Gaussian, decoupled, SO(4) singlet `(a,c)=(1/4,1/12)`,
  tetra‑chiral, and the **hypertetrahedral** point
  `a=(41−3√5)/168, b=−1/84, c=(13+√5)/168, d=1/84, e=−(1+√5)/168` (+ Galois twin)
  — matches `fourflavour.pdf` eq.(6) exactly.
- `interpretSolutions` on `n3_clean.json` → one class, cone dim **5 = 2 + 3**,
  ansatz `λ_iiii=u, λ_iijj=w`; on `n4_clean.json` → a q=0 dim‑14 (moduli 8)
  family, a q≠0 dim‑9 stratum, and a degenerate higher‑symmetry stratum
  (flattening mult `{1,1,4,4}`) — reproducing `metrics.json` and the
  four‑flavour break.

**Key practical finding.** At N ≥ 4 the "q=0 + isotropic" condition no longer pins
down the symmetric plane — it is a high‑dimensional family, and random multistart
lands on *generic, low‑symmetry* members (discrete stabiliser of order 8, generic
flattening spectrum). The special symmetric solutions (the `(u,w)` plane, the
hypertetrahedral point) are measure‑zero and are best obtained by the
**constructive road (a)**, not by sampling and squinting. The interpreter's job is
then to *confirm* and *classify*, which it does.

## Files
- `SigmaTools.wl` — the package (theory + exact sector solving + interpretation).
- `sigma_pipeline.wls` — end‑to‑end demo (`wolframscript -file sigma_pipeline.wls`).
- `_test_sigmatools.wls` — correctness tests (β, rotation, flattening invariance, exact N=3).
- `../N4_study/n{2,3,4}_clean.json` — the 196/89/94 solution sets, importable.
- `../N4_study/_npz_to_json.py` — regenerate the JSON from the saved `.npz`.
