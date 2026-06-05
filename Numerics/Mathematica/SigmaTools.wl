(* ::Package:: *)

(* ============================================================
   SigmaTools.wl
   ------------------------------------------------------------
   Unified toolkit for the RG-invariant maximal 1|N entropy locus

       Sigma = { lambda in Sym^4(R^N) : T = L_b T = L_b^2 T = ... = 0 }

   where
     lambda_{ijkl}            fully symmetric quartic coupling
     (rho'_{1|N})_{ij} = lam_{iklm} lam_{jklm}
     T_{ij} = rho_{ij} - (1/N) Tr[rho] delta_{ij}    (spin-2 part)
     beta_{ijkl} = -lam_{ijkl}
                   + sum_{ab}( lam_{ijab} lam_{abkl}
                             + lam_{ikab} lam_{abjl}
                             + lam_{ilab} lam_{abjk} )
   (one-loop, eps = kappa = 1).

   This package does THREE things:

     (1) THEORY   makeTheory[N] rebuilds T, beta, the Lie chain, and the
                  Sigma ansatz -- self-contained, so it also replaces the
                  helper library the search notebook expects but that is
                  missing from the repo.

     (2) SOLVE    - exact solving inside a symmetry sector (the rigorous
                    "symmetry road": restrict to the fixed subspace of a
                    candidate residual symmetry group H, then Solve the
                    small exact system).  Built in: the linear Sigma plane,
                    and the full S_N permutation sector (which produces the
                    cubic / Heisenberg / hypertetrahedral fixed points in
                    closed form).
                  - numeric multistart (NMinimize) for self-containedness.

     (3) INTERPRET  the heart of the toolkit: turn a pile of decimal
                    solution vectors into structure --
                      * O(N)-invariant fingerprint (rotation-proof)
                      * cluster solutions into distinct O(N) orbits
                      * gauge-fix (align) a representative to reveal which
                        components are zero  ->  read off the ansatz
                      * detect the residual symmetry group (cubic, tetra,
                        S_N, hypertetrahedral, enhanced continuous SO)
                      * RootApproximant the aligned components into closed
                        form, certified by exact resubstitution.

   Conventions match build_Tk_general.wl and locus_taylor.py.
   ============================================================ *)

BeginPackage["SigmaTools`"];

(* ---- theory ---- *)
sigKeys::usage          = "sigKeys[N] gives the sorted 4-tuples (1<=i<=j<=k<=l<=N) indexing the independent couplings.";
inflate::usage          = "inflate[vec, keys, N] builds the full NxNxNxN symmetric tensor from an independent-component vector.";
deflate::usage          = "deflate[lam, keys] reads the independent components from a symmetric tensor.";
rhoMatrix::usage        = "rhoMatrix[lam, N] = (rho'_{1|N})_{ij} = lam_{iklm} lam_{jklm}.";
tMatrix::usage          = "tMatrix[lam, N] = spin-2 traceless part of rho.";
betaTensor::usage       = "betaTensor[lam, N] = one-loop beta function tensor (eps=kappa=1).";
rotateTensor::usage     = "rotateTensor[lam, R] applies the O(N) rotation R on all four indices.";
makeTheory::usage       = "makeTheory[N] returns an Association with symbolic vars, keys, T, beta rule, Lie operator, and the Sigma ansatz substitution.";
lieChain::usage         = "lieChain[th, kMax] returns the list {T^(0),...,T^(kMax)} of symbolic matrices (Lie derivatives of T along beta).";
sigmaResidualReal::usage= "sigmaResidualReal[vec, keys, N, kMax] gives the numeric max |T^(0..kMax)| at a real component vector (Taylor-mode, no symbolic blowup).";

(* ---- solve ---- *)
permAnsatz::usage       = "permAnsatz[N] returns {amps, rule} for the full S_N sector: one amplitude per multiset shape (iiii,iiij,iijj,iijk,ijkl) and the substitution onto the couplings.";
solvePermSector::usage  = "solvePermSector[N, kMax] solves T=...=L^kMax T=0 exactly inside the S_N sector and returns the real solutions in shape-amplitude coordinates.";
solveFixedPoints::usage = "solveFixedPoints[N] solves beta=0 inside the S_N sector and intersects with T=0: the RG fixed points on M_{1|N} (Gaussian, decoupled, cubic, Heisenberg, hypertetrahedral, ...).";
multistartSolve::usage  = "multistartSolve[N, kMax, nStarts] runs an NMinimize/FindMinimum multistart on the unit sphere and returns accepted real solution vectors.";

(* ---- interpret ---- *)
flatteningMatrix::usage = "flatteningMatrix[lam, N] gives the ORTHONORMAL Sym^2(R^N) flattening (with sqrt(2) off-diagonal weights) whose eigenvalue spectrum is a true O(N) invariant.";
invariantFingerprint::usage = "invariantFingerprint[vec, keys, N] returns an Association of O(N)-invariant numbers: norm, singlet coeff s, spin-2 defect |q|, harmonic-trace defect, flattening spectrum (rounded) with multiplicities, and a normalized cubic invariant.";
clusterOrbits::usage    = "clusterOrbits[sols, keys, N, tol] groups the solution vectors into O(N) orbits by their invariant fingerprints; returns clusters with a representative each.";
alignTensor::usage      = "alignTensor[vec, keys, N] gauge-fixes the O(N) rotation by concentrating weight on the diagonal (maximizing sum_i lam_iiii^2), then sorts/sign-fixes axes; returns the aligned component vector (zeros become visible).";
residualSymmetry::usage = "residualSymmetry[vec, keys, N] reports the discrete signed-permutation stabilizer (order + generators) and the continuous (so(N)) stabilizer dimension of the aligned tensor.";
rationalizeVec::usage   = "rationalizeVec[vec, keys, N, deg] applies RootApproximant (max degree deg) to each component and certifies by exact resubstitution into the T-chain.";
interpretSolutions::usage = "interpretSolutions[sols, keys, N] runs the full pipeline (cluster -> align -> symmetry -> rationalize) and prints a structured report; returns the per-orbit data.";

(* ---- io ---- *)
importSols::usage       = "importSols[jsonfile] loads {N, keys, sols} from a JSON produced by npz_to_json; returns {sols, keys, N}.";

Begin["`Private`"];

(* ============================================================
   (1) THEORY
   ============================================================ *)

sigKeys[n_Integer] := DeleteDuplicates[Sort /@ Tuples[Range[n], 4]];

inflate[vec_List, keys_List, n_Integer] := Module[{lam = ConstantArray[0, {n, n, n, n}]},
  Do[ With[{k = keys[[a]], v = vec[[a]]},
        Do[lam[[Sequence @@ p]] = v, {p, Permutations[k]}]],
      {a, Length[keys]}];
  lam];

deflate[lam_, keys_List] := (lam[[Sequence @@ #]] &) /@ keys;

(* rho_{ij} = lam_{iklm} lam_{jklm}, T = rho - Tr/N *)
rhoMatrix[lam_, n_Integer] :=
  TensorContract[TensorProduct[lam, lam], {{2, 6}, {3, 7}, {4, 8}}];
tMatrix[lam_, n_Integer] := With[{rho = rhoMatrix[lam, n]},
  rho - (Tr[rho]/n) IdentityMatrix[n]];

(* beta channels, all four indices summed over a,b *)
betaTensor[lam_, n_Integer] := Module[{ch1, ch2, ch3},
  (* ch1_{ijkl} = lam_{ijab} lam_{abkl} *)
  ch1 = TensorContract[TensorProduct[lam, lam], {{3, 5}, {4, 6}}];
  (* ch2_{ijkl} = lam_{ikab} lam_{abjl}: contract idx 2 of first(=k) is kept...
     build directly by index gymnastics *)
  ch2 = Transpose[TensorContract[TensorProduct[lam, lam], {{3, 5}, {4, 6}}], {1, 3, 2, 4}];
  ch3 = Transpose[TensorContract[TensorProduct[lam, lam], {{3, 5}, {4, 6}}], {1, 4, 3, 2}];
  -lam + ch1 + ch2 + ch3];

(* rotate all four legs by R in O(N).  Index-by-index Dot: r.t contracts r's
   second index with t's first; the cyclic Transpose brings the next leg to the
   front, so after four passes every leg has been multiplied once and the index
   order is restored.  Avoids the rank-12 TensorProduct (16M entries at N=4). *)
rotateTensor[lam_, r_] := Module[{t = lam}, Do[t = Transpose[r . t, {4, 1, 2, 3}], {4}]; t];

(* ---- symbolic theory + Lie chain (for exact work) ---- *)
makeTheory[n_Integer] := Module[{keys, lamSym, lamF, vars, rho, tmat, betaF, betaRule, lieD, sigRule},
  keys = sigKeys[n];
  lamSym[k_] := lamSym[k] = Symbol["lam" <> StringJoin[ToString /@ k]];
  vars = lamSym /@ keys;
  lamF[i_, j_, k_, l_] := lamSym[Sort[{i, j, k, l}]];
  rho = Expand@Table[Sum[lamF[i, k, l, m] lamF[j, k, l, m], {k, n}, {l, n}, {m, n}], {i, n}, {j, n}];
  tmat = Expand[rho - (1/n) Tr[rho] IdentityMatrix[n]];
  With[{ct = Function[{p, q, r, t}, Sum[lamF[p, q, a, b] lamF[a, b, r, t], {a, n}, {b, n}]]},
    betaF[{i_, j_, k_, l_}] := -lamF[i, j, k, l] + ct[i, j, k, l] + ct[i, k, j, l] + ct[i, l, j, k]];
  betaRule = Thread[vars -> Expand[betaF /@ keys]];
  lieD[f_] := Expand[Sum[D[f, vars[[m]]] (vars[[m]] /. betaRule), {m, Length[vars]}]];
  (* Sigma ansatz: lam_iiii->u, lam_iijj->w, else 0 *)
  sigRule = Table[lamSym[k] -> With[{cnt = Sort[Table[Count[k, i], {i, n}], Greater]},
       Which[Take[cnt, 2] === {4, 0}, Global`u, Take[cnt, 2] === {2, 2}, Global`w, True, 0]], {k, keys}];
  <|"n" -> n, "keys" -> keys, "vars" -> vars, "lamSym" -> lamSym, "lamF" -> lamF,
    "rho" -> rho, "T" -> tmat, "betaRule" -> betaRule, "lieD" -> lieD, "sigmaRule" -> sigRule|>];

lieChain[th_Association, kMax_Integer] := Module[{lie},
  lie[m_] := Map[th["lieD"], m, {2}];
  NestList[lie, th["T"], kMax]];

(* ---- fast numeric Taylor-mode residual (mirrors locus_taylor.py) ---- *)
(* lambda(t) as truncated series; T^(k) = k! [t^k] T(lambda(t)).  No symbolic blowup. *)
sigmaResidualReal[vec_List, keys_List, n_Integer, kMax_Integer] := Module[
  {lam0, series, betaSeries, tSeries, conv, c, k, maxres},
  lam0 = inflate[vec, keys, n];
  (* series[[m]] = coefficient of t^(m-1), each an NxNxNxN tensor *)
  series = ConstantArray[ConstantArray[0, {n, n, n, n}], kMax + 1];
  series[[1]] = lam0;
  (* truncated convolution helper for two series of tensors via given contraction *)
  conv[contract_, deg_] := Sum[contract[series[[i + 1]], series[[deg - i + 1]]], {i, 0, deg}];
  Do[ (* fill series[[k+2]] = ([t^k] beta)/(k+1) *)
    With[{bk = -series[[k + 1]]
        + Sum[
            TensorContract[TensorProduct[series[[i + 1]], series[[k - i + 1]]], {{3, 5}, {4, 6}}]
          + Transpose[TensorContract[TensorProduct[series[[i + 1]], series[[k - i + 1]]], {{3, 5}, {4, 6}}], {1, 3, 2, 4}]
          + Transpose[TensorContract[TensorProduct[series[[i + 1]], series[[k - i + 1]]], {{3, 5}, {4, 6}}], {1, 4, 3, 2}],
          {i, 0, k}]},
      series[[k + 2]] = bk/(k + 1)],
    {k, 0, kMax - 1}];
  (* T(lambda(t)) coefficients: rho convolution then traceless *)
  maxres = 0.;
  Do[
    With[{rhoK = Sum[TensorContract[TensorProduct[series[[i + 1]], series[[k - i + 1]]], {{2, 6}, {3, 7}, {4, 8}}], {i, 0, k}]},
      With[{tK = (rhoK - (Tr[rhoK]/n) IdentityMatrix[n]) (k!)},
        maxres = Max[maxres, Max[Abs[Flatten[tK]]]]]],
    {k, 0, kMax}];
  maxres];


(* ============================================================
   (2) SOLVE  -- the symmetry road, done rigorously
   ============================================================ *)

(* full S_N sector: one amplitude per multiset shape of the index 4-tuple *)
shapeOf[k_, n_] := Sort[Select[Tally[k][[All, 2]], # > 0 &], Greater];
permAnsatz[n_Integer] := Module[{keys, shapes, ampOf, amps, rule},
  keys = sigKeys[n];
  shapes = DeleteDuplicates[shapeOf[#, n] & /@ keys];
  ampOf = AssociationThread[shapes -> (Symbol["a" <> StringJoin[ToString /@ #]] & /@ shapes)];
  amps = Values[ampOf];
  rule = Table[Symbol["lam" <> StringJoin[ToString /@ k]] -> ampOf[shapeOf[k, n]], {k, keys}];
  <|"amps" -> amps, "rule" -> rule, "shapes" -> shapes, "ampOf" -> ampOf|>];

solvePermSector[n_Integer, kMax_Integer:1] := Module[{th, chain, ans, eqs, sol},
  th = makeTheory[n];
  chain = lieChain[th, kMax];
  ans = permAnsatz[n];
  (* substitute the S_N ansatz; by S_N symmetry only a few distinct entries remain *)
  eqs = DeleteCases[DeleteDuplicates[Flatten[(chain /. ans["rule"])]], 0];
  sol = FullSimplify@Solve[Thread[eqs == 0], ans["amps"], Reals];
  <|"amps" -> ans["amps"], "shapes" -> ans["shapes"], "equations" -> eqs, "solutions" -> sol|>];

solveFixedPoints[n_Integer] := Module[{th, ans, betaShape, tShape, eqs, sol},
  th = makeTheory[n];
  ans = permAnsatz[n];
  (* reduced beta on the S_N sector: take beta of one representative coupling per shape *)
  betaShape = DeleteDuplicates[Table[
      Expand[(th["lamSym"][k] /. th["betaRule"]) /. ans["rule"]], {k, th["keys"]}]];
  (* T = 0 on the sector *)
  tShape = DeleteCases[DeleteDuplicates[Flatten[th["T"] /. ans["rule"]]], 0];
  eqs = DeleteCases[DeleteDuplicates[Join[betaShape, tShape]], 0];
  sol = FullSimplify@Solve[Thread[eqs == 0], ans["amps"], Reals];
  <|"amps" -> ans["amps"], "shapes" -> ans["shapes"], "solutions" -> sol|>];

(* numeric multistart, self-contained (Python Taylor solver is faster for sampling) *)
multistartSolve[n_Integer, kMax_Integer:3, nStarts_Integer:200, OptionsPattern[{"Tol" -> 10.^-6, "Seed" -> 1}]] :=
 Module[{keys, nv, obj, sols, x, x0, r, res, tol},
  SeedRandom[OptionValue["Seed"]];
  keys = sigKeys[n]; nv = Length[keys]; tol = OptionValue["Tol"];
  sols = {};
  Do[
    x0 = RandomVariate[NormalDistribution[], nv]; x0 = x0/Norm[x0];
    obj[xv_?(VectorQ[#, NumericQ] &)] := sigmaResidualReal[xv, keys, n, kMax]^2 + (Norm[xv]^2 - 1)^2;
    res = Quiet@FindMinimum[obj[Array[x, nv]], Transpose[{Array[x, nv], x0}],
        Method -> "QuasiNewton", MaxIterations -> 400, AccuracyGoal -> 14, PrecisionGoal -> 12];
    With[{xv = Array[x, nv] /. res[[2]]},
      r = sigmaResidualReal[xv, keys, n, kMax];
      If[r < tol && 0.5 < Norm[xv] < 2.0, AppendTo[sols, xv/Norm[xv]]]],
    {nStarts}];
  sols];


(* ============================================================
   (3) INTERPRET
   ============================================================ *)

(* orthonormal Sym^2 flattening: basis (i,i) weight 1, (i<j) weight sqrt2.
   Its spectrum is a genuine O(N) invariant (unlike the raw lam_(ij)(kl)). *)
flatteningMatrix[lam_, n_Integer] := Module[{pairs, wt, q},
  pairs = Flatten[Table[{i, j}, {i, n}, {j, i, n}], 1];
  wt[{i_, j_}] := If[i == j, 1, Sqrt[2]];
  q = Table[wt[p] wt[r] lam[[p[[1]], p[[2]], r[[1]], r[[2]]]], {p, pairs}, {r, pairs}];
  q];

(* singlet tensor D and its norm^2 *)
dTensor[n_] := Module[{id = IdentityMatrix[n]},
  TensorProduct[id, id]~addPerm~1];
addPerm[t_, _] := t + Transpose[t, {1, 3, 2, 4}] + Transpose[t, {1, 3, 4, 2}];
(* D_{ijkl} = d_ij d_kl + d_ik d_jl + d_il d_jk *)
dTen[n_] := Module[{id = IdentityMatrix[n]},
  TensorProduct[id, id] + Transpose[TensorProduct[id, id], {1, 3, 2, 4}] + Transpose[TensorProduct[id, id], {1, 3, 4, 2}]];

invariantFingerprint[vec_List, keys_List, n_Integer] := Module[
  {lam, nrm, dt, ss, tau, q, qn, hh, htr, spec, specR, mult, q2, q3, cub},
  lam = inflate[vec, keys, n];
  nrm = Sqrt[Total[Flatten[lam]^2]];
  dt = dTen[n];
  ss = Total[Flatten[lam dt]]/Total[Flatten[dt^2]];           (* singlet coefficient s *)
  tau = Table[Sum[lam[[i, j, k, k]], {k, n}], {i, n}, {j, n}]; (* spin-2 + spin-0 trace *)
  q = tau - (Tr[tau]/n) IdentityMatrix[n];
  qn = Sqrt[Total[Flatten[q]^2]];                              (* spin-2 defect *)
  hh = lam - ss dt;                                            (* approx harmonic part *)
  htr = Max[Abs[Flatten[Table[Sum[hh[[i, j, k, k]], {k, n}], {i, n}, {j, n}]]]];
  spec = Sort[Eigenvalues[N[flatteningMatrix[lam, n]]]];
  specR = Round[spec, 10.^-4];
  mult = Sort[Tally[specR][[All, 2]]];
  (* normalized cubic invariant of the flattening (trace of Q^3 / (trace Q^2)^{3/2}) *)
  q2 = Total[spec^2]; q3 = Total[spec^3];
  cub = If[q2 > 10.^-12, q3/q2^(3/2), 0.];
  (* isotropy defect of the harmonic part H = lam - sD :  C = H_iklm H_jklm - (1/N) tr *)
  Module[{cmat, isoDef},
   cmat = TensorContract[TensorProduct[hh, hh], {{2, 6}, {3, 7}, {4, 8}}];
   cmat = cmat - (Tr[cmat]/n) IdentityMatrix[n];
   isoDef = Sqrt[Total[Flatten[cmat]^2]]/Max[Sqrt[Total[Flatten[hh]^2]]^2, 10.^-12];
   <|"norm" -> nrm, "s" -> ss, "qdefect" -> qn, "harmTrace" -> htr, "isoDefect" -> isoDef,
     "spec" -> spec, "specRounded" -> specR, "mult" -> mult, "cubicInv" -> Round[cub, 10.^-4]|>]];

(* COARSE, qualitative signature for clustering by symmetry TYPE (not by the
   continuous moduli of a family): flattening multiplicity pattern, whether the
   spin-2 part vanishes, and whether the harmonic part is isotropic.  This puts a
   whole continuous family (e.g. the N=3 cubic/tetrahedral plane) in ONE cluster,
   while separating genuinely distinct strata (e.g. N=4 hypertetrahedral). *)
fpSignature[fp_Association, tol_] := {
   fp["mult"],
   fp["qdefect"] > 100 tol,
   fp["isoDefect"] > 100 tol};

clusterOrbits[sols_List, keys_List, n_Integer, tol_:10.^-3] := Module[{fps, sigs, groups},
  fps = invariantFingerprint[#, keys, n] & /@ sols;
  sigs = fpSignature[#, tol] & /@ fps;
  groups = GatherBy[Transpose[{sols, fps, Range[Length[sols]]}], fpSignature[#[[2]], tol] &];
  SortBy[<|"size" -> Length[#], "members" -> #[[All, 3]], "rep" -> #[[1, 1]],
          "fingerprint" -> #[[1, 2]]|> & /@ groups, -#["size"] &]];

(* gauge-fix: rotate to concentrate the quartic on the diagonal (reveals the
   tetra/cubic axes -> sparsity becomes visible), then sort axes by lam_iiii
   descending and fix signs.  The diagonal-concentration objective sum_i lam_iiii^2
   is maximized in the cubic-aligned (sparsest) frame, so we keep the global best
   over many starts.  obj carries a numeric guard so FindMinimum stays numeric
   (no symbolic MatrixExp differentiation) -- ~0.1s/align instead of ~7s. *)
Options[alignTensor] = {"Starts" -> 16};
alignTensor[vec_List, keys_List, n_Integer, OptionsPattern[]] := Module[
  {lam, gens, ng, rotFromA, obj, best, bestR, starts, lamR, ord, perm, signs, sgnFix},
  lam = inflate[vec, keys, n];
  gens = Normal /@ Flatten[Table[
     SparseArray[{{i, j} -> 1, {j, i} -> -1}, {n, n}], {i, n}, {j, i + 1, n}], 1];
  ng = Length[gens];
  rotFromA[av_] := MatrixExp[Sum[av[[m]] gens[[m]], {m, ng}]];
  obj[av_ /; VectorQ[av, NumericQ]] := With[{lm = rotateTensor[lam, rotFromA[av]]},
     -Sum[lm[[i, i, i, i]]^2, {i, n}]];
  best = Infinity; bestR = IdentityMatrix[n];
  starts = Join[{ConstantArray[0., ng]}, Table[RandomReal[{-2, 2}, ng], {OptionValue["Starts"]}]];
  Do[
    Module[{vars = Array[Unique["q"] &, ng], r},
      r = Quiet@FindMinimum[obj[vars], Transpose[{vars, s0}], MaxIterations -> 200];
      If[r[[1]] < best, best = r[[1]]; bestR = rotFromA[vars /. r[[2]]]]],
    {s0, starts}];
  lamR = rotateTensor[lam, bestR];
  (* sort axes by lam_iiii descending *)
  ord = Ordering[Table[lamR[[i, i, i, i]], {i, n}], All, Greater];
  perm = Normal@SparseArray[Table[{i, ord[[i]]} -> 1, {i, n}], {n, n}];
  lamR = rotateTensor[lamR, perm];
  (* fix signs: choose e_i -> -e_i to make sum of odd-in-i couplings >= 0 (cheap heuristic) *)
  signs = Table[
     With[{odd = Sum[If[OddQ[Count[k, i]], lamR[[Sequence @@ k]] Length[Permutations[k]], 0], {k, keys}]},
       If[odd < -10.^-9, -1, 1]], {i, n}];
  sgnFix = DiagonalMatrix[signs];
  lamR = rotateTensor[lamR, sgnFix];
  Chop[deflate[lamR, keys], 10.^-9]];

(* signed permutation group of R^N (hyperoctahedral B_N) *)
signedPerms[n_Integer] := Module[{perms, signs},
  perms = Permutations[Range[n]];
  signs = Tuples[{1, -1}, n];
  Flatten[Table[
    DiagonalMatrix[s] . Normal@SparseArray[Table[{i, p[[i]]} -> 1, {i, n}], {n, n}],
    {p, perms}, {s, signs}], 1]];

residualSymmetry[vec_List, keys_List, n_Integer, tol_:10.^-5] := Module[
  {lam, group, stab, gens, soDim, soGens, tangents},
  lam = inflate[alignTensor[vec, keys, n], keys, n];
  group = signedPerms[n];
  stab = Select[group, Max[Abs[Flatten[rotateTensor[lam, #] - lam]]] < tol &];
  (* continuous stabilizer: so(N) generators G with G.lam = 0 *)
  soGens = Flatten[Table[SparseArray[{{i, j} -> 1, {j, i} -> -1}, {n, n}], {i, n}, {j, i + 1, n}], 1];
  tangents = Table[
     With[{g = Normal[gg]},
       Flatten[ TensorContract[TensorProduct[g, lam], {{2, 3}}]
              + Transpose[TensorContract[TensorProduct[g, lam], {{2, 3}}], {2, 1, 3, 4}]
              + Transpose[TensorContract[TensorProduct[g, lam], {{2, 3}}], {2, 3, 1, 4}]
              + Transpose[TensorContract[TensorProduct[g, lam], {{2, 3}}], {2, 3, 4, 1}] ]],
     {gg, soGens}];
  soDim = Length[soGens] - MatrixRank[N[tangents], Tolerance -> tol];
  <|"discreteOrder" -> Length[stab], "discreteStab" -> stab,
    "continuousDim" -> soDim, "fullDim" -> Length[soGens]|>];

rationalizeVec[vec_List, keys_List, n_Integer, deg_:2] := Module[{rat, ratHi, resid},
  rat = Map[Quiet@Check[RootApproximant[#, deg], #] &, vec];
  (* certify by evaluating the (fast, Taylor-mode) T-chain on the EXACT algebraic
     numbers at high working precision -- avoids the symbolic Lie-chain blowup. *)
  ratHi = N[rat, 30];
  resid = sigmaResidualReal[ratHi, keys, n, 2];
  <|"rational" -> rat, "exactResidual" -> resid|>];

(* ---- local dimension of the solution variety (cone dim) at a point ---- *)
sigmaResidualVec[vec_List, keys_List, n_Integer, kMax_Integer] := Module[
  {lam0, series, k, out},
  lam0 = inflate[vec, keys, n];
  series = ConstantArray[ConstantArray[0, {n, n, n, n}], kMax + 1];
  series[[1]] = lam0;
  Do[With[{bk = -series[[k + 1]] + Sum[
        TensorContract[TensorProduct[series[[i + 1]], series[[k - i + 1]]], {{3, 5}, {4, 6}}]
      + Transpose[TensorContract[TensorProduct[series[[i + 1]], series[[k - i + 1]]], {{3, 5}, {4, 6}}], {1, 3, 2, 4}]
      + Transpose[TensorContract[TensorProduct[series[[i + 1]], series[[k - i + 1]]], {{3, 5}, {4, 6}}], {1, 4, 3, 2}], {i, 0, k}]},
     series[[k + 2]] = bk/(k + 1)], {k, 0, kMax - 1}];
  out = {};
  Do[With[{rhoK = Sum[TensorContract[TensorProduct[series[[i + 1]], series[[k - i + 1]]], {{2, 6}, {3, 7}, {4, 8}}], {i, 0, k}]},
     With[{tK = (rhoK - (Tr[rhoK]/n) IdentityMatrix[n]) (k!)},
       out = Join[out, Flatten[Table[tK[[i, j]], {i, n}, {j, i, n}]]]]], {k, 0, kMax}];
  out];

manifoldDimAt[vec_List, keys_List, n_Integer, kMax_Integer:3, h_:10.^-6] := Module[{nv, jac},
  nv = Length[keys];
  jac = Transpose[Table[
     (sigmaResidualVec[vec + h UnitVector[nv, a], keys, n, kMax]
      - sigmaResidualVec[vec - h UnitVector[nv, a], keys, n, kMax])/(2 h), {a, nv}]];
  nv - MatrixRank[jac, Tolerance -> 10.^-6]];

(* describe an aligned vector: group nonzero components by index-shape and list
   the distinct values within each shape (this is where the ansatz reveals itself). *)
describeAligned[aligned_List, keys_List, n_Integer, vtol_:10.^-4] := Module[{nz, byShape},
  nz = Select[Transpose[{keys, aligned}], Abs[#[[2]]] > 10.^-6 &];
  byShape = GatherBy[nz, shapeOf[#[[1]], n] &];
  Association[(shapeOf[#[[1, 1]], n] -> <|
       "components" -> (StringJoin[ToString /@ #[[1]]] & /@ #),
       "values" -> #[[All, 2]],
       "distinctValues" -> Union[Round[#[[All, 2]], vtol]],
       "uniform" -> (Length[Union[Round[Abs[#[[All, 2]]], vtol]]] == 1)|>) & /@ byShape]];

interpretSolutions[sols_List, keys_List, n_Integer, OptionsPattern[{"Tol" -> 10.^-3, "Degree" -> 2, "Dim" -> True}]] := Module[
  {clusters, out, idx, shapeName},
  shapeName[sh_] := StringJoin[Riffle[ToString /@ sh, "+"]];
  Print["============================================================"];
  Print["  INTERPRET: ", Length[sols], " solution vectors at N = ", n];
  Print["============================================================"];
  clusters = clusterOrbits[sols, keys, n, OptionValue["Tol"]];
  Print["  -> ", Length[clusters], " qualitative class(es) (by symmetry signature).\n"];
  idx = 0;
  out = Table[
    idx++;
    Module[{rep = c["rep"], fp = c["fingerprint"], aligned, sym, desc, mdim, odim, moduli, rat},
      Print["====== Class ", idx, "  (", c["size"], " of ", Length[sols], " solutions) ======"];
      Print["   flattening mult pattern : ", fp["mult"],
            "    spin-2 |q| ", ScientificForm[fp["qdefect"], 2],
            "    harmonic isotropy defect ", ScientificForm[fp["isoDefect"], 2]];
      aligned = alignTensor[rep, keys, n];
      sym = residualSymmetry[rep, keys, n];
      odim = sym["fullDim"] - sym["continuousDim"];
      If[OptionValue["Dim"],
        mdim = manifoldDimAt[rep, keys, n, 3]; moduli = mdim - odim,
        mdim = "-"; moduli = "?"];
      Print["   cone dim ", mdim, " = moduli ", moduli, " + O(", n, ")-orbit dim ", odim,
            "    (discrete stabilizer order ", sym["discreteOrder"], " in B_", n,
            ", continuous so dim ", sym["continuousDim"], ")"];
      desc = describeAligned[aligned, keys, n];
      Print["   aligned ansatz (nonzero index-shapes):"];
      Do[With[{info = desc[sh]},
        Print["       shape ", shapeName[sh], "  (", Length[info["components"]], " comps): ",
          If[info["uniform"], "ALL EQUAL = " <> ToString[NumberForm[info["values"][[1]], 6]],
             "values " <> ToString[NumberForm[#, 5] & /@ info["distinctValues"]]]]],
        {sh, Keys[desc]}];
      (* closed form: rationalize the DISTINCT aligned values (meaningful for isolated
         points / fixed points; for a moduli>0 family these are just one sampled point). *)
      rat = rationalizeVec[aligned, keys, n, OptionValue["Degree"]];
      If[TrueQ[moduli === 0],
        Print["   ISOLATED point -> closed form (exact T-chain residual ",
              ScientificForm[rat["exactResidual"], 2], "):"];
        Do[If[Abs[aligned[[a]]] > 10.^-6,
           Print["       lam", StringJoin[ToString /@ keys[[a]]], " = ", rat["rational"][[a]]]], {a, Length[keys]}],
        Print["   FAMILY (moduli ", moduli, ") -> ansatz above is the structure; the ",
              "shape-values are the free parameters (e.g. u, w). RootApproximant of a ",
              "generic family point is not meaningful."]];
      Print[""];
      <|"class" -> idx, "size" -> c["size"], "members" -> c["members"], "fingerprint" -> fp,
        "aligned" -> aligned, "description" -> desc, "symmetry" -> sym,
        "coneDim" -> mdim, "orbitDim" -> odim, "moduli" -> moduli, "rational" -> rat|>],
    {c, clusters}];
  out];


(* ============================================================
   IO
   ============================================================ *)
importSols[file_String] := Module[{d},
  d = Import[file, "RawJSON"];
  {N[d["sols"]], (# + 1 & /@ # & /@ d["keys"]), d["N"]}];  (* keys: 0-based -> 1-based *)

End[];
EndPackage[];
