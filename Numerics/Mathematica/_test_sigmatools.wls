#!/usr/bin/env wolframscript
(* correctness tests for SigmaTools *)
SetDirectory[DirectoryName[$InputFileName]];
Get["SigmaTools.wl"];
Needs["SigmaTools`"];

ok[label_, cond_] := Print[If[cond, "PASS", "**** FAIL"], "  ", label];

(* ---- 1. beta channels vs independent direct sum ---- *)
n = 3;
keys = sigKeys[n];
SeedRandom[1];
x = RandomReal[{-1, 1}, Length[keys]];
lam = inflate[x, keys, n];
betaDirect = Table[
   -lam[[i, j, k, l]]
    + Sum[lam[[i, j, a, b]] lam[[a, b, k, l]], {a, n}, {b, n}]
    + Sum[lam[[i, k, a, b]] lam[[a, b, j, l]], {a, n}, {b, n}]
    + Sum[lam[[i, l, a, b]] lam[[a, b, j, k]], {a, n}, {b, n}],
   {i, n}, {j, n}, {k, n}, {l, n}];
ok["betaTensor matches direct sum", Max[Abs[Flatten[betaTensor[lam, n] - betaDirect]]] < 10.^-10];

(* ---- 2. rotateTensor: T residual & flattening spectrum O(N)-invariant ---- *)
A = RandomReal[{-1, 1}, {n, n}]; A = A - Transpose[A];
R = MatrixExp[A];
lamR = rotateTensor[lam, R];
ok["rotateTensor preserves symmetry", Max[Abs[Flatten[lamR - Transpose[lamR, {2, 1, 3, 4}]]]] < 10.^-10];
r0 = sigmaResidualReal[x, keys, n, 0];
rR = sigmaResidualReal[deflate[lamR, keys], keys, n, 0];
ok["T residual O(N)-invariant", Abs[r0 - rR] < 10.^-8 r0 + 10.^-8];
spec0 = Sort[Eigenvalues[N[flatteningMatrix[lam, n]]]];
specR = Sort[Eigenvalues[N[flatteningMatrix[lamR, n]]]];
ok["flattening spectrum O(N)-invariant", Max[Abs[spec0 - specR]] < 10.^-8];

(* ---- 3. Sigma ansatz solves the chain at N=2,3,4 ---- *)
Do[
  With[{kk = sigKeys[nn]},
   Module[{xv, cls, iiii, iijj},
    cls = makeTheory[nn]["sigmaRule"]; (* not used; build ansatz vector directly *)
    iiii = Position[kk, k_ /; Sort[Tally[k][[All,2]], Greater][[1]] == 4][[All,1]];
    iijj = Position[kk, k_ /; (Sort[Tally[k][[All,2]], Greater] === {2,2})][[All,1]];
    xv = ConstantArray[0., Length[kk]];
    xv[[iiii]] = 0.37; xv[[iijj]] = -0.13;
    Print["  N=", nn, " Sigma-ansatz residual T^(0..2) = ",
      ScientificForm[sigmaResidualReal[xv, kk, nn, 2], 3]]
   ]],
  {nn, {2, 3, 4}}];

(* ---- 4. exact S_N sector fixed points at N=3 (expect Gaussian/decoupled/cubic/Heisenberg) ---- *)
Print["\n[exact] N=3 fixed points on M_{1|3} (S_3 sector):"];
fp3 = solveFixedPoints[3];
Print["  amplitudes (shapes ", fp3["shapes"], "):"];
Print["  ", Column[fp3["solutions"]]];

Print["\nDONE"];
