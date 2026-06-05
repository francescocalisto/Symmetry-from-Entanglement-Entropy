#!/usr/bin/env wolframscript
(* ============================================================
   sigma_pipeline.wls  --  end-to-end demonstration of SigmaTools

   Run:   wolframscript -file sigma_pipeline.wls
   (uses the JSON solution sets in ../N4_study/ ; regenerate them with
    ../N4_study/_npz_to_json.py if absent)

   Shows the two roads:
     (A) CONSTRUCTIVE / exact -- solve a symmetry sector in closed form
         (this is what *discovers* the loci; no decimals to squint at).
     (B) INTERPRET -- take a pile of numeric solution vectors and recover
         the structure: cluster by symmetry type, gauge-fix to reveal the
         ansatz, report dim = moduli + orbit, rationalize isolated points.
   ============================================================ *)

SetDirectory[DirectoryName[$InputFileName]];
Get["SigmaTools.wl"]; Needs["SigmaTools`"];
$Output = {OpenWrite["/dev/stdout", PageWidth -> Infinity]};
banner[s_] := (Print["\n", StringRepeat["#", 64]]; Print["##  ", s]; Print[StringRepeat["#", 64]]);

(* ---------- (A) the constructive road: exact symmetry-sector solving ---------- *)
banner["(A) EXACT fixed points on M_{1|N} inside the S_N sector"];

Print["\nN=3  (shapes ", solveFixedPoints[3]["shapes"], "):"];
Do[Print["   ", s], {s, solveFixedPoints[3]["solutions"]}];
Print["  -> Gaussian, decoupled (1/3,..), cubic (2/9,0,1/9,0),"];
Print["     Heisenberg (3/11,0,1/11,0), and the rotated 'rogue' P2, P3."];

Print["\nN=4  (shapes ", solveFixedPoints[4]["shapes"], "):"];
Do[Print["   ", s], {s, solveFixedPoints[4]["solutions"]}];
Print["  -> incl. the hypertetrahedral point  a4=(41-3Sqrt5)/168, a22=(13+Sqrt5)/168,"];
Print["     a1111=-(1+Sqrt5)/168, ...  (= eq.(6) of fourflavour.pdf, + Galois twin)."];

(* ---------- (B) the interpretation road: structure from decimals ---------- *)
banner["(B) INTERPRET numeric solution sets"];

Module[{sols, keys, n},
  {sols, keys, n} = importSols["../N4_study/n3_clean.json"];
  Print["\n>>> N=3: expect ONE class = the cubic/tetrahedral plane, dim 5 = 2 + 3."];
  interpretSolutions[Take[sols, UpTo[20]], keys, n];
];

Module[{sols, keys, n},
  {sols, keys, n} = importSols["../N4_study/n4_clean.json"];
  Print["\n>>> N=4: expect SEVERAL classes (q=0 family, q!=0 strata, a degenerate"];
  Print["    higher-symmetry stratum). Random sampling lands on GENERIC members of"];
  Print["    high-dim families, NOT the special symmetric points -- which is exactly"];
  Print["    why the constructive road (A) is needed to pin those down.  [~3-4 min]"];
  interpretSolutions[sols, keys, n];
];

Print["\nDONE"];
