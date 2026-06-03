(* ::Package:: *)

(* ============================================================
   build_Tk_general.wl

   Build  T  and  T^(k) = (d/dmu)^k T   for  k = 0,1,2
   for  N = 2,3,4,5  flavours.  For each N the notebook
     (1) prints T and each T^(k) cleanly and in order
         - N<=3: the full traceless matrix
         - N>=4: per-component term counts (the full T^(k) has hundreds of
                 terms) together with the matrix restricted to Sigma,
     (2) prints the reduced flow on Sigma, and
     (3) verifies that the conjectured Sigma locus

       Sigma = { lam_iiii = u,  lam_iijj = w (i!=j),  all else = 0 }

   solves   T = T^(1) = T^(2) = 0.

   Conventions (matching build_Tk.wl):
     - fully symmetric quartic  lam_{ijkl},  indices in {1..N}
     - one-loop beta (eps = kappa = 1):
         beta_{ijkl} = -lam_{ijkl}
                       + sum_{s,t,u channels} sum_{a,b} lam_{(pq)(ab)} lam_{(ab)(rt)}
     - (rho'_{1|N})_{ij} = sum_{klm} lam_{iklm} lam_{jklm}
     - T = rho - (1/N) Tr[rho] Id      (symmetric traceless spin-2 part)
     - Lie derivative along the flow:
         (d/dmu) F = sum_alpha beta_alpha (dF/dlam_alpha)

   Sanity: at N=2 the betas reproduce those in build_Tk.wl with
     a=lam1111, b=lam1112, c=lam1122, d=lam1222, e=lam2222.
   ============================================================ *)

ClearAll["Global`*"];

(* ------------------------------------------------------------
   makeTheory[n]:  generate all objects as functions of n.
   ------------------------------------------------------------ *)
makeTheory[n_Integer] := Module[
  {keys, lamSym, lamF, rho, Tmat, betaF, betaRule, vars, lieD},

  (* independent couplings: sorted 4-tuples 1<=i<=j<=k<=l<=n *)
  keys = DeleteDuplicates[Sort /@ Tuples[Range[n], 4]];

  (* one atomic symbol per independent coupling (e.g. lam1122 internally),
     displayed as a proper subscripted lambda \[Lambda]_{ijkl}.
     The symbol stays atomic so D[] and /. work normally. *)
  lamSym[k_] := lamSym[k] = With[
    {s = Symbol["lam" <> StringJoin[ToString /@ k]]},
    s /: MakeBoxes[s, fmt_] := SubscriptBox["\[Lambda]",
       RowBox[ToString /@ k]];
    s
  ];
  vars = lamSym /@ keys;

  (* fully symmetric accessor for ANY index order *)
  lamF[i_, j_, k_, l_] := lamSym[Sort[{i, j, k, l}]];

  (* reduced density matrix rho'_{1|N} *)
  rho = Expand@Table[
     Sum[lamF[i, k, l, m] lamF[j, k, l, m], {k, n}, {l, n}, {m, n}],
     {i, n}, {j, n}];

  (* traceless (spin-2) part *)
  Tmat = Expand[rho - (1/n) Tr[rho] IdentityMatrix[n]];

  (* one-loop beta for a given coupling, summing the 3 (s,t,u) channels.
     channelTerm[p,q,r,t] = sum_{a,b} lam_{(pq)(ab)} lam_{(ab)(rt)}. *)
  With[{channelTerm =
     Function[{p, q, r, t},
       Sum[lamF[p, q, a, b] lamF[a, b, r, t], {a, n}, {b, n}]]},
    betaF[{i_, j_, k_, l_}] := -lamF[i, j, k, l] +
       channelTerm[i, j, k, l] +
       channelTerm[i, k, j, l] +
       channelTerm[i, l, j, k]
  ];

  betaRule = Thread[vars -> Expand[betaF /@ keys]];

  (* Lie derivative along the RG flow on any scalar *)
  lieD[F_] := Expand[Sum[D[F, vars[[m]]] (vars[[m]] /. betaRule), {m, Length[vars]}]];

  <|
    "n" -> n, "keys" -> keys, "vars" -> vars, "lamSym" -> lamSym,
    "rho" -> rho, "T" -> Tmat, "betaRule" -> betaRule, "lieD" -> lieD
  |>
];

(* ------------------------------------------------------------
   sigmaRule[th]:  the conjectured Sigma ansatz as a substitution
   lam_iiii -> u,  lam_iijj -> w (i!=j),  all other couplings -> 0.
   ------------------------------------------------------------ *)
sigmaRule[th_] := Module[{n = th["n"]},
  Table[
    th["lamSym"][k] -> With[{cnt = Sort[Table[Count[k, i], {i, n}], Greater]},
      Which[
        Take[cnt, 2] === {4, 0}, Global`u,   (* lam_iiii *)
        Take[cnt, 2] === {2, 2}, Global`w,   (* lam_iijj *)
        True, 0
      ]
    ],
    {k, th["keys"]}
  ]
];

(* ------------------------------------------------------------
   checkN[n]:  build {T^(0),T^(1),T^(2)}, print the reduced flow on
   Sigma, and verify each T^(k) vanishes on Sigma.
   ------------------------------------------------------------ *)
checkN[n_Integer] := Module[
  {th, lie, Tlist, sig, bu, bw, results},

  th  = makeTheory[n];
  sig = sigmaRule[th];

  (* apply Lie derivative componentwise to a matrix *)
  lie[M_] := Map[th["lieD"], M, {2}];
  Tlist = NestList[lie, th["T"], 2];   (* {T^(0), T^(1), T^(2)} *)

  Print["================  N = ", n, "   (",
        Length[th["vars"]], " independent couplings)  ================"];

  (* ---- print T and each T^(k), cleanly and in order ----
     N<=3 : show the full traceless matrix (compact).
     N>=4 : the full T^(k) has hundreds of terms per entry, so instead show
            (i) the term count of each independent component (size indicator)
            and (ii) the matrix restricted to Sigma, which is what should vanish. ---- *)
  Do[
    Print["----  T^(", k, ")  ----"];
    If[n <= 3,
      Print[MatrixForm[Tlist[[k + 1]]]],
      (* n>=4 *)
      Print["  (full, unrestricted) term counts  T^(", k, ")_{ij}:"];
      Do[
        With[{poly = Expand[Tlist[[k + 1]][[i, j]]]},
          Print["    T^(", k, ")_{", i, j, "} : ",
                Which[poly === 0, 0, Head[poly] === Plus, Length[poly], True, 1],
                " terms"]
        ],
        {i, 1, n}, {j, i, n}
      ];
      Print["  (restricted to Sigma)  T^(", k, ")|_Sigma ="];
      Print[MatrixForm[Simplify[Tlist[[k + 1]] /. sig]]]
    ],
    {k, 0, 2}
  ];
  Print[""];

  (* reduced flow on Sigma: restrict beta of lam_1111 and lam_1122 *)
  bu = Expand[(th["lamSym"][{1, 1, 1, 1}] /. th["betaRule"]) /. sig];
  bw = Expand[(th["lamSym"][{1, 1, 2, 2}] /. th["betaRule"]) /. sig];
  Print["  reduced flow on Sigma:   u' = ", bu, " ,   w' = ", bw];

  (* verification *)
  results = Table[
    With[{Tk = Simplify[Tlist[[k + 1]] /. sig]},
      Print["  T^(", k, ")|_Sigma = 0 ?   ",
            If[Tk === ConstantArray[0, {n, n}], "YES  (OK)", "*** NONZERO ***"]];
      Tk === ConstantArray[0, {n, n}]
    ],
    {k, 0, 2}
  ];
  Print[""];
  results
];

(* ------------------------------------------------------------
   Run for N = 2,3,4,5.
   ------------------------------------------------------------ *)
allResults = Association[Table[n -> checkN[n], {n, {2, 3, 4, 5}}]];

Print["==================  SUMMARY  =================="];
Do[
  Print["N = ", n, ":  ",
    If[And @@ allResults[n],
       "Sigma solves  T = T^(1) = T^(2) = 0   (verified)",
       "*** FAILED ***"]],
  {n, {2, 3, 4, 5}}
];

Print["\nReusable objects:"];
Print["  th = makeTheory[N];   th[\"T\"], th[\"rho\"], th[\"vars\"], th[\"betaRule\"], th[\"lieD\"]"];
Print["  sigmaRule[makeTheory[N]]   gives the Sigma substitution (u = lam_iiii, w = lam_iijj)."];



