"""Dump n{2,3,4}_clean.npz -> n{2,3,4}_clean.json so SigmaTools.importSols can
read the solution sets in Mathematica. Needs only numpy.

Run with a numpy-capable interpreter, e.g.:
    /Users/<you>/miniconda3/bin/python3 _npz_to_json.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

for n in (2, 3, 4):
    path = os.path.join(HERE, f"n{n}_clean.npz")
    if not os.path.exists(path):
        print(f"skip N={n}: {path} not found")
        continue
    d = np.load(path)
    out = {
        "N": int(d["N"]),
        "k_max": int(d["k_max"]),
        "keys": [[int(i) for i in k] for k in d["keys"].tolist()],
        "sols": [[float(v) for v in row] for row in d["sols"]],
    }
    out_path = os.path.join(HERE, f"n{n}_clean.json")
    json.dump(out, open(out_path, "w"))
    print(f"N={n}: {len(out['sols'])} sols, {len(out['keys'])} keys -> {out_path}")
