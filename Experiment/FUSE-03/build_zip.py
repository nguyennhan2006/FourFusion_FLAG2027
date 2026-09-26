"""FUSE-03 candidate = FUSE-02 with g/Bn replaced by the duration-gated 003c/r2_mix fusion (a=0.5, b=-1.5,
the only variant that passed its pre-registered gate: OOF +1.25, 4/5 identity folds)."""
import zipfile, numpy as np, pandas as pd
from pathlib import Path
from common import cell_data, eer, E
F02 = E / "FUSE-02/out/submission_FUSE02.zip"
k = ("gender", "Bangla"); fn = "gender/sub_score_v4_Bangla_unheard.txt"
D = cell_data(k, ["003c", "r2_mix"])
w = 1 / (1 + np.exp(-(0.5 * np.log(np.maximum(D["dur"], .5)) - 1.5)))          # weight on 003c
s = w * D["R"]["003c"] + (1 - w) * D["R"]["r2_mix"]
kn = D["known"]
print(f"g/Bn pseudo: FUSE-02 {eer((D['R']['003c'] + D['R']['r2_mix'])[kn] / 2, D['y'][kn]):.2f} -> gated {eer(s[kn], D['y'][kn]):.2f}")
for lab in ["btc"]:
    Db = cell_data(k, ["003c", "r2_mix"], labels=lab); kb = Db["known"]
    print(f"   independent labels [{lab}]: FUSE-02 {eer((Db['R']['003c'] + Db['R']['r2_mix'])[kb] / 2, Db['y'][kb]):.2f} -> gated {eer(s[kb], Db['y'][kb]):.2f}")
out = Path("out/submission_FUSE03.zip")
with zipfile.ZipFile(F02) as zi, zipfile.ZipFile(out, "w") as zo:
    for n in zi.namelist():
        if n == fn:
            zo.writestr(n, "\n".join(f"{p} {-v:.6f}" for p, v in zip(D["t"].pair_id, s)) + "\n")
        else:
            zo.writestr(n, zi.read(n))
with zipfile.ZipFile(out) as zf:
    assert len(zf.namelist()) == 4
    d = pd.read_csv(zf.open(fn), sep=" ", header=None, names=["pid", "s"])
    assert (d.pid.values == D["t"].pair_id.values).all() and np.isfinite(d.s).all()
    assert abs(eer(-d.s.values[kn], D["y"][kn]) - eer(s[kn], D["y"][kn])) < 1e-9     # polarity round-trip
print("wrote + verified", out)
