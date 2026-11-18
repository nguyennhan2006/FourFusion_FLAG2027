r"""LID-AUDIT-01 (README §3): three LID votes per MAV-Celeb v1 / v2 voice file -> rule D1 (speaker level) and D2 (file level).
  W  Whisper-large-v3, V  VoxLingua107: FLAG_150 (Experiment/LIDSTRICT-01/out/mav.parquet)
  M  MMS-LID-4017: FLAG_152 (out/mms.parquet); a file M could not read counts as an M vote (no evidence = flagged)
    python Experiment/LID-AUDIT-01/run.py      -> report.txt, out/mav_d1.json (LIDSTRICT-01/merge.py reads it)

FourFusion reproduction repo (stage l_lid_audit): copy of Experiment/LID-AUDIT-01/run.py (md5 03f344846423687c95ef3be93e818b54).
Only change: the two inputs and the output folder can be set with FF_MAV_PARQUET, FF_MMS_PARQUET, FF_OUT (defaults = the original
team-machine paths); FF_OUT/out is created when missing. CHANGES.md lists it. mav_w.json (the only output the chain reads) depends on
FLAG_150's w_flag only; mms.parquet must still be given (row-count / 1:1 checks of votes()).
"""
import json, os
from pathlib import Path
import pandas as pd

H = Path(__file__).resolve().parent
K = ["src", "spk", "lang", "video", "file"]
MAV = Path(os.environ.get("FF_MAV_PARQUET", H.parent / "LIDSTRICT-01/out/mav.parquet"))   # original: inline in votes()
MMS = Path(os.environ.get("FF_MMS_PARQUET", H / "out/mms.parquet"))                       # original: inline in votes()
OUT = Path(os.environ.get("FF_OUT", H))                                                   # original: H (report.txt) and H / "out"


def votes():
    A = pd.read_parquet(MAV)
    B = pd.read_parquet(MMS)
    assert not A.duplicated(K).any() and not B.duplicated(K).any() and len(A) == len(B), (len(A), len(B))
    X = A.merge(B[[c for c in B.columns if c in K or c.startswith("m_")]], on=K, how="left", validate="1:1")
    X["W"] = X.w_flag.fillna(False).astype(bool)
    X["V"] = X.v_top1.isin(["bn", "as"]) | (X.v_pbn + X.v_pas >= 0.2)
    X["M"] = X.m_flag.isna() | X.m_flag.fillna(True).astype(bool)
    X["n"] = X[["W", "V", "M"]].sum(axis=1)
    X["status"] = X.n.map(lambda n: "CLEAN" if n == 0 else "NGHI" if n == 1 else "BENGALI")
    return X


def main():
    (OUT / "out").mkdir(parents=True, exist_ok=True)                                      # FF: the original folder already existed
    X = votes()
    bn = X[X.status == "BENGALI"]
    d1 = {v: sorted(set(bn.spk[bn.src == v])) for v in ("mav_v1", "mav_v2")}
    sole = X[(X.n == 1) & ~X.V].assign(p=lambda d: (d.w_pbn + d.w_pas).combine((d.m_pbn + d.m_pas).fillna(1), max))
    T = ["LID-AUDIT-01 report (README §3; W = Whisper-large-v3, V = VoxLingua107, M = MMS-LID-4017)", ""]
    T += ["files by status and source:", pd.crosstab(X.status, X.src, margins=True).to_string(), ""]
    T += ["vote patterns (W V M):", X.groupby(["W", "V", "M"]).size().rename("files").to_string(), ""]
    T += ["BENGALI files by language label:", pd.crosstab(bn.lang, bn.src).to_string() if len(bn) else "(none)", ""]
    T += [f"M flagged files: {int(X.M.sum())} (README §4: > 300 -> check M before use); M missing (read error): {int(X.m_flag.isna().sum())}",
          f"M top-1 languages: {X.m_top1s.dropna().str.split(',').explode().value_counts().head(10).to_dict()}", ""]
    T += [f"D1 (speaker level, main rule): speakers whose voice is dropped {({k: len(v) for k, v in d1.items()})} "
          f"of {X.groupby('src').spk.nunique().to_dict()}; voice files lost "
          f"{int(X[X.spk.isin(d1['mav_v1']) & (X.src == 'mav_v1') | X.spk.isin(d1['mav_v2']) & (X.src == 'mav_v2')].shape[0])}",
          f"D2 (file level, user's option): files dropped {len(bn)}; speakers with no file left "
          f"{int((X.assign(k=X.status != 'BENGALI').groupby(['src', 'spk']).k.sum() == 0).sum())}", ""]
    T += [f"layer 4 (listen): {len(sole)} NGHI files whose only vote is W or M; top 30 by max p(bn) + p(as):",
          sole.sort_values("p", ascending=False).head(30)[K + ["W", "M", "p"]].to_string(index=False) if len(sole) else "(none)"]
    (OUT / "report.txt").write_text("\n".join(T) + "\n", encoding="utf-8")
    sole.sort_values("p", ascending=False)[K + ["W", "M", "p"]].to_csv(OUT / "out/listen_files.csv", index=False)
    # user 3/11 14:30 ("chỉ whisper"): MAV voice files dropped = every file with a Whisper vote, file level; D1 / D2 only reported.
    # Paths in the eda-bundle VMETA form (v1/voices/<spk>/<lang>/<video>/<file>) = what ext_rows() filters on.
    w = X[X.W]
    files = sorted(f"{s[-2:]}/voices/{p}/{l}/{v}/{f}" for s, p, l, v, f in w[K].itertuples(index=False))
    (OUT / "out/mav_w.json").write_text(json.dumps(dict(exp="LID-AUDIT-01", rule="W", n_files=len(files),
                                                        speakers={v: int(w.spk[w.src == v].nunique()) for v in ("mav_v1", "mav_v2")},
                                                        files=files), indent=1))
    (OUT / "out/mav_d1.json").unlink(missing_ok=True)   # D1 was not chosen; LIDSTRICT-01/merge.py must never pick it up
    print("\n".join(T[:30]))


if __name__ == "__main__":
    main()
