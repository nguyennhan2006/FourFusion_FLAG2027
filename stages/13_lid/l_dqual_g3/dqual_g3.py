r"""l_dqual_g3 (FourFusion reproduction repo; NEW script = reduced copy of DQUAL-01, CPU, about a minute).

lid_strict.json (Experiment/LIDSTRICT-01/merge.py:47-49) reads ONE column of Experiment/DQUAL-01/persons.parquet: layer G3 of the
sources VoxCeleb2 and VoxBlink ("mixed identities": bottom 5 % of within-person consistency of the source, faces and voices).
Experiment/DQUAL-01/run.py computes G3 together with G1 / G2 / G4 for seven sources, and needs inputs that the 13.54 system must not
use (AVSpeech / LRS3 features through NEW_DIR, the AUXP-01 CC head through CC_MODELS). This script keeps only what G3 needs:
  * sources loaded as Experiment/EDA-NEWDATA/run.py:18-21 (cat), 41-48 (VoxCeleb2), 49-57 (VoxBlink):
      VoxCeleb2 = the FLAG_37 key rows (face_meta.csv / voice_meta.csv) minus T3ENC-01 `dropped_speakers()` = the FLAG_42 drop list
                  (VOX_LID lid_summary.json "speakers_dropped_ids"); LID_STRICT is NOT applied (lid_strict.json is built from this
                  table; on 14/10, when DQUAL-01 ran, it did not exist); arrays = FLAG_41 feats_vox2t3;
      VoxBlink  = every FLAG_46 row (feats_vb), no drop;
      face = [L2(FaRL ep16) | L2(CLIP-L)], voice = [L2(SimAM-ResNet100 pretrained) | L2(fine-tuned)];
  * G3 exactly as Experiment/DQUAL-01/run.py:48-51 (ids, row groups), 68-72 (cons), 94 (G3 rule), per source.
The lines marked "verbatim" are copied unchanged. Expected: G3 fails 454 of 4 852 (VoxCeleb2), 1 388 of 15 175 (VoxBlink), and the
same speakers as persons.parquet (sha256 of the sorted lists in --expected).
    python dqual_g3.py --vox2-arrays DIR --vox2-keys DIR --vox2-lid-summary FILE --vb DIR --out DIR [--expected expected.json]
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)          # verbatim: EDA-NEWDATA/run.py:14 (DQUAL-01: l2 = ED.l2n)


def cat(d, kind, part=None):                                                     # verbatim: EDA-NEWDATA/run.py:18-21 (part=None branch)
    names = ("farl16", "clipL") if kind == "face" else ("sam100p", "sam100f")
    if part is None:
        return np.concatenate([l2n(np.load(d / f"{kind}_{n}.npy").astype(np.float32)) for n in names], 1)
    raise NotImplementedError("only the part=None branch is used here")


def sources(a):
    """{name: dict(F, fspk, V, vspk)} for VoxCeleb2 and VoxBlink, as EDA-NEWDATA/run.py:41-57 (only the keys G3 reads)."""
    out = {}
    VK, X_ = a.vox2_keys, a.vox2_arrays
    fm = pd.read_csv(VK / "face_meta.csv", dtype={"video": str, "utt": str}); vm = pd.read_csv(VK / "voice_meta.csv", dtype={"video": str, "utt": str})
    drop = set(json.loads(Path(a.vox2_lid_summary).read_text())["speakers_dropped_ids"])   # = T3ENC-01/run.py dropped_speakers() without LID_STRICT
    kf, kv = ~fm.spk.isin(drop).values, ~vm.spk.isin(drop).values
    F, V = cat(X_, "face"), cat(X_, "voice")
    assert len(F) == len(fm) and len(V) == len(vm), "FLAG_41 arrays must follow the FLAG_37 key rows"   # FF: sanity check
    out["VoxCeleb2"] = dict(F=F[kf], fspk=fm.spk.values[kf].astype(str), V=V[kv], vspk=vm.spk.values[kv].astype(str))
    VB = a.vb
    fm = pd.read_csv(VB / "face_meta.csv", dtype={"video": str, "utt": str}); vm = pd.read_csv(VB / "voice_meta.csv", dtype={"video": str, "utt": str})
    F, V = cat(VB, "face"), cat(VB, "voice")
    assert len(F) == len(fm) and len(V) == len(vm), "FLAG_46 arrays must follow its meta rows"            # FF: sanity check
    out["VoxBlink"] = dict(F=F, fspk=fm.spk.values.astype(str), V=V, vspk=vm.spk.values.astype(str))
    return out


def g3_table(name, s):
    l2 = l2n
    fspk, vspk = s["fspk"], s["vspk"]                                            # verbatim: DQUAL-01/run.py:48-51
    ids = np.intersect1d(np.unique(fspk), np.unique(vspk))
    fi = pd.Series(np.arange(len(fspk))).groupby(fspk).apply(np.array)
    vi = pd.Series(np.arange(len(vspk))).groupby(vspk).apply(np.array)

    # G3: consistency with the identity's own centroid                          # verbatim: DQUAL-01/run.py:68-72
    def cons(X, idx):
        c = l2(np.stack([X[a].mean(0) for a in idx.values]))
        return pd.Series([float(np.mean(X[a] @ c[k])) for k, a in enumerate(idx.values)], index=idx.index)
    cf, cv = cons(s["F"], fi), cons(s["V"], vi)
    T = pd.DataFrame(dict(source=name, spk=ids, n_face=[len(fi[i]) for i in ids], n_voice=[len(vi[i]) for i in ids],
                          cons_face=cf.reindex(ids).values, cons_voice=cv.reindex(ids).values))   # DQUAL-01/run.py:87-91 (G3 columns)
    T["G3"] = (T.cons_face >= T.cons_face.quantile(.05)) & (T.cons_voice >= T.cons_voice.quantile(.05))   # verbatim: DQUAL-01/run.py:94
    return T


def digest(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vox2-arrays", required=True, type=Path, help="FLAG_41 feats_vox2t3: face_{farl16,clipL}.npy, voice_{sam100p,sam100f}.npy")
    ap.add_argument("--vox2-keys", required=True, type=Path, help="FLAG_37 key lists: face_meta.csv, voice_meta.csv (flag2027-vox2-keys)")
    ap.add_argument("--vox2-lid-summary", required=True, type=Path, help="FLAG_42 lid_summary.json (VOX_LID)")
    ap.add_argument("--vb", required=True, type=Path, help="FLAG_46 feats_vb: face_meta.csv, voice_meta.csv and the four T3 arrays")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expected", type=Path, default=None, help="expected.json of the stage (counts + sha256 of the sorted G3 lists)")
    ap.add_argument("--allow-mismatch", action="store_true", help="do not fail when the G3 lists differ from persons.parquet")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    S = sources(a)
    rows = []
    for name, s in S.items():
        T = g3_table(name, s)
        rows.append(T)
        print(name, "identities", len(T), "| G3 fails", int((~T.G3).sum()), "| q05 face / voice",
              round(float(T.cons_face.quantile(.05)), 6), round(float(T.cons_voice.quantile(.05)), 6), flush=True)
    P = pd.concat(rows, ignore_index=True)
    P.to_parquet(a.out / "persons_g3.parquet", index=False)
    rep = {}
    for name in ("VoxCeleb2", "VoxBlink"):
        t = P[P.source == name]
        rep[name] = dict(n_ids=int(len(t)), n_g3_fail=int((t.G3 == False).sum()), sha256_sorted_ids=digest(t.spk),
                         sha256_sorted_g3_fail=digest(t.spk[t.G3 == False]),
                         q05_face=float(t.cons_face.quantile(.05)), q05_voice=float(t.cons_voice.quantile(.05)))
    ok = None
    if a.expected is not None:
        E = json.loads(a.expected.read_text())["persons_g3"]
        ok = {n: all(rep[n][k] == E[n][k] for k in ("n_ids", "n_g3_fail", "sha256_sorted_ids", "sha256_sorted_g3_fail")) for n in rep}
        rep["match_persons_parquet"] = ok
    (a.out / "dqual_g3_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    if ok is not None and not all(ok.values()):
        msg = "G3 lists differ from Experiment/DQUAL-01/persons.parquet (lid_strict.json would change)"
        if not a.allow_mismatch:
            sys.exit("MISMATCH: " + msg)
        print("WARNING:", msg, flush=True)


if __name__ == "__main__":
    main()
