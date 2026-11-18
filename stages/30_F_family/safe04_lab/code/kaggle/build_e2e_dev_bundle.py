r"""Build kaggle/flag2027-e2e-dev.zip -> Kaggle dataset `flag2027-e2e-dev` (private): everything the end-to-end experiment
notebooks (FLAG_51 / 52 / 53, kaggle/build_e2e_lab.py) need besides `flag2027-lab-data` + `flag2027-eda-bundle` to turn a new
T3-side stream into a dev submission.zip, without re-extracting dev features (docs/PLAN_E2E_2026-10-10.md).

  e2edev/e2e_manifest.json
  e2edev/components_dev.npz     per dev file i (zip order of flag_lib.NAMES): pair ids, v11 Bangla inputs (6 B3 streams +
                                7 CA-FUSE conditions), v9 English inputs (7 CA-EN streams + 6 conditions), the 3 T3 streams of
                                v11's final head, the v11 / v9 scores (higher = same), raw dev T3 features + w2v2 gender probs
  e2edev/fusers/*.json          v11 Bangla fuser, v9 Bangla / English fusers, v12 English fuser
  e2edev/ref/*.zip              the submitted v9 / v11 / v12 zips
  e2edev/safe04/*.parquet       the SAFE-04 reference tables the in-notebook gate needs (streams, rel, relg, t3enc01f, bigb01f,
                                cafuse_en2; stack02_rows reduced to the face / voice person columns)

Every component is checked here: v11 Bangla re-scored from (B3, CO, T3) with the v11 fuser and v9 English from (EN7, ENCO)
with fuser_en.json must equal the submitted zips (max |diff| < 1e-5). The FLAG_50 dev run must reproduce this file.
    VOX_LID=... python build_e2e_dev_bundle.py          (CPU, about 10 min: the v6 members and B3Scorer on the dev files)
"""
import io, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd

R0 = Path(__file__).resolve().parents[1]
OUT = R0 / "kaggle/flag2027-e2e-dev.zip"
SCR = Path(r"C:/Users/ASUS/AppData/Local/Temp/claude/d--Sinh-vi-n-CNhan-FLAG-2027-FourFusion/969f2c71-effc-43ec-a610-c253be9d1ea8/scratchpad")
FEATS = [R0 / "kaggle/output/feats_eval", R0 / "kaggle/output/feats_av", R0 / "kaggle/output/feats_models"]
EXTD = [SCR / "in/eda_bundle", R0 / "kaggle/output/feats_models"]
T3D = R0 / "Experiment/T3ENC-01/feats_t3enc"
ZIPS = {"v9": R0 / "Experiment/CAFUSE-EN/submission_CA_EN_BN.zip", "v11": R0 / "Experiment/BIGB-01/submission_v11.zip",
        "v12": R0 / "Experiment/EN-T3-01/submission_v12.zip"}
FUSERS = {"v11_bangla.json": R0 / "Experiment/BIGB-01/cafuse_t3_fuser_v11.json", "v9_bangla.json": R0 / "BEST/v7_safe/cafuse_fuser.json",
          "v9_english.json": R0 / "BEST/v7_safe/fuser_en.json", "v12_english.json": R0 / "Experiment/EN-T3-01/fuser_en_t3.json"}
T3NPZ = [R0 / "Experiment/BIGB-01/dev_t3_streams_v11.npz", R0 / "Experiment/EN-T3-01/dev_t3_streams_en.npz"]
SAFE = SCR / "out/safe04"
B3_COLS = ["deep", "cca", "ib", "age", "rel", "relg"]
CO_COLS = ["is_en", "pf_m", "pv_m", "gdiff", "agef", "agev", "agegap"]
EN7_COLS = ["s007_deep", "s007_cca", "s010B_deep", "s010B_cca", "ib", "age", "acl"]
ENCO_COLS = ["pf_m", "pv_m", "gdiff", "agef", "agev", "agegap"]
T3_COLS = ["anP512_deep", "anP512_cca", "t3_relg"]
T3_RAW = [("face", "farl16"), ("face", "clipL"), ("voice", "sam100p"), ("voice", "sam100f")]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def components():
    sys.path.insert(0, str(R0 / "Experiment/E2E"))
    import system_v11 as SY
    fus = {k: json.loads(p.read_text()) for k, p in FUSERS.items()}
    sy = SY.System(R0 / "kaggle_upload", FEATS, EXTD, R0 / "BEST/v7_safe/members/s007_xall.pt", R0 / "BEST/v6_eval/_members",
                   fus["v9_bangla.json"], threads=10)
    read = lambda zp: {k: pd.read_csv(zipfile.ZipFile(zp).open(fn), sep=" ", header=None, names=["pid", "s"], dtype={"pid": str})
                       for k, fn in sy.V.NAMES.items()}
    ref = {k: read(z) for k, z in ZIPS.items()}
    t3 = {}
    for f in T3NPZ:
        z = np.load(f)
        t3.update({k: z[k] for k in z.files if k not in ("streams", "info")})
    F = dict(v9_bangla=fus["v9_bangla.json"], v9_english=fus["v9_english.json"], v11_bangla=fus["v11_bangla.json"])
    comp, checks, files = {}, {}, []
    for i, (c, k) in enumerate(sy.cells):
        tag = "_".join(k)
        d = sy.file_components(c, k)
        for name, r in ref.items():
            assert (r[k].pid.values == d["pid"]).all(), (name, c)
        d["t3"] = t3[tag].astype(np.float64)
        v9, v11 = SY.scores(d, F)
        r9, r11 = -ref["v9"][k].s.values.astype(np.float64), -ref["v11"][k].s.values.astype(np.float64)
        chk = dict(v9_reproduced=float(np.abs(v9 - r9).max()), v11_reproduced=float(np.abs(v11 - r11).max()),
                   level0_half_file_max_abs_diff=sy.selftest(c, k))
        if d["is_en"]:
            chk["v12_english_reproduced"] = float(np.abs(SY.apply(fus["v12_english.json"], np.concatenate([d["en7"], d["t3"]], 1), d["enco"])
                                                         + ref["v12"][k].s.values).max())
        assert all(v < 1e-5 for v in chk.values()), (c, chk)
        d.update(v11=r11, v9=r9)                            # the submitted scores (rounded to 6 decimals in the zips)
        for kind, nm in T3_RAW:
            d[f"{kind}_{nm}"] = np.load(T3D / f"{kind}_{nm}_{tag}.npy").astype(np.float32)
            assert len(d[f"{kind}_{nm}"]) == len(d["pid"]), (c, nm)
        comp[i] = d
        files.append(dict(i=i, key=c, zip_name=d["zip_name"], protocol=k[0], lang=k[1], is_en=d["is_en"], rows=len(d["pid"])))
        checks[c] = chk
        log(f"{c}: {len(d['pid'])} trials | checks {chk}")
    return comp, files, checks, SY


def safe04_tables():
    out = {}
    for s in range(1, 6):
        out[f"streams_fold{s}.parquet"] = pd.read_parquet(SAFE / f"streams_fold{s}.parquet")
        out[f"relonly_multi_fold{s}.parquet"] = pd.read_parquet(SAFE / f"relonly_multi_fold{s}.parquet")[["relc_fia_eb"]]
        out[f"relonly_g_fold{s}.parquet"] = pd.read_parquet(SAFE / f"relonly_g_fold{s}.parquet")[["relg_fia_eb"]]
        for t in ("t3enc01f", "bigb01f", "cafuse_en2"):
            out[f"{t}_fold{s}.parquet"] = pd.read_parquet(SAFE / f"{t}_fold{s}.parquet")
    out["stack02_rows.parquet"] = pd.read_parquet(SAFE / "stack02_rows.parquet")[["fold", "data", "y", "fp", "vp"]]
    return out


def main():
    assert os.environ.get("VOX_LID"), "the components carry v11's Bengali-filtered T3 head: set VOX_LID like build_v11.py"
    t0 = time.time()
    comp, files, checks, SY = components()
    man = dict(created=time.strftime("%Y-%m-%d %H:%M"), kind="e2e-dev", split="dev", files=files, checks=checks,
               columns=dict(b3=B3_COLS, co=CO_COLS, en7=EN7_COLS, enco=ENCO_COLS, t3=T3_COLS, t3_raw=[f"{a}_{b}" for a, b in T3_RAW]),
               note="FourFusion private: components of the dev files (organiser data); do not redistribute.")
    tabs = safe04_tables()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        b = io.BytesIO(); SY.save_components(b, comp); z.writestr("e2edev/components_dev.npz", b.getvalue())
        for name, p in FUSERS.items():
            z.write(p, f"e2edev/fusers/{name}")
        for name, p in ZIPS.items():
            z.write(p, f"e2edev/ref/submission_{name}.zip")
        for name, df in tabs.items():
            b = io.BytesIO(); df.to_parquet(b); z.writestr(f"e2edev/safe04/{name}", b.getvalue())
        z.writestr("e2edev/safe04/fold_persons.json", (R0 / "Experiment/STACK-02/fold_persons.json").read_text())
        man["safe04_tables"] = sorted(tabs)
        z.writestr("e2edev/e2e_manifest.json", json.dumps(man, indent=1))
    log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
