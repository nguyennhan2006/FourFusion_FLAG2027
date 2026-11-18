"""FVBIND-01 gate 2, step 1 (14/11, README section 3 + section 3b registered before any gate-2 number): block v_fvb = B2 voice embedding
(ImageBind audio + B2 LoRA + linear head, state fvbB2_state.pt of FLAG_189) minus mu_v (mean over aux training clips, stored in the state:
a constant from external data). Rows:
  every voice row of the RFS bundle (aux = audio-cache centre 3 s clip, 3 x 2 s chunks as in B2 training; MAV = whole file, v1 / v2 roots
  resolved separately), every v4 voice row (whole file), every voice of the four Evaluation lists (index_<proto>_<lang>.txt of FLAG_47rs).
Output (layout of rfs_train.XBLK "_AV", read by rfs_train.load_extra and patch_163i_factor):
  vox/voice_fvb.npy (every bundle row with v_src vox = FLAG_37 + FLAG_53 rows), vox2b/voice_fvb.npy (0 rows), vb/voice_fvb.npy,
  ext_v1_complete_voice_fvb.npy, ext_v2_complete_voice_fvb.npy, voice_fvb_train.npy, eval/voice_fvb_<name>.npy + eval/index_<name>.txt.
Wiring check (before anything else): the first 32 v4 rows reproduce fvbB2_emb.npz (gate-1 embeddings), cos >= 0.999.
Smoke: FLAG_SMOKE=1 (synthetic bundle + waves, stand-in model); FVB_STANDIN=1 (real data, stand-in model, FVB_STANDIN_N rows per part)."""
import argparse, os, shutil, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fvbind as FV

R, log, SMOKE, STANDIN = FV.R, FV.R.log, FV.SMOKE, FV.STANDIN
NS = int(os.environ.get("FVB_STANDIN_N", 200))


def mav_roots(base):
    """version folder (v1 / v2 holding faces + voices) -> its parent; every MAV v_path is '<ver>/voices/...'."""
    roots = {}
    for d, dirs, files in os.walk(base):
        p = Path(d)
        if p.name in ("v1", "v2") and {"faces", "voices"} <= set(dirs):
            roots.setdefault(p.name, p.parent)
        dirs[:] = [x for x in dirs if x not in ("faces", "voices")] if len(p.relative_to(base).parts) < 7 else []
    return roots


EVAL_RE = __import__("re").compile(r"index_(gender|no_gender)_(English|Bangla)\.txt")


def do_eval(a, out, files):
    """Evaluation lists (index of FLAG_47rs: pair_id voice face): voice path matched by suffix INSIDE the list's protocol folder
    (the same <lang>_test/voices/<n>.wav exists under gender/ and no_gender/); index_train.txt is not an Evaluation list."""
    ev = out / "eval"; ev.mkdir(exist_ok=True)
    idxs = sorted({p for p in Path(a.evidx).rglob("index_*.txt") if EVAL_RE.fullmatch(p.name)}, key=lambda p: p.name)
    names = sorted({p.name for p in idxs}); assert len(names) == 4, f"Evaluation index files {names}"
    byname = {}
    for p in Path(a.evroot).rglob("*"):
        if p.is_file() and p.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a"): byname.setdefault(p.name, []).append(p)
    for nm in names:
        proto = EVAL_RE.fullmatch(nm).group(1)
        q = [p for p in idxs if p.name == nm][0]; rows = [l.split() for l in q.read_text().splitlines() if l.strip()]
        if STANDIN: rows = rows[:NS]
        paths = []
        for r in rows:
            c = [p for p in byname.get(Path(r[1]).name, []) if p.as_posix().endswith(r[1].replace("\\", "/")) and f"/{proto}/" in p.as_posix()]
            assert len(c) == 1, f"{nm}: voice {r[1]} -> {c}"
            paths.append(c[0])
        u = sorted(set(paths)); Y = dict(zip(u, files(u)))
        A = np.stack([Y[p] for p in paths]).astype(np.float32); assert np.isfinite(A).all(), nm
        tag = nm[len("index_"):-len(".txt")]
        np.save(ev / f"voice_fvb_{tag}.npy", A)
        if STANDIN: (ev / nm).write_text("\n".join(" ".join(r) for r in rows) + "\n")
        else: shutil.copy(q, ev / nm)
        log(f"  eval {tag} ({proto} folder): {len(rows)} pairs, {len(u)} voices, e.g. {paths[0]}")


def main():
    ap = argparse.ArgumentParser()
    for k in ("bundle", "cache", "mavbase", "v4root", "state", "emb", "ib_ckpt", "out", "evidx", "evroot"): ap.add_argument("--" + k, default="")
    a = ap.parse_args()
    C = R.core(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t00 = time.time()
    E, V4 = R.load_bundle(a.bundle)
    src = np.asarray(E["v_src"]).astype(str); n = len(src)

    # ---- model + B2 state
    ib, MT = FV.load_model(a.ib_ckpt); params = C.add_lora(ib, MT, 8, 16)
    if STANDIN:
        D = 1280; head = nn.Linear(1024, D).to(C.DEVICE); mu_v = np.zeros(D, np.float32); log("!! stand-in model / head (smoke)")
    else:
        st = torch.load(a.state, map_location="cpu", weights_only=False)
        assert len(st["params"]) == len(params), (len(st["params"]), len(params))
        for p, q in zip(params, st["params"]):
            assert p.shape == q.shape; p.data.copy_(q.to(p.device, p.dtype))
        D = st["head"]["weight"].shape[0]; head = nn.Linear(1024, D).to(C.DEVICE); head.load_state_dict(st["head"])
        mu_v = np.asarray(st["mu_v"], np.float32); log(f"B2 state: {len(params)} LoRA tensors, head 1024 -> {D}, info {st.get('info')}")
    ib.eval(); head.eval()

    def post(X):   # 1024 ImageBind -> head -> minus mu_v; NaN rows stay NaN
        X = np.asarray(X, np.float32); ok = np.isfinite(X).all(1)
        with torch.no_grad():
            Y = head(torch.tensor(np.nan_to_num(X), device=C.DEVICE)).float().cpu().numpy()
        Y = Y - mu_v; Y[~ok] = np.nan
        return Y

    def files(paths):
        if SMOKE:
            with torch.no_grad():
                return post(np.stack([ib({MT.AUDIO: C.ib_audio_clips(FV.synth_wave(p))[None].to(C.DEVICE)})[MT.AUDIO].float().cpu().numpy()[0]
                                      for p in paths]))
        return post(C.embed_audio(ib, MT, [str(p) for p in paths]))

    # ---- wiring check: B2 state reproduces the gate-1 v4 embeddings
    if not STANDIN:
        Z = np.load(a.emb); r32 = [int(x) for x in Z["v4rows"][:32]]
        got = files([Path(a.v4root) / V4["v_path"][r] for r in r32]) + mu_v; ref = Z["e4"][:32]
        cs = (R.U(got) * R.U(np.asarray(ref, np.float32))).sum(1)
        log(f"wiring (state -> fvbB2_emb.npz e4, 32 v4 rows): cos mean {cs.mean():.5f} min {cs.min():.5f}"); assert cs.min() >= 0.999, "wiring"
    if os.environ.get("FVB_ONLY") == "eval":   # 15/11: Evaluation lists only (the bundle part of the first FLAG_192 run stopped at this stage)
        do_eval(a, out, files); log(f"FVB extract (eval only) done [{time.time() - t00:.0f}s]"); return

    X = np.full((n, D), np.nan, np.float32)
    # ---- aux rows: audio cache centre 3 s clips, 3 x 2 s chunks (B2 training input), batches of 64
    aux = np.nonzero(np.isin(src, ["vox", "vb"]))[0]
    if STANDIN: aux = np.concatenate([np.nonzero(src == s)[0][:NS] for s in ("vox", "vb")])
    t0 = time.time(); waves = {}
    if SMOKE:
        waves = {int(i): FV.synth_wave(("v", int(i))) for i in aux}
    else:
        idx = {}
        for sub in ("vox", "vb"):
            for p in sorted((Path(a.cache) / sub).glob("*.npz")):
                Zc = np.load(p)
                for r, k in enumerate(zip(Zc["spk"].astype(str), Zc["video"].astype(str), Zc["utt"].astype(str))): idx[(sub, *k)] = (p, r)
        by = {}
        for i in aux:
            k = ("vb" if src[i] == "vb" else "vox", str(E["v_spk"][i]), str(E["v_vid"][i]).split("/", 1)[-1], str(E["v_utt"][i]))
            if k in idx: by.setdefault(idx[k][0], []).append((int(i), idx[k][1]))
        log(f"aux: {sum(len(v) for v in by.values())} of {len(aux)} rows in the audio cache ({len(by)} cache files) [{time.time() - t0:.0f}s]")
    done = 0
    def run_batch(rows, ws):
        fb = torch.stack([C.ib_audio_clips(w)[:3] for w in ws])
        with torch.no_grad():
            e = ib({MT.AUDIO: fb.half().to(C.DEVICE)})[MT.AUDIO].float().cpu().numpy()   # as core.embed_audio
        X[rows] = post(e)
    items = sorted(waves.items()) if SMOKE else None
    if SMOKE:
        for lo in range(0, len(items), 64):
            run_batch([i for i, _ in items[lo:lo + 64]], [w for _, w in items[lo:lo + 64]])
    else:
        for p, its in by.items():
            Zc = np.load(p); W = Zc["wav"]; nn_ = Zc["n"] if "n" in Zc.files else None
            ws = [(i, W[r][:max(int(nn_[r]) if nn_ is not None else W.shape[1], 8000)].astype(np.float32) / 32768.0) for i, r in its]
            for lo in range(0, len(ws), 64):
                run_batch([i for i, _ in ws[lo:lo + 64]], [w for _, w in ws[lo:lo + 64]])
            done += len(its); log(f"  aux {done}/{len(aux)} [{time.time() - t0:.0f}s]")
    log(f"aux done: finite {np.isfinite(X[aux]).all(1).mean():.4f} [{time.time() - t0:.0f}s]")

    # ---- MAV rows: whole file, v1 / v2 roots resolved separately
    mav_all = np.nonzero(src == "mav")[0]; vp = np.asarray(E["v_path"]).astype(str)
    if SMOKE:
        ver_all = np.where(np.arange(len(mav_all)) < len(mav_all) // 2, "v1", "v2")
    else:
        ver_all = np.array([q.replace("\\", "/").split("/")[0] for q in vp[mav_all]])
    assert set(ver_all) <= {"v1", "v2"}, f"MAV v_path does not start with v1 / v2: {vp[mav_all][:3]}"
    nv1 = int((ver_all == "v1").sum()); assert (ver_all[:nv1] == "v1").all(), "bundle MAV rows must be v1 then v2"
    VER = dict(zip(mav_all.tolist(), ver_all.tolist()))
    mav = np.concatenate([mav_all[:NS], mav_all[nv1:nv1 + NS]]) if STANDIN else mav_all
    if SMOKE:
        X[mav] = files([("m", int(i)) for i in mav])
    else:
        roots = mav_roots(Path(a.mavbase)); log(f"MAV roots {roots}"); assert {"v1", "v2"} <= set(roots), f"MAV roots {roots}: attach v1 and v2"
        t0 = time.time()
        for lo in range(0, len(mav), 2000):
            rr = mav[lo:lo + 2000]
            X[rr] = files([roots[VER[int(i)]] / vp[i] for i in rr])
            log(f"  mav {lo + len(rr)}/{len(mav)} finite {np.isfinite(X[mav[:lo + len(rr)]]).all(1).mean():.4f} [{time.time() - t0:.0f}s]")
    vm = np.array([VER[int(i)] for i in mav])
    cov = {s: float(np.isfinite(X[mav[vm == s]]).all(1).mean()) for s in ("v1", "v2")}
    log(f"MAV coverage {cov}")
    if not STANDIN: assert min(cov.values()) >= 0.99, f"MAV coverage {cov} < 0.99"

    # ---- v4 rows: whole file, must be complete
    n4 = len(V4["v_path"]); r4 = list(range(n4 if not STANDIN else min(n4, NS)))
    X4 = np.full((n4, D), np.nan, np.float32)
    X4[r4] = files([("4", r) for r in r4]) if SMOKE else files([Path(a.v4root) / V4["v_path"][r] for r in r4])
    log(f"v4: finite {np.isfinite(X4).all(1).mean():.4f}")
    if not STANDIN: assert np.isfinite(X4).all(), "v4 rows must be complete"

    # ---- write the bundle layout
    a1 = (src == "vox").sum(); a2 = a1 + (src == "vb").sum()
    if not SMOKE:   # the real bundle is vox (FLAG_37 + FLAG_53), vb, mav in blocks (rfs_train.load_extra asserts the same)
        assert (src[:a1] == "vox").all() and (src[a1:a2] == "vb").all() and (src[a2:] == "mav").all(), "bundle source blocks"
    sel = {"vox": np.nonzero(src == "vox")[0], "vox2b": np.zeros(0, int), "vb": np.nonzero(src == "vb")[0], "mav": mav_all}
    for sub in ("vox", "vox2b", "vb"):
        (out / sub).mkdir(exist_ok=True); np.save(out / sub / "voice_fvb.npy", X[sel[sub]].astype(np.float16))
    np.save(out / "ext_v1_complete_voice_fvb.npy", X[mav_all[:nv1]].astype(np.float16))
    np.save(out / "ext_v2_complete_voice_fvb.npy", X[mav_all[nv1:]].astype(np.float16))
    np.save(out / "voice_fvb_train.npy", X4.astype(np.float32))
    for nm, sl in (("vox+vox2b", sel["vox"]), ("vb", sel["vb"]), ("mav", mav_all)):
        k = np.isfinite(X[sl]).all(1)
        log(f"  {nm}: rows {k.size}, finite {k.mean():.4f}, identity EER {R._id_eer(X[sl], np.asarray(E['v_spk'])[sl].astype(str), np.asarray(E['v_vid'])[sl].astype(str)):.2f}")

    if a.evidx and not SMOKE:
        do_eval(a, out, files)
    log(f"FVB extract done [{time.time() - t00:.0f}s]: {sorted(str(p.relative_to(out)) for p in out.rglob('*.np*'))}")


if __name__ == "__main__":
    main()
