"""FVBIND-01 (README registered 14/11 before any number): ImageBind audio tower + LoRA C1 recipe re-trained on our auxiliary persons
(VoxCeleb2 + VoxBlink of the RFS pool, clean_ext / lid_strict; audio cache centre 3 s, BN42-clean) with
  B0  = LoRA C1 recipe (64 persons, 3 x 2 s, 1 200 steps, anchor = per-person ImageBind face),
  B1  = 2 048 persons per step (GradCache), one random 2 s chunk, 600 steps, same anchor,
  B2  = B1 + anchor [C5s | FaRL] and a linear voice head 1024 -> anchor dim,
  IB0 = no LoRA (reference).
Per arm: embeddings of every v4 voice row and of every MAV voice row the SAFE-04 tables use (ImageBind whole-file inference), then the
standalone stream <unit(a_f - mu_f), unit(e_v - mu_v)> on the fold tables -> fvb<arm>_fold<s>.parquet (+ fvb<arm>_state.pt, emb npz).
Lives next to rfs_train.py (imports it). Run by FLAG_189:
    python fvbind.py --bundle B --work W --out O --lid L --tables T --ib_ckpt CK --cache AC --mav MAV --v4root V4 --arms B0,B1
Smoke: FLAG_SMOKE=1 (synthetic bundle + waves, tiny stand-in model); FVB_STANDIN=1 (real data, stand-in model, few persons / rows)."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
if not (Path(__file__).resolve().parent / "rfs_train.py").exists():   # repo layout (local runs): Experiment/RFS-01/train
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RFS-01" / "train"))
import rfs_train as R

SMOKE = os.environ.get("FLAG_SMOKE") == "1"
STANDIN = SMOKE or os.environ.get("FVB_STANDIN") == "1"
log = R.log
ARMS = {"B0": dict(B=64, chunks=3, steps=1200, warm=60, anchor="ibf"),
        "B1": dict(B=2048, chunks=1, steps=600, warm=30, anchor="ibf"),
        "B2": dict(B=2048, chunks=1, steps=600, warm=30, anchor="c5f"),
        "IB0": dict(steps=0, anchor="ibf")}
AUX = ("vox", "vox2b", "vb")


# ------------------------------------------------------------------ stand-in model (smoke only: wiring, not numbers)
class _MT:
    AUDIO = "audio"


class _Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(204, 32); self.att = nn.MultiheadAttention(32, 4, batch_first=True); self.out = nn.Linear(32, 1024)


class TinyIB(nn.Module):
    def __init__(self):
        super().__init__()
        self.modality_trunks = nn.ModuleDict({"audio": _Trunk()})

    def forward(self, d):
        x = d["audio"].float(); n, s = x.shape[:2]; t = self.modality_trunks["audio"]
        h = t.inp(x.reshape(n * s, 128, 204)); h = t.att(h, h, h)[0].mean(1)
        return {"audio": t.out(h).reshape(n, s, -1).mean(1)}


def load_model(ck):
    C = R.core()
    if STANDIN:
        torch.manual_seed(0); return TinyIB().to(C.DEVICE).eval(), _MT
    return C.load_imagebind(ck)


def ckpt_blocks(ib, MT):   # as train_lora: checkpoint every audio block when gradients are on
    for blk in getattr(ib.modality_trunks[MT.AUDIO], "blocks", []):
        if not getattr(blk, "_rfs_ck", False):
            def fw(*a, _f=blk.forward, **k):
                from torch.utils.checkpoint import checkpoint
                return checkpoint(_f, *a, use_reentrant=False, **k) if torch.is_grad_enabled() else _f(*a, **k)
            blk.forward = fw; blk._rfs_ck = True


# ------------------------------------------------------------------ data
def synth_wave(key, n=48000):
    return np.random.default_rng(abs(hash(key)) % (1 << 31)).standard_normal(n).astype(np.float32) * 0.1


def train_clips(E, lid, a, max_per=3, seed=1):
    """aux voice rows kept by clean_ext, <= max_per videos per person, read from the audio cache (vox/, vb/) -> waves, person keys."""
    rng = np.random.default_rng(seed); keep, _ = R.clean_ext(E, lid); byp = {}
    for i in np.nonzero(keep)[0]:
        if E["v_src"][i] in AUX: byp.setdefault((str(E["v_src"][i]), str(E["v_spk"][i])), []).append(i)
    keys = sorted(byp)
    if STANDIN: keys = [keys[j] for j in rng.permutation(len(keys))[:int(os.environ.get("FVB_STANDIN_P", 200))]]
    sel = []
    for k in keys:
        rows = list(byp[k]); rng.shuffle(rows); seen = set()
        for i in rows:
            if E["v_vid"][i] not in seen and len(seen) < max_per: seen.add(E["v_vid"][i]); sel.append(i)
    wavs = {}
    if SMOKE:
        wavs = {i: synth_wave(("v", i)) for i in sel}
    else:
        idx = {}
        for sub in ("vox", "vb"):
            for p in sorted((Path(a.cache) / sub).glob("*.npz")):
                Z = np.load(p)
                for r, k in enumerate(zip(Z["spk"].astype(str), Z["video"].astype(str), Z["utt"].astype(str))): idx[(sub, *k)] = (p, r)
        by_file = {}; miss = []
        for i in sel:   # bundle v_vid = "<spk>/<video>" (assemble.py), the cache keys the bare video id
            k = ("vb" if E["v_src"][i] == "vb" else "vox", str(E["v_spk"][i]), str(E["v_vid"][i]).split("/", 1)[-1], str(E["v_utt"][i]))
            if k in idx: by_file.setdefault(idx[k][0], []).append((i, idx[k][1]))
            else: miss.append(k)
        if len(miss) > 0.2 * len(sel):
            log(f"  cache key check: missing e.g. {miss[:3]}; cache keys e.g. {list(idx)[:2]} / {[k for k in idx if k[0] == 'vb'][:2]}")
        for p, items in by_file.items():
            Z = np.load(p); W = Z["wav"]; n = Z["n"] if "n" in Z.files else None
            for i, r in items: wavs[i] = W[r][:max(int(n[r]) if n is not None else W.shape[1], 8000)].astype(np.float32) / 32768.0
    rows = sorted(wavs); src = {s: f"{sum(E['v_src'][i] == s for i in rows)}/{sum(E['v_src'][i] == s for i in sel)}" for s in AUX}
    log(f"train clips: {len(rows)} of {len(sel)} selected rows found in the cache ({len(keys)} persons); found / selected by source {src}")
    assert len(rows) >= 0.5 * len(sel), "audio cache does not cover the selected rows"   # README section 1: rows outside the cache are dropped
    return rows, [wavs[i] for i in rows]


def person_anchor(E, names, X):
    """per-person mean of unit face rows of X (rows of E), unit again; persons without a finite face -> ok False."""
    out = np.zeros((len(names), X.shape[1]), np.float32); cnt = np.zeros(len(names))
    for s, k, x in zip(E["f_src"], E["f_spk"], X):
        j = names.get((str(s), str(k)))
        if j is not None and np.isfinite(x).all(): out[j] += R.U(x[None])[0]; cnt[j] += 1
    ok = cnt > 0; out[ok] /= cnt[ok, None]
    return R.U(out), ok


def c5f(E_or_V):
    return (np.hstack([R.U(np.nan_to_num(np.asarray(E_or_V["f_c5s"], np.float32))),
                       R.U(np.nan_to_num(np.asarray(E_or_V["f_farl"], np.float32)))]) / np.sqrt(2)).astype(np.float32)


# ------------------------------------------------------------------ training
def train_big(ib, MT, params, FB, pid, anchor, steps, B, warm, head=None, lr=1e-4, tau=0.05, seed=1, mb=256, scale=256.0):
    """GradCache InfoNCE: embeddings of the whole batch without grad -> loss -> grad w.r.t. embeddings -> per-chunk re-forward + backward."""
    C = R.core(); dev = C.DEVICE; rng = np.random.default_rng(seed); torch.manual_seed(seed)
    groups = [{"params": params, "lr": lr, "weight_decay": 0.0}]
    if head is not None: groups.append({"params": list(head.parameters()), "lr": 1e-3, "weight_decay": 1e-4})
    opt = torch.optim.AdamW(groups); base = [g["lr"] for g in opt.param_groups]
    byp = {}
    for i, p in enumerate(pid): byp.setdefault(int(p), []).append(i)
    people = np.array(sorted(byp)); B = min(B, len(people)); A = torch.tensor(anchor, dtype=torch.float32, device=dev)
    ckpt_blocks(ib, MT); t0 = time.time(); skipped = 0; dts = []; t = 0
    X = lambda ci, ch: torch.from_numpy(np.ascontiguousarray(FB[ci, ch]))[:, None].to(dev)      # (n, 1 clip, 1, 128, 204)
    while t < steps:
        ts = time.time()
        for g, b in zip(opt.param_groups, base): g["lr"] = b * min(1.0, (t + 1) / warm)
        ps = rng.choice(people, B, replace=False); ci = np.array([byp[int(p)][rng.integers(len(byp[int(p)]))] for p in ps])
        ch = rng.integers(0, FB.shape[1], B)
        with torch.no_grad():
            emb = torch.cat([ib({MT.AUDIO: X(ci[lo:lo + 2 * mb], ch[lo:lo + 2 * mb])})[MT.AUDIO].float() for lo in range(0, B, 2 * mb)])
        emb.requires_grad_(True); opt.zero_grad()
        loss = R.nce(A[torch.as_tensor(ps, device=dev)], head(emb) if head is not None else emb, tau); loss.backward()
        ge = emb.grad.detach() * scale
        for lo in range(0, B, mb):
            a = ib({MT.AUDIO: X(ci[lo:lo + mb], ch[lo:lo + mb])})[MT.AUDIO].float(); (a * ge[lo:lo + mb]).sum().backward()
        if all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):
            for p in params: p.grad /= scale
            opt.step()
        else:
            skipped += 1
        dts.append(time.time() - ts)
        if t == 10:                                                     # registered time rule (README section 2)
            tm = float(np.mean(dts[1:11]))
            if tm > 30: steps = max(200, int(5 * 3600 / tm)); log(f"  time rule: {tm:.1f} s/step > 30 s -> {steps} steps")
            else: log(f"  time rule: {tm:.1f} s/step <= 30 s -> keep {steps} steps")
        if t % 25 == 0 or t == steps - 1:
            log(f"  big step {t}/{steps} loss {loss.item():.4f} skipped {skipped} [{time.time() - t0:.0f}s, last {dts[-1]:.1f}s]")
        t += 1
    return dict(steps=steps, skipped=skipped, sec_per_step=float(np.mean(dts[1:])) if len(dts) > 1 else None)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    for k in ("bundle", "work", "out", "lid", "tables", "ib_ckpt", "cache", "mav", "v4root"): ap.add_argument("--" + k, default="")
    ap.add_argument("--arms", default="B0,B1"); a = ap.parse_args()
    import pandas as pd
    C = R.core(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t00 = time.time()
    arms = [x for x in a.arms.split(",") if x]; assert set(arms) <= set(ARMS), arms
    E, V4 = R.load_bundle(a.bundle, a.work); lid = json.loads(Path(a.lid).read_text())
    need_c5 = any(ARMS[x]["anchor"] == "c5f" for x in arms)
    if not SMOKE:
        R.load_extra(E, V4, (["f_c5s", "f_farl"] if need_c5 else []) + ["v_ibv"])
    elif need_c5:                                                      # synthetic: stand-in blocks
        for D in (E, V4): D["f_c5s"], D["f_farl"] = D["f_clip"], D["f_ibf"]
    AF = {"ibf": (np.asarray(E["f_ibf"], np.float32), np.asarray(V4["f_ibf"], np.float32))}
    if need_c5: AF["c5f"] = (c5f(E), c5f(V4))

    # SAFE-04 rows: v4 voice rows (all) + the MAV voice rows of the fold tables
    tabs = sorted(Path(a.tables).rglob(os.environ.get("RFS_LAB_GLOB", "ct01r2_fold*.parquet")))
    if os.environ.get("RFS_LAB_FOLDS"): tabs = [p for p in tabs if int(p.stem[-1]) in {int(x) for x in os.environ["RFS_LAB_FOLDS"].split(",")}]
    mf, mv = np.nonzero(E["f_src"] == "mav")[0], np.nonzero(E["v_src"] == "mav")[0]
    T = {p.stem[-1]: pd.read_parquet(p)[["fold", "data", "protocol", "y", "fi", "vi"]].reset_index(drop=True) for p in tabs}
    mrows = sorted({int(mv[int(v)]) for d in T.values() for v in d.vi[d.data != "v4"]})
    v4rows = list(range(len(V4["v_path"])))
    if STANDIN and not SMOKE: mrows, v4rows = mrows[:int(os.environ.get("FVB_STANDIN_N", 300))], v4rows[:int(os.environ.get("FVB_STANDIN_N", 300))]
    log(f"SAFE-04 tables {[p.name for p in tabs]}; eval rows: v4 {len(v4rows)}, MAV {len(mrows)}")
    if SMOKE:
        def embed(ib, MT, rows, kind):
            with torch.no_grad():
                return np.stack([ib({MT.AUDIO: C.ib_audio_clips(synth_wave((kind, r)))[None].to(C.DEVICE)})[MT.AUDIO].float().cpu().numpy()[0] for r in rows])
    else:
        def embed(ib, MT, rows, kind):
            ps = [str(Path(a.v4root) / V4["v_path"][r]) for r in rows] if kind == "v4" else [str(Path(a.mav) / E["v_path"][r]) for r in rows]
            return C.embed_audio(ib, MT, ps).astype(np.float32)

    rows, wavs = train_clips(E, lid, a) if any(ARMS[x]["steps"] for x in arms) else ([], [])
    names = {}; pid = np.array([names.setdefault((str(E["v_src"][i]), str(E["v_spk"][i])), len(names)) for i in rows], int)
    FB = None
    if rows:
        t0 = time.time(); FB = np.zeros((len(rows), 3, 1, 128, 204), np.float16)
        for j, w in enumerate(wavs):
            FB[j] = C.ib_audio_clips(w)[:3].numpy().astype(np.float16)
            if j % 5000 == 0: log(f"  fbank {j}/{len(rows)} [{time.time() - t0:.0f}s]")
        del wavs; log(f"fbanks {FB.shape} ({FB.nbytes / 2**30:.1f} GB) [{time.time() - t0:.0f}s]")
    summ = {}
    for arm in arms:
        cfg = ARMS[arm]; t0 = time.time(); log(f"== arm {arm} {cfg}")
        ib, MT = load_model(a.ib_ckpt); head = None; info = {}; anc = None
        if FB is not None:
            anc_all, ok = person_anchor(E, names, AF[cfg["anchor"]][0]); remap = -np.ones(len(names), int); remap[ok] = np.arange(ok.sum())
            m = remap[pid] >= 0; idx = np.nonzero(m)[0]; P = remap[pid][m]; anc = anc_all[ok]
            log(f"  {len(idx)} clips with an anchor, {len(anc)} persons, anchor dim {anc.shape[1]}")
        if cfg["steps"]:
            params = C.add_lora(ib, MT, 8, 16)
            if not STANDIN:                                            # registered wiring check: LoRA = 0 reproduces the stored v_ibv
                ref = V4["v_ibv"][:32] if "v_ibv" in V4 else E["v_ibv"][mv[:32]]
                got = embed(ib, MT, list(range(32)), "v4") if "v_ibv" in V4 else embed(ib, MT, list(mv[:32]), "mav")
                cs = (R.U(np.asarray(got, np.float32)) * R.U(np.asarray(ref, np.float32))).sum(1)
                log(f"  wiring (LoRA = 0 vs stored v_ibv, 32 rows): cos mean {cs.mean():.5f} min {cs.min():.5f}"); assert cs.min() >= 0.99, "wiring check failed"
                info["wiring_cos_min"] = float(cs.min())
            steps = int(os.environ.get("FVB_STEPS", cfg["steps"]))
            if cfg["anchor"] == "c5f":
                head = nn.Linear(1024, anc.shape[1]).to(C.DEVICE)
            if arm == "B0":
                sk = R.train_lora(ib, MT, params, [torch.from_numpy(FB[i]) for i in idx], P, anc, steps, B=cfg["B"], warm=cfg["warm"])
                info.update(steps=steps, skipped=int(sk))
            else:
                info.update(train_big(ib, MT, params, FB[idx], P, anc, steps, B=cfg["B"], warm=cfg["warm"], head=head))
            info.update(clips=int(len(idx)), persons=int(len(anc)))
        # mu_f / mu_v: means over the aux training persons / 2 000 aux training clips (3 chunks): constants from external data
        assert anc is not None, "no aux training clips: run IB0 together with a trained arm"
        mu_f = anc.mean(0)
        with torch.no_grad():
            sel = np.random.default_rng(7).permutation(len(idx))[:2000]
            ev = torch.cat([ib({MT.AUDIO: torch.from_numpy(FB[idx[sel[lo:lo + 64]]]).to(C.DEVICE)})[MT.AUDIO].float() for lo in range(0, len(sel), 64)])
            mu_v = (head(ev) if head is not None else ev).cpu().numpy().mean(0)
        if cfg["steps"]:
            torch.save({"params": [p.data.detach().cpu() for p in params], "head": head.state_dict() if head is not None else None,
                        "mu_f": mu_f, "mu_v": mu_v, "info": info, "cfg": cfg}, out / f"fvb{arm}_state.pt")
        e4, em = embed(ib, MT, v4rows, "v4"), embed(ib, MT, mrows, "mav")
        if head is not None:
            with torch.no_grad():
                hf = lambda X: head(torch.tensor(np.nan_to_num(X), device=C.DEVICE)).cpu().numpy() * np.where(np.isfinite(X).all(1, keepdims=True), 1, np.nan)
                e4, em = hf(e4), hf(em)
        np.savez(out / f"fvb{arm}_emb.npz", v4rows=np.array(v4rows), e4=e4, mrows=np.array(mrows), em=em, mu_f=mu_f, mu_v=mu_v)
        V4E = dict(zip(v4rows, e4)); MVE = dict(zip(mrows, em)); F4, FE = AF[cfg["anchor"]][1], AF[cfg["anchor"]][0]
        for s, D in T.items():
            m4 = (D.data == "v4").to_numpy(); fi, vi = D.fi.to_numpy().astype(int), D.vi.to_numpy().astype(int)
            fa_ = np.where(m4[:, None], F4[np.where(m4, fi, 0)], FE[mf[np.where(m4, 0, fi)]])
            nanv = np.full(1024 if head is None else len(mu_v), np.nan, np.float32)
            va = np.stack([V4E.get(int(v), nanv) if q else MVE.get(int(mv[v]), nanv) for v, q in zip(vi, m4)])
            sc = (R.U(fa_ - mu_f) * R.U(va - mu_v)).sum(1).astype(np.float32); sc[~np.isfinite(va).all(1)] = np.nan
            o = D.copy(); o[f"fvb{arm}"] = sc; o.to_parquet(out / f"fvb{arm}_fold{s}.parquet")
            for (d, pr), g in o.groupby(["data", "protocol"]):
                x = g[f"fvb{arm}"].to_numpy(); k = np.isfinite(x)
                if k.sum() > 10 and len(set(g.y.to_numpy()[k])) == 2:
                    log(f"  fold {s} {d:6s} {pr:9s}: n {len(g)} finite {k.mean():.3f} EER {R.eer(x[k], g.y.to_numpy()[k]):.2f}")
        summ[arm] = dict(info, sec=round(time.time() - t0)); log(f"arm {arm} done [{time.time() - t0:.0f}s]")
        del ib; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if "v_lora" in V4:                                                 # reference: the existing LoRA C1 (v4 rows only)
        mf4, ml = np.nanmean(AF["ibf"][1], 0), np.nanmean(np.asarray(V4["v_lora"], np.float32), 0)
        for s, D in T.items():
            m4 = (D.data == "v4").to_numpy(); o = D.copy(); o["fvbC1ref"] = np.nan
            fi, vi = D.fi.to_numpy()[m4].astype(int), D.vi.to_numpy()[m4].astype(int)
            o.loc[m4, "fvbC1ref"] = (R.U(AF["ibf"][1][fi] - mf4) * R.U(np.asarray(V4["v_lora"], np.float32)[vi] - ml)).sum(1)
            o.to_parquet(out / f"fvbC1ref_fold{s}.parquet")
    (out / f"fvb_summary_{'_'.join(arms)}.json").write_text(json.dumps(summ, indent=1, default=str))
    log(f"FVBIND done {arms} [{time.time() - t00:.0f}s]: {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
