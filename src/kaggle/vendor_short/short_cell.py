# ============================================================================ SHORT-02: voice embeddings of SHORT crops (labelled validation + training rows)
# Dev has 37 % Bangla / 26 % English voices under 4 s; SAFE-04 validation has none. Two products, every crop drawn with a
# fixed seed from the signal the existing embedding was computed on (v4: the whole clip; MAV-Celeb: its 12 s centre crop):
#   trial crops   every voice of a SAFE-04 trial (trial_voices.json: 2 887 v4 + 10 418 MAV-Celeb rows), cropped to 2 s and
#                 4 s, through every voice encoder of v11's Bangla streams: btc (organiser ECAPA, XALL + rel), ecapa192 (rel),
#                 iba (ImageBind), w2v2 age / gender (age stream, P(male) voice, relg mask), sam100p / sam100f (T3 head)
#   train crops   every v4 row and every MAV-Celeb voice row, ONE crop of U(2, 6) s, through sam100p / sam100f (arm C)
# Arrays (rows = every v4 row / every VMETA row; NaN where not extracted) -> WORK/feats_short; resumable per chunk.
TRIAL = json.loads(__TRIAL_JSON__)
CONDS = {"c2": 2.0, "c4": 4.0}
FS_ = WORK / "feats_short"; PARTS_ = FS_ / "parts"
PARTS_.mkdir(parents=True, exist_ok=True)
for prev in ([] if LOCAL else IN.rglob("feats_short")):                 # resume: chunks of an earlier run
    if prev == FS_:
        continue
    for f in prev.glob("parts/*.npz"):
        if not (PARTS_ / f.name).exists():
            shutil.copy(f, PARTS_ / f.name)
SR_ = 16000
assert "train" in S and len(S["train"]["wav"]) == TRIAL["n_v4"], f"the organiser train list must be attached ({TRIAL['n_v4']} rows)"
assert set(EXT) == {"v1_complete", "v2_complete"}, f"attach MAV-Celeb v1 + v2 (raw) and flag2027-ext-meta (found {list(EXT)})"
assert len(EXT["v1_complete"].voice) == TRIAL["n_v1"] and len(EXT["v1_complete"].voice) + len(EXT["v2_complete"].voice) == TRIAL["n_ext"]
TV4, TEXT = set(TRIAL["v4"]), set(TRIAL["ext"])


def signal(part, i):
    """The signal the existing embedding of this row was computed on."""
    if part == "v4":
        sp = S["train"]; return load_audio(sp["root"] / sp["wav"][i])
    src, j = ("v1_complete", i) if i < TRIAL["n_v1"] else ("v2_complete", i - TRIAL["n_v1"])
    rows = EXT[src]
    return centre_crop(load_audio(io.BytesIO(rows.read(rows.voice[j]))), rows.L)


def crop(w, sec, seed):
    L = int(sec * SR_)
    if len(w) <= L:
        return w
    off = np.random.RandomState(seed).randint(len(w) - L + 1)
    return w[off:off + L]


def train_len(seed):
    return float(np.random.RandomState(seed).uniform(2.0, 6.0))


VOICE_ENCS = None


SMOKE_N = int(os.environ.get("SHORT_SMOKE_N", "0")) if (LOCAL or os.environ.get("FLAG_SMOKE") == "1") else 0   # smoke only (local or kapi --smoke): rows per part


class _StubIB:                                                          # local smoke test on a small-RAM PC only
    __call__ = lambda self, w, use_amp=True: {"": np.random.RandomState(len(w)).rand(1024).astype(np.float32)}


def encoders():
    global VOICE_ENCS
    if VOICE_ENCS is None:
        ib = _StubIB() if ((LOCAL or os.environ.get("FLAG_SMOKE") == "1") and os.environ.get("FLAG_SMOKE_STUB_IB")) else IBAudio()
        VOICE_ENCS = dict(btc=OrganiserVoice(CACHE / "ecapa_btc"), ecapa=Ecapa(), iba=ib, w2v2=W2V2AgeGender(),
                          sam100p=build("sam100pA"), sam100f=build("sam100fA"))
    return VOICE_ENCS


def embed_all(waves, full=True):
    """waves of one length (or any) -> dict of arrays; full=False: SimAM only (train crops)."""
    E = encoders(); out = {}
    for n in ("sam100p", "sam100f"):
        out[n] = np.stack([E[n](w)[""] for w in waves])
    if full:
        out["btc"] = np.stack([e for s in range(0, len(waves), 32) for e in E["btc"](waves[s:s + 32])])
        out["ecapa192"] = np.stack([r[0] for s in range(0, len(waves), 32) for r in E["ecapa"].batch(waves[s:s + 32])])
        out["iba"] = np.stack([E["iba"](w)[""] for w in waves])
        g = [E["w2v2"](w) for w in waves]
        out["w2v2agage"] = np.stack([x["age"] for x in g]); out["w2v2aggen"] = np.stack([x["gen"] for x in g])
    return {k: v.astype(np.float32) for k, v in out.items()}


import concurrent.futures as _cf
CH = 1000
for part, n in (("v4", TRIAL["n_v4"]), ("ext", TRIAL["n_ext"])):
    for c0 in range(0, SMOKE_N or n, CH):
        f = PARTS_ / f"{part}_{c0 // CH:03d}.npz"
        if f.exists():
            continue
        t0 = time.time(); rows = list(range(c0, min(SMOKE_N or n, c0 + CH)))
        with _cf.ThreadPoolExecutor(8) as ex:
            sigs = list(ex.map(lambda i: signal(part, i), rows))
        base = 0 if part == "v4" else 10 ** 6
        R_ = {"row": np.array(rows), "dur": np.array([len(w) / SR_ for w in sigs], np.float32)}
        tr = [crop(w, train_len(base + i), base + i) for i, w in zip(rows, sigs)]
        for k, v in embed_all(tr, full=False).items():
            R_[f"train_{k}"] = v
        trial = [(i, w) for i, w in zip(rows, sigs) if i in (TV4 if part == "v4" else TEXT)]
        R_["trial_row"] = np.array([i for i, _ in trial], int)
        for ci, (cn, sec) in enumerate(CONDS.items()):
            if trial:
                for k, v in embed_all([crop(w, sec, (ci + 1) * 10 ** 7 + base + i) for i, w in trial]).items():
                    R_[f"{cn}_{k}"] = v
        np.savez(f, **R_)
        log(f"  {part} chunk {c0 // CH}: {len(rows)} rows, {len(trial)} trial voices [{time.time() - t0:.0f}s]")
VOICE_ENCS = None
release()


def assemble(part, n):
    out = {}
    for f in sorted(PARTS_.glob(f"{part}_*.npz")):
        Z = np.load(f)
        for k in Z.files:
            if k in ("row", "trial_row"):
                continue
            idx = Z["trial_row"] if k.startswith(tuple(CONDS)) else Z["row"]
            if not len(idx):
                continue
            if k not in out:
                out[k] = np.full((n,) + Z[k].shape[1:], np.nan, np.float32)
            out[k][idx] = Z[k]
    for k, v in out.items():
        np.save(FS_ / f"{part}_{k}.npy", v)
    return {k: v.shape for k, v in out.items()}


SHAPES = {p: assemble(p, n) for p, n in (("v4", TRIAL["n_v4"]), ("ext", TRIAL["n_ext"]))}
(FS_ / "trial_voices.json").write_text(json.dumps(TRIAL))
(FS_ / "feats_manifest.json").write_text(json.dumps(dict(created=time.strftime("%Y-%m-%d %H:%M"), conds=CONDS, shapes=SHAPES,
    train_crop="U(2, 6) s, seed = row (+1e6 for MAV-Celeb)", broken={k: len(v) for k, v in BROKEN.items()}), indent=1, default=str))
log("short-crop arrays:", SHAPES)
EXTRA_LINKS["Experiment/SHORT-02/feats_short"] = FS_
