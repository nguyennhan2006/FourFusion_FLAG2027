"""Generate FLAG_63w_aux_voice_feats.ipynb: ENC-SCALE-01 stage A (Experiment/ENC-SCALE-01/README.md; PLAN_SAT_2026-10-17 S4):
WeSpeaker ResNet293-LM (rn293, 256-d) and ECAPA1024-LM (ecapa1024, 192-d) of every auxiliary voice row (VoxCeleb2 FLAG_37 + FLAG_53,
VoxBlink-clean), on the centre 12 s of the clip (the T3 cut), aligned with each source's voice_meta.csv (NaN = not fetched).
The audio is re-fetched exactly as FLAG_63v (same shard index / WebDataset tars; LID-dropped speakers never fetched); the wiring check
is FLAG_63v's (frozen SimAM on the re-fetched audio must reproduce the stored arrays). The MAV-Celeb rows already have both encoders
(FLAG_47b, flag2027-feats-screen2-p).
    python build_vfeat_notebook.py
Smoke test on Kaggle CPU (kapi --smoke): FLAG_SMOKE=1;AUDIO_SMOKE_N=6;AUDIO_SMOKE_TARS=1;AUDIO_WIRE_N=4
"""
import build_audiocache_notebook as AC
import build_eval_notebooks as B12

INTRO = """
# FLAG 2027 — NB-63w: ResNet293-LM + ECAPA1024-LM of the ~21k auxiliary identities (ENC-SCALE-01 stage A)

**Why.** Track 3 (CB 17.70) gets most of its Bangla margin from **voice-encoder diversity**: sim, sam100ft, ecapa, r293 and their
concatenation `cat4`, each in its own branch. Our CC head uses SimAM p + ft only; ENC-CC-01 screened `cat4` WITHOUT auxiliary heads
(+0.04, a weak screen) because the auxiliary voices lacked the two other encoders. This notebook adds them for every auxiliary voice
row, so ENC-SCALE-01 stage B can train `cat4` (and one-branch-per-encoder) CC heads WITH the auxiliary sources.

| output | rows |
|---|---|
| `feats_aux_voice/<vox / vox2b / vb>/voice_rn293.npy`, `voice_ecapa1024.npy` | same order as the source's `voice_meta.csv` (NaN = not fetched or LID-dropped) |

**Wiring check:** frozen SimAM (sam100p / sam100f) on the re-fetched centre 12 s reproduces the stored arrays (mean cosine >= 0.99).
**Rules:** VoxBlink CC BY-NC-SA: notebook, output and dataset **private**. No Bengali voice is fetched (each source's LID rule).
**Inputs:** `flag2027-frames-index-p`, `flag2027-lab-vb-p`, `flag2027-lab-data-p`, `flag2027-feats-vox2b-p`. **GPU T4**, **Internet ON**,
secret `HF_TOKEN`. **Time:** about 6-8 h (resumable per shard / tar). Upload to `VFEAT_UPLOAD` when set.
"""

# modelscope.cn timed out from Kaggle (smoke 1, 16/10 15:50): the official zips come from an attached dataset when one has them,
# else from ModelScope with retries. Patched into the T3 cell source itself, since the wiring cell re-executes it.
_WS_OLD = '''    import requests
    tree = requests.get(WESPK_TREE, timeout=120).json()["Data"]
    url = [it["Url"] for it in tree if it.get("Key") == zip_name][0]
    z = fetch(url, d / zip_name, size)
'''
_WS_NEW = '''    import requests, shutil as _sh
    hit = [p for p in Path("/kaggle/input").rglob(zip_name) if p.stat().st_size == size] if Path("/kaggle/input").exists() else []
    if hit:                                                            # a weights dataset is attached: no network
        z = d / zip_name
        if not (z.exists() and z.stat().st_size == size):
            _sh.copy(hit[0], z)
    else:
        for _a in range(6):                                            # modelscope.cn times out from Kaggle now and then
            try:
                tree = requests.get(WESPK_TREE, timeout=120).json()["Data"]
                url = [it["Url"] for it in tree if it.get("Key") == zip_name][0]
                z = fetch(url, d / zip_name, size)
                break
            except Exception as _e:
                if _a == 5:
                    raise
                print(f"wespeaker {zip_name}: download failed ({_e!r}); retry {_a + 1}/5 in {30 * (_a + 1)} s", flush=True)
                time.sleep(30 * (_a + 1))
'''
assert AC.T3CELL.count(_WS_OLD) == 1
T3CELL = AC.T3CELL.replace(_WS_OLD, _WS_NEW)

ENV = AC.ENV.replace('OUT = WORK / "audio_cache"\nfor d in (OUT, OUT / "vb", OUT / "vox"):',
                     'OUT = WORK / "feats_aux_voice"\nfor d in (OUT, OUT / "vb", OUT / "vox", OUT / "chunks_vb", OUT / "chunks_vox"):')
assert ENV != AC.ENV
ENV = ENV.replace('log("audio cache | smoke"', 'log("aux voice features (rn293, ecapa1024) | smoke"')

PIP = AC.PIP.replace('[p for p in IN.rglob("audio_cache") if p.is_dir() and (p / "vb").exists()]',
                     '[p for p in IN.rglob("feats_aux_voice") if p.is_dir() and (p / "chunks_vb").exists()]')
PIP = PIP.replace('list((prev / "vb").glob("*.npz")) + list((prev / "vox").glob("*.npz")) + list(prev.glob("wire_*.npz"))',
                  'list((prev / "chunks_vb").glob("*.npz")) + list((prev / "chunks_vox").glob("*.npz")) + list(prev.glob("wire_*.npz"))')
assert PIP.count("feats_aux_voice") == 1 and PIP.count("chunks_vb") == 2

ENCS = r'''
# ============================================================================ the two encoders (T3 notebooks' loader; official ModelScope zips)
exec(compile(T3CELL_SRC, "t3enc_cell", "exec"))                       # SimAMVoice, wespeaker_dir, fetch, SAM_ZIPS
SAM_ZIPS.update({"rn293A": ("voxceleb_resnet293_LM.zip", 124116126), "ecapa1024A": ("voxceleb_ECAPA1024_LM.zip", 67665905)})
VENC = {"rn293": SimAMVoice("rn293A"), "ecapa1024": SimAMVoice("ecapa1024A")}
DIMS_V = {"rn293": 256, "ecapa1024": 192}


def save_emb(path, items):
    """items = [((spk, video, utt), float wav)] -> npz: keys + both embeddings of the centre 12 s (the T3 cut)."""
    keys = [k for k, _ in items]
    E = {n: np.zeros((len(items), d), np.float32) for n, d in DIMS_V.items()}
    for i, (_, w) in enumerate(items):
        c = centre_crop(w, L_WIRE)
        for n, enc in VENC.items():
            E[n][i] = enc(c)[""]
    k = np.array(keys, str).reshape(-1, 3)
    tmp = Path(str(path) + ".part.npz")
    np.savez(tmp, spk=k[:, 0], video=k[:, 1], utt=k[:, 2], **E)
    tmp.replace(path)
'''

VB = AC.VB.replace('    zf = OUT / "vb" / f"{part}_{shard}.npz"', '    zf = OUT / "chunks_vb" / f"{part}_{shard}.npz"')
VB = VB.replace('    tmp = zf.with_suffix(".part.npz"); save_chunk(tmp, got); tmp.replace(zf)', '    save_emb(zf, got)')
assert VB != AC.VB and "save_emb(zf, got)" in VB and 'OUT / "chunks_vb"' in VB

VOX = AC.VOX.replace('for f in sorted((OUT / "vox").glob("tar*.npz")):', 'for f in sorted((OUT / "chunks_vox").glob("tar*.npz")):')
VOX = VOX.replace('tars = [t for t in range(779) if not (OUT / "vox" / f"tar{t:04d}.npz").exists()]',
                  'tars = [t for t in range(779) if not (OUT / "chunks_vox" / f"tar{t:04d}.npz").exists()]')
VOX = VOX.replace('            path = OUT / "vox" / f"tar{t:04d}.npz"; tmp = path.with_suffix(".part.npz")\n            save_chunk(tmp, items); tmp.replace(path)',
                  '            save_emb(OUT / "chunks_vox" / f"tar{t:04d}.npz", items)')
assert VOX.count("chunks_vox") == 3, VOX.count("chunks_vox")

ASSEMBLE = r'''
# ============================================================================ chunks -> one array per source and encoder, in voice_meta.csv row order
ASM = {}
for s in META:
    for n, d in DIMS_V.items():
        ASM[(s, n)] = np.full((len(META[s]), d), np.nan, np.float32)
for sub in ("chunks_vb", "chunks_vox"):
    for f in sorted((OUT / sub).glob("*.npz")):
        Z = np.load(f)
        for i, k in enumerate(zip(Z["spk"], Z["video"], Z["utt"])):
            for s in (("vb",) if sub == "chunks_vb" else ("vox", "vox2b")):     # a VoxCeleb2 clip may be in both lists
                r = ROW[s].get(k)
                if r is not None:
                    for n in DIMS_V:
                        ASM[(s, n)][r] = Z[n][i]
done = {}
for (s, n), A in ASM.items():
    (OUT / s).mkdir(exist_ok=True)
    np.save(OUT / s / f"voice_{n}.npy", A.astype(np.float16))
    done[f"{s}/{n}"] = dict(rows=len(A), finite=int(np.isfinite(A).all(1).sum()), wanted=len(WANT[s]))
log("assembled:", json.dumps(done))
'''

DONE = r'''
# ============================================================================ manifest + (optional) upload to a private dataset of this account
cover = {s: done[f"{s}/rn293"]["finite"] / max(1, done[f"{s}/rn293"]["wanted"]) for s in META}
man = dict(created=time.strftime("%Y-%m-%d %H:%M"), encoders={"rn293": "voxceleb_resnet293_LM", "ecapa1024": "voxceleb_ECAPA1024_LM"},
           cut="centre 12 s", arrays=done, coverage=cover, wiring=WIRING, smoke=SMOKE, complete=bool(min(cover.values()) >= 0.97))
(OUT / "vfeat_manifest.json").write_text(json.dumps(man, indent=1, default=str))
log("manifest:", json.dumps(man, default=str))
UP = os.environ.get("VFEAT_UPLOAD", "")
if UP and not SMOKE and man["complete"] and ok:
    try:
        import kagglehub
        stage = Path("/tmp/vfeat_upload"); shutil.rmtree(stage, ignore_errors=True); stage.mkdir(parents=True)
        for s in META:
            (stage / s).mkdir()
            for n in DIMS_V:
                os.symlink(OUT / s / f"voice_{n}.npy", stage / s / f"voice_{n}.npy")
        os.symlink(OUT / "vfeat_manifest.json", stage / "vfeat_manifest.json")
        kagglehub.dataset_upload(UP, str(stage), version_notes="ENC-SCALE-01 stage A: rn293 + ecapa1024 of the auxiliary voices")
        log("uploaded ->", UP)
    except Exception as e:
        log(f"upload to {UP} failed ({e!r}); the arrays are in the notebook output")
else:
    log(f"no upload (VFEAT_UPLOAD={UP!r}, smoke={SMOKE}, complete={man['complete']}, wiring ok={ok}): attach the output and run again if incomplete")
'''


def build():
    t3src = "T3CELL_SRC = " + repr(T3CELL) + "\n"
    cells = [("markdown", INTRO), ("code", ENV), ("code", PIP),
             ("markdown", "## 1. Voice rows of each source (LID-dropped speakers excluded)"), ("code", AC.INDEX),
             ("markdown", "## 2. The two encoders"), ("code", t3src + ENCS),
             ("markdown", "## 3. VoxBlink: re-fetch (FLAG_63's shard index), FLAG_46's utterance cut, embed the centre 12 s"), ("code", VB),
             ("markdown", "## 4. VoxCeleb2: stream the WebDataset tars, embed the centre 12 s"), ("code", VOX),
             ("markdown", "## 5. Wiring check (frozen SimAM vs the stored arrays)"), ("code", AC.WIRECELL),
             ("markdown", "## 6. One array per source and encoder"), ("code", ASSEMBLE),
             ("markdown", "## 7. Manifest + upload"), ("code", DONE)]
    B12.nb(cells, "FLAG_63w_aux_voice_feats.ipynb")
    print("FLAG_63w_aux_voice_feats.ipynb built")


if __name__ == "__main__":
    build()
