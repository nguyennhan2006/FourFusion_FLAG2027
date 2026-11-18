"""FACTOR-01 step 3 (13/11, Experiment/FACTOR-01/README.md): patch a FLAG_163i copy (already through patch_163i_eval.py) IN PLACE so it
scores an RFX artifact whose projector reads extra blocks (manifest 'add', e.g. f_c5s + v_rl).

    python kaggle/patch_163i_factor.py <nb.ipynb>

- cell 4: scorer = make_xscorer(L)(ART); make_xscorer is copied verbatim from Experiment/RFS-01/train/rfs_train.py (one source).
- new cell after 6: XTRA(proto, lang, t) reads the extra blocks of that trial list from the attached kernel outputs:
    f_c5s = face_clipC5_<proto>_<lang>.npy (FLAG_65xc on the same lists, C5s adapter), v_rl = [l2n(voice_rn293) | l2n(voice_ecapa1024)]
    _<proto>_<lang>.npy (FLAG_47b re-run), rows = index_<proto>_<lang>.txt (pair_id voice face); mapped by pair_id, every pair present,
    media paths checked against the list (suffix match), every row finite. Same row layout as rfs_train._xpart (each array l2n, then concat).
- cell 8: x.update(XTRA(...)) before scoring (the level-0 check slices the extra blocks with the rest).
Without 'add' in the artifact manifest the notebook behaves as before (XTRA returns {}).
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
nbp = Path(sys.argv[1]); nb = json.loads(nbp.read_text(encoding="utf-8"))
src = (ROOT / "Experiment/RFS-01/train/rfs_train.py").read_text(encoding="utf-8")
m = re.search(r"# >>> XSCORER.*?\n(def make_xscorer.*?)# <<< XSCORER", src, re.S)
assert m, "XSCORER block not found in rfs_train.py"
XS = m.group(1)

code = lambda c: "".join(c["source"])
c4 = nb["cells"][4]; s4 = code(c4)
assert "scorer = L.FinalScorer(ART)" in s4, "cell 4 is not the scorer cell"
c4["source"] = s4.replace("scorer = L.FinalScorer(ART)", XS + "\nscorer = make_xscorer(L)(ART)   # FACTOR-01: wider projector input when manifest has 'add'\n"
                          "print('FACTOR-01 extra blocks:', scorer.xf, scorer.xv, flush=True)")

XTRA = r'''print("[cell] factor extra blocks", flush=True)
# FACTOR-01 step 3: extra blocks of each trial list from the attached kernel outputs (FLAG_65xc C5s faces, FLAG_47b rn293 / ecapa1024)
def _l2n(X):
    X = np.asarray(X, np.float32); return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

_XFILES = None
def _one(pat):   # one walk of the input tree, skipping the media / artifact datasets (hundreds of thousands of files)
    global _XFILES
    if _XFILES is None:
        import os
        _XFILES, skip = [], ('raw-v4', 'mavceleb', 'evalphase', '-art-p')
        for r, ds, fs in os.walk(Path('/kaggle/input')):
            ds[:] = [d for d in ds if not any(s in d for s in skip)]
            _XFILES += [Path(r) / f for f in fs if f.endswith(('.npy', '.txt'))]
    hit = sorted({p.parent for p in _XFILES if p.name == pat})
    assert len(hit) == 1, f'{pat}: need exactly one folder, found {hit}'
    return hit[0]

def XTRA(proto, lang, t, fp, vp):
    if not (scorer.xf or scorer.xv):
        return {}
    name = f'{proto}_{lang}'; want = [str(p) for p in t.pair_id]; out = {}
    def rows(folder, arrs):
        idx = [l.split() for l in (folder / f'index_{name}.txt').read_text().splitlines() if l.strip()]
        pos = {r[0]: i for i, r in enumerate(idx)}
        assert set(want) <= set(pos), f'{name}: {len(set(want) - set(pos))} pairs missing in {folder}'
        sel = np.array([pos[p] for p in want])
        for j, p in enumerate(want[:50] + want[-50:]):          # media paths of the index = paths of the list (suffix)
            r = idx[pos[p]]; k = want.index(p)
            assert str(vp[k]).replace('\\', '/').endswith(r[1]) and str(fp[k]).replace('\\', '/').endswith(r[2]), (name, p, r, vp[k], fp[k])
        X = np.concatenate([_l2n(np.load(folder / f'{a}_{name}.npy'))[sel] for a in arrs], 1)
        assert len(np.load(folder / f'{arrs[0]}_{name}.npy')) == len(idx) and np.isfinite(X).all(), (name, arrs)
        return X
    for k in scorer.xf + scorer.xv:
        if k == 'f_c5s':
            out[k] = rows(_one(f'face_clipC5_{name}.npy'), ['face_clipC5'])
        elif k == 'v_rl':
            out[k] = rows(_one(f'voice_rn293_{name}.npy'), ['voice_rn293', 'voice_ecapa1024'])
        elif k == 'v_fvb':   # FVBIND-01 gate 2: B2 voice block of the Evaluation lists (FLAG_192 output eval/, index copied from FLAG_47rs)
            out[k] = rows(_one(f'voice_fvb_{name}.npy'), ['voice_fvb'])
        elif k.startswith('v_wh'):   # WHL-01: Whisper-large-v3 layer blocks of the Evaluation lists (FLAG_200e output), v_wh1624 = layers 16 + 24
            ls = ['16', '24'] if k == 'v_wh1624' else [k[4:]]
            out[k] = rows(_one(f'voice_whl{ls[0]}_{name}.npy'), [f'voice_whl{l}' for l in ls])
        else:
            raise AssertionError(f'extra block {k}: no Evaluation source wired (FACTOR-01 step 3 wires f_c5s, v_rl)')
        print(f'  {name}: {k} {out[k].shape}', flush=True)
    return out
'''
nb["cells"].insert(7, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": XTRA})
c8 = nb["cells"][9]; s8 = code(c8)
anchor = "        if d['csv'] is not None: x['xf'], x['xv'] = d['csv']\n"
assert s8.startswith('print("[cell] score+level0+zip"') and anchor in s8, "cell 8 layout changed"
c8["source"] = s8.replace(anchor, anchor + "        x.update(XTRA(proto, lang, d['t'], d['fp'], d['vp']))   # FACTOR-01 extra blocks\n")
nbp.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("patched", nbp, "(XScorer + XTRA)")
