"""EVALPATH-U: sanity of a (dev or Evaluation) submission zip, with file-name / pair-count parameters (UNIFIED-01/check_zip.py hard-codes dev).
    python check_eval_zip.py <zip> [--dev | --counts 'name=n,name=n'] [--map 'devname=evalname,...'] [--ref ZIP --min 0.999] [--peer ZIP ...]
  --dev      expected pairs = the four dev files; --counts overrides (names as in <zip>); neither = no count check (file list must still be 4 names)
  --map      renames the REFERENCE zip's names to <zip>'s names (Evaluation file names unknown until 11/11)
  --ref/min  per file: same pair ids, Spearman >= min (R11: 0.999); max |diff| and mean |diff| are printed, never judged
  --peer     other zips of the same pairs (F1s eval, RFS eval, U): same file names, same pair-id sets
Final line OK / FAIL."""
import argparse
import zipfile
import numpy as np
from scipy.stats import spearmanr

DEV = {'no_gender/sub_score_v4_English_heard.txt': 1008, 'no_gender/sub_score_v4_Bangla_unheard.txt': 1406,
       'gender/sub_score_v4_English_heard.txt': 982, 'gender/sub_score_v4_Bangla_unheard.txt': 1468}


def read(z, n):
    return dict(l.split() for l in z.read(n).decode().splitlines() if l.strip())


ap = argparse.ArgumentParser()
ap.add_argument('zip'); ap.add_argument('--dev', action='store_true'); ap.add_argument('--counts'); ap.add_argument('--map', default='')
ap.add_argument('--ref'); ap.add_argument('--min', type=float, default=0.999); ap.add_argument('--peer', nargs='*', default=[])
a = ap.parse_args()
z = zipfile.ZipFile(a.zip)
names = sorted(n for n in z.namelist() if n.endswith('.txt'))
counts = DEV if a.dev else dict((k, int(v)) for k, v in (x.split('=') for x in a.counts.split(','))) if a.counts else None
ok = len(names) == 4 and (counts is None or sorted(counts) == names)
if not ok:
    print(f'file list {names} (expected 4' + (f' = {sorted(counts)}' if counts else '') + ')')
ren = dict(x.split('=') for x in a.map.split(',') if x)             # dev name -> this zip's name
inv = {v: k for k, v in ren.items()}
ref = zipfile.ZipFile(a.ref) if a.ref else None
for n in names:
    d = read(z, n); v = np.array([float(x) for x in d.values()])
    line = f'{n}: pairs {len(d)}' + (f'/{counts[n]}' if counts and n in counts else '') + f' finite {bool(np.isfinite(v).all())} sd {v.std():.3f}'
    ok &= bool(np.isfinite(v).all()) and v.std() > 0 and (counts is None or counts.get(n) == len(d))
    if ref is not None:
        rn = inv.get(n, n)
        if rn not in ref.namelist():
            line += f' | ref has no {rn}'; ok = False
        else:
            r = read(ref, rn); same = set(d) == set(r)
            if same:
                x = np.array([float(d[p]) for p in r]); y = np.array([float(r[p]) for p in r])
                rho = spearmanr(x, y)[0]; dd = np.abs(x - y)
                line += f' | vs ref spearman {rho:.6f} (min {a.min}) max|diff| {dd.max():.2e} mean {dd.mean():.2e}'
                ok &= rho >= a.min
            else:
                line += ' | pair ids differ from ref'; ok = False
    for p in a.peer:
        pz = zipfile.ZipFile(p)
        if n not in pz.namelist() or set(read(pz, n)) != set(d):
            line += f' | PEER MISMATCH {p}'; ok = False
    print(line)
print('OK' if ok else 'FAIL')
