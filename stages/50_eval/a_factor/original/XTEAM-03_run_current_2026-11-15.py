"""XTEAM-03: dev zip = RF-092 score + F1 score per pair (both are -logit, weights 1:1, no test statistics)."""
import os
import sys
import zipfile
from pathlib import Path

H = Path(__file__).parent
# argv[1] = our own reproduction (submission.zip of onnhqunhh/flag-rf-092-tailap-p); default = the RF team's zip (CB 18.61)
RF = Path(sys.argv[1]) if len(sys.argv) > 1 else H / 'rf092_dev.zip'
# argv[2] / argv[3] (UNIFIED-01, 3/11): F1s zip and output path; defaults = XTEAM-03 (F1, CB 20.17)
F1 = Path(sys.argv[2]) if len(sys.argv) > 2 else H.parent / 'CLEANTUNE-01' / 'submission_cleantune01_F1_CA_dev.zip'
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else H / 'submission_xteam03_rf092_plus_f1_dev.zip'
# FUSEDIM-02 (15/11): XT_W = share w of the F1 zip -> 2 [w F1 + (1 - w) RF], fixed before any test score; unset = 1:1 sum unchanged
W = float(os.environ['XT_W']) if os.environ.get('XT_W') else None
comb = (lambda r, f: r + f) if W is None else (lambda r, f: 2 * (W * f + (1 - W) * r))


def read(z, n):
    return [l.split() for l in z.read(n).decode().splitlines() if l.strip()]


with zipfile.ZipFile(RF) as a, zipfile.ZipFile(F1) as b, zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as o:
    assert sorted(a.namelist()) == sorted(b.namelist())
    for n in b.namelist():
        ra, rb = dict(read(a, n)), read(b, n)
        assert len(ra) == len(rb) and all(p in ra for p, _ in rb), n
        o.writestr(n, ''.join(f'{p} {comb(float(ra[p]), float(s)):.8f}\n' for p, s in rb))   # F1 pair order
        print(n, len(rb))
print('wrote', OUT, '(1:1 sum)' if W is None else f'(F1 share w = {W})')
