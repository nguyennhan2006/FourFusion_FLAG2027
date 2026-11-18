"""Organiser Evaluation zip -> dev-shaped copy for the FLAG notebooks (they look for <folder>/<Lang>_test.txt).
- every '<Lang>_valid' in member names and list lines -> '<Lang>_test'
- gender/English_valid.txt (BTC): 'English_valid/faces 00000.jpg' (space instead of '/') -> 'English_test/faces/00000.jpg'
- pair ids, row order and media bytes unchanged; the organiser lists kept as btc_original/<g>/<Lang>_valid.txt.orig
usage: python -I build_evalphase.py <btc zip> <out zip>
"""
import sys, zipfile, hashlib, json

src, dst = sys.argv[1], sys.argv[2]
zi = zipfile.ZipFile(src)
names = set(zi.namelist())
rep = {}
with zipfile.ZipFile(dst, "w", allowZip64=True) as zo:
    for info in zi.infolist():
        n = info.filename
        if n.endswith("/"):
            continue
        data = zi.read(n)
        if n.endswith("_valid.txt"):
            g = n.split("/")[1]; lang = n.split("/")[-1].split("_")[0]
            zo.writestr(f"evaluation_phase/btc_original/{g}/{lang}_valid.txt.orig", data, zipfile.ZIP_DEFLATED)
            raw = data.decode("utf-8").splitlines()
            out, fixed = [], 0
            for ln in raw:
                t = ln.split()
                if not t:
                    continue
                if len(t) == 4 and t[2].endswith("/faces") and t[3].lower().endswith(".jpg"):
                    t = [t[0], t[1], t[2] + "/" + t[3]]; fixed += 1
                assert len(t) == 3, (n, ln)
                for p in t[1:]:
                    assert f"evaluation_phase/{g}/{p}" in names, (n, p)
                out.append(" ".join([t[0]] + [p.replace(f"{lang}_valid/", f"{lang}_test/") for p in t[1:]]))
            ids = [l.split()[0] for l in out]
            assert len(set(ids)) == len(ids), n
            zo.writestr(f"evaluation_phase/{g}/{lang}_test.txt", "\n".join(out) + "\n", zipfile.ZIP_DEFLATED)
            rep[f"{g}/{lang}"] = dict(rows=len(out), fixed=fixed,
                                      sha_btc=hashlib.sha256(data).hexdigest()[:16])
            continue
        new = n.replace("_valid/", "_test/").replace("_valid_faces.csv", "_test_faces.csv").replace("_valid_voices.csv", "_test_voices.csv")
        zo.writestr(new, data, info.compress_type)
    zo.writestr("evaluation_phase/README_FOURFUSION.txt",
                "Organiser Evaluation data (evaluation_phase.zip, 11/11/2026), renamed *_valid -> *_test so the FLAG notebooks "
                "read it like dev; gender/English list: 416 lines 'English_valid/faces NNNNN.jpg' fixed to '.../faces/NNNNN.jpg'. "
                "Ids, row order and media bytes unchanged. Original lists: btc_original/*.orig\n" + json.dumps(rep, indent=1) + "\n")
# csv rows == list rows
zc = zipfile.ZipFile(dst)
for k, v in rep.items():
    g, lang = k.split("/")
    for kind in ("faces", "voices"):
        n = len(zc.read(f"evaluation_phase/{g}/features/{lang}_test_{kind}.csv").decode().splitlines())
        assert n == v["rows"], (k, kind, n, v["rows"])
print(json.dumps(rep, indent=1))
print("members", len(zc.namelist()))
