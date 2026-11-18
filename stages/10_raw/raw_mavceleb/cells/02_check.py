print("[cell] raw_mavceleb: count files and check the layout of the dataset folder", flush=True)
for name, (_, _, cnt) in ZIPS.items():
    stem = name[:-4]
    root = DS / stem
    got = {"jpg": 0, "wav": 0, "csv": 0, "files": 0}
    for dp, dn, fn in os.walk(root):
        for f in fn:
            got["files"] += 1
            ext = f.rsplit(".", 1)[-1].lower()
            if ext in got:
                got[ext] += 1
    ver = "v1" if "_v1_" in stem else "v2"
    layout = {d: (root / ver / d).is_dir() for d in ("faces", "voices")}
    REP["zips"][name].update(extracted=got, expected=cnt, layout=layout, counts_ok=(got == cnt) or SMOKE)
    print(stem, got, "expected", cnt, "layout", layout, flush=True)
    assert all(layout.values()), f"{stem}: {ver}/faces and {ver}/voices must sit directly under {root}"
    assert got == cnt or SMOKE, f"{stem}: file counts differ from the release"
meta = DS / "mavceleb_v1_complete-001/meta_v1.csv"
REP["meta_v1_sha256"] = sha256_file(meta) if meta.exists() else None
assert REP["meta_v1_sha256"] == META_V1_SHA256, "mavceleb_v1_complete-001/meta_v1.csv missing or different"
stray = sorted(str(p) for p in DS.rglob("*.zip"))
assert not stray, f"no zip may stay inside the dataset folder (r08 would read it): {stray[:3]}"
REP["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
(STATE / "raw_mavceleb_check.json").write_text(json.dumps(REP, indent=1))
print("raw_mavceleb: OK ->", DS, flush=True)
