print("[cell] raw_mavceleb: extract each zip into the folder named after it (= Kaggle's auto-extraction)", flush=True)
# About 25 GB written; one marker per zip makes the cell resumable. Disk: the extracted tree (~25 GB) + the runner's export copy.
for name, src in SRC.items():
    stem = name[:-4]                                  # mavceleb_v1_complete-001 / mavceleb_v2_complete-002
    mark = STATE / f"{stem}.done"
    dst = DS / stem
    if mark.exists() and not SMOKE:
        print(f"{stem}: already extracted ({mark.read_text().strip()})", flush=True)
        continue
    if dst.exists():
        shutil.rmtree(dst)                            # a partial extraction of an interrupted run
    dst.mkdir(parents=True)
    t = time.time()
    with zipfile.ZipFile(src) as z:
        infos = z.infolist()
        if SMOKE:                                     # smoke: the meta table + the first 200 face and 200 voice members only
            infos = ([i for i in infos if i.filename.endswith(".csv")] + [i for i in infos if "/faces/" in i.filename][:200]
                     + [i for i in infos if "/voices/" in i.filename][:200])
        for k, info in enumerate(infos):
            z.extract(info, dst)
            if k % 50000 == 0:
                print(f"  {stem}: {k}/{len(infos)} members ({time.time() - t:.0f}s)", flush=True)
    if not SMOKE:
        mark.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"{stem}: {len(infos)} members extracted in {time.time() - t:.0f}s", flush=True)
