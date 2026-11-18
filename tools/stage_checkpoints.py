"""Copy every checkpoint of configs/data_map.json (class 'checkpoint') into a folder laid out for FF_DATA (the Drive folder), and
check each copy against the sha256 recorded in the map. Team use: the sources are the team package and the team machine.

    python tools/stage_checkpoints.py --dest <folder> --package <team package root, holds datasets/<owner>__<slug>.zip> \
        [--folder <artifact id>=<folder on this machine> ...]

For each checkpoint artifact:
  * a Kaggle dataset kept in the team package as datasets/<owner>__<slug>.zip is copied to <dest>/datasets/<owner>__<slug>.zip
    (the runner reads this layout directly, see runner/ffcore.py saved_candidates) and its sha256 is compared with 'zip_sha256';
  * an artifact given with --folder is copied to <dest>/<path of the artifact in data_map.json>/ and every file listed in its
    'files' is compared with its sha256;
  * an artifact already present in <dest> (zip or folder) is checked, not copied again.
Prints one line per artifact and exits with code 1 when a copy does not match or a checkpoint has no source.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def check_files(folder, files):
    """Compare the files listed in data_map 'files' (relative path or bare name -> sha256) with the copies under folder."""
    bad = []
    for name, want in files.items():
        if len(want) != 64:                                  # descriptive keys such as "xjf_scores.npz (English only)"
            continue
        hits = [folder / name] if (folder / name).is_file() else sorted(folder.rglob(Path(name).name))
        if not any(sha256_of_file(h) == want for h in hits):
            bad.append(name)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--folder", action="append", default=[], help="<artifact id>=<folder on this machine>")
    args = ap.parse_args()

    folders = dict(x.split("=", 1) for x in args.folder)
    data_map = json.loads((REPO / "configs" / "data_map.json").read_text(encoding="utf-8"))["artifacts"]
    checkpoints = {k: v for k, v in data_map.items() if v.get("class") == "checkpoint"}
    failed = 0
    for aid, entry in sorted(checkpoints.items()):
        kind, owner, slug = (aid.split("/") + ["", ""])[:3]
        zip_name = f"{owner}__{slug}.zip"
        zip_dest = args.dest / "datasets" / zip_name
        folder_dest = args.dest / entry["path"]
        if kind == "datasets" and entry.get("zip_sha256") and (zip_dest.is_file() or (args.package / "datasets" / zip_name).is_file()):
            if not zip_dest.is_file():
                zip_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.package / "datasets" / zip_name, zip_dest)
            ok = sha256_of_file(zip_dest) == entry["zip_sha256"]
            print(f"{'OK ' if ok else 'BAD'} {aid}: datasets/{zip_name} (zip sha256 {'matches' if ok else 'DIFFERS'})", flush=True)
            failed += not ok
        elif aid in folders or (folder_dest.is_dir() and any(folder_dest.iterdir())):
            if aid in folders and not (folder_dest.is_dir() and any(folder_dest.iterdir())):
                shutil.copytree(folders[aid], folder_dest, dirs_exist_ok=True)
            bad = check_files(folder_dest, entry.get("files", {}))
            print(f"{'OK ' if not bad else 'BAD'} {aid}: {entry['path']}/ ({len(entry.get('files', {}))} recorded files; "
                  f"{'all match' if not bad else 'differ: ' + str(bad)})", flush=True)
            failed += bool(bad)
        else:
            print(f"--- {aid}: no source (not in the package, no --folder given); status: {entry.get('status', '-')}", flush=True)
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
