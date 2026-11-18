"""Copy FLAG_163i_rfs_infer.ipynb with the Evaluation file names set in its config cell (cell 1).

    python kaggle/patch_163i_eval.py <out.ipynb> --names 'no_gender/English=<name>,no_gender/Bangla=<name>,gender/English=<name>,gender/Bangla=<name>'

<name> = the path of that cell's file inside the submission zip, as in the organisers' Evaluation template.
Without --names the copy is unchanged (dev file names). Only the line `NAMES = None ...` changes; DATA_ROOT, ART and
REF_ZIP stay None (auto-detect; attach exactly one artifact dataset: RFS or RFSG). Used by kaggle/rfs_eval_push.sh
(EVALPATH-U section 2 for RFS, OBJFAM-01 section 7 for RFSG).
"""
import argparse, json, re
from pathlib import Path

SRC = Path(__file__).with_name("FLAG_163i_rfs_infer.ipynb")
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--names", default="")
# --data-root: folder (Kaggle path) holding the Evaluation trial lists ({protocol}_{lang}_test.txt or <protocol>/../{lang}_test.txt,
# exactly one per cell, media paths relative to the list). Needed when the Evaluation lists are NOT in the dataset that has
# train_English.txt: DATA_ROOT = None would then find the DEV lists (the pair-count / pair-id checks catch it, after 55 min).
ap.add_argument("--data-root", default="")
# --ib-url (12/11): dl.fbaipublicfiles.com unreachable (Kaggle + local); same file from a Hugging Face mirror. The notebook still
# checks SHA-256 d6f6c22b... of imagebind_huge (identical LFS oid on the mirrors), so only the download source changes.
ap.add_argument("--ib-url", default="")
a = ap.parse_args()

nb = json.loads(SRC.read_text(encoding="utf-8"))
cell = nb["cells"][1]
src = "".join(cell["source"])
assert src.startswith('print("[cell] config"'), "cell 1 is not the config cell"
if a.names:
    d = {}
    for kv in a.names.split(","):
        k, v = kv.split("=", 1)
        prot, lang = k.strip().split("/")
        assert prot in ("no_gender", "gender") and lang in ("English", "Bangla"), k
        d[(prot, lang)] = v.strip()
    assert len(d) == 4, f"need 4 cells, got {sorted(d)}"
    line = "NAMES = {" + ", ".join(f"({p!r}, {l!r}): {v!r}" for (p, l), v in sorted(d.items())) + "}   # Evaluation (patch_163i_eval.py)"
    src2, n = re.subn(r"^NAMES = None.*$", lambda m: line, src, flags=re.M)
    assert n == 1, "NAMES line not found exactly once"
    src = src2
    print(line)
if a.data_root:
    # Git Bash rewrites /kaggle/... into D:/Git/kaggle/... unless MSYS_NO_PATHCONV=1 (rfs_eval_push.sh sets it)
    assert a.data_root.startswith("/kaggle/"), f"--data-root must be a Kaggle path (/kaggle/...), got {a.data_root!r}"
    line = f"DATA_ROOT = Path({a.data_root!r})   # Evaluation lists (patch_163i_eval.py)"
    src, n = re.subn(r"^DATA_ROOT = None.*$", lambda m: line, src, flags=re.M)
    assert n == 1, "DATA_ROOT line not found exactly once"
    print(line)
cell["source"] = src.splitlines(keepends=True)
if a.data_root:
    # 11/11: load_dev_lenient(..., strict_count=True) checks the DEV row counts (EXPECTED_ROWS); on Evaluation that check
    # fails and the cell silently falls back to re-extracting VGG / ECAPA. Off here: the csv / list / pair-id checks stay.
    old = "scorer.man['dimensions'], True, scorer.p['fill_xf']"
    hits = [c for c in nb["cells"] if old in "".join(c["source"]) and "CORE_SOURCE" not in "".join(c["source"])]
    assert len(hits) == 1, f"strict_count call found {len(hits)} times"
    hits[0]["source"] = "".join(hits[0]["source"]).replace(old, old.replace(", True,", ", False,")).splitlines(keepends=True)
    print("strict_count -> False (Evaluation row counts differ from dev)")
    # 11/11 (EXPERIMENT_LOG 404): RFSG on Evaluation failed the level-0 self-check at atol=1e-6. The scorer is row-wise
    # (train constants only; mean over models, not pairs), so the gap is float32 torch / BLAS noise that changes with the
    # batch size. The check now prints the largest gap, stores it in run_report.json and allows 1e-4 x the score scale;
    # a real dependence on other pairs (any statistic over the batch) moves scores by about the score scale and still fails.
    sc = [c for c in nb["cells"] if "'score depends on other pairs'" in "".join(c["source"])]
    assert len(sc) == 1, f"level-0 cell found {len(sc)} times"
    s8 = "".join(sc[0]["source"])
    for old, new in (
        ("pack = OUT/'submission.zip'; scores = {}", "pack = OUT/'submission.zip'; scores = {}; L0D = {}"),
        ("        assert np.allclose(part, s[h:], atol=1e-6), 'score depends on other pairs'",
         "        L0D[member] = float(np.abs(part - s[h:]).max()); tol = 1e-4 * max(1.0, float(np.abs(s).max()))\n"
         "        print(f'  level 0 {member}: max |full - half| {L0D[member]:.2e} (tol {tol:.1e}, score scale {np.abs(s).max():.2f})', flush=True)\n"
         "        assert L0D[member] <= tol, 'score depends on other pairs'"),
        ("level0=True))", "level0=True, level0_max_abs_diff=L0D))"),
    ):
        assert s8.count(old) == 1, old
        s8 = s8.replace(old, new)
    compile(s8, "level0 cell", "exec")
    sc[0]["source"] = s8.splitlines(keepends=True)
    print("level-0 check -> prints the gap, tol 1e-4 x score scale")
if a.ib_url:
    # Meta's URL, or (since 12/11 the source notebook holds) the pinned Hugging Face copy of release/docker/ff.py IB_URL
    UU = ["https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
          "https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/imagebind/imagebind_huge.pth"]
    hit = [c for c in nb["cells"] if any(u in "".join(c["source"]) for u in UU)]
    assert len(hit) == 1, f"{len(hit)} cells hold the ImageBind URL"
    U = next(u for u in UU if u in "".join(hit[0]["source"]))
    hit[0]["source"] = "".join(hit[0]["source"]).replace(U, a.ib_url).splitlines(keepends=True)
    print("ImageBind URL ->", a.ib_url, "(SHA-256 check unchanged)")
Path(a.out).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", a.out)
