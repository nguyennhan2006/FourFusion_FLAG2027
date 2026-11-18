# ==== FourFusion end-to-end orchestrator (embedded verbatim in the notebook; split into cells at "# %% <title>") ====
# Every original Kaggle run of the 16.18 system is one STAGE. A stage reads its inputs from a private copy of /kaggle/input
# (VIN: real directories + file symlinks, exact Kaggle layout datasets/<owner>/<slug>, notebooks/<owner>/<slug>) and writes
# into its own /kaggle/working (RUN dir). What it leaves there becomes an ARTIFACT (ART/<artifact id>/ + _DONE.json), which
#   - feeds the later stages of the same session;
#   - is saved as the notebook output, so the NEXT session (this output attached as input) continues where this one stopped.
# Artifact ids are the ids of the original Kaggle sources ("datasets/<owner>/<slug>", "notebooks/<owner>/<slug>").
# Where an artifact comes from: made in this session > output of an earlier session of this notebook > dataset / kernel output
# attached by hand (the original intermediate datasets: the "already imported" branch) > run the stage that makes it.

# %% bộ điều phối: đường dẫn, trạng thái, bí mật
import os, sys, json, time, shutil, subprocess, re, glob
from pathlib import Path

T0 = time.time()
KIN = Path(os.environ.get("FF_KIN", "/kaggle/input"))
KOUT = Path(os.environ.get("FF_KOUT", "/kaggle/working"))
TMP = Path(os.environ.get("FF_TMP", "/kaggle/tmp" if Path("/kaggle/tmp").is_dir() else "/tmp"))
ART = KOUT / "ff_art"          # artifacts made in this session (part of the notebook output)
CODE = KOUT / "ff_code"        # stage code captured from the cells below (part of the notebook output)
VIN = TMP / "ff_vin"           # virtual /kaggle/input of the running stage
RUNS = TMP / "ff_run"          # /kaggle/working of each stage
for _p in (ART, CODE, VIN, RUNS):
    _p.mkdir(parents=True, exist_ok=True)
STATE_FILE = ART / "_STATE.json"
STATE = {"sessions": [], "stages": {}}
for _q in sorted(KIN.glob("notebooks/*/*/ff_art/_STATE.json")) + sorted(KIN.glob("*/ff_art/_STATE.json")):
    try:                                                    # history of the earlier sessions (report only)
        STATE["sessions"] += json.loads(_q.read_text()).get("sessions", [])
    except Exception:
        pass
STATE["sessions"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": MODE, "target": TARGET, "smoke": SMOKE, "only": ONLY})
HAS_GPU = shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
for _k in ("HF_TOKEN",):                                    # Kaggle secret -> environment of the stages (never printed)
    if not os.environ.get(_k):
        try:
            from kaggle_secrets import UserSecretsClient
            _v = UserSecretsClient().get_secret(_k)
            if _v:
                os.environ[_k] = _v
        except Exception:
            pass
print(f"KIN {KIN} | KOUT {KOUT} | TMP {TMP} | GPU {HAS_GPU} | MODE {MODE} | SMOKE {SMOKE} | ONLY {ONLY} | "
      f"HF_TOKEN {'có' if os.environ.get('HF_TOKEN') else 'không'}", flush=True)


def save_state():
    STATE_FILE.write_text(json.dumps(STATE, indent=1, ensure_ascii=False))


def elapsed_min():
    return (time.time() - T0) / 60


# %% bộ điều phối: tìm artifact
def _inp(e):
    """Input entry -> (path in the stage's /kaggle/input, artifact id)."""
    return (e[0], e[1]) if isinstance(e, (list, tuple)) else (e, e)


def _attached(aid):
    """A dataset / kernel output attached by hand, matched by slug (the owner may differ: copies under another account are fine)."""
    kind, owner, slug = aid.split("/")
    cands = [KIN / kind / owner / slug] + sorted((KIN / kind).glob(f"*/{slug}")) + [KIN / slug]
    return next((p for p in cands if p.is_dir()), None)


def _imported(aid):
    """Output of an earlier session of this notebook (its /kaggle/working/ff_art attached as input)."""
    for pat in (f"notebooks/*/*/ff_art/{aid}/_DONE.json", f"datasets/*/*/ff_art/{aid}/_DONE.json",   # dataset = output of a session
                f"datasets/*/*/*/ff_art/{aid}/_DONE.json", f"*/ff_art/{aid}/_DONE.json"):                # on another account
        hit = sorted(KIN.glob(pat))
        if hit:
            return hit[0].parent
    return None


def locate(aid, allow_old=True):
    """-> (path, how), how in made | imported | attached; (None, None) when nowhere. allow_old=False: only made in this session."""
    p = ART / aid
    if (p / "_DONE.json").exists():
        return p, "made"
    if aid in RAW:                                          # raw data is always attached by hand
        q = _attached(aid)
        return (q, "attached") if q else (None, None)
    if not allow_old or not USE_IMPORTED:
        return None, None
    q = _imported(aid)
    if q:
        return q, "imported"
    q = _attached(aid)
    return (q, "attached") if q else (None, None)


def _forced(aid):
    return aid in FORCE or PRODUCER.get(aid) in FORCE


def have(aid):
    return locate(aid, allow_old=not _forced(aid))[0] is not None


def mirror(src, dst):
    """Directory tree as real directories + file symlinks (Path.rglob / os.walk never follow directory symlinks).
    Falls back to hard links, then copies, where symlinks are not allowed (local Windows test)."""
    n = 0
    for root, _dirs, files in os.walk(src, followlinks=True):
        r = Path(root)
        d = dst / r.relative_to(src)
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            t = d / f
            if os.path.lexists(t):
                continue
            try:
                os.symlink(r / f, t)
            except OSError:
                try:
                    os.link(r / f, t)
                except OSError:
                    shutil.copy2(r / f, t)
            n += 1
    return n


# %% bộ điều phối: kế hoạch
def stage_outputs(name):
    return [S[name]["kernel"]] + list(S[name].get("exports", {}))


PRODUCER = {a: n for n in S for a in stage_outputs(n)}


def make_plan(targets):
    """Stages needed for `targets` in dependency order, and what must be attached by hand."""
    order, blocked, seen = [], [], set()

    def visit(aid, via):
        if aid in seen:
            return
        seen.add(aid)
        if have(aid):
            return
        prod = PRODUCER.get(aid)
        if prod is None:
            blocked.append((aid, via, "dữ liệu gốc / chưa có stage sinh ra -> gắn vào input"))
            return
        if MODE == "infer" and S[prod]["kind"] == "train":
            blocked.append((aid, via, f"MODE='infer' không học lại ({prod}) -> gắn dataset / output đã học vào input"))
            return
        for e in S[prod]["inputs"]:
            visit(_inp(e)[1], prod)
        if prod not in order:
            order.append(prod)

    for t in targets:
        visit(t, "TARGET")
    return order, blocked


def show_plan():
    global PLAN, BLOCKED
    PLAN, BLOCKED = make_plan([TARGET] if isinstance(TARGET, str) else list(TARGET))
    tot = sum(S[n]["minutes"] for n in PLAN)
    print(f"Đích: {TARGET} | MODE {MODE} | {len(PLAN)} stage cần chạy, ~{tot / 60:.1f} giờ theo lượt gốc"
          f"{' (SMOKE: ngắn hơn nhiều)' if SMOKE else ''}:", flush=True)
    for n in PLAN:
        sel = "" if not ONLY or n in ONLY else "   (ngoài ONLY: phiên khác chạy)"
        print(f"  {n:16s} {S[n]['kind']:6s} {'GPU' if S[n].get('gpu') else 'CPU'} ~{S[n]['minutes']:4.0f} phút  {S[n]['title']}{sel}", flush=True)
    if BLOCKED:
        print(f"\nThiếu {len(BLOCKED)} artifact (stage cần nó sẽ dừng ở trạng thái 'blocked'):", flush=True)
        for aid, via, why in BLOCKED:
            print(f"  {aid:66s} cho {via:12s} {why}", flush=True)


# %% bộ điều phối: chạy một stage
CELL_SEP = "\n# <<<FF-CELL>>>\n"
RUNNER = '''import sys
from IPython.core.interactiveshell import InteractiveShell
sh = InteractiveShell.instance(colors="NoColor")
cells = open(sys.argv[1], encoding="utf-8").read().split(%r)
for i, c in enumerate(cells):
    if not c.strip():
        continue
    r = sh.run_cell(c, store_history=False)
    if not r.success:
        e = r.error_in_exec or r.error_before_exec
        print(f"[ff] cell {i} of the stage failed: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
''' % CELL_SEP


class StageBlocked(Exception):
    pass


def _build_vin(name):
    shutil.rmtree(VIN, ignore_errors=True)
    VIN.mkdir(parents=True)
    rows = []
    for e in S[name]["inputs"]:
        vid, aid = _inp(e)
        p, how = locate(aid, allow_old=not _forced(aid))
        n = mirror(p, VIN / vid)
        rows.append(f"    {vid:66s} {how:9s} {n:7d} file  <- {p}" + ("" if vid == aid else f"  (artifact {aid})"))
    print("  input:\n" + "\n".join(rows), flush=True)


def _stage_code(name, run_dir):
    """Stage code with Kaggle's two path constants pointed at the stage's own input view and working dir."""
    src = (CODE / f"{name}.ipy").read_text(encoding="utf-8")
    src = src.replace("/kaggle/input", VIN.as_posix()).replace("/kaggle/working", run_dir.as_posix())
    p = run_dir / "_stage.ipy"
    p.write_text(src, encoding="utf-8")
    (run_dir / "_runner.py").write_text(RUNNER, encoding="utf-8")
    return p


def _collect(name, run_dir, minutes):
    """Kernel artifact = everything the stage left in its /kaggle/working (minus scratch); exports = files re-packed as old datasets."""
    st = S[name]
    kdir = ART / st["kernel"]
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    skip = set(st.get("scratch", [])) | {"_stage.ipy", "_runner.py"}
    for x in run_dir.iterdir():
        if x.name not in skip:
            shutil.move(str(x), str(kdir / x.name))
    done = {"stage": name, "minutes": round(minutes, 1), "mode": MODE, "smoke": SMOKE, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (kdir / "_DONE.json").write_text(json.dumps(done))
    for aid, spec in st.get("exports", {}).items():         # "subdir" (the dataset = that folder's content) or {"dst": "src or glob"}
        edir = ART / aid
        shutil.rmtree(edir, ignore_errors=True)
        if isinstance(spec, str):
            assert (kdir / spec).is_dir(), f"{name}: export {aid}: thiếu thư mục {spec}"
            shutil.copytree(kdir / spec, edir)
            (edir / "_DONE.json").write_text(json.dumps({**done, "export_of": f"{st['kernel']}/{spec}"}))
            continue
        for dst, src in spec.items():
            hits = sorted(kdir.glob(src)) if any(c in src for c in "*?[") else [kdir / src]
            assert hits and all(h.exists() for h in hits), f"{name}: export {aid}: thiếu {src}"
            for h in hits:
                t = edir / dst / h.name if dst.endswith("/") else edir / dst
                t.parent.mkdir(parents=True, exist_ok=True)
                (shutil.copytree if h.is_dir() else shutil.copy2)(h, t)
        (edir / "_DONE.json").write_text(json.dumps({**done, "export_of": st["kernel"]}))


def run(name, sha=None, ncell=None):
    """Run stage `name` when the plan needs it; never trains in MODE='infer'. The stage code is loaded (load_code) only when
    the stage really runs. Returns the status."""
    st = S[name]
    rec = STATE["stages"].setdefault(name, {})
    if name not in PLAN:
        have_ = [a for a in stage_outputs(name) if have(a)]
        msg = "đã có output" if have_ else "không cần cho đích của phiên này"
        print(f"[stage] {name}: {msg}, không chạy.", flush=True)
        rec.update(status="present" if have_ else "not_needed")
        return rec["status"]
    if ONLY and name not in ONLY:
        print(f"[stage] {name}: ngoài ONLY (phiên khác chạy), bỏ qua.", flush=True)
        rec.update(status="not_selected")
        return "not_selected"
    miss = [_inp(e)[1] for e in st["inputs"] if not have(_inp(e)[1])]
    if miss:
        print(f"[stage] {name}: thiếu input {miss} -> blocked.", flush=True)
        rec.update(status="blocked", missing=miss)
        save_state()
        return "blocked"
    est = st["minutes"] * (0.1 if SMOKE else 1.0)
    left = TIME_BUDGET_H * 60 - elapsed_min()
    if est > left:
        print(f"[stage] {name}: cần ~{est:.0f} phút, phiên này còn {left:.0f} phút -> để phiên sau "
              f"(gắn output phiên này vào input rồi chạy lại notebook).", flush=True)
        rec.update(status="deferred", est_min=est)
        save_state()
        return "deferred"
    if st.get("gpu") and not HAS_GPU and not SMOKE:
        raise StageBlocked(f"stage {name} cần GPU: bật Accelerator GPU, hoặc chạy SMOKE=True để thử trên CPU.")
    print(f"\n[stage] {name}: CHẠY ({st['title']}; ước ~{est:.0f} phút; phiên đã dùng {elapsed_min():.0f} phút)", flush=True)
    if sha is not None:
        load_code(name, sha, ncell)
    _build_vin(name)
    run_dir = RUNS / name
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    script = _stage_code(name, run_dir)
    env = {**os.environ, **{k: str(v) for k, v in st.get("env", {}).items()}, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           "FF_STAGE": name, "FF_MODE": MODE, "FF_SMOKE": "1" if SMOKE else "0"}
    env.setdefault("MPLBACKEND", "Agg")
    if SMOKE:
        env.update({k: str(v) for k, v in st.get("smoke_env", {}).items()})
    t = time.time()
    p = subprocess.Popen([sys.executable, "-u", str(run_dir / "_runner.py"), str(script)], cwd=run_dir, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    tail = []
    for line in p.stdout:
        print(f"  | {line}", end="", flush=True)
        tail = (tail + [line])[-40:]
    code = p.wait()
    minutes = (time.time() - t) / 60
    if code != 0:
        rec.update(status="failed", minutes=round(minutes, 1), tail="".join(tail[-15:]))
        save_state()
        raise RuntimeError(f"stage {name} lỗi (mã {code}) sau {minutes:.0f} phút; xem log phía trên.")
    _collect(name, run_dir, minutes)
    shutil.rmtree(run_dir, ignore_errors=True)
    rec.update(status="done", minutes=round(minutes, 1))
    save_state()
    print(f"[stage] {name}: XONG sau {minutes:.0f} phút -> {[str(ART / a) for a in stage_outputs(name)]}", flush=True)
    return "done"


def out(aid):
    """Path of an artifact for the visualisation cells (None when not available in this session)."""
    return locate(aid)[0]


def keep_code(name, append=True):
    """Write the code in the `if False:` block of the CURRENT cell into ff_code/<name>.ipy (4-space indent removed).
    The block is an original cell, verbatim (only the patches listed in the stage header applied); it is not executed here
    but by run(<stage>) in a fresh process."""
    raw = get_ipython().history_manager.input_hist_raw[-1]
    assert "keep_code(" in raw, "không đọc được mã của cell đang chạy (lịch sử IPython)"
    lines = raw.split("\n")
    i = next(k for k, l in enumerate(lines) if l.rstrip() == "if False:")
    body = lines[i + 1:]
    assert all(l.startswith("    ") or l == "" for l in body), "khối mã phải thụt đúng 4 dấu cách"
    with open(CODE / f"{name}.ipy", "a" if append else "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(l[4:] for l in body).rstrip("\n") + CELL_SEP)


import hashlib


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_code(name, sha, ncell):
    """Stage code: written by the keep_code cells just above (full notebook), else taken from the code dataset attached as input
    (its ff_e2e_code_manifest.json; the Kaggle notebook stays under the API size limit). The sha256 must match this notebook."""
    p = CODE / f"{name}.ipy"
    if p.exists() and _sha(p) == sha:
        print(f"  mã {name}: {ncell} cell gốc, từ các cell ngay trên (sha256 {sha[:12]})", flush=True)
        return
    if p.exists():                                          # inline cells present but different: keep them for diagnosis
        shutil.copy2(p, CODE / f"{name}.inline_mismatch.ipy")
    for k in range(1, 6):
        for m in sorted(KIN.glob("/".join(["*"] * k) + "/ff_e2e_code_manifest.json")):
            q = m.parent / f"{name}.ipy"
            if q.exists() and _sha(q) == sha:
                shutil.copy2(q, p)
                print(f"  mã {name}: {ncell} cell gốc, từ dataset mã {m.parent} (sha256 {sha[:12]})", flush=True)
                return
    raise FileNotFoundError(f"không thấy mã stage {name} có sha256 {sha[:12]}: gắn dataset mã cùng phiên bản với notebook này "
                            f"(file ff_e2e_code_manifest.json), hoặc dùng bản notebook đầy đủ.")


def find_ref(name, depth=5):
    """A reference file attached anywhere in /kaggle/input (depth-limited search)."""
    for k in range(1, depth + 1):
        hit = sorted(KIN.glob("/".join(["*"] * k) + "/" + name))
        if hit:
            return hit[0]
    return None


def report():
    print(f"\nPhiên dùng {elapsed_min():.0f} phút. Trạng thái stage:", flush=True)
    for name in S:
        r = STATE["stages"].get(name, {})
        outs = "".join("x" if have(a) else "-" for a in stage_outputs(name))
        print(f"  {name:16s} {S[name]['kind']:6s} {r.get('status', '-'):13s} {str(r.get('minutes', '')):>7} phút  output {outs}", flush=True)
    pend = [n for n in PLAN if STATE["stages"].get(n, {}).get("status") in ("deferred", "blocked", "not_selected")]
    if pend:
        print(f"\nCòn {len(pend)} stage chưa xong: {pend}.\nPhiên sau: gắn OUTPUT của phiên này (notebook) vào input, giữ nguyên các "
              f"dataset gốc, chạy lại notebook (có thể đặt ONLY = [...] để chia việc cho nhiều tài khoản).", flush=True)
    save_state()
