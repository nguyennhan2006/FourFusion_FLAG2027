"""Run one notebook inside the container the way Kaggle runs it, then save its outputs.

    python /opt/fourfusion/docker/run_nb.py <in.ipynb> <stage>

- The notebook runs in /kaggle/working inside the container (a Linux file system, as on Kaggle). Several notebooks symlink
  input folders there; a bind mount from a Windows host cannot hold symlinks, so /kaggle/working is not the host folder.
- Cell output is streamed to the console line by line (the notebooks print `[cell] ...` / progress lines with flush=True).
- At the end, success or not, every regular file of /kaggle/working is copied to /kaggle/out (the host's stage folder).
  Symlinks are skipped: they point into /kaggle/input. The executed notebook is written to /kaggle/out/<stage>.executed.ipynb
  after every cell.

Exit code 0 only when every cell ran."""
import os
import shutil
import sys
import time

import papermill as pm

WORK, OUT = "/kaggle/working", "/kaggle/out"


def save_outputs():
    n = size = 0
    for root, dirs, files in os.walk(WORK, followlinks=False):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
        for f in files:
            src = os.path.join(root, f)
            if os.path.islink(src):
                continue
            dst = os.path.join(OUT, os.path.relpath(src, WORK))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n += 1; size += os.path.getsize(src)
    print(f"[docker] saved {n} files ({size / 1e6:.1f} MB) -> {OUT}", flush=True)


def main():
    src, stage = sys.argv[1], sys.argv[2]
    os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    os.chdir(WORK)
    t0, ok = time.time(), False
    print(f"[docker] run {src} (stage {stage})", flush=True)
    try:
        pm.execute_notebook(src, os.path.join(OUT, f"{stage}.executed.ipynb"), kernel_name="python3", cwd=WORK,
                            stdout_file=sys.stdout, stderr_file=sys.stderr, progress_bar=False,
                            request_save_on_cell_execute=True, execution_timeout=None)
        ok = True
    except pm.PapermillExecutionError as e:
        print(f"[docker] FAILED in cell {e.exec_count}: {e.ename}: {e.evalue}", flush=True)
    finally:
        save_outputs()
    print(f"[docker] {'ALL CELLS OK' if ok else 'STOPPED'} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
