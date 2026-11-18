"""Stage table of the 16.18 system (C1FIX-01 = FACTOR_ENFUSE c5rl with the fixed LoRA C1), read by build_notebook.py.

Every stage = one original Kaggle run (exact code from the package FACTOR_ENFUSE_c5rl, notebooks/1_train or notebooks/2_eval),
or a new small step (kind "new"). Artifact ids are the ids of the original Kaggle sources ("datasets/<owner>/<slug>",
"notebooks/<owner>/<slug>"): a stage's output IS what the original run left in its /kaggle/working, so downstream stages read
the same paths as in the original runs.

Fields: name, src ("pkg:<notebook path>" | "new"), kernel (artifact id of the stage's /kaggle/working), kind ("train" = learns
weights / tables, "infer" = Evaluation path, "data" = features from media), gpu, minutes (wall time of the original run),
title, role (markdown), inputs (artifact ids; a pair (vin_id, artifact_id) mounts artifact_id at the path vin_id), smoke_env,
patches [(old, new, count)], drop_cells [regex on the cell's first line], exports {artifact id: {dst: src}}, viz (code)."""
import json
import os
from pathlib import Path

PKG = Path(os.environ.get("FF_PKG", "E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl"))
RUNS = json.loads((PKG / "notebooks/RUNS.json").read_text(encoding="utf-8"))

RAW = {
    "datasets/ngnnguynchn/flag2027-raw-v4-p": "MAV-Celeb v4 của BTC: train_set.zip, dev_set.zip, meta_file_train_set.csv (nguyên bản BTC)",
    "datasets/ngnnguynchn/flag2027-evalphase-p": "dữ liệu Evaluation của BTC (đổi tên danh sách *_valid -> *_test, sửa đường dẫn danh sách cùng giới / English)",
    "datasets/chnnhnnguyn/external-data-mavceleb": "MAV-Celeb v1 / v2 (media + meta)",
}
IMAGEBIND_DEAD = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
IMAGEBIND_HF = "https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/imagebind/imagebind_huge.pth"
GLOBAL_PATCHES = [(IMAGEBIND_DEAD, IMAGEBIND_HF, None)]   # same file (sha256 d6f6c22b...), Meta's link is dead since 12/11/2026

K = {  # kernel ids of the Evaluation runs
    "65xbs": "notebooks/thanhhuyen11/flag-2027-nb-65xbs-b3sx-eval",
    "65xcs": "notebooks/thanhhuyen11/flag-2027-nb-65xcs-c5sx-eval",
    "47rs": "notebooks/thanhhuyen11/flag-2027-nb-47rs-screen2-eval",
    "50cc": "notebooks/thanhhuyen11/flag-2027-nb-50cc-e2e-eval",
    "112f0sx": "notebooks/thanhhuyen11/flag-2027-nb-112f0sx-eval",
    "112f1sx": "notebooks/thanhhuyen11/flag-2027-nb-112f1sx-eval",
    "163i": "notebooks/thanhhuyen11/flag-2027-nb-163i-rfsfxc5rl-eval",
    "final": "notebooks/ff/fourfusion-1618-final",
}
C1FIX_ART = "datasets/ngnnguynchn/flag2027-c1fix-rfx-art-p"    # RFX artefact of 16.18 (kit C1FIX01 rfx/art)


def run_inputs(key):
    r = RUNS[key]
    return [f"datasets/{d}" for d in r["datasets"]] + [f"notebooks/{k}" for k in (r.get("kernel_inputs") or [])]


def pkg(name, key, nbfile, **kw):
    d = dict(name=name, src=f"pkg:{nbfile}", kernel=f"notebooks/{key}", inputs=run_inputs(key), patches=[], drop_cells=[],
             exports={}, smoke_env={}, env={})
    d.update(kw)
    return d


SRC_NB = Path(__file__).resolve().parent / "src_nb"      # exact notebooks of runs outside the package (pull_runs.py)


def pulled(name, key, **kw):
    """A run that is not in the package: its notebook as pulled from Kaggle (kernels pull) + its kernel-metadata.json."""
    slug = key.split("/")[1]
    m = json.loads((SRC_NB / slug / "kernel-metadata.json").read_text(encoding="utf-8"))
    d = dict(name=name, src=f"nb:{slug}", kernel=f"notebooks/{key}", patches=[], drop_cells=[], exports={}, smoke_env={}, env={},
             inputs=[f"datasets/{x}" for x in m.get("dataset_sources", [])] + [f"notebooks/{x}" for x in m.get("kernel_sources", [])])
    d.update(kw)
    return d


S = []

# ------------------------------------------------------------------------------------------------ Tier 1: Evaluation path
S.append(pkg("e65xbs", "thanhhuyen11/flag-2027-nb-65xbs-b3sx-eval", "2_eval/01_flag-2027-nb-65xbs-b3sx-eval.ipynb",
    kind="infer", gpu=True, minutes=16, title="Mặt CLIP-L + adapter B3s cho 4 danh sách thi",
    role="Nạp adapter LoRA B3s đã học (`facelora_state.pt`, không học lại), nhúng mặt của 4 danh sách thi; chép mảng train / v1 / v2 / "
         "VoxCeleb2 / VoxBlink từ dataset adapter. Chỉ dùng để nối dây cho 112f0sx / 112f1sx (kiểm khớp hàng); không vào số của zip cuối.",
    smoke_env={"FLAG_SMOKE": "1", "FLAG_SMOKE_ROWS": "24"},
    viz='''
d = need(K["65xbs"])
if d is not None:
    E_B3, _ = embedding_figure(d / "feats_clipb3c_x", "face_clipB3", "face", "Stage e65xbs: mặt CLIP-L + adapter B3s")
    show_json(d / "feats_clipb3c_x/feats_manifest.json", ["created", "checks", "failed"])
'''))
S.append(pkg("e65xcs", "thanhhuyen11/flag-2027-nb-65xcs-c5sx-eval", "2_eval/02_flag-2027-nb-65xcs-c5sx-eval.ipynb",
    kind="infer", gpu=True, minutes=21, title="Mặt CLIP-L + adapter C5s cho 4 danh sách thi",
    role="Nạp adapter LoRA C5s (`facelora_state.pt`), nhúng mặt thi. Vào zip qua hai đường: khối mặt [farl16 ‖ clipC5] của đầu CA (`trCA`) "
         "và khối `f_c5s` của RFX.",
    smoke_env={"FLAG_SMOKE": "1", "FLAG_SMOKE_ROWS": "24"},
    viz='''
d = need(K["65xcs"])
if d is not None:
    E_C5, _ = embedding_figure(d / "feats_clipc5c_x", "face_clipC5", "face", "Stage e65xcs: mặt CLIP-L + adapter C5s")
    if "E_B3" in globals():
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for i in range(4):
            ax.hist((unit(E_B3[i]) * unit(E_C5[i])).sum(1), 50, density=True, histtype="step", color=COL[i], label=LAB[i])
        ax.set_title("cos giữa mặt B3s và mặt C5s của cùng một ảnh thi"); ax.legend(fontsize=7); plt.show()
'''))
S.append(pkg("e47rs", "thanhhuyen11/flag-2027-nb-47rs-screen2-eval", "2_eval/03_flag-2027-nb-47rs-screen2-eval.ipynb",
    kind="infer", gpu=True, minutes=240, title="Giọng ResNet293-LM và ECAPA-TDNN-1024-LM (đông cứng) cho thi + train",
    role="Hai encoder giọng WeSpeaker pretrained VoxCeleb2, không học lại. Mảng thi vào giọng CAT4 của đầu CA và khối `v_rl` của RFX; "
         "mảng train cho hằng số chuẩn hoá z của `trCA`. Phần MAV v1 / v2 (gần hết 4 giờ) không ảnh hưởng zip.",
    smoke_env={"FLAG_SMOKE": "1", "FLAG_SMOKE_ROWS": "24"},
    exports={"datasets/ngnnguynchn/flag2027-feats-screen2-p": "feats_screen2"},   # same 21 files as the old FLAG_47b dataset (cos 1.000000)
    viz='''
d = need(K["47rs"])
if d is not None:
    embedding_figure(d / "feats_screen2", "voice_rn293", "voice", "Stage e47rs: giọng ResNet293-LM")
    embedding_figure(d / "feats_screen2", "voice_ecapa1024", "voice", "Stage e47rs: giọng ECAPA-TDNN-1024-LM")
'''))
S.append(pkg("e50cc", "thanhhuyen11/flag-2027-nb-50cc-e2e-eval", "2_eval/04_flag-2027-nb-50cc-e2e-eval.ipynb",
    kind="infer", gpu=True, minutes=221, title="Các luồng S4 (XALL) + đầu T3 v11 + điều kiện giới / tuổi",
    role="Trích ImageBind / FaRL / CLIP-L / SimAM / tuổi / giới cho thi + train; chấm `s007_xall.pt` (4 luồng XALL) và đầu T3 v11 "
         "(`anP512_deep`, `anP512_cca`) đã lưu; học bộ phân loại giới mặt (logistic, 23 018 mặt) ngay trong lượt; ghi `components_eval.npz` "
         "và chuyển tiếp bảng SAFE-04 (`e2e_eval/safe04`) cho bộ trộn F1s′.",
    viz='''
d = need(K["50cc"])
if d is not None:
    C = np.load(d / "e2e_eval/components_eval.npz"); MAN = json.loads((d / "e2e_eval/e2e_manifest.json").read_text())
    print("file:", [(f["key"], f["rows"]) for f in MAN["files"]])
    names = ["XALL_deep0", "XALL_cca", "XALL_ib", "XALL_age", "anP512_deep", "anP512_cca"]
    Z = [np.c_[C[f"b3_{i}"][:, :4], C[f"t3_{i}"][:, :2]] for i in range(4)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.5))
    for k, nm in enumerate(names):
        logit_hist([z[:, k] for z in Z], f"luồng {nm}", ax.flat[k])
    fig.suptitle("6 luồng điểm đầu vào của F1s′ trên 4 file thi (không có nhãn: xem hình dạng / lệch giữa file)"); fig.tight_layout(); plt.show()
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    im = ax[0].imshow(np.corrcoef(Z[3].T), vmin=-1, vmax=1, cmap="RdBu_r"); ax[0].set_xticks(range(6), names, rotation=60, fontsize=7)
    ax[0].set_yticks(range(6), names, fontsize=7); ax[0].set_title("tương quan 6 luồng (cùng giới · Bangla)"); fig.colorbar(im, ax=ax[0])
    ax[1].bar(range(4), [C[f"co_{i}"][:, 3].mean() for i in range(4)], color=COL); ax[1].set_xticks(range(4), LAB, rotation=20, fontsize=7)
    ax[1].set_title("tỉ lệ cặp lệch giới dự đoán (ô cùng giới phải gần 0)")
    for i in range(4):
        ax[2].hist(C[f"co_{i}"][:, 6], 40, density=True, histtype="step", color=COL[i], label=LAB[i])
    ax[2].set_title("lệch tuổi mặt − giọng dự đoán"); ax[2].legend(fontsize=7); fig.tight_layout(); plt.show()
    show_json(d / "e2e_eval/e2e_manifest.json", ["created", "kind", "split"], width=400)
'''))
# ------------------------------------------------------------------------------------------------ Tier 2: RFX training (16.18 variant)
import zipfile as _zf

KITZIP = Path(os.environ.get("FF_C1FIX_KIT", str(Path(__file__).resolve().parents[2] / "Experiment/C1FIX-01/kit/C1FIX01_kit.zip")))
_kit = lambda n: _zf.ZipFile(KITZIP).read("C1FIX01_kit/" + n).decode("utf-8").replace("\r\n", "\n")
C1FIX_LORA = "datasets/ngnnguynchn/flag2027-c1fix-lora-p"     # lora_C1.pt + v4_extra.npz of 16.18 (VoxCeleb2-only, fixed cache key)


def _sub(text, old, new):
    assert text.count(old) == 1, f"patch target found {text.count(old)} times: {old[:80]!r}"
    return text.replace(old, new)


def noup(var, val):
    """Original runs uploaded their output to a dataset; here the output becomes an artifact."""
    return (f"os.environ['{var}'] = '{val}'", f"# end-to-end: không tải lên dataset (output thành artifact). Gốc: os.environ['{var}'] = '{val}'", 1)


TRAIN_ONLY = ('SPLIT_FILTER = None            # e.g.', 'SPLIT_FILTER = ["train"]      # end-to-end: chỉ tập train (không đụng dev); gốc None. e.g.', 1)

VIZ_LANES = '''
d = need(K_{k})
if d is not None:
    r = d / "{sub}"; rows = []
    for src in ("vox", "vox2b", "vb"):
        for f in sorted((r / src).glob("*.npy")):
            X = np.load(f, mmap_mode="r"); rows.append((src, f.stem, len(X), float(np.isfinite(np.asarray(X[:, 0], float)).mean())))
    print(pd.DataFrame(rows, columns=["nguồn", "khối", "hàng", "tỉ lệ có giá trị trong làn này"]).to_string(index=False))
    show_json(r / "feats_manifest.json", ["created", "checks", "failed", "broken"], width=600)
'''
S.append(pkg("t160a", "tranthuytien2511/flag-2027-nb-160a-whv3", "1_train/08_flag-2027-nb-160a-whv3.ipynb",
    kind="data", gpu=True, minutes=214, title="Whisper-large-v3 (khối `whv3`) cho v4 train + MAV v1 / v2",
    role="Encoder Whisper-large-v3 (30 giây đầu, trung bình trạng thái ẩn) cho giọng v4 train và MAV v1 / v2 (12 giây giữa). Khối `v_whv3` của RFX.",
    patches=[noup("WH_UPLOAD", "tranthuytien2511/flag2027-feats-whv3-a-p"), TRAIN_ONLY],
    exports={"datasets/tranthuytien2511/flag2027-feats-whv3-a-p": "feats_whv3_a"}, smoke_env={"FLAG_SMOKE": "1", "FLAG_SMOKE_ROWS": "16"},
    viz='''
d = need("datasets/tranthuytien2511/flag2027-feats-whv3-a-p")
if d is not None:
    X = np.load(d / "voice_whv3_train.npy").astype(np.float32); rows, ids = train_index(d, "voice")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)); identity_panel(ax[0], X[rows], ids, "giọng (Whisper-large-v3)")
    pca_panel(ax[1], [("v4 train", X, "0.6")] + [(f"MAV {v}", np.load(d / f"ext_{v}_complete_voice_whv3.npy").astype(np.float32), c) for v, c in (("v1", "#1f77b4"), ("v2", "#ff7f0e"))])
    fig.tight_layout(); plt.show()
'''))
S.append(pkg("t160b", "thanhhuyen11/flag-2027-nb-160b-whv3", "1_train/09_flag-2027-nb-160b-whv3.ipynb",
    kind="data", gpu=True, minutes=138, title="Whisper-large-v3 cho VoxCeleb2 / VoxBlink (làn chẵn)",
    role="Cùng encoder trên đoạn 3 giây của kho audio VoxCeleb2 / VoxBlink, các khối chẵn của danh sách kho (làn c làm khối lẻ).",
    patches=[noup("WH_UPLOAD", "thanhhuyen11/flag2027-feats-whv3-b-p")],
    exports={"datasets/thanhhuyen11/flag2027-feats-whv3-b-p": "feats_whv3_b"}, smoke_env={"FLAG_SMOKE": "1", "FLAG_SMOKE_ROWS": "16"},
    viz=VIZ_LANES.replace("K_{k}", '"datasets/thanhhuyen11/flag2027-feats-whv3-b-p"').replace('d / "{sub}"', "d")))
S.append(pkg("t160c", "phuonggmaii/flag-2027-nb-160c-whv3", "1_train/10_flag-2027-nb-160c-whv3.ipynb",
    kind="data", gpu=True, minutes=122, title="Whisper-large-v3 cho VoxCeleb2 / VoxBlink (làn lẻ)",
    role="Như t160b, các khối lẻ.",
    patches=[noup("WH_UPLOAD", "phuonggmaii/flag2027-feats-whv3-c-p")],
    exports={"datasets/phuonggmaii/flag2027-feats-whv3-c-p": "feats_whv3_c"}, smoke_env={"FLAG_SMOKE": "1", "FLAG_SMOKE_ROWS": "16"},
    viz=VIZ_LANES.replace("K_{k}", '"datasets/phuonggmaii/flag2027-feats-whv3-c-p"').replace('d / "{sub}"', "d")))
S.append(pkg("t162a", "nguynchnngn1/flag-2027-nb-162a-rfs", "1_train/11_flag-2027-nb-162a-rfs.ipynb",
    kind="data", gpu=True, minutes=434, title="Các khối RFS (mặt CLIP-L / ImageBind / tuổi; giọng SimAM / ResNet221 / ECAPA / ImageBind / tuổi) cho v4 train + MAV",
    role="Bộ trích của công thức RF-092 trên v4 train và MAV v1 / v2. Bỏ dataset kiểm `flag2027-rf092-dev-emb-p` (chỉ để so với embedding dev của lượt "
         "tái lập RF-092; không có thì phép kiểm tự bỏ qua).",
    patches=[noup("RFS_UPLOAD", "nguynchnngn1/flag2027-feats-rfs-a-p"), TRAIN_ONLY],
    exports={"datasets/nguynchnngn1/flag2027-feats-rfs-a-p": "feats_rfs_a"}, smoke_env={"FLAG_SMOKE": "1", "FLAG_SMOKE_ROWS": "16"},
    viz='''
d = need("datasets/nguynchnngn1/flag2027-feats-rfs-a-p")
if d is not None:
    blocks = [("face", "rfclip"), ("face", "rfibf"), ("voice", "rfsam"), ("voice", "rfr221"), ("voice", "rfxv"), ("voice", "rfibv")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for a, (m, b) in zip(ax.flat, blocks):
        X = np.load(d / f"{m}_{b}_train.npy").astype(np.float32); rows, ids = train_index(d, m); identity_panel(a, X[rows], ids, f"{m} {b}")
    fig.suptitle("t162a: tách người theo từng khối RFS trên v4 train"); fig.tight_layout(); plt.show()
'''))
S[-1]["inputs"] = [a for a in S[-1]["inputs"] if "rf092-dev-emb" not in a]
S.append(pkg("t162b", "phanththui/flag-2027-nb-162b-rfs", "1_train/12_flag-2027-nb-162b-rfs.ipynb",
    kind="data", gpu=True, minutes=242, title="Các khối RFS cho VoxCeleb2 / VoxBlink (làn chẵn)",
    role="Bộ trích RFS trên khung mặt + đoạn 3 giây của VoxCeleb2 / VoxBlink (mặt: chỉ số chẵn; giọng: khối chẵn của kho). Giữ chốt thời gian 8 giờ của lượt gốc.",
    patches=[noup("RFS_UPLOAD", "phanththui/flag2027-feats-rfs-b-p")],
    exports={"datasets/phanththui/flag2027-feats-rfs-b-p": "feats_rfs_b"}, smoke_env={"FLAG_SMOKE": "1", "FLAG_SMOKE_ROWS": "16"},
    viz=VIZ_LANES.replace("K_{k}", '"datasets/phanththui/flag2027-feats-rfs-b-p"').replace('d / "{sub}"', "d")))
S.append(pkg("t162c", "phanvnnnnn6789/flag-2027-nb-162c-rfs", "1_train/13_flag-2027-nb-162c-rfs.ipynb",
    kind="data", gpu=True, minutes=237, title="Các khối RFS cho VoxCeleb2 / VoxBlink (làn lẻ)",
    role="Như t162b, chỉ số / khối lẻ.",
    patches=[noup("RFS_UPLOAD", "phanvnnnnn6789/flag2027-feats-rfs-c-p")],
    exports={"datasets/phanvnnnnn6789/flag2027-feats-rfs-c-p": "feats_rfs_c"}, smoke_env={"FLAG_SMOKE": "1", "FLAG_SMOKE_ROWS": "16"},
    viz=VIZ_LANES.replace("K_{k}", '"datasets/phanvnnnnn6789/flag2027-feats-rfs-c-p"').replace('d / "{sub}"', "d")))
S.append(pkg("t163b", "ngnnguynchn/flag-2027-nb-163-rfs-b", "1_train/14_flag-2027-nb-163-rfs-b.ipynb",
    kind="data", gpu=False, minutes=4, title="Ghép bundle RFS (`ext.npz`, `v4.npz`)",
    role="`assemble.py`: ghép 6 dataset đặc trưng (t160a/b/c, t162a/b/c) + meta MAV + danh sách v4 thành `ext.npz` (MAV + VoxCeleb2 + VoxBlink) và `v4.npz` (6 485 hàng v4 train).",
    patches=[noup("RFS_UPLOAD", "ngnnguynchn/flag2027-rfs-bundle-p")],
    exports={"datasets/ngnnguynchn/flag2027-rfs-bundle-p": "rfs_bundle"}, smoke_env={"FLAG_SMOKE": "1"},
    viz='''
d = need("datasets/ngnnguynchn/flag2027-rfs-bundle-p")
if d is not None:
    E = np.load(d / "ext.npz", allow_pickle=False); V = np.load(d / "v4.npz", allow_pickle=False)
    t = pd.DataFrame({"mặt": pd.Series(E["f_src"]).value_counts(), "giọng": pd.Series(E["v_src"]).value_counts()})
    t["người (giọng)"] = pd.Series({s: len(set(E["v_spk"][E["v_src"] == s])) for s in t.index}); print(t.to_string())
    fin = {k: float(np.isfinite(E[k].astype(np.float32)).all(1).mean()) for k in ("f_clip", "f_ibf", "v_sam", "v_r221", "v_xv", "v_whv3")}
    print("tỉ lệ hàng đủ giá trị:", {k: round(v, 4) for k, v in fin.items()})
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, k in zip(ax, ("v_sam", "v_r221", "v_xv", "v_whv3")):
        X = V[k].astype(np.float32); keep = ~pd.Series(V["v_path"]).duplicated().values; identity_panel(a, X[keep], V["v_spk"][keep], f"giọng {k[2:]}")
    fig.suptitle("t163b: tách người theo khối giọng trên v4 train"); fig.tight_layout(); plt.show()
'''))
S.append(dict(name="c1fix_lora", src="new", kernel="notebooks/ff/c1fix-lora-train", kind="train", gpu=True, minutes=60,
    title="LoRA C1 của ImageBind-audio, đã sửa khoá tra kho audio (C1FIX)",
    role="Đúng `c1fix_train.py` của đồng đội (lượt 16.18): chọn tối đa 3 video / người VoxCeleb2 trong bundle (bỏ người bị cờ `lid_strict.json`), "
         "đọc đoạn 3 giây từ kho audio với **khoá đã sửa** (phần sau dấu `/` của `v_vid`), học LoRA (hạng 8, alpha 16, 1 200 bước, lô 64) để giọng "
         "khớp ImageBind-mặt trung bình của người; nhúng 6 485 giọng v4. Chỉ đổi 3 hằng đường dẫn của máy đồng đội; ImageBind lấy bản sao HF cùng sha256.",
    inputs=["datasets/ngnnguynchn/flag2027-rfs-bundle-p", "datasets/ngnnguynchn/flag2027-lid-strict-p",
            "datasets/duongphihoang/flag2027-audio-cache-p", "datasets/ngnnguynchn/flag2027-raw-v4-p"],
    patches=[], drop_cells=[], exports={C1FIX_LORA: "lora"}, scratch=["out"], smoke_env={"RFS_LORA_STEPS": "3", "RFS_LORA_BATCH": "4"}, env={},
    cells=["print(\"[cell] c1fix lora: thư mục mã\", flush=True)\nimport os\nos.makedirs(\"code\", exist_ok=True)",
           "%%writefile code/flag_final_core_092.py\n" + _kit("code/flag_final_core_092.py"),
           "%%writefile code/rfs_train.py\n" + _kit("code/rfs_train.py"),
           "%%writefile c1fix_train.py\n" + _sub(_kit("c1fix_train.py"),
               'DATA = Path(r"D:/Download/FLAG 2027/_tam/c1fix_data"); BUNDLE, CACHE, LID = DATA / "bundle", DATA / "cache", DATA / "lid/lid_strict.json"\n'
               'IBCK = Path(r"D:/Download/imagebind_huge-001.pth"); V4ROOT = Path(r"D:/Download/FLAG 2027/_tam/realdata/train_set/train_set")',
               '_P = json.loads(Path("ff_paths.json").read_text())   # end-to-end: paths found by the next cell (were the teammate\'s local folders)\n'
               'BUNDLE, CACHE, LID = Path(_P["bundle"]), Path(_P["cache"]), Path(_P["lid"])\n'
               'IBCK = Path(_P["ibck"]); V4ROOT = Path(_P["v4root"])'),
           f'''print("[cell] c1fix lora: tìm input, tải ImageBind, chạy c1fix_train.py, đặt LoRA vào chỗ RFX đọc", flush=True)
import os, sys, json, hashlib, shutil, subprocess
from pathlib import Path
import numpy as np, torch
IN, W = Path("/kaggle/input"), Path("/kaggle/working")
find_all = lambda n: sorted(IN.rglob(n))
bundle = next(p.parent for p in find_all("ext.npz") if (p.parent / "v4.npz").exists())
lid = find_all("lid_strict.json")[0]
cache = next(p.parent.parent for p in find_all("tar0000.npz") if p.parent.name == "vox")
v4root = None
for d, dirs, files in os.walk(IN):
    if "train_English.txt" in files and Path(d, "faces").is_dir():
        v4root = Path(d); break
assert v4root, "thiếu raw-v4 (ngnnguynchn/flag2027-raw-v4-p)"
ck = Path("/tmp/ff_imagebind_huge.pth")
if not ck.exists():
    torch.hub.download_url_to_file("{IMAGEBIND_HF}", str(ck) + ".part", progress=False); os.replace(str(ck) + ".part", ck)
h = hashlib.sha256()
with open(ck, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
assert h.hexdigest().startswith("d6f6c22bedcc9070"), h.hexdigest()
Path("ff_paths.json").write_text(json.dumps(dict(bundle=str(bundle), cache=str(cache), lid=str(lid), ibck=str(ck), v4root=str(v4root))))
print(json.loads(Path("ff_paths.json").read_text()), flush=True)
p = subprocess.Popen([sys.executable, "-u", "c1fix_train.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
assert p.wait() == 0, "c1fix_train.py lỗi"
L = W / "lora"; L.mkdir(exist_ok=True)
shutil.copy2("out/lora_C1_fixed.pt", L / "lora_C1.pt")
v = np.load("out/v4_lora_fixed.npy").astype(np.float32)
assert v.shape == (6485, 1024) and np.isfinite(v).all(), v.shape
np.savez(L / "v4_extra.npz", v_lora=v)
shutil.copy2("out/v4_ibv_zeroshot.npy", W / "v4_ibv_zeroshot.npy")
print("LoRA C1 (sửa khoá):", sorted(x.name for x in L.iterdir()), flush=True)'''],
    viz='''
dl, db = need(C1FIX_LORA, "datasets/ngnnguynchn/flag2027-rfs-bundle-p") or (None, None)
k = out("notebooks/ff/c1fix-lora-train")
if dl is not None and k is not None:
    V = np.load(db / "v4.npz", allow_pickle=False); spk_f, spk_v = V["f_spk"].astype(str), V["v_spk"].astype(str)
    F = unit(V["f_ibf"].astype(np.float32)); P = sorted(set(spk_f)); fm = unit(np.stack([F[spk_f == p].mean(0) for p in P])); pi = {p: i for i, p in enumerate(P)}
    rng = np.random.default_rng(0); vi = rng.integers(0, len(spk_v), 6000); fj = rng.integers(0, len(P), 6000)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, (lab, X) in zip(ax, [("ImageBind-audio gốc", np.load(k / "v4_ibv_zeroshot.npy")), ("ImageBind-audio + LoRA C1 sửa", np.load(dl / "v4_extra.npz")["v_lora"])]):
        Xu = unit(X); pos = (Xu[vi] * fm[[pi[spk_v[i]] for i in vi]]).sum(1); neg = (Xu[vi] * fm[fj]).sum(1); neg = neg[np.array(P)[fj] != spk_v[vi]]
        a.hist(neg, 60, density=True, alpha=.6, label="giọng–mặt khác người"); a.hist(pos, 60, density=True, alpha=.6, label="giọng–mặt cùng người")
        a.set_title(f"{lab}: EER giọng–mặt (chỉ khối ib) {eer(pos, neg):.1f} %"); a.legend()
    fig.suptitle("c1fix_lora: LoRA kéo giọng về phía mặt trung bình của người (v4 train, 70 người đã thấy khi học bộ trộn)"); fig.tight_layout(); plt.show()
'''))
S.append(pkg("c1fix_rfx", "ngnnguynchn/flag-2027-nb-163-rfsfxc5rl-p", "1_train/19_flag-2027-nb-163-rfsfxc5rl-p.ipynb",
    kernel="notebooks/ff/c1fix-rfx-train", kind="train", gpu=True, minutes=9,
    title="Projector + bộ trộn RFX (mode P, khối thêm `f_c5s` + `v_rl`) với LoRA C1 đã sửa",
    role="Notebook 19 gốc (projector InfoNCE 3 seed, bộ trộn logistic, `RFS_CFG add f_c5s + v_rl`), **khác bản gốc ở LoRA**: dùng LoRA C1FIX (stage "
         "`c1fix_lora`) gắn đúng vào đường dẫn dataset LoRA cũ; tương đương `run_rfx.py` của đồng đội. Ra artifact RFX của hệ 16.18.",
    patches=[noup("RFS_UPLOAD", "ngnnguynchn/flag2027-rfsfxc5rl-art-p")],
    exports={C1FIX_ART: "rfs_art"}, smoke_env={"FLAG_SMOKE": "1"},
    viz='''
d = need(C1FIX_ART)
if d is not None:
    FR = json.loads((d / "fuse_report.json").read_text(encoding="utf-8")); show_json(d / "manifest_final.json", None, width=700)
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    for j, pr in enumerate(["no_gender", "gender"]):
        v = FR["reserve_eer"][pr]["per_projector"]; ax[0].bar(np.arange(len(v)) + 0.4 * j, v, 0.4, label=f"{pr} (trung bình {np.mean(v):.2f})")
    ax[0].set_title("EER trên 10 người giữ riêng, từng projector"); ax[0].legend()
    x = np.arange(2); ax[1].bar(x - .2, [FR["oof_eer"][p]["t3"] for p in ["no_gender", "gender"]], .4, label="luồng t3 riêng")
    ax[1].bar(x + .2, [FR["oof_eer"][p]["fused"] for p in ["no_gender", "gender"]], .4, label="sau bộ trộn RFX")
    ax[1].set_xticks(x, ["không ràng buộc giới", "cùng giới"]); ax[1].set_title("EER ngoài mẫu (người v4)"); ax[1].legend(); fig.tight_layout(); plt.show()
'''))
_i = S[-1]["inputs"].index("datasets/onnhqunhh/flag2027-rfs-lora-p")
S[-1]["inputs"][_i] = ("datasets/onnhqunhh/flag2027-rfs-lora-p", C1FIX_LORA)
S[-1]["inputs"] = [a for a in S[-1]["inputs"] if a != "datasets/trngsnam/flag2027-feats-peface01-pe0-p"]   # only the f_pel block reads it; c5rl does not

# ------------------------------------------------------------------------------------------------ Tier 3: F1s' training
def adapter_export(arr, sub):
    return {"./": f"{sub}/*{arr}*.npy", "facelora_state.pt": f"{sub}/facelora_state.pt", "train_log.json": f"{sub}/train_log.json",
            "bn42_clean.json": f"{sub}/bn42_clean.json"}


VIZ_ADAPTER = '''
d = need("{ds}")
if d is not None:
    tl = json.loads((d / "train_log.json").read_text()) if (d / "train_log.json").exists() else {{}}
    print({{k: tl[k] for k in list(tl)[:12] if not isinstance(tl[k], (list, dict))}})
    curves = {{k: v for k, v in tl.items() if isinstance(v, list) and len(v) > 5 and all(isinstance(x, (int, float)) for x in v[:5])}}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for k, v in list(curves.items())[:4]:
        ax[0].plot(np.asarray(v, float), lw=1, label=k)
    ax[0].set_title("nhật ký học adapter"); ax[0].legend(fontsize=7)
    X = np.load(d / "face_{arr}_train.npy")
    groups = [("VoxCeleb2 (vox)", np.load(d / "vox_face_{arr}.npy", mmap_mode="r")[::40], "#1f77b4"), ("VoxCeleb2 (vox2b)", np.load(d / "vox2b_face_{arr}.npy", mmap_mode="r")[::40], "#2ca02c"),
              ("VoxBlink", np.load(d / "vb_face_{arr}.npy", mmap_mode="r")[::80], "#d62728"), ("v4 train", X, "0.6")]
    pca_panel(ax[1], [(a, np.asarray(b, np.float32), c) for a, b, c in groups]); fig.tight_layout(); plt.show()
'''
S.append(pkg("t153b", "phanvnhtrng/flag-2027-nb-153b-b3s", "1_train/03_flag-2027-nb-153b-b3s.ipynb",
    kind="train", gpu=True, minutes=494, title="Adapter LoRA B3s trên CLIP-L (khối 18–23) + mảng mặt mọi nguồn",
    role="LoRA nhỏ (0.98 triệu tham số) trên CLIP-L, học để mặt khớp giọng [SimAM-ResNet100 ‖ bản ft] đông cứng: 4 600 bước, mỗi bước 16 người "
         "VoxCeleb2 + 16 người VoxBlink (lọc `lid_strict.json`); rồi nhúng mọi hàng (v4 train, MAV v1 / v2, VoxCeleb2, VoxBlink, 4 danh sách thử). "
         "Chỉ để nối dây (e65xbs, đầu B0s); không vào số của zip cuối.",
    patches=[noup("FL3_UPLOAD", "phanvnhtrng/flag2027-feats-ls-b3s-p")],
    exports={"datasets/phanvnhtrng/flag2027-feats-ls-b3s-p": adapter_export("clipB3", "feats_lidstrict_b3s")},
    smoke_env={"FLAG_SMOKE": "1", "FLAG_SMOKE_ROWS": "16", "SMOKE_IDS": "8"},
    viz=VIZ_ADAPTER.format(ds="datasets/phanvnhtrng/flag2027-feats-ls-b3s-p", arr="clipB3")))
S.append(pkg("t153c", "quynhnhu82/flag-2027-nb-153c-c5s", "1_train/04_flag-2027-nb-153c-c5s.ipynb",
    kind="train", gpu=True, minutes=712, title="Adapter LoRA C5s trên CLIP-L (khối 12–23, MLP + chú ý) + mảng mặt mọi nguồn",
    role="Như t153b với adapter lớn hơn (3.54 triệu tham số, hạng 16). Vào zip qua khối mặt của đầu CA và khối `f_c5s` của RFX. "
         "Lượt gốc dùng gần hết 12 giờ: chạy riêng một phiên (`ONLY = ['t153c']`).",
    patches=[noup("FL3_UPLOAD", "quynhnhu82/flag2027-feats-ls-c5s-p")],
    exports={"datasets/quynhnhu82/flag2027-feats-ls-c5s-p": adapter_export("clipC5", "feats_lidstrict_c5s")},
    smoke_env={"FLAG_SMOKE": "1", "FLAG_SMOKE_ROWS": "16", "SMOKE_IDS": "8"},
    viz=VIZ_ADAPTER.format(ds="datasets/quynhnhu82/flag2027-feats-ls-c5s-p", arr="clipC5")))
VIZ_FOLDTAB = '''
ps = [out(k) for k in {kernels}]
if any(p is None for p in ps):
    print("chưa có output của các lượt fold:", [k for k, p in zip({kernels}, ps) if p is None])
else:
    T = pd.concat([pd.read_parquet(f) for p in ps for f in sorted(p.rglob("{glob}"))], ignore_index=True)
    cols = [c for c in {cols} if c in T.columns]; print(len(T), "cặp SAFE-04; cột", cols)
    R = [(c, cell, f, eer_scores(g[c].values, g.y.values)) for c in cols for (cell, f), g in T.groupby([T.data.astype(str) + " " + T.protocol.astype(str), "fold"]) if g.y.nunique() == 2]
    R = pd.DataFrame(R, columns=["luồng", "ô", "fold", "EER"]).groupby(["luồng", "ô"]).EER.mean().unstack(0).round(2); print(R.to_string())
    R.plot.bar(figsize=(12, 3.6), title="EER out-of-fold theo ô SAFE-04 (trung bình fold)"); plt.tight_layout(); plt.show()
'''
F123, F45 = "notebooks/nguynlcnhtho/flag-2027-nb-164-cleancomp01-f123", "notebooks/dngxunqun/flag-2027-nb-164-cleancomp01-f45"
S.append(pulled("t164_f123", "nguynlcnhtho/flag-2027-nb-164-cleancomp01-f123", kind="train", gpu=True, minutes=16,
    title="CLEANCOMP-01 fold 1–3: học lại cầu XALL (S6) + đầu T3 v11 P512 theo fold, ghi điểm out-of-fold",
    role="Lượt không có trong gói (mã kéo từ Kaggle, đúng bản đã chạy). Mỗi fold SAFE-04: học cầu XALL (3 mạng × 1 200 bước) trên người v4-train của fold + MAV, "
         "đầu T3 v11 P512 trên mảng T3 (VoxCeleb2 bỏ 608 người bị cờ), chấm cặp giữ ra → bảng `cleancomp01_fold<s>.parquet`.",
    viz=VIZ_FOLDTAB.format(kernels=[F123, F45], glob="cleancomp01_fold*.parquet", cols=["XALL_deep0", "XALL_cca", "XALL_ib", "XALL_age", "anP512_deep", "anP512_cca"])))
S.append(pulled("t164_f45", "dngxunqun/flag-2027-nb-164-cleancomp01-f45", kind="train", gpu=True, minutes=12,
    title="CLEANCOMP-01 fold 4–5", role="Như t164_f123, fold 4 và 5."))
S.append(dict(name="cleancomp_tables", src="new", kernel="notebooks/ff/cleancomp-tables", kind="train", gpu=False, minutes=1,
    title="Bảng SAFE-04 CLEANCOMP (thay cột XALL / anP512 / rel bằng điểm out-of-fold mới)",
    role="Bản chuyển của `Experiment/CLEANCOMP-01/build_tables.py` (bước chạy ở máy, nay thành stage): chép bộ bảng SAFE-04 gốc (`e2e-dev-p`), thay các cột "
         "`XALL_*`, `anP512_*`, `relc_fia_eb`, `relg_fia_eb`, `t3_relg` bằng điểm của t164_f123 / t164_f45 (kiểm khớp hàng y / fi / vi / data). "
         "**Khác bản gốc:** đọc thư mục dataset thay vì file zip ở máy.",
    inputs=["datasets/ngnnguynchn/flag2027-e2e-dev-p", F123, F45], patches=[], drop_cells=[], smoke_env={}, env={},
    exports={"datasets/ngnnguynchn/flag2027-cleancomp-e2e-p": "e2edev"},
    cells=['''print("[cell] cleancomp tables: thay cột bằng điểm out-of-fold của FLAG_164", flush=True)
import io, shutil
from pathlib import Path
import numpy as np, pandas as pd
IN, OUT = Path("/kaggle/input"), Path("/kaggle/working")
BASE = next(p for p in IN.rglob("e2edev") if p.is_dir() and (p / "e2e_manifest.json").exists())
SRC = {int(p.stem.split("fold")[-1]): p for p in IN.rglob("cleancomp01_fold*.parquet")}
assert sorted(SRC) == [1, 2, 3, 4, 5], sorted(SRC)
NEW = {"streams": ["XALL_deep0", "XALL_cca", "XALL_ib", "XALL_age"], "relonly_multi": ["relc_fia_eb"], "relonly_g": ["relg_fia_eb"],
       "bigb01f": ["anP512_deep", "anP512_cca"], "t3enc01f": ["t3_relg"]}
out = OUT / "e2edev"; rep = []
for f in sorted(BASE.rglob("*")):
    if not f.is_file():
        continue
    dst = out / f.relative_to(BASE); dst.parent.mkdir(parents=True, exist_ok=True); data = f.read_bytes()
    for t, cols in NEW.items():
        if f.stem.startswith(f"{t}_fold"):
            s = int(f.stem.split("fold")[-1]); old, new = pd.read_parquet(io.BytesIO(data)), pd.read_parquet(SRC[s])
            key = [k for k in ("y", "fi", "vi") if k in old.columns]
            assert len(old) == len(new) and all((old[k].values == new[k].values).all() for k in key), (t, s, "row alignment")
            if "data" in old.columns:
                assert (old.data.values == new.data.values).all(), (t, s)
            d = {c: float(np.corrcoef(old[c].values, new[c].values)[0, 1]) for c in cols}
            for c in cols:
                old[c] = new[c].values
            b = io.BytesIO(); old.to_parquet(b); data = b.getvalue()
            rep.append(f"fold {s} {t}: corr(old, new) " + ", ".join(f"{c} {v:.4f}" for c, v in d.items()))
    dst.write_bytes(data)
print("\\n".join(sorted(rep)))
(out / "CLEANCOMP.txt").write_text("CLEANCOMP-01: SAFE-04 tables with S6 / rel / relg / v11 T3 columns relearned under LID_STRICT=1\\n" + "\\n".join(sorted(rep)) + "\\n")'''],
    viz='''
d = need("datasets/ngnnguynchn/flag2027-cleancomp-e2e-p")
if d is not None:
    show_text(d / "CLEANCOMP.txt", 30)
'''))
S.append(pkg("t164f", "dngxunqun/flag-2027-nb-164f-cleancomp01-final", "1_train/05_flag-2027-nb-164f-cleancomp01-final.ipynb",
    kind="train", gpu=True, minutes=9, title="CLEANCOMP-01 bản cuối: cầu XALL `s007_xall.pt` (5 mạng) + đầu T3 v11 `v11_t3head.pt` (3 mạng)",
    role="Học trên mọi người train: XALL (221 người, 26 244 hàng, đặc trưng không gian BTC vgg / btc) và đầu T3 v11 P512 (4 508 danh tính: 70 v4 + 151 MAV + "
         "4 287 VoxCeleb2, 59 971 hàng; luật lọc Bengali chặt). Dùng ở e50cc.",
    exports={"datasets/ngnnguynchn/flag2027-cleancomp-final-p": "lab_out/final"},
    viz='''
d = need("datasets/ngnnguynchn/flag2027-cleancomp-final-p")
if d is not None:
    show_json(d / "final_info.json", None, width=1500)
    T = np.load(d / "v11_head/t3_streams.npz"); print({k: T[k].shape for k in T.files})
'''))
S.append(pulled("t155s_f12", "nguyenchonnhan/flag-2027-nb-155s-ct02ca-f12", kind="train", gpu=True, minutes=78,
    title="Đầu CA theo fold (fold 1–2): điểm out-of-fold `trCA` cho bộ trộn F1s′",
    role="Lượt không có trong gói (mã kéo từ Kaggle). CLEANTUNE-01 vòng 2 chỉ nhánh CA (`CT_R2_ONLY=CA`), 3 seed mỗi fold, mặt [farl16 ‖ clipC5], "
         "giọng CAT4 → `ct01r2_fold<s>.parquet`."))
S.append(pulled("t155s_f34", "nguyenchonnhan/flag-2027-nb-155s-ct02ca-f34", kind="train", gpu=True, minutes=79,
    title="Đầu CA theo fold (fold 3–4)", role="Như t155s_f12."))
S.append(pulled("t155s_f5", "locnguyentruongan/flag-2027-nb-155s-ct02ca-f5", kind="train", gpu=True, minutes=69,
    title="Đầu CA theo fold (fold 5)", role="Như t155s_f12."))
K155 = ["notebooks/nguyenchonnhan/flag-2027-nb-155s-ct02ca-f12", "notebooks/nguyenchonnhan/flag-2027-nb-155s-ct02ca-f34",
        "notebooks/locnguyentruongan/flag-2027-nb-155s-ct02ca-f5"]
S.append(dict(name="gather_r2ca", src="new", kernel="notebooks/ff/gather-r2ca", kind="train", gpu=False, minutes=1,
    title="Gom 5 bảng fold `ct01r2` thành dataset bảng (bước `sat_push.sh 111sr`)",
    role="Bước gom ở máy (`gather_tables ct01r2`) nay thành stage: chép `ct01r2_fold1..5.parquet` của ba lượt t155s vào `tables/`.",
    inputs=K155, patches=[], drop_cells=[], smoke_env={}, env={},
    exports={"datasets/ngnnguynchn/flag2027-ct02s-r2ca-tables-p": "ds"},
    cells=['''print("[cell] gom bảng ct01r2", flush=True)
import shutil
from pathlib import Path
IN, OUT = Path("/kaggle/input"), Path("/kaggle/working")
fs = sorted({p.name: p for p in IN.rglob("ct01r2_fold[0-9].parquet")}.values())
print([str(p) for p in fs]); assert len(fs) == 5, f"{len(fs)} bảng fold (cần 5)"
(OUT / "ds/tables").mkdir(parents=True, exist_ok=True)
for p in fs:
    shutil.copy2(p, OUT / "ds/tables" / p.name)'''],
    viz=VIZ_FOLDTAB.format(kernels=["notebooks/ff/gather-r2ca"], glob="ct01r2_fold*.parquet", cols=["trCA", "trCAb"])))
S.append(pkg("t112sf0s", "locnguyentruongan/flag-2027-nb-112s-f0s", "1_train/06_flag-2027-nb-112s-f0s.ipynb",
    kind="train", gpu=True, minutes=60, title="Đầu B0s (mặt [farl16 ‖ clipB3], giọng [sam100p ‖ sam100f]), 5 seed + bộ trộn English của F0s",
    role="Chỉ để nối dây: 112s-f1s và e112f0sx cần zip F0 / mô hình B0s; điểm English của nó bị tính lại ở stage cuối.",
    exports={"datasets/ngnnguynchn/flag2027-evalu-f0s-p": {"ct01g1_final_models.pt": "e2e_out/ct01g1_final_models.pt"}},
    smoke_env={"FL3Z_SMOKE": "1"}))
S.append(pkg("t112sf1s", "locnguyentruongan/flag-2027-nb-112s-f1s", "1_train/07_flag-2027-nb-112s-f1s.ipynb",
    kind="train", gpu=True, minutes=154, title="Đầu CA cuối (10 seed × 18 560 bước) → `ct01r2_final_models.pt`",
    role="Mặt [farl16 ‖ clipC5], giọng CAT4 = [sam100p ‖ sam100f ‖ rn293 ‖ ecapa1024]; học trên 477 danh tính chính (29 809 hàng) + VoxCeleb2 / VoxBlink "
         "phụ. Dùng ở e112f1sx (`trCA`).",
    exports={"datasets/ngnnguynchn/flag2027-evalu-f1s-p": {"ct01r2_final_models.pt": "e2e_out/ct01r2_final_models.pt"}},
    smoke_env={"FL3Z_SMOKE": "1"},
    viz='''
d = need("notebooks/locnguyentruongan/flag-2027-nb-112s-f1s")
if d is not None:
    show_text(next(iter(d.rglob("ct01r2_audit_A0.txt")), d / "x"), 10)
    show_json(next(iter(d.rglob("progress.json")), d / "x"), None, width=900)
'''))

S.append(pkg("e112f0sx", "thanhhuyen11/flag-2027-nb-112f0sx-eval", "2_eval/05_flag-2027-nb-112f0sx-eval.ipynb",
    kind="infer", gpu=False, minutes=2, title="Đầu B0s đã lưu → điểm English của F1s (nối dây)",
    role="Chấm đầu B0s đã lưu, kiểm A0 (không người nói Bengali trong hàng học). 112f1sx bắt buộc có đúng một zip F0; điểm English của "
         "nó bị tính lại ở stage cuối, nên stage này không vào số của zip cuối.",
    smoke_env={"FL3Z_SMOKE": "1"},
    viz='''
d = need(K["112f0sx"])
if d is not None:
    H0 = np.load(d / "e2e_out/ct01g1_final_eval.npz"); print(str(H0["info"])[:400])
    logit_hist([H0[f"trB0c_{i}"] for i in range(4)], "điểm đầu B0s (`trB0c`) trên 4 file thi"); plt.show()
    show_text(d / "e2e_out/ct01g1_audit_A0.txt", 12)
'''))
S.append(pkg("e112f1sx", "thanhhuyen11/flag-2027-nb-112f1sx-eval", "2_eval/06_flag-2027-nb-112f1sx-eval.ipynb",
    kind="infer", gpu=False, minutes=4, title="Đầu CA đã lưu + bộ trộn F1s′ (NOREL) → f1s zip",
    role="Chấm đầu CA (`trCA`, 10 projector, mặt [farl16 ‖ clipC5], giọng CAT4); fit lại bộ trộn Bangla F1s′ (logistic, 7 luồng × 7 điều kiện) "
         "trên bảng out-of-fold SAFE-04 (37 070 cặp; `NOREL=1`); ghi zip F1s′.",
    smoke_env={"FL3Z_SMOKE": "1"},
    viz='''
d = need(K["112f1sx"])
if d is not None:
    e = d / "e2e_out"; HC = np.load(e / "ct01r2_final_eval.npz"); print(str(HC["info"])[:400])
    FU = json.loads((e / "cleantune01_fuser_bangla_CA.json").read_text()); F1 = zip_logits(e / "submission_cleantune01_F1_CA_eval.zip")
    st = FU["streams"]; k = len(st); coef = np.array(FU["coef"]); nc = (len(coef) - k) // (k + 1)
    cn = ["là English", "giới mặt", "giới giọng", "lệch giới", "tuổi mặt", "tuổi giọng", "lệch tuổi"][:nc]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    logit_hist([HC[f"trCA_{i}"] for i in range(4)], "điểm đầu CA (`trCA`)", ax[0])
    ax[1].barh(range(k + nc), coef[:k + nc], color=["#4c72b0"] * k + ["#999999"] * nc); ax[1].set_yticks(range(k + nc), st + cn, fontsize=7)
    ax[1].set_title("hệ số bộ trộn Bangla F1s′ (đã chuẩn hoá)")
    m = np.abs(coef[k + nc:]).max(); im = ax[2].imshow(coef[k + nc:].reshape(k, nc), cmap="RdBu_r", vmin=-m, vmax=m)
    ax[2].set_yticks(range(k), st, fontsize=7); ax[2].set_xticks(range(nc), cn, rotation=60, fontsize=7); ax[2].set_title("tương tác luồng × điều kiện")
    fig.colorbar(im, ax=ax[2]); fig.tight_layout(); plt.show()
    logit_hist(F1, "logit F1s′ trong zip F1s′ (English sẽ được tính lại ở stage cuối)"); plt.show()
'''))
S.append(pkg("e163i", "thanhhuyen11/flag-2027-nb-163i-rfsfxc5rl-eval", "2_eval/07_flag-2027-nb-163i-rfsfxc5rl-eval.ipynb",
    kind="infer", gpu=True, minutes=31, title="RFX: trích lại mặt / giọng của file thi + chấm bằng artifact RFX (C1FIX)",
    role="Trích lại mọi khối của RFX từ media thi (mặt `xf`, ImageBind, CLIP-L, tuổi; giọng `xv`, ImageBind-audio, ResNet221, ImageBind-audio + "
         "LoRA C1, SimAM, Whisper-large-v3, tuổi; thêm `f_c5s` từ e65xcs và `v_rl` từ e47rs), chấm bằng artifact RFX. **Khác bản gốc:** "
         "artifact là bản C1FIX (LoRA C1 đã sửa khoá + RFX học lại; zip 16.18), gắn đúng vào đường dẫn của dataset art cũ.",
    viz='''
d = need(K["163i"])
if d is not None:
    r = d / "FLAG_RF_E2E"; EMB = np.load(r / "embeddings.npz"); show_text(r / "run_report.json", 25)
    RFX = zip_logits(r / "submission.zip")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].barh(range(len(EMB.files)), [EMB[k].shape[1] if EMB[k].ndim > 1 else 1 for k in EMB.files]); ax[0].set_yticks(range(len(EMB.files)), EMB.files, fontsize=7)
    ax[0].set_title(f"số chiều từng khối ({len(EMB['face_xf'])} file không trùng)")
    ax[1].hist((unit(EMB["voice_lora"]) * unit(EMB["voice_ibv"])).sum(1), 60); ax[1].set_title("cos(ImageBind-audio + LoRA C1, ImageBind-audio gốc)")
    ax[2].hist(EMB["face_age"], 40, alpha=.6, label="tuổi từ mặt"); ax[2].hist(EMB["voice_age"], 40, alpha=.6, label="tuổi từ giọng"); ax[2].legend()
    fig.tight_layout(); plt.show()
    logit_hist(RFX, "logit RFX trên 4 file thi"); plt.show()
'''))
_i = S[-1]["inputs"].index("datasets/ngnnguynchn/flag2027-rfsfxc5rl-art-p")
S[-1]["inputs"][_i] = ("datasets/ngnnguynchn/flag2027-rfsfxc5rl-art-p", C1FIX_ART)

S.append(dict(name="final", src="new", kernel=K["final"], kind="infer", gpu=False, minutes=1,
    title="Ghép F1s′ + RFX → zip nộp", inputs=[K["50cc"], K["112f1sx"], K["163i"]], patches=[], drop_cells=[], exports={},
    smoke_env={}, env={},
    role="`rebuild_final.py` của gói (nguyên văn): Bangla = F1s′ + RFX; English = (bộ trộn Bangla F1s′ áp lên cặp English) + RFX; "
         "lưu −logit. Ra `submission.zip`.",
    cells=['''print("[cell] ghép: chép đầu vào", flush=True)
import shutil
from pathlib import Path
IN, OUT = Path("/kaggle/input"), Path("/kaggle/working")
K50 = IN / "notebooks/thanhhuyen11/flag-2027-nb-50cc-e2e-eval"
K112 = IN / "notebooks/thanhhuyen11/flag-2027-nb-112f1sx-eval"
K163 = IN / "notebooks/thanhhuyen11/flag-2027-nb-163i-rfsfxc5rl-eval"
RIN = OUT / "rebuild_in"; RIN.mkdir(exist_ok=True)
for src, dst in [(K112 / "e2e_out/submission_cleantune01_F1_CA_eval.zip", "f1s_eval.zip"), (K50 / "e2e_eval/components_eval.npz", "components_eval.npz"),
                 (K50 / "e2e_eval/e2e_manifest.json", "e2e_manifest.json"), (K112 / "e2e_out/cleantune01_fuser_bangla_CA.json", "cleantune01_fuser_bangla_CA.json"),
                 (K112 / "e2e_out/ct01r2_final_eval.npz", "ct01r2_final_eval.npz"), (K163 / "FLAG_RF_E2E/submission.zip", "rfx_eval.zip")]:
    shutil.copyfile(src, RIN / dst); print("  ", dst, "<-", src)''',
           "%%writefile rebuild_final.py\n" + (PKG / "rebuild/rebuild_final.py").read_text(encoding="utf-8"),
           '''print("[cell] ghép: chạy rebuild_final.py", flush=True)
import subprocess, sys, hashlib
r = subprocess.run([sys.executable, "rebuild_final.py", str(RIN), str(OUT / "submission.zip"), str(OUT / "no_reference.zip")], capture_output=True, text=True)
print(r.stdout, r.stderr)
assert r.returncode == 0 and (OUT / "submission.zip").exists(), "ghép lỗi"
print("sha256 submission.zip:", hashlib.sha256((OUT / "submission.zip").read_bytes()).hexdigest())'''],
    viz='''
d = need(K["final"])
if d is not None:
    FIN = zip_logits(d / "submission.zip")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    logit_hist(FIN, "logit cuối trên 4 file thi", ax[0])
    k163 = need(K["163i"])
    if k163 is not None:
        RFX = zip_logits(k163 / "FLAG_RF_E2E/submission.zip")
        for i in range(4):
            rf = RFX[i].reindex(FIN[i].index)
            ax[1].scatter(FIN[i] - rf, rf, s=3, c=COL[i], label=f"{LAB[i]}: Spearman {spearmanr(FIN[i] - rf, rf)[0]:.2f}")
        ax[1].set_xlabel("phần F1s′"); ax[1].set_ylabel("phần RFX"); ax[1].legend(fontsize=7); ax[1].set_title("hai họ đóng góp vào logit cuối")
    fig.tight_layout(); plt.show()
    for name, ref in REFERENCE_ZIPS.items():
        if ref and Path(ref).exists():
            R = zip_logits(ref)
            print(f"so với {name}: Spearman theo file", [round(float(spearmanr(FIN[i], R[i].reindex(FIN[i].index))[0]), 5) for i in range(4)],
                  "| lệch tuyệt đối lớn nhất", [round(float(np.abs(FIN[i] - R[i].reindex(FIN[i].index)).max()), 6) for i in range(4)])
'''))

# ------------------------------------------------------------------------------------------------ order (each stage after its producers)
ORDER = ["t153b", "t153c", "t164_f123", "t164_f45", "cleancomp_tables", "t164f",
         "e65xbs", "e65xcs", "e47rs", "e50cc",
         "t155s_f12", "t155s_f34", "t155s_f5", "gather_r2ca", "t112sf0s", "t112sf1s",
         "t160a", "t160b", "t160c", "t162a", "t162b", "t162c", "t163b", "c1fix_lora", "c1fix_rfx",
         "e112f0sx", "e112f1sx", "e163i", "final"]
assert sorted(ORDER) == sorted(s["name"] for s in S), set(ORDER) ^ {s["name"] for s in S}
S.sort(key=lambda s: ORDER.index(s["name"]))
_made = set()
for _s in S:                                    # every input made by an earlier stage, or attached by hand (no producer)
    for _e in _s["inputs"]:
        _a = _e[1] if isinstance(_e, tuple) else _e
        _prod = [x["name"] for x in S if x["kernel"] == _a or _a in x.get("exports", {})]
        assert not _prod or _prod[0] in _made, f"{_s['name']} needs {_a} from {_prod[0]}, which comes later"
    _made.add(_s["name"])
