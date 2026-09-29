# Chạy FLAG trên Kaggle (T4 ×2)

## ⭐ Giai đoạn 2 (hiện tại) — theo [../docs/PLAN_KAGGLE.md](../docs/PLAN_KAGGLE.md)

Sinh lại notebook sau khi sửa code: `python build_v2_notebooks.py` (nhúng sẵn `flag_lib.py`, `flag_extract.py`, `flag_v2.py`).

| Bước | Notebook | Input (Add Input) | Settings | Thời gian dự kiến trên T4 | Output |
|---|---|---|---|---|---|
| 1 | `FLAG_04_extract.ipynb` | dataset **raw** (upload thư mục `Input/`: 2 zip + `meta_file_train_set.csv`) | GPU, **Internet ON** | ~1–1.5 giờ | `/kaggle/working/feats/` → tạo dataset **`flag2027-feats-v2`** |
| 2 | `FLAG_05_ablate.ipynb` | `flag2027-feats-v2` + `flag2027-features` | GPU, Internet OFF | ~2–4 giờ | `ablation.csv`, `p0_unimodal.csv`, `submissions/sub_{base,crops_mix,dann}.zip` |
| 3 | `FLAG_06_submit.ipynb` | như bước 2 | GPU, Internet OFF | ~30 phút | `submissions/r2_*.zip` + `*_scores.npz` |
| 4 | `FLAG_07_redimnet.ipynb` (vòng 3) | dataset **raw** + `flag2027-features` + `flag2027-feats-v2` | GPU, **Internet ON** | ~45–60 phút | `feats_v3/`, `submissions/r3_*.zip` + `*_scores.npz` + `*_matrix.npz` |

| 5 | `FLAG_08_external.ipynb` (EXT-01) | `flag2027-features` + `flag2027-feats-v2` (**không cần** raw v4) | GPU, **Internet ON** | ~1.5–2 giờ (tải ~26 GB từ Drive) | `feats_ext/`, `overlap.csv`, `ablation_ext.csv`, `submissions/ext_*.zip` + npz |
| 6 | `FLAG_10_models.ipynb` (MODEL-01, [PLAN_MODELS](../docs/PLAN_MODELS.md) S1 + probe ImageBind) — sinh bằng `python build_models_notebook.py`. **Chạy tiếp:** đính kèm output của lượt trước, chỉ encoder còn thiếu được chạy | dataset **raw** v4; tuỳ chọn: v1/v2 (zip hoặc thư mục) + dataset nhỏ `flag2027-ext-meta` (upload `kaggle/flag2027-ext-meta.zip`, 0.4 MB: 4 file `ext_v{1,2}_complete_{face,voice}_meta.csv` của `output/feats_ext_full/`) | GPU, **Internet ON** | ~2 giờ có v1/v2, ~40 phút chỉ v4 (tải ~9 GB checkpoint vào `/tmp`) | `feats_models/` → dataset **`flag2027-feats-models`**; `probe.csv` / `probe.json` |
| 8 | `FLAG_12_features.ipynb` (E3 của [PLAN_EVAL](../docs/PLAN_EVAL.md): mọi đặc trưng pipeline Evaluation cần). **Code đầy đủ trong cell**, không import file .py; sinh bằng `python build_eval_notebooks.py`. Đã chạy trên Kaggle 28/09: khớp tuyệt đối mảng đang dùng. Lỗi w2v2 dưới transformers 5.0 + Jupyter đã sửa và kiểm chứng (`run_notebook_local.py --jupyter` với transformers 5.0.0). **Chạy tiếp:** đính kèm output lượt trước | dataset **raw** v4 (sau này: dữ liệu Evaluation). Tuỳ chọn: v1/v2 + `flag2027-ext-meta` (ArcFace / ECAPA cho STRESS-01); `flag2027-feats-v2` + `flag2027-feats-models` để tự so sánh | GPU, **Internet ON** | ~1 giờ 15 phút v4, thêm ~1 giờ v1/v2 | `feats_eval/` → dataset **`flag2027-feats-eval`**; `check_vs_reference.csv` |
| 7 | `FLAG_11_layers.ipynb` (S3 + S4 của [OPEN_DIRECTIONS](../docs/OPEN_DIRECTIONS.md): các lớp khác của chính encoder BTC + TTA) — cùng builder `build_models_notebook.py` | như FLAG_10 (raw v4 bắt buộc; v1/v2 + `flag2027-ext-meta` tuỳ chọn). Raw v4 có sẵn CSV của BTC để notebook tự kiểm fc7 / 192 trích lại (phải ≈ 1.0000) | GPU, **Internet ON** | ~45 phút | `feats_layers/` → dataset **`flag2027-feats-layers`** (fc6, pool5, fc7 ảnh lật; ECAPA pooling 3072, 192 TTA) |

Sau mỗi bước 3/4/5: **không nộp ngay**. Tải `submissions/` về `kaggle/`, chấm trước bằng selector (`SEL_LABELS=arc python Experiment/SEL-01_selector/run.py name=path.zip`) rồi thử fusion (`Experiment/FUSE-0x/`).

**Cách chạy:** bấm **Save Version → Save & Run All (Commit)**, không chạy từng cell tương tác. Như vậy phiên không bị ngắt khi tắt tab, và output được lưu lại.
Sau bước 1: mở version vừa chạy → tab **Output** → **New Dataset** → đặt tên `flag2027-feats-v2`.

**Chống OOM / giữ chất lượng (đã có sẵn trong code):**
- Audio được sắp theo độ dài và gom batch theo tổng số giây đã padding (≤ 240 s). Gặp OOM thì tự chia đôi batch; nếu một file vẫn OOM thì chạy file đó trên CPU.
- ECAPA và ArcFace chạy fp32. Notebook assert cosine giữa embedding chạy batch và chạy từng file (bs=1) > 0.999.
- Ảnh mặt được detect + align bằng SCRFD và notebook in tỉ lệ detect. insightface được cài bằng `--no-deps` để không đè `onnxruntime-gpu`.
- SSL (WavLM/XLS-R) chạy theo cửa sổ 20 s, ở fp16. Notebook in cosine so với fp32. Nếu một encoder SSL lỗi thì bỏ qua encoder đó, không làm hỏng cả run.
- Mọi array được ghi ngay khi xong; chạy lại notebook sẽ bỏ qua những gì đã có. NB-2 cũng resume theo `ablation.csv`.

**Nộp CodaBench theo thứ tự:** `sub_base` → `sub_crops_mix` → `sub_dann` (mỗi lượt chỉ đổi một biến). `sub_dann` dùng audio dev không nhãn để adapt (transductive), nên phải ghi rõ trong system description.

---

## Giai đoạn 1 (đã xong — giữ lại để đối chiếu)

## 1. Upload dataset — 1 file

| File | Dung lượng |
|---|---|
| `kaggle/flag2027-features.zip` | **35 MB** (giải nén 140 MB) |

Kaggle → **Datasets → New Dataset** → kéo thả `flag2027-features.zip` → đặt tên gì cũng được → Create.
Notebook **tự dò file theo tên** (`rglob`) nên không phụ thuộc Kaggle lồng thư mục thế nào — chỉ cần Add Input là chạy.
Cấu trúc sau khi Kaggle giải nén:

```text
/kaggle/input/flag2027-features/
├── train/  train_English_faces.csv, train_English_voices.csv, train_English.txt, meta_file_train_set.csv
└── dev/    {no_gender,gender}_{English,Bangla}_{faces,voices}.csv + _test.txt
```

Không cần upload ảnh/wav thô (2.4 GB) — chỉ cần khi làm Pipeline D (re-extract feature).

## 2. Upload notebook — 2 file

| File | Dùng khi nào |
|---|---|
| `kaggle/FLAG_01_sweep.ipynb` | chạy trước: sweep kiến trúc → augmentation → ensemble size → xuất submission |
| `kaggle/FLAG_02_submit.ipynb` | chạy sau: dán config thắng, train ensemble trên 70 speaker, chỉ xuất zip |
| `kaggle/FLAG_03_extract.ipynb` | **ưu tiên hiện tại**: trích lại feature từ raw wav/jpg (ECAPA 6144-d, ArcFace 512-d) và chạy ablation A–E |

Kaggle → **Code → New Notebook → File → Import Notebook** → chọn file → **Add Input** = dataset ở bước 1.

**Settings:** Accelerator **GPU T4 ×2**. Internet **Off** cho NB 01/02; **Internet ON** cho NB 03 (phải tải encoder pretrained từ HuggingFace).

### Nếu gặp lỗi speechbrain trên Kaggle

Kaggle cài sẵn `speechbrain` bản cũ; bản < 1.1.1 gọi `torchaudio.list_audio_backends()` lúc import, mà API này đã bị gỡ khỏi torchaudio 2.9+. Notebook nay dùng `pip install -U "speechbrain>=1.1.1"` (chữ `-U` là mấu chốt — thiếu nó pip báo *already satisfied* và giữ bản cũ), kèm shim + dọn `sys.modules`.

Nếu vẫn lỗi `partially initialized module 'speechbrain'`: đó là module hỏng còn sót trong kernel → **Restart kernel rồi Run All**. Notebook tự dọn được trường hợp này, nhưng restart luôn là cách chắc chắn.

### Riêng cho FLAG_03_extract — chỉ cần **1 dataset**

Upload thư mục `Input/` (hoặc 2 file `train_set.zip` + `dev_set.zip`), ~2.4 GB. Kaggle có thể tự giải nén — notebook xử lý được cả hai dạng. CSV feature của BTC nằm sẵn trong `train_set/features/` và `dev_set/*/features/` nên **không cần** attach thêm `flag2027-features`. Nó trích **ECAPA 192-d** (để đối chứng với CSV của BTC) và **ECAPA 6144-d** (layer attentive-stat-pooling, đúng thứ đội hạng 1 FAME 2026 dùng), cộng **ArcFace R100 512-d** cho khuôn mặt; rồi chạy ablation 5 run đổi **một biến mỗi lần**.

Notebook tự ghi `flag_lib.py` ở cell đầu — không cần upload file `.py` nào.

`flag_lib` tự dò dữ liệu theo tên file và hỗ trợ **cả hai cách đặt tên**: dạng phẳng (`no_gender_English_faces.csv`, như `flag2027-features`) và dạng zip gốc của BTC (`dev_set/no_gender/features/English_test_faces.csv`).

## 3. Thứ tự chạy

1. `FLAG_01_sweep.ipynb` → Run All. Các mục: (1) sweep kiến trúc 26 cfg → (2) augmentation 13 biến thể → (3) **partial CORAL** α → (4) ensemble size → (5) xuất `submission_*.zip`. Kết quả ghi ra `sweep.csv`, `aug_sweep.csv`, `alpha.csv`, `ensemble.csv`.
2. Tải các `submission_*.zip` về, nộp CodaBench.
3. Nếu muốn train lại/ghép khác: `FLAG_02_submit.ipynb`, sửa `CFG_FINAL` và `N_SEEDS` ở cell "Paste the config".

## 4. Những gì đã chốt (notebook không kiểm tra lại)

| Đã biết | Nguồn |
|---|---|
| InfoNCE > linear CCA > CE+OPL (loss của FOP) | EXP-003 |
| Fuse deep ensemble + CCA 4-dim có lợi | EXP-003b |
| Per-file **mean** centering có lợi; **full CORAL hỏng** (41.36) | EXP-004 |
| **AS-norm tốt trên internal nhưng hại trên CodaBench** → score = cosine thô | EXP-003b vs 003c |
| ⚠️ Dedup theo voice-clip **làm tệ hơn** (int_g 32.26 không dedup vs 33.02 có) — giả định từ EDA-0/1 đã bị lật, nên nó vẫn nằm trong grid | EXP-005 |
| Chọn model theo `int_g`, không theo `int_ng` | EXP-000d/e calibration |
| **Fusion phải cho CCA trọng số tường minh w≈0.25** — trung bình đều n+1 thành viên làm CCA loãng dần theo n (1/4 → 1/11) và trông như "ensemble lớn hại" | EXP-007 |
| Augmentation **không** giúp trên internal (`none` thắng 13 biến thể); partial CORAL xấu đi đơn điệu theo α | EXP-006 (Kaggle) |

Baseline cần vượt: **31.34** (25.60 / 27.88 / 34.01 / 37.87).

⚠️ Bản notebook trước (chạy 2026-09-25) có 2 lỗi đã sửa: cell submit hard-code `n=10` thay vì đọc `ensemble.csv`, và fusion trung bình đều làm loãng CCA. Nếu bạn còn giữ output cũ, **đừng nộp** 3 zip đó.

## 5. Lưu ý khi đọc kết quả

- Internal val **chỉ có tiếng Anh**. Một augmentation có thể không cải thiện internal mà vẫn kéo được 2 cell Bangla — nên quy tắc là: **chọn cái không làm hại internal, rồi đo Bangla bằng CodaBench**.
- `int_g` của sweep không so trực tiếp được với EER CodaBench (lệch ~+4 ở `no_gender`, ~0…+4 ở `gender`); chỉ dùng để **xếp hạng**.
- GPU khiến sweep nhanh ~20–30× so với CPU local, nên grid trong notebook rộng hơn nhiều so với những gì đã chạy ở máy.
- **Partial CORAL (α)**: EXP-004 cho thấy α=1 (whitening đầy đủ) mất 10 điểm vì covariance của file dev chính là cấu trúc speaker trong đó. Mục 3 kiểm tra α nhỏ. Thử nghiệm nhanh local đã cho α=0.1 tệ hơn α=0 → nhiều khả năng kết luận là **giữ α=0**; nếu vậy notebook không xuất zip α.
- Cell cuối mỗi notebook kiểm tra zip: đúng 4 file, đúng thứ tự pair_id, không NaN. Luôn đọc output đó trước khi nộp.

## 6. Nếu muốn tận dụng cả 2 GPU

Notebook mặc định dùng `cuda:0`. Cách đơn giản nhất để dùng cả hai: mở **2 notebook session**, mỗi session chạy một nửa `GRID` (sửa `GRID = GRID[:len(GRID)//2]` và `GRID[len(GRID)//2:]`), rồi gộp `sweep.csv`. Model rất nhỏ (MLP 2 nhánh) nên dữ liệu vào/ra GPU là nút cổ chai chứ không phải compute — chia việc theo session hiệu quả hơn DataParallel.
