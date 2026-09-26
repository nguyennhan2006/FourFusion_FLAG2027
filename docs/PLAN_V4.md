# Kế hoạch v4 — sau mốc 28.69

Lập 2026-09-26, cập nhật cùng ngày sau khi rà soát [deep-research-report_26-09-2026.md](deep-research-report_26-09-2026.md).
Thay [PLAN_V3.md](PLAN_V3.md) (giữ làm lịch sử). Nguồn số liệu: [../Experiment/EXPERIMENT_LOG.md](../Experiment/EXPERIMENT_LOG.md).

| | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---:|---:|---:|---:|---:|
| FOP baseline của nhóm (000c) | 42.17 | 36.31 | 38.98 | 43.38 | 50.00 |
| **FUSE-03 (hiện hành)** | **28.69** | **23.61** | **26.88** | **29.12** | **35.15** |

Tái tạo: `python BEST/v3_28.69/build_best.py` (công thức nằm trong `BEST_CONFIG`, rank-corr 1.000000 với zip đã nộp).

**Lịch làm việc** (theo [trang challenge](https://mavceleb.github.io/dataset/competition.html), khớp Evaluation Plan; trang ICASSP có hai mục FLAG mâu thuẫn nên không dùng):

| Mốc | Ngày | Ghi chú |
|---|---|---|
| Progress phase | 15/09 – **10/11/2026** | 150 lượt nộp tổng, tối đa 15/ngày; còn lại = 150 − số đã nộp |
| Evaluation phase | 11/11 – 18/11 | **15 lượt tổng**, ngân sách riêng |
| Kết quả | 23/11 | |
| System description (2 trang, template ICASSP) + **link code chạy được** | **27/11** | thiếu một trong hai là bị loại |
| Challenge paper | 10/12 | |

Không nộp CodaBench cho các candidate mà SEL-01b xếp là hòa (xem §3).

## 0. Triết lý v4

```text
v1–v3:  tìm encoder tốt hơn → tìm kiến trúc tốt hơn → giải shift Bangla
v4:     encoder unimodal đã đủ mạnh
            ↓
        nút thắt là cầu nối face↔voice (học từ 70 người)
            ↓
        dữ liệu cặp ngoài khớp quần thể, kiểm bằng validation có nhãn thật  +  fusion các hệ bổ sung  +  SEL-01b
```

Bằng chứng: **nhận dạng một modality tốt hơn không làm ghép chéo tốt hơn**, ở cả hai phía.

| Encoder | Unimodal EER | Cross-modal (proxy internal) |
|---|---|---|
| Face VGG fc7 (BTC) → ArcFace (có align) | 7.77 → **1.70** | 26.16 → **33.92** |
| Voice BTC-192 → ReDimNet2 đa ngôn ngữ | 4.77 → **3.17** | 26.20 → **27.01**; dev tệ hơn 4.5–6.7 |

Và ánh xạ mặt–giọng **phụ thuộc quần thể**: thêm MAV-Celeb v3 (người châu Âu, En/German) làm tệ cả 4 cell.

## 1. Các hướng còn mở, theo information gain

### A. Dữ liệu cặp ngoài — kiểm bằng validation song ngữ có nhãn thật ⭐⭐⭐

**A1. EXT-01 vòng 2 (đang chạy, nhánh MIX):** v1 (Urdu) riêng, v2 (Hindi) riêng, v1+v2 (cap 150 / 400), + v3 đối chứng — dữ liệu ngoài **trộn chung** vào train với v4. Không chạy lại các mix này.

**A2. Validation song ngữ v1/v2 — validation chính cho giả thuyết dữ liệu ngoài.** Mỗi người trong v1/v2 có cả
English lẫn Urdu/Hindi, nên dựng được bài toán giống FLAG nhưng **có nhãn thật**:

```text
identity của v1 (hoặc v2)  →  chia speaker-disjoint (5 fold theo người)
train  : v4 train + English của các identity train ngoài
held-out identities, cùng tập mặt:
    heard   : trial (face, voice English)          ← đo có hy sinh English không
    unheard : trial (face, voice Urdu / Hindi)     ← đo chuyển giao cross-lingual
trial   : 50% target mỗi file, như dev FLAG
```

| Kết quả trên held-out | Kết luận |
|---|---|
| unheard tốt lên, heard giữ nguyên (trong guardrail) | có bằng chứng cho cầu nối chuyển giao được → mới đưa sang SEL-01b / v4 |
| unheard tốt lên, heard giảm vượt guardrail | **trade-off**, không phải "dữ liệu ngoài giúp cầu nối" |
| cả hai giảm | bỏ nguồn / recipe đó |

Guardrail heard = luật chấp nhận §3 (không tệ hơn > 0.5 ở cell đang dùng).
Validation này cũng là **phép kiểm độc lập của SEL-01b**: nếu recipe nào thắng trên held-out có nhãn thật mà SEL-01b chấm hòa hoặc thua trên dev, selector đang đo tương quan với leaderboard chứ không phải khả năng tổng quát cross-lingual.

**A3. Nhánh PRETRAIN → FINE-TUNE (build kế tiếp của FLAG_08):** `PT(v1) → FT(v4)`, `PT(v2) → FT(v4)`; chỉ thêm `PT(v1+v2) → FT(v4)` nếu hai nhánh đơn có tín hiệu. So với MIX bằng cùng ma trận A2 + SEL-01b.

Còn mở trước khi build A2/A3:
- Nhãn giới tính của v1/v2: cần cho cell kiểu `gender` (negative cùng giới). Chỉ lấy từ metadata của dataset; nếu không có thì A2 chỉ báo cell kiểu `no_gender`, **không** suy giới từ khuôn mặt.
- A2 chạy được local trên CPU khi đã tải `feats_ext/` từ lượt Kaggle hiện tại (feature đã trích, chỉ train MLP).

Metadata mỗi hàng ngoài: `source`, `spk`, `lang`, `video`, `dur` (đã có trong `ext_<source>_*_meta.csv`). Nhóm quần thể chỉ theo **thiết kế dataset** (v1 Pakistan, v2 Ấn Độ, v3 châu Âu).

### B. Thuộc tính mềm — chỉ là phép thử bác bỏ (EDA), không phải feature chấm điểm

`D_pitch` kiểu "khoảng cách pitch mặt–giọng" không định nghĩa được (mặt không có F0). Phiên bản có nghĩa:

```text
trên identity train, trong từng giới:
    ridge  face embedding → log-F0 median của speaker,   leave-speaker-out (cross-fit)
trên identity held-out:
    r = |F0_pred(face) − F0_observed(voice)|
bằng chứng chỉ khi: r(positive) < r(negative cùng giới) ổn định (bootstrap theo identity)
           VÀ thắng đối chứng: xáo F0 trong cùng giới
```

Chạy trên v1/v2 trước (nhãn identity thật, nhiều người hơn), rồi mới v4. Tương tự cho tuổi (`buffalo_l genderage`). Kết quả chỉ dùng để trả lời "có cue sinh lý chung không"; không đưa vào score nếu chưa qua A2.

### C. Fusion các hệ bổ sung — giữ, không mở rộng

- **Gate g/Bn đóng băng** như trong `BEST_CONFIG`. Kiểm tra đối chứng (log 41): OOF gain của trọng số tĩnh đổi từ −0.42 đến +1.09 chỉ vì cách chọn lưới; gate độ dài (+1.25) nằm trong vùng dao động đó; ở ng/Bn độ dài bị xáo còn tốt hơn độ dài thật. ⇒ gate **không** phải một gain đã xác lập, chỉ là cách đổi trọng số đã được CB xác nhận một lần.
- **Không** fit quality gate / router nhiều tham số trên dev. Đặc trưng chất lượng (độ dài, độ bất đồng giữa hai hệ, chất lượng mặt/giọng) chỉ dùng để **phân tích** cấu trúc lỗi bổ sung, cho tới khi có validation có nhãn thật (A2).
- Chỉ thêm hệ vào fusion khi nó sai ở chỗ khác các hệ đang có; ≤ 5 ứng viên khai báo trước mỗi vòng; trọng số mặc định 0.5/0.5.

## 2. Đã đóng — không quay lại

| Hướng | Vì sao đóng | Nguồn |
|---|---|---|
| Thay face encoder (ArcFace, AdaFace, MagFace, face recognition SOTA) | ArcFace có align: face↔face 1.7 EER nhưng cross-modal 33.92 | NB-2 P0/P2 |
| Thay voice encoder vì unimodal EER tốt hơn (ReDimNet, CAM++, ERes2Net) | ReDimNet2: speaker EER 3.17 nhưng cross-modal phẳng, dev tệ hơn | NB-4 / FUSE-04 |
| WavLM L4–9 / XLS-R pooling mù | Đơn lẻ kém mọi cell, hại fusion English | NB-2, r2 |
| CORAL, partial CORAL | Xóa luôn identity | 004, 006c |
| AS-norm | Internal dự đoán sai dấu | 003b → 003c |
| Gender tường minh / GRL / same-gender negatives | Không có lợi ổn định | NB-2 P4, GEN-01 |
| Làm mượt score bằng đồ thị kNN | Mức fusion: English tệ hơn | GRAPH-01 |
| Công thức FAME26 (linear + dropout .9 + AAM); shared center / AAM | Dropout .9 phá; AAM ≈ InfoNCE (`ours+aam` 30.66 vs 30.53; NB-2 P3 −0.34) | 009, NB-2 |
| MSE alignment (XM-ALIGN) | Trên embedding L2-norm, ‖z_f − z_v‖² = 2 − 2cos: chỉ là biến đổi affine của số hạng positive đã có trong InfoNCE. XM-ALIGN chỉ đạt 34.2 (đơn) / 33.5 (fusion) trên FAME 2026 → chỉ là reference | [arXiv:2512.06757](https://arxiv.org/abs/2512.06757) |
| Quality gate / router > 2 tham số fit trên dev | Nhiễu OOF do chọn lưới (~1.5 EER) lớn hơn khác biệt cần đo | log 41 |
| Dual-LoRA / adversary neo theo ngôn ngữ | Cần LoRA trong backbone trên audio thô, bài toán speaker verification; không hợp pipeline feature đóng băng | [arXiv:2604.26327](https://arxiv.org/abs/2604.26327) |
| MAV-Celeb v3 làm dữ liệu train | Hại cả 4 cell | EXT-01 vòng 1 |
| Trọng số tĩnh tối ưu cho English | Overfit (OOF tệ hơn 0.5/0.5) | FUSE-03 |
| Giải thích "crop giúp vì Bangla ngắn" | r2_mix thua ở 0–3 s, thắng lớn ở câu dài; gate cho 003c trọng số tăng theo độ dài | FUSE-03 / DUR-02 |

**Treo, chờ BTC xác nhận luật:** ImageBind. Luật chỉ cho "a pretrained encoder for faces or voices"; ImageBind là mô hình đã căn chỉnh sẵn image–audio (audio học từ AudioSet ghép ảnh), tức mang sẵn alignment chéo — không tự diễn giải là hợp lệ. License CC-BY-NC 4.0. Nếu được xác nhận: chỉ một **probe đóng băng**; gần ngẫu nhiên thì dừng, không fine-tune.

**Ưu tiên thấp, chỉ khi rảnh:** WavLM/XLS-R scalar mix trên **mean** 25 layer — đã có sẵn trong `feats_v2` (không cần trích lại; muốn thêm std thì phải trích lại). Coi là ablation đa dạng, không phải lời giải.

## 3. Luật chấp nhận và luật dùng dev không nhãn

**Chấp nhận thay đổi** (giữ nguyên, không dùng ngưỡng của report):
- Representation / recipe mới: pseudo-EER tốt hơn ≥ 1.0 ở ít nhất một cell, không tệ hơn > 0.5 ở cell nào dùng nó, cùng chiều với nhãn độc lập `--labels=btc`; với dữ liệu ngoài thì phải qua A2 trước.
- Fusion: trọng số mặc định; chỉ fit khi OOF theo pseudo-identity thắng ≥ 0.25 **và** ≥ 4/5 fold, trong ngân sách ≤ 2 tham số (đã dùng hết).
- **Vùng hòa:** chênh lệch pseudo-EER ≤ ~0.4 (cặp chênh ≤ 0.41 là nơi SEL-01b từng xếp sai thứ tự) ⇒ **chưa xác định**, không xếp hạng, không tốn lượt nộp.
- Không dùng: ΔEER ≤ −0.5, P_boot ≥ 0.8 (sẽ loại FUSE-01, vốn cho g/En −1.84 trên CB), OOF ≥ 0.3 cho gate (thấp hơn nhiễu).

**Độ tin của selector** là thuộc tính của selector, không của candidate:
`selector_reliability.rho_historical` = Spearman(xếp hạng SEL-01b, xếp hạng CB) trên các hệ lịch sử có cả hai số, báo theo từng cell + overall (hiện 0.95–0.996; tính trong `SEL-01_selector/run.py`). Độ tin của một candidate là phân phối bootstrap theo identity của Δpseudo-EER so với hệ đang dùng (`pseudo_eval.compare`).

**Dev không nhãn** — dev của phase này cũng là tập được chấm, và pseudo-label SEL-01b trùng nhãn thật ở 96–99% trial:

| Việc | Được? |
|---|---|
| Chọn giữa vài hệ khai báo trước bằng `pseudo_eval.py` | ✅ |
| Fit ≤ 2 tham số fusion, out-of-fold theo pseudo-identity | ✅ khai báo |
| Per-file mean centering (thống kê bậc 1 của file dev) | ✅ khai báo |
| Train bất kỳ model nào trên pseudo-label | ❌ |
| Sinh score nộp từ pseudo-identity / cấu trúc danh sách trial | ❌ rò rỉ protocol, không phải ghép mặt–giọng, không chắc có ở test |

Thực thi trong code: `build_best.py::check_policy` chỉ chấp nhận combiner `rank_fusion` / `duration_gate`, score phải đến từ zip thành phần (hệ train trên nhãn thật), và đếm số tham số fit trên dev. `pseudo_eval.py` chỉ đọc nhãn để chấm, không xuất nhãn cho train.

## 4. Hạ tầng

| Thành phần | Trạng thái |
|---|---|
| `BEST/v3_28.69/build_best.py` — đọc `BEST_CONFIG`, kiểm policy, so rank-corr với zip đã nộp | ✅ |
| `Experiment/SEL-01_selector/pseudo_eval.py` — `evaluate(zip)`, `compare(a, b)`, `group_folds` | ✅ |
| `Experiment/FUSE-03/gate_controls.py` — đối chứng gate (lưới tĩnh, độ dài thật / xáo, bất đồng) | ✅ |
| `kaggle/flag_extract.load_audio` — một hàm đọc audio cho v4 và dữ liệu ngoài, khớp `librosa.load(sr=16000, mono=True)` | ✅ |
| `kaggle/flag_extract.fetch_zip` — danh sách id (bản sao riêng → id BTC); zip xóa sau khi trích thành công | ✅ |
| A2 validation song ngữ + A3 PT→FT | ☐ build sau khi có kết quả vòng MIX; cần nhãn giới v1/v2 |
| Build lại `FLAG_08_external.ipynb` với lib mới + smoke test local | ☐ cùng lúc với A3 |
| `FUSE-03/common.py` dùng lại `pseudo_eval` | ☐ khi có FUSE-05 (tên FUSE-04 đã dùng cho ReDimNet) |

Mọi thí nghiệm mới trên dev báo cáo: pseudo-EER 4 cell (nhãn `arc`, kiểm chéo `btc`), coverage, CI bootstrap theo identity, và OOF nếu có fit tham số. Tên mới tránh trùng: không tạo package `flag_lib/` (đã có `kaggle/flag_lib.py`).

## 5. Khai báo trong system description (hạn 27/11, kèm link code)

1. Dev không nhãn được dùng để chọn hệ thành phần và fit 2 tham số fusion (pseudo-label từ cụm ArcFace / ECAPA-192 + đồng xuất hiện trong trial); không model nào được train trên dev.
2. Per-file mean centering trên dev.
3. Nếu bản cuối dùng dữ liệu ngoài: MAV-Celeb v1 / v2 (và v3 nếu có) làm dữ liệu train ngoài, đã loại trùng identity với v4 train + dev; không có Bangla.
4. Encoder: VGGFace fc7, speechbrain `yangwang825/ecapa-tdnn-vox2` (đều là encoder của BTC), speechbrain `spkrec-ecapa-voxceleb` (192 và 6144), ArcFace `buffalo_l` (chỉ cho pseudo-label).
5. Code: `BEST/` + notebook Kaggle, đóng gói thành repo công khai trước 27/11.
