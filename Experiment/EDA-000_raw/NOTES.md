# EDA-000 — Raw data, speaker, geometry, cross-modal, gender, language, domain shift

- **Ngày:** 2026-09-21
- **Phạm vi:** EDA-0 → EDA-6 trên feature CSV có sẵn (train + 4 file dev). EDA-7 (FOP anatomy) **chưa làm** vì checkpoint 000c chưa có local và cần internal val của EXP-001.
- **Chạy lại:** `python run_eda_012.py && python run_eda_34.py && python run_eda_3b.py && python run_eda_56.py` (cwd = thư mục này; `data/` được giải nén từ `Input/*.zip`, chỉ CSV/TXT). Tổng ~25 phút CPU.
- **Hàm dùng chung:** [eda_utils.py](eda_utils.py) — `load_train`, `speaker_split(seed)`, `build_trials(same_gender=…)`, `eer_from_scores`, `RidgeCCA`. **EXP-001 phải dùng đúng `speaker_split` + `build_trials` này** để số so được với bảng dưới.
- Output: [out/](out/) — `eda_012.json`, `eda_34.json`, `eda_3b.json`, `eda_56.json`, PNG, 2 zip submission tham chiếu.

---

## Tóm tắt 1 trang — bằng chứng → quyết định

| # | Bằng chứng (số) | Kết luận | Hành động |
|---|---|---|---|
| 1 | **CCA tuyến tính** (fit 56 spk, đo 14 spk unseen, 3 seed): k=32 → **no_gender 29–30, gender 35–36**. FOP 000c trên CodaBench: 36.31 / 43.38. | Một phép chiếu tuyến tính không train gì đã ≥ FOP baseline ở cả hai protocol (với caveat internal ≠ CB). **Bottleneck là training/generalization của FOP, không phải feature.** | **Nộp `out/submission_cca_k32.zip` 1 lần** để calibrate internal↔CB. Pipeline A/C lên ưu tiên cao nhất; D (re-extract) xuống cuối. |
| 2 | CCA EER tăng đơn điệu theo k: k=4 → 24/35.5, k=32 → 30/35, **k=64 → 37/40, k=128 → 45/46** (train EER ở k=128 vẫn 7.5). | **Overfit theo speaker cực nhanh**: chỉ 56 speaker, chiều shared thật sự dùng được ≈ 8–32. FOP embed 128 + CE 56 lớp là công thức overfit. | Pipeline A: thử embed_dim 32/64, weight decay lớn, dropout trên input, early-stop theo val gender. Thêm biến thể "FOP + CCA-init". |
| 3 | Cell **gender phẳng ~35 với mọi k**; cell no_gender giảm mạnh khi k nhỏ. Gender probe: face 93%, voice 92%, trong CCA space 92–93%. | Phần lớn lợi thế ở `no_gender` đến từ gender; **identity cross-modal thực (ngoài gender) trần tuyến tính ≈ 35 EER**. | Chọn model theo `int_g` (hoặc mean) — không bao giờ theo `int_ng` đơn lẻ. |
| 4 | Projection-out hướng gender (centroid m−f) trong CCA: gender 40.2 → 40.3, no_gender 37.6 → 39.5. | Gender **không nằm trên 1 hướng tuyến tính**; xoá tuyến tính không giúp. | Pipeline F (adversarial) chỉ khi B thất bại; B (same-gender negatives) hợp lý hơn F-linear. |
| 5 | Same-gender neg EER 40.3 vs cross-gender neg 33.5 (CCA-64, gap ≈ 5–7 điểm); neg same/cross mean distance 1.97 / 2.04, pos 1.81. | Gender shortcut tồn tại nhưng ở mức tuyến tính chỉ ~6 điểm — tương đương gap CB của FOP (7). | B1/B2 kỳ vọng thu ≤ 5 điểm ở cell gender, không hơn. |
| 6 | Voice train: **4029 vector unique / 6485 hàng**; 1267 nhóm trùng (max 24). id006/id052 có 2 mẫu, 3 speaker ≤ 5 voice unique. | Nhiều face crop dùng chung 1 audio → batch chứa duplicate voice; CE có thể memorize. | Sampler: 1 mẫu / voice-clip / batch, hoặc weight 1/dup. Loại id006/id052 khỏi internal val (không đủ mẫu). |
| 7 | Face: 89% giá trị = 0, không âm (post-ReLU fc7), norm 119±19, PCA 90% cần 526 chiều, effective dim 74. Voice: zero-mean, norm 137±14, PCA 90% cần 85 chiều, effective dim 53. | Cả hai modality đều có identity signal mạnh unimodal: **face–face EER 7.8, voice–voice 4.5** (raw cosine, centred). Không có NaN/Inf/const-dim. Data sạch. | Không cần sửa pipeline data. Standardize + PCA-256 face là bước tiền xử lý hợp lý. |
| 8 | Dev vs train (PCA-64 fit train): centroid lệch **face ≈ 1.0 sd, voice ≈ 2.1–2.4 sd** ngay cả với English; nhưng kNN-dist tới train (0.55) ≈ held-out speakers (0.52/0.54), MMD 0.02–0.03 ≈ ref 0.03. | Shift là **mean-offset**, không phải out-of-manifold. | Pipeline E (unsupervised per-file mean-centering / z-norm / AS-norm) rất rẻ và có cơ sở. Đưa vào EXP-002 luôn. |
| 9 | Voice En vs Bn dev: language probe 88%, centroid lệch 2.6 sd, kNN Bn→En 0.42 vs Bn→Bn 0.19. Face En vs Bn: probe 61%, MMD 0.005. **96% face Bangla có face English với cos > 0.6** (train same-speaker chỉ 0.46). | **Dev English và Bangla là cùng một tập người** (face gần như trùng), chỉ voice đổi ngôn ngữ. Language shift nằm hoàn toàn ở voice và lớn. | Cell Bangla: per-language voice centering (E) trước; multilingual voice (D) sau. Kỳ vọng E thu được phần lớn gap En–Bn 2.7 điểm. |
| 10 | Gender-probe trên dev: face/voice predicted-gender agree 83–89% ở protocol `gender`, 68% ở `no_gender`; Bangla ~65% nam, English ~52% nam. | Protocol `gender` đúng là cặp cùng giới; Bangla lệch nam. | Internal val gender protocol đúng như thiết kế EXP-001. |

**Quyết định bottleneck:** thứ tự = **(1) overfit/capacity + model selection → (2) score normalization (E) → (3) gender-aware negatives (B) → (4) voice language shift (D)**. Kiến trúc fusion không phải ưu tiên.

---

## EDA-0 — Integrity

| | face | voice |
|---|---|---|
| shape | 6485 × 4096 | 6485 × 192 |
| NaN / Inf / const dim | 0 / 0 / 0 | 0 / 0 / 0 |
| unique vectors | 6041 (391 nhóm trùng, max 5) | **4029 (1267 nhóm, max 24)** |
| trùng vượt speaker | 0 | 0 |
| norm mean ± std [min, max] | 119.1 ± 19.4 [50, 191] | 137.2 ± 14.5 [94, 208] |
| value range | [0, 38.4]; **88.9% = 0** | [−55, 51]; 49.5% âm |

- Cột cuối CSV = speaker int (0..69) = cột cuối txt (không phải video idx). `label` luôn 1. Face/voice align hàng ✓.
- Dev: không NaN; face/voice unique < số hàng (mỗi file ~5% face trùng, ~20% voice trùng — cùng cơ chế nhiều face/1 audio). Không overlap dev↔train. Overlap `no_gender`↔`gender` cùng ngôn ngữ: face 26–74, voice 98–180 vector → cùng pool. Voice En↔Bn overlap 0.

## EDA-1 — Speaker

- 70 speaker, 40 m / 30 f; mẫu: min 2 (id006, id052), median 88.5, max 226 (id002), ratio 113×. Mẫu theo giới cân (3260 m / 3225 f).
- Voice unique / speaker: min 1, median 54.5, max 163. Face unique / speaker: min 2, median 88.5.
- Unimodal raw cosine (centred, loại positive trùng clip), 5000 pos / 5000 neg:

| | no_gender EER | gender EER | AUC (ng) |
|---|---:|---:|---:|
| face–face | 7.83 | 10.28 | 97.7 |
| voice–voice | 4.54 | 5.65 | 98.8 |

- Positive trùng voice-clip chỉ 1.6% trial ngẫu nhiên → leak nhỏ ở mức trial, nhưng lớn ở mức batch (38% hàng là bản sao).
- Cos tới centroid speaker: face 0.68 (min 0.43), voice 0.77 (min 0.55).

## EDA-2 — Geometry

| | face | voice |
|---|---|---|
| PCA cum @ 32 / 64 / 128 / 256 | 0.50 / 0.63 / 0.73 / 0.82 | 0.66 / 0.84 / 0.97 / – |
| dims for 90 / 95 / 99% | 526 / 968 / 2087 | 85 / 110 / 147 |
| effective dim (PR) | 74 | 53 |
| top-1 PC share | 6.6% | 5.2% |
| mean \|corr\| (300 dims) | 0.044 | 0.091 |
| mean cos random pairs raw → centred | 0.150 → 0.002 | 0.098 → −0.001 |

Face raw có anisotropy (cos 0.15 giữa cặp ngẫu nhiên do ReLU non-neg) → **centering trước cosine** cải thiện EER 9.9 → 7.8. Hình: [out/eda2_pca.png](out/eda2_pca.png), [out/eda2_norms.png](out/eda2_norms.png).

## EDA-3 — Cross-modal alignment (RidgeCCA, speaker-disjoint 56/14, 3 seed)

Mean EER (%) trên 14 speaker val:

| pca_face | reg | k=4 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| 128 | 1.0 | 24.3 / 35.5 | 27.2 / 38.4 | 28.8 / 36.8 | 29.8 / 36.1 | 30.4 / 35.6 | 33.6 / 37.1 | 36.7 / 39.5 |
| 256 | 10.0 | 26.9 / 37.7 | 27.1 / 37.2 | 28.5 / 36.8 | 28.4 / 35.0 | **29.1 / 34.9** | 30.4 / 35.7 | 33.7 / 38.4 |

(no_gender / gender). k=128, reg 0.1: 44.6 / 46.1 (train 7.5). Shuffle-label sanity: 50.x. Canonical corr top-8 (70 spk): 0.60 0.56 0.49 0.46 0.44 0.43 0.41 0.41. Toàn bộ sweep: [out/eda3b_sweep.csv](out/eda3b_sweep.csv), hình [out/eda3b_k_sweep.png](out/eda3b_k_sweep.png), phân phối [out/eda3_cca_dist.png](out/eda3_cca_dist.png).

Submission tham chiếu (fit 70 spk, nộp `+d²`): `out/submission_cca_k32.zip` (pca 256 / reg 10 / k 32) — **khuyến nghị nộp**; `out/submission_cca_k4.zip` (k=4, gần như gender-only) — chỉ để đối chứng nếu còn budget.

## EDA-4 — Gender

- Probe logistic (PCA-128, speaker-disjoint): face **93.4 ± 2.4%**, voice **91.7 ± 2.6%**; trong CCA-64: face 91.9%, voice 92.8%.
- CCA-64 val, negative tách theo giới (3 seed): EER same-gender-neg **40.3** vs cross-gender-neg **35.5**; distance pos 1.81 / neg-same 1.97 / neg-cross 2.04. Hình [out/eda4_gender_neg.png](out/eda4_gender_neg.png).
- Project-out hướng gender: không cải thiện (gender 40.2→40.3, no_gender 37.6→39.5).
- Raw unimodal negatives: face cos same-gender vs cross-gender Cohen d và voice — xem `eda_34.json` (`*_raw_gender_gap_cohen_d`).

## EDA-5 — Language (dev, unsupervised)

| | face | voice |
|---|---|---|
| MMD En vs Bn (PCA-64) | 0.005 | **0.033** |
| centroid dist En–Bn (train sd) | 0.59 | **2.58** |
| language probe (5-fold, lạc quan) | 61% | **88%** |
| kNN5 Bn→En / En→En / Bn→Bn | 0.196 / 0.200 / 0.172 | **0.415** / 0.182 / 0.189 |

Face Bangla: nearest English face cos mean **0.84**, 96% > 0.6 (train same-speaker mean 0.46, diff 0.00) → cùng identity, có thể cùng video. Hình [out/eda56_pca_scatter.png](out/eda56_pca_scatter.png).

## EDA-6 — Train → Dev

| set | face norm | face centroid shift (sd) | face var ratio | voice norm | voice centroid shift (sd) | voice var ratio | kNN5→train (f / v) |
|---|---|---|---|---|---|---|---|
| train (held-out ref) | 119 | 0 | 1.00 | 137 | 0 | 1.00 | 0.518 / 0.540 |
| ng/English | 117 | 0.98 | 0.50 | 137 | 2.18 | 0.81 | 0.553 / 0.557 |
| ng/Bangla | 116 | 1.05 | 0.54 | 140 | 2.39 | 0.82 | 0.553 / 0.543 |
| g/English | 119 | 1.01 | 0.52 | 138 | 2.13 | 0.82 | 0.548 / 0.550 |
| g/Bangla | 115 | 1.03 | 0.53 | 140 | 2.41 | 0.81 | 0.554 / 0.548 |

Var ratio face 0.5 là do ít speaker hơn (between-speaker variance giảm) — bình thường. Hình [out/eda6_knn_dist.png](out/eda6_knn_dist.png).

## Hạn chế

- Internal 14 speaker unseen ≠ dev CodaBench (speaker khác, có thể khó hơn); std giữa seed 1–4 điểm. Cần 1 lần nộp CCA để calibrate.
- CCA là tuyến tính; trần 35 ở cell gender là trần **tuyến tính**, không phải trần của feature.
- Language probe trên dev không speaker-disjoint (dev không có nhãn) → 88% là lạc quan.
- EDA-7 (gate k, per-layer AUC của FOP) còn nợ — làm trong EXP-001 khi có checkpoint.
