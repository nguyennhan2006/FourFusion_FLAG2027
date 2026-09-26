# EXP-000d — CCA linear reference (không train deep)

- **Ngày:** 2026-09-22
- **Parent:** EDA-000 ([../EDA-000_raw/NOTES.md](../EDA-000_raw/NOTES.md))
- **Pipeline:** – (tham chiếu tuyến tính, ngoài A–F)
- **Giả thuyết:** feature gốc đã mang đủ identity cross-modal; một phép chiếu tuyến tính chọn theo internal `int_g` sẽ ≥ FOP 000c trên CodaBench.
- **Thay đổi duy nhất so với parent:** fit trên toàn 70 speaker thay vì 56.

## Cấu hình

| Tham số | Giá trị |
|---|---|
| tiền xử lý | standardize (mean/std train) → face PCA-256, voice giữ 192 |
| model | `eda_utils.RidgeCCA(k=32, reg=10.0, pca_x=256)` |
| score | `+ (2 − 2·cos)` trong CCA space, lower = same |
| train | không có (closed-form) |
| chọn config | sweep pca∈{128,256} × reg∈{0.01,0.1,1,10} × k∈{4..64}, 3 seed 56/14, chọn theo `int_g` (bảng tại EDA-000 §3) |

## Kết quả internal (speaker-disjoint, 3 seed)

| split seed | EER no_gender | EER gender | mean |
|---|---:|---:|---:|
| 1,2,3 (mean) | 29.1 | 34.9 | 32.0 |

## CodaBench

| ng/En | ng/Bn | g/En | g/Bn | Overall | file nộp |
|---:|---:|---:|---:|---:|---|
| 33.73 | 34.00 | 34.42 | 41.01 | **35.79** | out/submission_EXP000d.zip |

Score orientation đã nộp: `+d²` (lower = same) ☑

## Kết luận

- Giả thuyết: ✅ đúng — 35.79 < FOP 000c 42.17 (−6.38) và < organizer FOP ref 36.92.
- **Calibration internal↔CB:** `int_g` 34.9 ↔ `g/En` 34.4 (khớp); `int_ng` 29.1 ↔ `ng/En` 33.7 (internal lạc quan 4.6). → Từ nay chọn model theo `int_g`; `int_ng` chỉ để tham khảo.
- Gap heard→unheard: no_gender +0.27, gender **+6.59**. Language shift ở voice chỉ gây hại khi không còn gender shortcut → cell `g/Bn` 41.0 là mục tiêu của E (per-language voice centering) rồi D.
- Bước tiếp theo: EXP-001 (internal val + EDA-7 trên ckpt 000c) và EXP-002 = CCA + per-file/per-language centering (E) — kỳ vọng kéo g/Bn xuống trước khi quay lại deep model.

## Artifacts

- code: [../EDA-000_raw/run_eda_3b.py](../EDA-000_raw/run_eda_3b.py) (sweep) + snippet k=32 trong EDA-000 NOTES
- submission: out/submission_EXP000d.zip (= EDA-000_raw/out/submission_cca_k32.zip)

---

## Phụ lục — EXP-000e (k=4) và EXP-000f (hybrid), 2026-09-22

| EXP | config | ng/En | ng/Bn | g/En | g/Bn | Overall |
|---|---|---:|---:|---:|---:|---:|
| 000d | k=32, pca 256, reg 10 | 33.73 | 34.00 | **34.42** | **41.01** | 35.79 |
| 000e | k=4, pca 128, reg 1 | **28.97** | **29.02** | 37.47 | 42.37 | 34.46 |
| 000f | hybrid: 000e cho `no_gender/*`, 000d cho `gender/*` | 28.97 | 29.02 | 34.42 | 41.01 | **33.35** |

- Hybrid khớp dự đoán (33.36) → cell độc lập, ghép per-cell là chiến lược nộp chuẩn từ nay.
- Files: `out/submission_EXP000e_k4.zip`, `out/submission_EXP000f_hybrid.zip`.
