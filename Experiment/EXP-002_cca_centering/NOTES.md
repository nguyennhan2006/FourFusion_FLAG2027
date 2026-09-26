# EXP-002 — Pipeline E trên CCA linear: per-file centering + AS-norm

- **Ngày:** 2026-09-22
- **Parent:** EXP-000f (CCA hybrid 33.35)
- **Pipeline:** E
- **Giả thuyết:** dev lệch train chỉ ở centroid (EDA-6, voice 2.1–2.4 sd) → trừ mean của chính file dev (unsupervised) rồi AS-norm score với cohort = cùng file sẽ giảm EER, đặc biệt `g/Bn`.
- **Thay đổi duy nhất so với parent:** thêm bước tiền xử lý/hậu xử lý; model CCA giữ nguyên (k4: pca128/reg1; k32: pca256/reg10).

## Cấu hình
| Tham số | Giá trị |
|---|---|
| center | `X − mean(X_file) + mean(X_train)` cho face và voice (`file`), hoặc chỉ voice (`file_voice_only`) |
| AS-norm | cohort top-200 trong cùng file, cả 2 phía, score = −normalized cos |
| internal | val 14 speaker được center bằng mean của chính val (mô phỏng) |

## Kết quả internal (3 seed, mean ± std)
| cfg | center | snorm | int_g | int_ng |
|---|---|---|---:|---:|
| k32 | none | raw | 34.67 ± 2.2 | 27.67 ± 0.4 |
| k32 | file | raw | 33.23 | 26.73 |
| k32 | file | **asnorm** | **32.62 ± 1.4** | 26.64 |
| k32 | voice-only | asnorm | 32.70 | 26.82 |
| k4 | none | raw | 35.55 ± 3.7 | 24.52 |
| k4 | file | **asnorm** | **33.09 ± 2.6** | **23.38** |

Toàn bộ: `out/exp002_internal_summary.csv`.

## CodaBench (chỉ nếu nộp)
| ng/En | ng/Bn | g/En | g/Bn | Overall | file nộp |
|---:|---:|---:|---:|---:|---|
| | | | | | out/submission_EXP002_hybrid.zip (ng ← k4_file_asnorm, g ← k32_file_asnorm) |

Score orientation: `−asnorm_cos` (lower = same) ☑ (cùng chiều +d²)

## Kết luận
- Internal: ✅ centering −1 đến −1.5, AS-norm thêm −0.6 đến −2.4; giảm cả std giữa seed.
- Chưa đo được tác dụng trên language shift (internal chỉ English) → cần nộp.

## Artifacts
- code: [run.py](run.py); các zip thành phần trong `out/` (8 biến thể) nếu muốn nộp riêng.
