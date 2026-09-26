# EXP-000c — FOP source-faithful baseline

- **Ngày:** 2026-09-21 (submission zip lúc 02:27)
- **Parent:** – (baseline gốc)
- **Pipeline:** FOP gốc (Saeed et al., ICASSP 2022), chi tiết trong [PIPELINE.md](PIPELINE.md)
- **Giả thuyết:** implement trung thành FOP + speaker-disjoint selection cho một baseline tin cậy để so sánh về sau.
- **Thay đổi duy nhất so với parent:** –

## Cấu hình

| Tham số | Giá trị |
|---|---|
| lr / wd / batch | 1e-5 / 0.01 / 128 (Adam) |
| embed dim / dropout / fusion | 128 / 0.5 / gated |
| loss | CE + α·OPL, OPL = (1 − pos_cos) + 0.7·\|neg_cos\| |
| α sweep | {0, 0.1, 0.5, 1, 2, 5} |
| epochs | ≤ 50, chọn best epoch theo internal EER, rồi retrain fresh trên 70 id |
| seed(s) | 1 |
| split internal (seed) | 56 / 14 speaker, seed 1, negative ngẫu nhiên (**không** ép cùng giới) |

## Kết quả internal (speaker-disjoint)

| split seed | EER no_gender | EER gender | mean | AUC | best α / epoch |
|---|---:|---:|---:|---:|---|
| 1 | (điền từ output notebook) | – (chưa đo) | – | | |

Seen-train EER (sanity): (điền từ notebook)

## CodaBench

| ng/En | ng/Bn | g/En | g/Bn | Overall | file nộp |
|---:|---:|---:|---:|---:|---|
| 36.31 | 38.98 | 43.38 | 50.00 | **42.17** | out/submission_baseline.zip |

Score orientation đã nộp: `+squared_L2` (lower = same) ☑ đã kiểm tra — nộp `−d²` cho 57.83.

Organizer FOP reference: 32.54 / 38.12 / 32.99 / 44.01 → 36.92.

## Kết luận

- Baseline chạy được, kém organizer reference 5.25 điểm.
- `gender/Bangla` = 50.00 (random) và `gender/English` cao hơn `no_gender/English` 7 điểm → model đang dựa nhiều vào gender shortcut.
- Internal validation không phát hiện được vì negative không ép cùng giới → phải sửa trước (EXP-001).
- Bước tiếp theo: EXP-001 theo [docs/RESEARCH_PLAN.md §2](../../docs/RESEARCH_PLAN.md).

## Artifacts

- notebook: [FLAG_2027_Baseline.ipynb](FLAG_2027_Baseline.ipynb) — chạy trên Kaggle, đọc từ `/kaggle/input`
- checkpoint: chưa copy về local (chỉ có trong Kaggle `RUN_DIR`)
- metadata json: `run_metadata_EXP000c.json` (Kaggle, chưa copy về)
- submission: [out/submission_baseline.zip](out/submission_baseline.zip)
