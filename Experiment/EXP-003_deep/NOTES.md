# EXP-003 — Deep cross-modal projection (A / B / C sweep) + EXP-003b ensemble/E

- **Ngày:** 2026-09-22
- **Parent:** EXP-000f (33.35), EXP-002 (E)
- **Pipeline:** A (CE+OPL, FOP-like), AB (OPL same-gender), B (InfoNCE same-gender neg), C (InfoNCE SupCon), rồi E + ensemble
- **Giả thuyết:** (1) FOP thua CCA vì loss CE+OPL không ép alignment; InfoNCE trực tiếp sẽ vượt CCA ở cell gender. (2) Same-gender negatives giúp cell gender nhưng hại cell no_gender.
- **Thay đổi so với parent:** model MLP 2 nhánh (PCA-256 face / 192 voice → 512 → emb, BN, dropout 0.5), AdamW lr 1e-3 wd 1e-2, cosine, batch 256, **dedup theo voice-clip mỗi epoch**, chọn theo `int_g`.

## Cấu hình sweep (40 epoch, 3 split seed, CPU ~13 s/run)
`train.py` — 15 cfg: {A, AB, B, C} × emb {32, 64, 128} + C+CE, A no-dedup, C τ=0.2. Kết quả `out/sweep_*.csv`.

## Kết quả internal — sweep (mean 3 seed; best-epoch theo int_g, `last` = epoch 40)
| cfg | int_g | last_g | int_ng |
|---|---:|---:|---:|
| B_infonce_sg_e128 | **32.67** | 33.5 | 30.5 |
| C_infonce_e128 | **33.22** | 34.0 | **24.6** |
| C_infonce_e32 / e64 | 33.6 / 33.6 | 34.2 | 24.4 / 24.2 |
| C + CE 0.5 | 34.3 | 35.0 | 24.5 |
| A_ce_opl e32 / e64 / e128 | 36.8 / 37.7 / 37.7 | 38–38.6 | 25.6–26.7 |
| AB_ce_opl_sg | 36.8–37.1 | 37.5–38.1 | 27–29 |
| *CCA k32 / k4* | 34.9 / 35.5 | | 29.1 / 24.3 |

## EXP-003b — ensemble + E (epoch cố định: C 15, B 10; 3 train-seed / split; fusion = mean z-score)
| recipe | center | snorm | int_g | int_ng |
|---|---|---|---:|---:|
| **C_ens3 + k4** | file | asnorm | **30.02** | **21.76** |
| C_ens3 + k32 | file | asnorm | 31.01 | 23.32 |
| C_ens3 | file | asnorm | 31.22 | 23.08 |
| C_ens3 + B_ens3 + k32 | file | asnorm | 31.22 | 24.98 |
| C (1 seed) | none | raw | 34.42 | 25.22 |
| k32 (000d) / k4 (000e) | none | raw | 34.67 / 35.55 | 27.67 / 24.52 |

Toàn bộ: `out/combine_internal.csv`.

## CodaBench
| ng/En | ng/Bn | g/En | g/Bn | Overall | file nộp |
|---:|---:|---:|---:|---:|---|
| 25.79 | 28.88 | 33.81 | 39.37 | **31.96** | out/submission_EXP003_C_ens3+k4_file_asnorm.zip |

Score = −(fused z-score), lower = same ☑. Sanity: 4 file, pair_id khớp, không NaN, corr với 000f ≈ 0.6–0.7.

## Kết luận
- H1 ✅: InfoNCE 33.2 < CCA 34.9 < CE+OPL 37 ở cell gender. FOP-style loss là bottleneck, không phải hyperparameter (đã sweep lr/emb/dedup).
- H2 ✅: B int_g 32.7 (tốt nhất single) nhưng int_ng 30.5; sau ensemble/E thì C+k4 vượt B ở cả hai cell → B không cần cho submission.
- E (centering + AS-norm) cộng thêm −3 ở gender, −1.5 ở no_gender trên deep; ensemble 3 seed −0.3…−1.
- Bug đã sửa trong quá trình: cache PCA theo tập train (trước đó dùng nhầm PCA all-data → +6 EER).
- CB 31.96: thắng 000f ở cả 4 cell nhưng lợi ích tập trung ở English (−3.2 / −0.6); Bangla −0.1 / −1.6. Gap ngôn ngữ tăng (+3.1 / +5.6). int_g lệch CB +3.8 (trước ≈ 0).
- Bước tiếp theo: EXP-004 tune C (hid/dropout/τ/wd, epochs, PCA dim), SupCon với hard-negative mining, per-language centering cho Bangla.

## Artifacts
- code: [train.py](train.py), [combine.py](combine.py); zip các recipe khác trong `out/`.
