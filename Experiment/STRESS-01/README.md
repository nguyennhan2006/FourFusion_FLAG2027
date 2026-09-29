# STRESS-01: gom cụm (SET-13) có hỏng khi file Evaluation khác dev không? (E2 / E4 trong [PLAN_EVAL](../../docs/PLAN_EVAL.md))

- **Ngày:** 2026-09-28. **Code:** [run.py](run.py) (phần A), [run_b.py](run_b.py) (phần B). Kết quả: [results.csv](results.csv).
- **Câu hỏi:** gom cụm đáng 5.7 điểm trên dev. File Evaluation trông thế nào thì nó mất tác dụng hoặc gây hại? Có dấu hiệu nào, không cần nhãn, báo trước được không?

## Cách làm (phần A: người v4 giữ ra, tiếng Anh, nhãn thật)

- **Dữ liệu:** 5 split, mỗi split train 40 người v4, giữ ra 30 người.
- **Thành phần:** s007′, tức s007 (MLP × 3 + CCA-4) + ImageBind + tuổi, đúng phần khác nhau giữa SET-13 và INDEP-01.
- **File mô phỏng:** dựng như danh sách dev. Mỗi ảnh và mỗi giọng xuất hiện một lần; mặt và giọng của một người lấy từ các dòng khác nhau.
- **Mỗi file được chấm 5 cách:**

| Nhánh | Là gì |
|---|---|
| L0 | INDEP-01: từng cặp, hằng số từ cặp train |
| L1 | SET-13 bỏ cụm: trừ mean và z-score theo file |
| L2 | SET-13: L1 + trung bình khối theo cụm ước lượng (ngưỡng hiệu chỉnh trên người train) |
| L2a | L2 với ngưỡng giọng tự thích nghi: số cụm giọng gần số cụm mặt nhất |
| L2o | L2 với cụm theo danh tính thật (trần) |

- **Kịch bản:** 20 file mỗi kịch bản (5 split × 2 lần lặp × 2 protocol).
  - m1 … m32: 30 người, mỗi người m ảnh và m giọng;
  - pos20: chỉ 20% cặp dương;
  - dom: một người chiếm khoảng 26% file;
  - p5 / p10: file chỉ có 5 hoặc 10 người.

## Kết quả phần A (EER trung bình)

| Kịch bản | Protocol | L0 | L1 | **L2** | L2a | L2o | L2 − L0 |
|---|---|---:|---:|---:|---:|---:|---:|
| m1 | ng / g | 28.60 / 31.21 | 23.09 / 29.64 | **22.53 / 29.26** | 23.09 / 29.05 | 23.09 / 29.64 | −6.1 / −2.0 |
| m2 | ng / g | 22.24 / 31.19 | 20.23 / 27.82 | 21.55 / 29.13 | 21.55 / 29.13 | 20.20 / 28.45 | −0.7 / −2.1 |
| m4 | ng / g | 24.21 / 33.76 | 22.52 / 33.16 | **19.07 / 30.52** | 18.90 / 30.69 | 18.27 / 30.00 | −5.1 / −3.2 |
| m8 | ng / g | 22.81 / 33.12 | 22.53 / 32.89 | **18.99 / 29.40** | 19.11 / 29.57 | 18.11 / 28.69 | −3.8 / −3.7 |
| m16 | ng / g | 21.91 / 34.99 | 21.02 / 34.45 | **16.78 / 28.43** | 16.99 / 28.62 | 16.43 / 27.35 | −5.1 / −6.6 |
| m32 | ng / g | 22.92 / 33.63 | 22.27 / 33.34 | **16.99 / 28.10** | 16.71 / 27.79 | 16.39 / 26.75 | −5.9 / −5.5 |
| pos20 | ng / g | 24.39 / 33.79 | 23.86 / 34.22 | **18.60 / 27.85** | 18.98 / 28.30 | 17.72 / 27.43 | −5.8 / −5.9 |
| dom | ng / g | 25.62 / 33.55 | 23.21 / 34.22 | **17.90 / 27.90** | 17.90 / 28.20 | 12.91 / 26.98 | −7.7 / −5.7 |
| p5 | ng / g | 23.19 / 31.06 | 22.63 / 25.88 | **16.16 / 24.22** | 15.03 / 23.47 | 14.42 / 23.85 | −7.0 / −6.8 |
| p10 | ng / g | 24.91 / 34.16 | 22.14 / 31.13 | **16.20 / 22.97** | 15.91 / 21.99 | 15.93 / 22.80 | −8.7 / −11.2 |

## Đọc kết quả

1. **Gom cụm tốt hơn chấm từng cặp ở mọi kịch bản**, kể cả một mẫu mỗi người, 5 người mỗi file, và một người chiếm phần lớn file. Lý do ở m1: dù mỗi người chỉ có 1 mẫu, cụm vẫn có tác dụng như L1 (trừ mean và z theo file), và L1 đã tốt hơn L0.
2. **15/200 file gom cụm thua L0 quá 1 EER. Tất cả là file tí hon** (30–120 trial: m1, m2, m4, p5), nơi EER dao động ±5–10. Không dấu hiệu chẩn đoán nào tách được chúng (tỉ lệ cụm giọng/mặt 1.07 so với 1.12 trung bình; cụm lớn nhất 0.09 so với 0.10).
3. **Hai quy tắc `auto` tạm thời đều gây hại** và đã bị bỏ khỏi `BEST/v6_eval`:
   - "cụm lớn nhất > 25% file → INDEP" làm mất 5.7 (g) và 2.6 (ng) ở dom, 4.4 ở p5 g;
   - "cụm trung vị < 2 → INDEP" làm mất 6.1 và 2.0 ở m1.
4. **Ngưỡng giọng tự thích nghi (L2a) không hơn** ngưỡng hiệu chỉnh trên train: chênh ±0.3, chỉ nhỉnh hơn một chút ở p5 / p10. Không đưa vào pipeline.
5. **L1 (thống kê file, không cụm) ≥ L0 (INDEP-01) ở gần hết kịch bản**, chênh lớn nhất ở file nhỏ: m1 ng −5.5, p5 g −5.2. Vì vậy **bản dự phòng khi cụm hỏng nên là L1 (`noclus`), không phải INDEP-01**. INDEP-01 chỉ còn dùng khi luật cấm dùng các mẫu test khác.
6. **Phần A không có kịch bản nào mà giọng bị gộp nhầm nhiều**: tỉ lệ cụm giọng/mặt của từng file nằm trong 0.875–1.64, không file nào dưới 0.8, vì người v4 là tiếng Anh, cùng miền với ngưỡng. Rủi ro gộp nhầm ở ngôn ngữ khác hoặc file đông người (kiểu SET-04) là việc của phần B.

## Phần B: người mới, ngôn ngữ khác, file đông người ([run_b.py](run_b.py), [results_b.csv](results_b.csv))

- **Thiết kế:** người MAV-Celeb v1 (Urdu + English) và v2 (Hindi + English), trừ 2 người trùng với v4. Thành phần s007′ là bản production (5 mạng train trên đủ 70 người v4, lấy từ cache của `BEST/v6_eval`).
- **Đặc trưng:** ArcFace / ECAPA-192 thật, từ output Kaggle của FLAG_12.
- **Cặp dương** luôn lấy mặt **khác video** với giọng.
- 10 file mỗi kịch bản (5 lần lặp × 2 protocol).

| Kịch bản | Protocol | L0 | L1 | **L2** | L2a | L2o | L2 − L0 | ARI mặt / giọng | cụm giọng / mặt |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| Urdu, 30 người × 16 | ng / g | 25.89 / 36.96 | 24.30 / 36.79 | **20.94 / 31.44** | 21.52 / 32.03 | 18.59 / 30.52 | −5.0 / −5.5 | 0.99 / 0.91 | 1.42 |
| Hindi, 30 × 16 | ng / g | 24.94 / 39.60 | 25.40 / 39.84 | **20.41 / 34.26** | 21.10 / 34.79 | 20.28 / 32.69 | −4.5 / −5.3 | 0.99 / 0.93 | 1.36 |
| English (v1/v2), 30 × 16 | ng / g | 25.68 / 36.13 | 25.43 / 36.35 | **19.15 / 31.77** | 19.89 / 33.24 | 18.63 / 31.27 | −6.5 / −4.4 | 0.99 / 0.91 | 1.44 |
| 60 người × 8 | ng / g | 24.42 / 37.58 | 24.25 / 38.33 | **21.08 / 34.33** | 21.33 / 34.92 | 19.58 / 33.25 | −3.3 / −3.3 | 0.99 / 0.89 | 1.27 |
| 100 người × 8 | ng / g | 26.30 / 39.05 | 25.40 / 38.70 | **21.55 / 36.40** | 21.70 / 36.05 | 20.35 / 34.35 | −4.8 / −2.7 | 0.99 / 0.86 | 1.15 |
| 150 người × 8 | ng / g | 26.27 / 37.87 | 25.20 / 37.53 | **22.03 / 34.37** | 22.27 / 34.80 | 20.67 / 33.93 | −4.2 / −3.5 | 0.99 / 0.81 | 1.14 |
| Urdu, 30 × 2 | ng / g | 25.33 / 36.67 | 24.00 / 36.00 | **20.00 / 34.67** | 20.00 / 34.67 | 20.67 / 33.33 | −5.3 / −2.0 | 0.99 / 0.80 | 1.06 |
| Hindi, 30 × 2 | ng / g | 28.67 / 40.18 | 28.67 / 42.00 | **24.00 / 34.00** | 24.00 / 34.67 | 22.67 / 32.67 | −4.7 / −6.2 | 0.99 / 0.84 | 1.06 |

**Đọc kết quả phần B:**
1. **Gom cụm vẫn tốt hơn ở mọi kịch bản** (−2.0 đến −6.5): ngôn ngữ chưa dùng để hiệu chỉnh ngưỡng (Urdu, Hindi), file đông gấp 5 lần dev, và 2 mẫu mỗi người. L2 cách trần (cụm đúng danh tính, L2o) chỉ 0.1–2.
2. **Cụm mặt gần như hoàn hảo ở mọi nơi** (ARI 0.99). **Cụm giọng kém dần khi file đông lên** (ARI 0.91 ở 30 người, 0.81 ở 150 người), vì có nhiều người giọng gần nhau hơn. Tuy vậy lợi ích vẫn còn 3–5 EER.
3. **Không lần nào giọng bị gộp nhầm theo kiểu SET-04.** Ngưỡng 0.70 luôn cho **nhiều** cụm giọng hơn cụm mặt (tỉ lệ 1.06–1.44). Nó thiên về tách thừa, và tách thừa chỉ làm giảm một phần lợi ích. Cảnh báo "giọng / mặt < 0.8" không lần nào bật.
4. **Ngưỡng giọng tự thích nghi (L2a) vẫn không hơn** (tệ hơn 0–1.5).

## Kết luận chung (A + B, 280 file)

- **SET-13 (gom cụm) là lựa chọn cho mọi dạng file Evaluation đã thử được.** `BEST/v6_eval --mode auto` giữ nguyên: gom cụm mọi file, cảnh báo khi giọng / mặt < 0.8, đổi sang `noclus` bằng tay nếu cần.
- **Dạng file duy nhất chưa thử:** người Evaluation nói Bangla. Đây cũng là ngôn ngữ của dev, nơi cụm giọng Bangla ra 39–41 cụm cho khoảng 30 người, tức tách thừa chứ không gộp nhầm.
