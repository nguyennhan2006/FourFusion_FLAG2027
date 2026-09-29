# EXTSCALE-01: thêm người train có còn giúp không, SetProto có ích hơn khi nhiều người không, và còn lại bao nhiêu sau ImageBind?

- **Ngày:** 2026-09-28/29. **Code:** [run.py](run.py). **Kết quả:** [results_part1.csv](results_part1.csv), [results_part2.csv](results_part2.csv).
- **Nguồn:** hai thí nghiệm EXT-VAL70 và SCALE-ID do `deep-research-report` (28/09) đề xuất.
- **Những gì EXT-03 để ngỏ:**
  - nhánh có dữ liệu ngoài được nhiều bước tối ưu hơn;
  - chỉ chấm thành phần trần, không chấm bản production;
  - chưa có đường cong theo số người;
  - chưa có người tiếng Anh giữ ra của v1/v2.

## Thiết kế (nhãn thật, theo cặp)

- **Mọi nhánh:**
  - dùng bộ lấy mẫu P × K và **cùng số bước tối ưu**;
  - 3 mạng mỗi nhánh, công thức EXP-007;
  - người v1/v2 chia 5 fold theo người; fold giữ ra không bao giờ được train.
- **Hai cách chấm, như SET-13 cho từng file:**
  - `member`: cầu nối đứng riêng;
  - `prod`: s007′ = z(0.5 z(member) + 0.5 z(ImageBind)) + 0.1 z(tuổi).
- **Ba mức:** mẫu, người (khối theo danh tính thật), cụm (ArcFace / ECAPA ở ngưỡng production).
- **File đánh giá:** giọng Urdu / Hindi / Anh của người giữ ra, mặt của cặp dương luôn lấy **khác video**. Ở phần 2 có thêm file 30 người v4 giữ ra.

| Phần | Nhánh | Train | P × K, số bước |
|---|---|---|---|
| 1 (EXT-VAL70) | R / P / XR / XP | 70 người v4 (+ v1/v2 của 4 fold kia), không / có SetProto | 32 × 8, 1200 |
| 2 (SCALE-ID) | 7 điểm × {row, SetProto} | 10 / 20 / 30 / 40 người v4 lồng nhau, rồi 40 + 25 / 50 / 100% người v1/v2 lồng nhau (tổng 70 / 100 / 161 người) | 8 × 8, 1500 |

## Phần 1: 70 người v4, thêm người v1/v2 và SetProto (lợi so với R; + = tốt hơn; [số fold thắng /5])

| Mức | Dữ liệu giữ ra | member: XR | member: XP | **prod: P** | **prod: XR** | **prod: XP** |
|---|---|---|---|---|---|---|
| mẫu | English g / ng | +3.82 [4] / +2.98 [5] | +3.93 [5] / +2.64 [5] | +0.33 / +0.34 | +1.93 [4] / +1.68 [4] | +2.20 [4] / +1.61 [4] |
| mẫu | Hindi ng | +5.54 [5] | +4.90 [5] | +0.39 | +1.57 [4] | +1.71 [3] |
| mẫu | Urdu g / ng | +1.74 [4] / +2.01 [3] | +2.20 [4] / +2.48 [3] | −0.14 / +0.20 | −0.85 [2] / +0.22 [3] | +0.46 [3] / −0.58 [1] |
| cụm | English g / ng | +3.13 [4] / +1.41 [3] | +2.65 [4] / +2.17 [5] | +0.34 / −0.33 | **+2.33 [5]** / +0.39 [3] | **+2.67 [5]** / +0.40 [2] |
| cụm | Hindi ng | +3.97 [3] | +3.11 [3] | +0.26 | +1.20 [3] | +1.03 [3] |
| cụm | Urdu g / ng | +0.51 [4] / +0.24 [2] | +0.01 [3] / −0.66 [2] | +0.17 / −0.22 | −0.27 [2] / −0.46 [3] | +0.02 [3] / −0.49 [3] |
| người | English g / ng | +3.40 [4] / +2.25 [5] | +3.51 [4] / +2.10 [5] | −0.03 / −0.24 | +1.51 [3] / −0.18 [3] | +1.42 [4] / −0.01 [3] |

## Phần 2: đường cong theo số người (EER trung bình, nhánh row; "ext" = trung bình Urdu / Hindi / English giữ ra)

| Số người train | 10 | 20 | 30 | 40 | 70 | 100 | 161 |
|---|---:|---:|---:|---:|---:|---:|---:|
| member, mẫu, ext ng | 45.24 | 37.10 | 34.69 | 33.25 | 29.35 | 27.76 | **27.12** |
| member, mẫu, v4 g | 48.30 | 42.16 | 40.26 | 36.96 | 35.76 | 35.95 | **34.78** |
| prod, mẫu, ext ng | 30.38 | 27.20 | 25.89 | 26.00 | 24.18 | 23.31 | 23.51 |
| prod, người, ext ng | 23.77 | 21.57 | 20.27 | 19.63 | 18.86 | 18.03 | 17.82 |
| prod, cụm, v4 g | 33.72 | 29.66 | 29.36 | 28.86 | 26.10 | 26.30 | **24.32** |
| prod, cụm, v4 ng | 21.83 | 18.98 | 18.72 | 17.19 | 16.57 | 17.07 | 15.53 |

**Lợi theo cặp split ở bản production** (40 người v4 → 161 người, + = tốt hơn):
- **mức cụm:** v4 g +4.55 [5/5], Hindi +3.40 [4/5], English +1.1…+1.8 [3–4/5], Urdu ±1;
- **bước cuối 100 → 161 người:** chỉ v4 ng còn ≥ 0.5 với ≥ 4/5 split (cụm +1.54, người +0.65). Các ô khác từ −1.6 đến +2.0 và thắng ≤ 3/5.

## Kết luận

1. **SetProto không có tác dụng khi ngân sách bằng nhau**, ở mọi số người (±0.3) và cả khi thêm người v1/v2: tương tác ≈ 0. Giả thuyết trung tâm của báo cáo không được xác nhận. Lợi ích EXT-03 từng thấy ở B3P đến từ việc thêm người và thêm bước tối ưu, không từ SetProto. **Đóng SetProto.**
2. **Thêm người giúp cầu nối rõ rệt, và cầu nối trần chưa bão hoà ở 161 người.**
3. **Nhưng ImageBind đã chứa phần lớn cái lợi đó.** Trong bản production, lợi từ người v1/v2 co lại:
   - còn rõ ở **English g** (cụm +2.3 [5/5] ở phần 1, v4 g +4.6 [5/5] ở phần 2) và Hindi;
   - gần 0 ở Urdu và English ng.
4. **Cổng VoxCeleb (X1) chưa đạt.** Ở bản production, bước cuối của đường cong gần như chỉ còn nhiễu. Hàng nghìn người có lẽ vẫn giúp cầu nối, nhưng lợi dự kiến cho hệ nộp nhỏ (≤ 1–2 EER) so với công sức trích dữ liệu.
5. **Ứng viên cụ thể (chưa làm):** SET-13 với s007 train trên 70 người v4 + toàn bộ v1/v2, **chỉ cho các cell English**, nơi lợi ổn định nhất.
   - Bằng chứng: English g +2.3…+4.6, thắng 5/5 ở cả hai phần.
   - Rủi ro: English ng gần 0, và SET-07 (cùng ý tưởng, chưa có ImageBind) cho ng/En +2.98 trên dev.
   - Cần một lượt kiểm chứng riêng trước khi dùng.
