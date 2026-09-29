# VAL-70 + nhiễu seed — vì sao dev CodaBench không dùng được để quyết định chênh lệch cỡ 1 EER

- **Ngày:** 2026-09-28
- **Bối cảnh:** SET-07, SET-08, SET-09 đều qua validation nhãn thật, nhưng đều tệ hơn SET-02 trên CodaBench (+0.52 / +0.62 / +0.81). g/Bn tệ hơn 2+ điểm cả ba lần.
- **Code:** [run.py](run.py) (VAL-70), [seed_noise.py](seed_noise.py) (SET-02 dựng lại, chỉ đổi seed của s007).

## 1. VAL-70: validation với thành phần train trên đủ 70 người (giống bản nộp)

- **Thiết kế.** Thành phần s007 của SET-02 / 08 / 09 train trên toàn bộ 70 người v4 (n = 5, seed như `dev_scores`).
- **File đánh giá.** File giống dev dựng từ người v1/v2, không ai trong số họ nằm trong dữ liệu train:
  - 30 người mỗi file, tối đa 15 giọng mỗi người;
  - mặt lấy khác video;
  - 10 file cho mỗi nguồn × ngôn ngữ.
- **Thước đo.** Mức mẫu, và mức người (khối theo danh tính thật, b = 0.99).

| Δ so với SET-02 (mẫu / người) [số file tốt hơn] | SET-08 | SET-09 |
|---|---|---|
| English ng | −0.38 / −0.76 | **−3.06 [20/20] / −2.41 [16/20]** |
| English g (v1) | −0.95 / −1.31 | **−1.94 [9/10] / −1.18 [7/10]** |
| Urdu ng | −0.95 / −1.17 | **−3.17 [10/10] / −2.51 [8/10]** |
| Urdu g | −1.07 / −1.83 | **−2.57 [10/10] / −3.85 [9/10]** |
| Hindi ng | −0.09 / −0.34 | **−2.64 [10/10] / −1.63 [7/10]** |
| CodaBench (dev) | +0.62 | +0.81 |

→ VAL-70 **không** tái tạo được dấu trên CodaBench. Nó đồng ý với validation v4 giữ ra (MODEL-01): SET-09 tốt hơn.

## 2. Nhiễu seed: SET-02 dựng lại, chỉ đổi seed của s007

- **Thiết kế.** Cùng công thức, n = 5, seed0 = 1001 / 2001 / 3001. Mọi thứ khác lấy nguyên từ `_dryrun`.
- **Chấm.** Dev chấm bằng pseudo-arc. Pseudo-arc khớp CodaBench ≤ 1 EER mỗi cell trên 4 hệ; ở đây nó chỉ dùng để **đo nhiễu của thước đo dev**, không để chọn hệ.

| Δ so với SET-02 gốc | ng/En | ng/Bn | g/En | g/Bn | trung bình |
|---|---:|---:|---:|---:|---:|
| seed 1001 | −0.72 | +0.29 | +0.95 | +4.77 | **+1.32** |
| seed 2001 | +0.21 | −0.50 | −0.42 | +3.32 | **+0.65** |
| seed 3001 | +0.72 | −0.14 | −0.21 | +2.49 | **+0.72** |
| SET-09 (thay đổi thật, CodaBench) | +1.69 | −1.00 | +0.41 | +2.11 | +0.81 |

## Kết luận

1. **Chênh lệch của SET-08 / SET-09 trên dev nằm trong nhiễu seed.**
   - Chỉ đổi seed đã làm dev tệ đi 0.65–1.32, và g/Bn tệ đi 2.5–4.8 ở **cả ba** lần.
   - SET-02 gốc là một lần "rút trúng" trên file g/Bn. Nó được chọn trên chính các file dev này (qua pseudo-label), nên điểm dev của nó bị phồng theo kiểu **winner's curse**.
   - Hệ quả: mọi thay đổi s007 đều trông tệ hơn trên dev, kể cả khi hoà hoặc tốt hơn thật.
2. **Hệ quả cho Evaluation phase** (người mới, không có chọn lọc):
   - Nên kỳ vọng SET-02 tệ hơn 21.68 khoảng 0.7–1.3.
   - Quyết định chênh lệch cỡ ±1 phải dựa vào validation độc lập: v4 giữ ra + VAL-70, với nhiều người và nhiều file. Không dựa vào dev CodaBench (4 file × khoảng 30 người, đã bị chọn lọc).
3. **SET-09 (FaRL):** 2/2 validation độc lập nói tốt hơn ở mọi ô. Dev cho +0.81, nằm trong nhiễu seed → **ứng viên hợp lệ cho cấu hình Evaluation**. Quyết định cuối thuộc về nhóm.
4. **Việc nên làm tiếp (rẻ):**
   - **Giảm phương sai theo seed** cho mọi thành phần: n lớn hơn, hoặc trung bình nhiều bộ seed. Đo bằng VAL-70.
   - Nhiễu seed là rủi ro thật cho Evaluation, lớn cỡ ±1 EER tổng.

## 3. Lượt 2–3 (28/09): các ứng viên, cùng file, cùng thành phần (cache)

Δ so với SET-02, **mức người** (mức mẫu trong ngoặc); [số file tốt hơn] là số file tốt hơn ở mức người.

| Hệ | English g | English ng | Urdu g | Urdu ng | Hindi ng |
|---|---|---|---|---|---|
| AGE.1 (SET-10) | −0.83 (−0.52) [7/10] | −0.03 (−0.46) [12/20] | −0.51 (−0.44) [5/10] | −0.21 (−0.30) [5/10] | −1.42 (−0.45) [9/10] |
| FARL5+AGE.1 (SET-11) | −1.23 (−2.47) | −2.71 (−3.46) | −4.27 (−2.95) | −2.75 (−3.48) | −2.31 (−3.13) |
| **SET-12** (IB) | −8.32 (−6.13) [10/10] | −6.76 (−6.48) [20/20] | −7.00 (−5.67) [10/10] | −5.69 (−6.20) [10/10] | −7.59 (−5.41) [10/10] |
| **SET-13** (IB + tuổi) | **−9.02** (−6.33) [10/10] | **−6.86** (−6.76) [20/20] | −7.21 (−5.90) [10/10] | **−6.05** (−6.28) [10/10] | **−7.76** (−5.71) [10/10] |
| SET-14 (FaRL + IB + tuổi) | −8.24 (−6.52) [10/10] | −6.14 (−7.20) [20/20] | **−7.54** (−6.50) [10/10] | −5.49 (−6.91) [10/10] | −6.09 (−6.15) [10/10] |

**Đọc kết quả:**
- Các bản ImageBind **thắng ở mọi file, mọi ô**, cả English lẫn Urdu/Hindi, với biên độ 5–9 EER. Biên độ đó lớn hơn nhiều so với nhiễu seed (khoảng 1) lẫn lời nguyền người thắng trên dev.
- **SET-13 tốt nhất ở mức người** (mức nộp bài), trung bình −7.4, so với SET-12 −7.1 và SET-14 −6.7.
- g/Bn trên dev (SET-12: +5.52) là cell duy nhất nói ngược, và đó đúng là cell nhạy với seed nhất (nhật ký 61).

## 4. Nhiễu seed trong VAL-70: n = 5 hay n = 15 (28/09, `VAL_SEEDVAR=1`)

- **Thiết kế.** 3 bộ seed khác nhau (0 / 1000 / 2000), mỗi bộ n = 5, cho s007 / s010B / r2mix. n = 15 = trung bình ma trận của cả 3 bộ. Công thức SET-02 và SET-13, cùng file VAL-70.
- **Kết quả** ([out/seed_var_summary.csv](out/seed_var_summary.csv)). "Chênh giữa 3 bộ" = max − min trên cùng một file, trung bình theo file.

| SET-13, mức người | English g | English ng | Urdu g | Urdu ng | Hindi ng |
|---|---:|---:|---:|---:|---:|
| n = 5, trung bình 3 bộ | 26.43 | 19.56 | 28.78 | 19.00 | 19.20 |
| chênh giữa 3 bộ (một file) | 2.05 | 0.99 | 1.51 | 1.11 | 1.17 |
| n = 15 | 26.22 | 19.53 | 28.91 | 19.13 | 19.03 |

- **Đọc kết quả:**
  - n = 15 cho **cùng kỳ vọng** với n = 5 (chênh −0.44…+0.26 ở cả SET-02 lẫn SET-13).
  - Nhưng một lần n = 5 là một lần "rút thăm": trên từng file, EER mức người lệch 1–2.7 tuỳ bộ seed. Mức mẫu chỉ lệch 0.4–0.7, vì gom cụm dồn lỗi về mức người.
- **Quyết định đề xuất:** dùng **n = 15** cho Evaluation. Kỳ vọng không đổi, bỏ được phần may rủi theo seed, chỉ tốn thêm khoảng 1 giờ CPU.
