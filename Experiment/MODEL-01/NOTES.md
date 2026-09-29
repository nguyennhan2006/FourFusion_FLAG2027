# MODEL-01 — encoder mới làm stream của cầu nối (S1) + ImageBind làm stream (P1 / IB-02)

- **Ngày:** 2026-09-27
- **Kế hoạch:** [docs/OPEN_DIRECTIONS.md](../../docs/OPEN_DIRECTIONS.md) S1a, S1b, P1 bước 2. Giao thức và cổng ở §1 của tài liệu đó.
- **Code:** [run.py](run.py) (harness, chạy tiếp được), [summary.py](summary.py) (hiệu ghép cặp so với CTRL + cổng), [build_set09.py](build_set09.py) (ứng viên nộp).
- **Feature:** FLAG_10 lượt 1 (vitage, farl, ibv, iba). SigLIP2 / AdaFace / w2v2ag (và S2) tự được thêm khi có lượt 2. Lớp BTC / TTA (S3 / S4) tự được thêm khi có FLAG_11.

## Thiết kế (đăng ký trước, không chỉnh trọng số)

- **Dữ liệu và cách chấm.** v4 giữ ra 30 người, 5 split ghép cặp; cầu nối train trên 40 người còn lại. Người Urdu (v1) / Hindi (v2) giữ ra theo fold, cặp đúng lấy **khác video**.
- **Thước đo.** Chính: EER mức mẫu. Phụ: mức cụm (v4, cụm ArcFace / ECAPA như SET-02) và mức người (Urdu / Hindi).
- **Chuẩn hoá.** z-score trên cả ma trận mặt × giọng của file, nên điểm không phụ thuộc cách ghép trial.
- **CTRL** = s007 của SET-02: MLP (EXP-007, n = 2) + 0.25 CCA-4 trên VGG + BTC-192.
- **Các nhánh:**
  - IB: ImageBind zero-shot, cosine đã trừ mean theo file.
  - `<S>`: cùng công thức với mặt = stream S.
  - `CTRL+<X>.w`: (1 − w) z(CTRL) + w z(X).
  - `CAT<S>`: 256 PC của VGG, cộng PCA-32 của S ở cùng độ lớn.
- **Cổng:** mức mẫu, v4 g tốt hơn ≥ 0.5 ở ≥ 4/5 split; mọi ô khác không tệ hơn quá 0.5 (trung bình).

## Kết quả (5 split; Δ so với CTRL, âm = tốt hơn; [k/5] = số split tốt hơn)

| Nhánh | v4 ng mẫu | **v4 g mẫu** | Urdu ng / g mẫu | Hindi ng mẫu | v4 ng / g cụm | Urdu ng / g người | Cổng |
|---|---:|---:|---:|---:|---:|---:|---|
| CTRL (EER tuyệt đối) | 29.17 | 37.84 | 33.02 / 43.05 | 32.61 | 23.25 / 33.23 | 26.36 / 36.28 | – |
| IB (đứng riêng) | −4.51 [4] | −3.51 [4] | −7.74 / −7.65 | −4.61 | −6.78 / −8.94 | −10.85 / −9.97 | qua* |
| CTRL+IB.25 | −4.05 [5] | −3.07 [5] | −4.69 / −4.01 | −3.55 | −3.39 / −4.16 | −5.66 / −2.39 | qua* |
| **CTRL+IB.5** | **−6.49 [5]** | **−5.55 [5]** | −8.07 / −7.21 | −5.97 | −6.15 / −6.60 | −8.09 / −7.61 | qua* |
| farl (thay VGG) | −0.30 | +0.31 [1] | −1.13 / −0.29 | −1.59 | +0.06 / −0.84 | +0.53 / +2.54 | trượt |
| CTRL+farl.25 | −1.63 [5] | −0.97 [5] | −1.73 / −0.71 | −1.62 | −1.06 / −0.31 | −2.26 / +0.13 | **qua** |
| **CTRL+farl.5** | **−2.17 [5]** | **−1.55 [5]** | −3.01 / −1.46 | −2.89 | −0.95 / −1.37 | −2.35 / +0.43 | **qua** |
| CATfarl | −2.00 [5] | −1.33 [4] | −2.53 / −1.09 | −1.91 | −0.43 / −0.97 | −5.23 / −1.55 | **qua** |
| vitage (thay VGG) | +0.19 | +1.51 [1] | −1.57 / −0.84 | −1.23 | +1.03 / +3.39 | −0.43 / +0.37 | trượt |
| CTRL+vitage.25 | −1.54 [5] | −0.77 [5] | −2.15 / −1.21 | −1.76 | −1.01 / **+0.82** | −3.10 / −1.31 | trượt (g cụm) |
| CTRL+vitage.5 | −1.98 [5] | −0.65 [3] | −3.52 / −1.91 | −2.91 | −0.73 / −0.09 | −2.55 / −2.15 | trượt (g) |
| CATvitage | −1.34 [5] | −0.17 [1] | −2.39 / −0.67 | −2.53 | +0.28 / +0.60 | −2.75 / −1.93 | trượt (g) |

\* ImageBind qua cổng nhưng **chưa được dùng**: mô hình đã căn chỉnh sẵn ảnh–âm thanh, chờ BTC trả lời câu 7. Licence CC-BY-NC.

## Đọc kết quả

1. **ImageBind là tín hiệu mạnh nhất ta từng thấy.** Nó chưa train gì mà đứng riêng đã hơn cầu nối, trộn 0.5 thì −5.5…−8 ở mọi ô, cả mức mẫu lẫn mức cụm.
   - Trên v4, một phần lợi thế có thể là hiệu ứng buổi quay: v4 không có id video, và IB-01 cho thấy cặp cùng video phồng 7–11 điểm.
   - Nhưng Urdu/Hindi đều là cặp khác video, và ở đó lợi thế còn lớn hơn (−7…−8). Vậy tín hiệu là thật.
   - → **Việc quan trọng nhất bây giờ là có câu trả lời của BTC cho câu 7.**
2. **FaRL là stream bổ sung hợp lệ (hướng S) đầu tiên qua cổng.**
   - Thay hẳn VGG thì hoà; nhưng trộn vào thì tốt hơn đều ở mọi split: −2.2 / −1.6 ở v4, và cả Urdu/Hindi.
   - Ở mức cụm lợi ích nhỏ hơn: −0.95 / −1.37 (g 3/5).
3. **ViT-age giúp ng và Urdu/Hindi nhưng không giúp g của v4.**
   - Đúng như cảnh báo trong kế hoạch: nếu stream chỉ thắng ở ng thì đó phần lớn là thông tin giới lặp lại.
   - Tuổi dạng embedding không thêm gì ở protocol gender trên English. S2 (so khớp tuổi tường minh) vẫn chờ w2v2ag.
4. **Chọn ứng viên nộp cho FaRL:** CTRL+farl.5 có g tốt nhất trong ba nhánh qua cổng, cả mức mẫu (−1.55) lẫn mức cụm (−1.37) → dựng **SET-09** bằng [build_set09.py](build_set09.py).

## Lượt 2 (28/09): SigLIP2, AdaFace, S2 tuổi, S3 lớp BTC, S4 TTA — cùng giao thức, cùng CTRL

| Nhánh | v4 ng mẫu | **v4 g mẫu** | Urdu ng / g mẫu | v4 ng / **g** cụm | Urdu g người | Cổng |
|---|---:|---:|---:|---:|---:|---|
| **CTRL+AGE.1** (so khớp tuổi, w 0.1, không học tham số) | −0.39 [5] | −0.56 [5] | −0.38 / −0.53 | −1.21 / **−1.35 [5]** | +0.23 | **qua** |
| CTRL+AGE.25 | −1.01 [5] | **−1.47 [5]** | −0.69 / −1.21 | −1.27 / **−2.41 [4]** | +0.53 | trượt sát (Urdu g người +0.53) |
| CTRL+btcfacefc6.25 (S3) | −0.52 [4] | −0.54 [4] | −1.11 / −0.52 | −1.01 / +0.01 | −0.46 | qua (sát) |
| CTRL+siglip2.25 / .5 | −1.6 / −1.9 | −0.65 / −0.51 [3] | −2.4…−3.7 / −1.3…−2.6 | ≈ 0 / ≈ 0 | −3.2…−4.0 | trượt (g) |
| CTRL+voice:btcvoiceasp (S3) | −0.9…−1.6 | −0.2…−0.3 | −1.2…−1.9 | +0.4 / +0.4…+1.6 | +0.3…+0.9 | trượt |
| CTRL+btcfacepool5 (S3) | −0.7…−0.8 | −0.3…−0.4 | −0.9…−1.7 | −0.2…−0.6 / +0.1…+1.2 | −1.4…−2.0 | trượt (g) |
| TTAface / TTAvoice / TTAboth (S4) | ≈ −0.2 | ≈ −0.2…−0.4 | ≈ −0.3 | −0.4 / −0.6 | ≈ 0 | trượt (lợi < 0.5) |
| adaface (thay VGG) | +9.06 | +5.99 | +8.3 / +2.4 | +11.8 / +9.1 | +7.2 | trượt: tệ hẳn |
| CTRL+adaface.25 / .5 | ≈ 0…+1.6 | +0.6…+1.5 | | | | trượt |

**Đọc lượt 2:**
1. **Tuổi tường minh (S2) là cách hợp lệ đầu tiên cải thiện đúng protocol gender ở mức cụm, ổn định:** AGE.1 g cụm −1.35 ở 5/5 split. Nó không có tham số nào được fit, nên gần như không thể overfit 70 người. Stream tuổi dạng embedding (ViT-age) thì trượt ở g: phép so khớp tường minh hiệu quả hơn là để cầu nối tự học.
2. **AdaFace đúng như dự đoán:** thay VGG thì tệ đi 6–12 EER; trộn vào cũng tệ hơn. Đóng hẳn.
3. **SigLIP2** mạnh ở ng và Urdu/Hindi nhưng không giúp g của v4, giống ViT-age: thông tin giới lặp lại.
4. **S3 (lớp khác của VGG / ECAPA của BTC) và S4 (TTA)** cho lợi nhỏ, dưới ngưỡng. Riêng fc6 qua cổng sát ngưỡng, nhưng g cụm ≈ 0.
5. **Luật:** tuổi giọng lấy từ audeering, train trên Common Voice **tiếng Đức** ("de-validated collection"), aGender, TIMIT, VoxCeleb2 → không có tiếng Bengali → hướng S. Licence CC-BY-NC-SA, phải khai báo.
6. **Bước tiếp theo:** kiểm lại AGE.1 / AGE.25 / FaRL / FaRL + AGE bằng VAL-70 (thành phần train trên 70 người). Đó là cổng cho cấu hình Evaluation (nhật ký 61).
