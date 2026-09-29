# SWA-01: trung bình trọng số các epoch cuối có giảm may rủi theo seed không? (SEED-SWA của deep-research-report 28/09)

- **Ngày:** 2026-09-29. **Code:** [run.py](run.py), dùng lại phần dựng file và chấm điểm của [../EXTSCALE-01/run.py](../EXTSCALE-01/run.py). **Kết quả:** [results.csv](results.csv).
- **Cách làm:**
  - 5 split v4 (40 người train, 30 giữ ra) và fold v1/v2 tương ứng (người chưa thấy).
  - Thành phần s007 công thức EXP-007, 10 seed mỗi split.
  - `raw` (bình thường) so với `avg` (`avg_from=30`: trung bình trọng số sau các epoch 31–40, BatchNorm tính lại trên dòng train).
  - Chấm từng mạng riêng (đo may rủi theo seed) và theo tổ hợp 5 / 10 mạng, ở cả `member` và `prod` (s007′ như SET-13).

## Kết quả (bản production, trung bình 5 split)

| Mức | Dữ liệu | EER raw / avg | Độ lệch theo seed raw / avg | Tổ hợp 10 mạng raw / avg |
|---|---|---|---|---|
| cụm | v4 g / ng | 27.14 / 27.15 · 17.53 / 17.54 | 1.67 / 1.70 · 1.09 / 0.97 | 26.90 / 27.09 · 17.05 / 17.05 |
| cụm | English g / ng | 32.88 / 32.87 · 20.78 / 20.79 | 1.29 / 1.25 · 1.08 / 1.11 | 32.67 / 32.67 · 20.76 / 20.78 |
| người | Hindi ng | 21.12 / 21.21 | 1.50 / 1.49 | 21.20 / 21.06 |
| người | Urdu g / ng | 31.64 / 31.63 · 18.75 / 18.76 | 1.62 / 1.62 · 1.02 / 1.04 | 31.70 / 31.70 · 19.09 / 19.09 |

Trên toàn bộ các ô của bản production: EER đổi trung bình **+0.03**, độ lệch theo seed đổi **+1%**, tổ hợp 10 mạng đổi **+0.02**.

## Kết luận

- **Không có tác dụng → đóng.** Tiêu chí giữ lại (độ lệch giảm ≥ 20% mà EER không tệ hơn 0.2, hoặc EER tốt hơn ≥ 0.5) không đạt ở ô nào. Lý do có thể: lịch học cosine đã đưa lr về gần 0 ở 10 epoch cuối, nên trọng số của chúng gần như trùng nhau.
- **May rủi theo seed là có thật:** một mạng đơn lẻ lệch 0.7–1.7 EER tuỳ seed. **Cách giảm đúng là tổ hợp nhiều mạng**: tổ hợp 10 mạng tốt hơn trung bình mạng đơn khoảng 0.2–0.5 ở mức cụm v4. Điều này củng cố quyết định dùng 15 mạng mỗi thành phần cho Evaluation (nhật ký 69).
