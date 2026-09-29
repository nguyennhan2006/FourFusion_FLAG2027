# ATTR-01 — có cue pitch chung giữa mặt và giọng không? (phép thử bác bỏ, EDA)

- **Ngày:** 2026-09-26
- **Kế hoạch:** [docs/PLAN_V4.md](../../docs/PLAN_V4.md) §1 B. Không phải feature chấm điểm; không đưa vào score nếu chưa qua EXT-02.
- **Giả thuyết:** sau khi cố định giới, khuôn mặt dự đoán được pitch của giọng đủ để phân biệt người đó với người
  khác cùng giới. Nếu không → đóng hướng "thuộc tính pitch" cho `g/En`.

## Thiết kế (đăng ký trước, [run_v4.py](run_v4.py))

- F0: `flag_extract.f0_stats` (YIN, cổng voicing theo RMS) mỗi câu → log-F0 của người = median qua các câu.
- Mặt: trung bình VGG fc7 (feature BTC) của người, chuẩn hoá + PCA-16 fit **bên trong** mỗi fold leave-one-speaker-out.
- Trong từng giới: ridge (α = 10) mặt → log-F0, dự đoán leave-one-speaker-out.
- Phép thử cặp: r = |pred(mặt s) − F0(giọng t)|; AUC của −r cho t = s so với t ≠ s cùng giới.
- Null: xáo F0 giữa các người cùng giới, fit lại toàn bộ, 1000 lần → p-value.
- Sanity bắt buộc: F0 phải tách giới rõ (AUC > 0.9), nếu không thì bộ trích F0 hỏng và không đọc gì khác.
- Độ nhạy (chỉ báo cáo, không dùng để chọn): α ∈ {1, 100}, k ∈ {8, 32}.

v4 chỉ có 70 người (~35 mỗi giới) → đây là **pilot**; phép thử chính chạy trên v1/v2 khi có F0 từ NB-6
(`kaggle/FLAG_09_ext_meta.ipynb`, CPU, tải lại zip và lưu F0 + metadata).

## Kết quả — pilot v4 train (2026-09-26), [out/attr01_v4.json](out/attr01_v4.json)

Sanity: F0 hợp lệ 6485/6485 câu; median 138 Hz (nam) / 205 Hz (nữ); AUC(F0 → nữ) 0.959 → bộ trích F0 dùng được.

| Giới | n | Spearman(pred, F0) | pair AUC | null AUC (mean / p95) | p |
|---|---:|---:|---:|---|---:|
| nam | 40 | 0.10 | 0.504 | 0.494 / 0.543 | 0.35 |
| nữ | 30 | 0.45 | 0.553 | 0.488 / 0.558 | 0.061 |

Độ nhạy (200 hoán vị): nam 0.49–0.54 (p ≥ 0.06); nữ 0.50–0.57, chỉ α = 1 có p = 0.045 (1/4 cấu hình, chưa hiệu chỉnh đa so sánh).

**Kết luận pilot:** không có bằng chứng về cue pitch chung. Nam: không có gì. Nữ: xu hướng yếu, không đạt ngưỡng
và không ổn định theo cấu hình → **chưa bác bỏ được, chưa chấp nhận được** với 30 người. Không đưa vào score.
Phép thử có sức mạnh hơn: cùng script trên v1/v2 (138 + người có nhãn giới). Zip v1/v2 đã có local
(`D:/Sinh viên CNhan/download/04092026`), nên F0 lấy bằng `flag_extract.extract_ext_meta` trên CPU, không cần NB-6.
