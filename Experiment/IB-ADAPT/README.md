# IB-ADAPT: dữ liệu cặp hiện có có làm ImageBind tốt hơn không? (E5a trong [PLAN_EVAL](../../docs/PLAN_EVAL.md))

- **Ngày:** 2026-09-28. **Code:** [run.py](run.py), kết quả [results.csv](results.csv).
- **Câu hỏi:** trước khi tốn GPU fine-tune ImageBind bằng LoRA (hướng của đội hạng 2 FAME 2026), kiểm tra rẻ trên CPU xem dữ liệu cặp mặt–giọng của ta có đủ để **chỉnh** không gian chung của ImageBind hay không.

## Cách làm

- Embedding ImageBind **đông cứng** (FLAG_10). Mỗi phía một ma trận tuyến tính 1024 × 1024, khởi tạo bằng ma trận đơn vị và kéo về đó bằng λ‖W − I‖² (λ = 0.01).
- Loss: InfoNCE đối xứng, nhiều cặp dương (τ = 0.05); 1500 bước, batch 32 người × 2 cặp.
- Mọi siêu tham số **đăng ký trước**, không chỉnh sau khi xem kết quả.

| Nhánh | Train trên |
|---|---|
| ZS | không train: chính số hạng ImageBind của SET-13 |
| AD4 | 40 người v4 của split |
| ADX | 40 người v4 + người v1/v2 của 4 fold còn lại (khoảng 122 người; mặt và giọng lấy khác video) |
| ZS+ADX | 0.5 z(ZS) + 0.5 z(ADX) |

**Đánh giá:**
- 5 split, cosine trừ mean của file, z-score trên ma trận, giống SET-13.
- Người giữ ra: 30 người v4, và khoảng 30 người v1/v2 thuộc fold giữ ra. Với v1/v2, cặp dương luôn lấy mặt **khác video** với giọng, nên hiệu ứng cùng buổi quay (IB-01) không giúp được.
- Hai mức: mẫu (mỗi trial riêng) và người (khối theo danh tính thật, b = 0.99).

## Kết quả (trung bình 5 split; Δ so với ZS, âm = tốt hơn)

| Dữ liệu | Mức | ZS | AD4 − ZS | ADX − ZS | ZS+ADX − ZS | ADX tốt hơn ≥ 1 |
|---|---|---:|---:|---:|---:|---|
| v4 ng | người | 18.73 | +0.58 | −0.02 | −1.57 | 2/5 |
| v4 g | người | 27.04 | +3.47 | **+3.83** | +1.43 | 1/5 |
| Urdu ng | người | 16.83 | +0.58 | −1.45 | −1.25 | 3/5 |
| Urdu g | người | 33.17 | +1.33 | +1.83 | +0.94 | 1/5 |
| Hindi ng | người | 20.86 | −1.83 | −2.25 | −1.30 | 4/5 |
| English (v1) ng | người | 14.47 | +5.36 | **+4.86** | +1.39 | 0/5 |
| English (v2) ng | người | 22.20 | +2.51 | −1.21 | −1.67 | 3/5 |
| v4 ng | mẫu | 24.76 | +1.55 | −0.18 | −1.22 | 0/5 |
| Urdu ng | mẫu | 28.15 | +1.54 | −3.35 | −3.19 | 3/5 |

**Cổng cho LoRA:** ADX phải tốt hơn ZS ≥ 1 EER ở mức người, trên **cả** Urdu lẫn v4, ở ≥ 4/5 split. **Không qua** (Urdu ng 3/5, Urdu g 1/5, v4 ng 2/5, v4 g 1/5).

## Kết luận

1. **Chỉ với 40 người v4, chỉnh ImageBind làm nó tệ đi** ở gần như mọi ô: AD4 +0.6…+5.4. Đây đúng là chế độ dữ liệu của bản nộp (70 người).
2. **Thêm khoảng 120 người v1/v2, kết quả lẫn lộn.** Ô tốt lên là các ngôn ngữ đã có trong dữ liệu train (Hindi, Urdu ng). Ô tệ đi là protocol g và English v1: adapter học giới tính và đặc điểm của tập train thay vì mối liên hệ mặt–giọng.
3. **LoRA (E5) hoãn lại.** LoRA có nhiều tham số hơn một ma trận tuyến tính, nên với cùng dữ liệu còn dễ overfit hơn. Đội hạng 2 FAME 2026 có nhiều người hơn hẳn.
   - Chỉ nên làm lại nếu có dữ liệu lớn hơn nhiều (hướng X1 VoxCeleb, hàng nghìn người). Khi đó chạy lại chính script này với dữ liệu mới làm bước sàng lọc trước.
4. **ZS+ADX** tốt lên chút ít ở vài ô ng (−1.2…−1.7) nhưng tệ đi ở v4 g (+1.4) và English v1 (+1.4). Không đủ để thay số hạng ImageBind của SET-13.
