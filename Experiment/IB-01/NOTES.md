# IB-01 — ImageBind zero-shot mặt ↔ giọng, cặp đúng khác video (nhãn thật, không train)

- **Ngày:** 2026-09-27
- **Kế hoạch:** [docs/OPEN_DIRECTIONS.md](../../docs/OPEN_DIRECTIONS.md) P1, bước 1. [docs/PLAN_MODELS.md](../../docs/PLAN_MODELS.md) §4 X3.
- **Feature:** `face_ibv` / `voice_iba` từ [FLAG_10_models](../../kaggle/FLAG_10_models.ipynb) (ImageBind-huge, embedding 1024 chiều trong không gian chung).
- **Luật:** ImageBind đã được căn chỉnh sẵn ảnh–âm thanh, nên chỉ dùng để probe cho tới khi BTC trả lời câu 7. Licence CC-BY-NC 4.0.

## Vì sao cần thí nghiệm này

Probe trong notebook cho 23.73 / 34.02 (ng / g, trừ mean) trên v4 train. Nhưng v4 không có id video, nên một cặp đúng có thể là mặt và giọng **của cùng một video**. ImageBind mã hoá cả bối cảnh, nên có thể nhận ra buổi quay thay vì nhận ra người. v1/v2 có id video, nên ở đây ta ép cặp đúng phải **khác video** (giống EXT-02/03), và dùng cặp cùng video làm đối chứng.

## Thiết kế (cố định trước khi chạy, [run.py](run.py))

- Mỗi nguồn × ngôn ngữ giọng: tối đa 15 giọng mỗi người.
- Mỗi giọng có 1 cặp đúng (cùng người, khác video) và 1 cặp sai (người khác; ở `g` thì cùng giới, chỉ tính được cho v1 vì chỉ v1 có nhãn giới).
- 5 seed. Điểm = cosine thô, hoặc cosine sau khi trừ mean mặt / mean giọng của cả nguồn.
- Không có tham số nào được fit.

## Kết quả ([results.csv](results.csv)), EER %, trung bình 5 seed (sd ≤ 0.7)

| Nguồn / giọng | protocol | khác video, trừ mean | cùng video, trừ mean | khác video, thô |
|---|---|---:|---:|---:|
| v1 English | ng | **25.64** | 18.50 | 29.48 |
| v1 English | g | **38.10** | 27.31 | 39.57 |
| v1 Urdu | ng | **26.98** | 18.78 | 31.10 |
| v1 Urdu | g | **39.48** | 28.10 | 40.89 |
| v2 English | ng | **28.63** | 21.29 | 33.38 |
| v2 Hindi | ng | **30.07** | 20.89 | 33.75 |

**Tham chiếu:** cầu nối đã train của ta, ARCH-01 MLP + CCA-4 (bản production, train trên 40 người v4), mức mẫu, cũng với cặp đúng khác video:
Urdu 33.02 / 43.05 (ng / g), Hindi 32.61. Hai phép đo không ghép cặp, vì cách lấy trial khác nhau, nên chỉ đọc chiều và độ lớn, không đọc từng con số.

## Đọc kết quả

1. **Hiệu ứng buổi quay là có thật và lớn:** cặp cùng video tốt hơn 7–11 điểm. Con số 23.73 / 34.02 trên v4 bị phồng. Con số trung thực là cột "khác video".
2. **ImageBind có tín hiệu thật, và không cần train:**
   - ng khoảng 26–30. Riêng ở Urdu/Hindi, con số này **tốt hơn cầu nối đã train của ta ở mức mẫu** khoảng 3–6 điểm (ng).
   - g khoảng 38–39.5: tín hiệu ngoài giới tính yếu, nhưng vẫn tốt hơn cầu nối (Urdu g 39.48 so với 43.05).
3. **Gần như không nhạy ngôn ngữ:** Urdu/Hindi chỉ tệ hơn English cùng nguồn 1.3–1.4 điểm. Điều này khớp với việc hạng 2 FAME26 fine-tune trên tiếng Ả Rập mà vẫn tốt trên các ngôn ngữ khác.
4. **Trừ mean giúp 1.4–4.5 điểm**, cùng chiều với file centering của ta.

**Kết luận:** P1 bước 1 qua (xa mức "≥ khoảng 45% thì dừng").
- **Bước tiếp theo (IB-02):** trộn ImageBind như một stream với cầu nối, `z(M_cầu nối) + w · z(M_ImageBind)`, w ∈ {0.25, 0.5} đăng ký trước. Dùng harness ARCH-01 (v4 giữ ra 30 + Urdu/Hindi, cặp đúng khác video). Đo mức mẫu là chính, mức cụm là phụ.
- Chỉ được nộp khi BTC xác nhận câu 7.
- **Rủi ro cần nhớ:** nếu file dev / Evaluation có nhiều cặp đúng cùng video, ImageBind sẽ trông tốt hơn thực chất. Vì vậy validation phải dùng cặp khác video.
