# IB-PROBE: CCA trên ImageBind đông cứng (IB-CCA16 của deep-research-report 28/09)

- **Ngày:** 2026-09-29. **Code:** [run.py](run.py), sinh từ `../IB-ADAPT/run.py`, cùng fold, cùng file đánh giá, cùng mức đo.
- **Cách làm:** ridge CCA k = 4 / 16 / 64, mỗi phía chuẩn hoá rồi PCA 128. Fit trên 40 người v4 + người v1/v2 của 4 fold còn lại (mặt và giọng khác video). So với cosine zero-shot (ZS, số hạng ImageBind của SET-13).

| Dữ liệu | Mức | ZS | CCA4 − ZS | CCA16 − ZS | CCA64 − ZS | ZS+CCA16 − ZS |
|---|---|---:|---:|---:|---:|---:|
| v4 ng / g | người | 18.73 / 27.04 | +5.5 / +9.3 | +4.7 / +5.4 | +13.2 / +9.2 | −0.9 / −0.5 |
| Urdu ng / g | người | 16.83 / 33.17 | +3.0 / +0.7 | +6.6 / +1.9 | +11.9 / +7.4 | +1.6 / +0.5 |
| Hindi ng | người | 20.86 | +2.2 | −0.7 | +5.4 | −2.2 |
| English v1 / v2 ng | người | 14.47 / 22.20 | +6.5 / +0.9 | +5.7 / −0.4 | +9.7 / +8.6 | +1.9 / −2.3 |

(+ = tệ hơn ZS.)

## Kết luận

- **CCA đứng riêng gần như luôn tệ hơn zero-shot**, tệ nhất là CCA64 (+5…+13), dấu hiệu overfit. ImageBind đã căn chỉnh sẵn ảnh–âm thanh tốt hơn mọi phép chiếu ta học được từ khoảng 160 người.
- **Trộn ZS với CCA16** chênh −2.3…+1.9 và không qua cổng ở mức người (v4 ng thắng 2/5, Urdu ng 1/5). **Đóng**, cùng kết luận với IB-ADAPT: với dữ liệu hiện có, nên dùng ImageBind nguyên trạng.
