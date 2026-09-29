# EXT-02 — validation song ngữ có nhãn thật + PRETRAIN → FINE-TUNE

- **Ngày:** 2026-09-26
- **Parent:** EXP-007 (recipe), EXT-01 (dữ liệu ngoài, nhánh MIX)
- **Kế hoạch:** [docs/PLAN_V4.md](../../docs/PLAN_V4.md) §1 A2 + A3
- **Giả thuyết:** dữ liệu cặp ngoài cùng quần thể làm cầu nối face↔voice **chuyển giao được sang ngôn ngữ chưa nghe**
  mà không hy sinh English. Đo trên người chưa thấy, **có nhãn thật**, không dựa vào dev FLAG.

## Thiết kế (cố định trước khi xem kết quả)

| | |
|---|---|
| Nguồn | mỗi nguồn ngoài riêng (`v3_train` có sẵn local; `v1_train`, `v2_complete` khi tải `feats_ext/`) |
| Chia | 5 fold theo identity của nguồn ngoài; identity của fold không vào bất kỳ tập train nào |
| File đánh giá (mỗi fold) | **heard** = (mặt, giọng English) · **unheard** = (mặt, giọng Urdu/Hindi/German); 1500 + 1500 trial, 50% target; mặt của trial dương lấy từ **video khác** với giọng (không có shortcut cùng phiên) |
| Chấm | đúng như bản nộp dev: centring theo file, n_models (3) × MLP + CCA k=4 trọng số 0.25 |
| Recipe | EXP-007 (hid512 / emb128 / drop0.3 / 40 ep); fine-tune: 20 ep, lr 3e-4 (chọn trước, không tune) |
| Cap | 150 hàng / người ngoài (như EXT-01) |

| Arm | Train | Ý nghĩa |
|---|---|---|
| `v4` | v4 train | đối chứng, giống nhau ở mọi fold |
| `mix_en` | v4 + English của người ngoài (fold train) | thêm người, cùng ngôn ngữ |
| `mix_all` | v4 + mọi hàng của người ngoài | = nhánh MIX của FLAG_08. ⚠️ ngôn ngữ "unheard" của người held-out đã **được thấy** trong train → số unheard lạc quan so với Bangla |
| `pt_en>ft` | pretrain English ngoài → fine-tune v4 | PT→FT, không lộ ngôn ngữ thứ hai |
| `pt_all>ft` | pretrain mọi hàng ngoài → fine-tune v4 | PT→FT có ngôn ngữ thứ hai |

Mọi arm cùng seed, cùng trial → hiệu với `v4` là **paired theo fold**.

**Luật đọc (PLAN_V4 A2):** Δunheard ≤ −1.0 và Δheard ≤ +0.5 → `transfer`; Δunheard ≤ −1.0 nhưng heard tệ hơn > 0.5 →
`trade-off`; cả hai tệ hơn → `drop`; còn lại → `no clear effect`. Kèm số fold cải thiện và sd của hiệu qua fold.

Giới hạn: chỉ cell kiểu `no_gender` cho tới khi có nhãn giới của v1/v2 (NB-6 `FLAG_09_ext_meta`); mỗi fold chỉ
~10–17 người held-out → sd giữa fold lớn, đọc chiều + số fold, không đọc con số lẻ.

## Chạy

```bash
cd Experiment/EXT-02_bilingual
python run.py v3_train --smoke            # plumbing: 1 model, 2 fold, 300 trial
python run.py v3_train                    # ~1–1.5 giờ CPU
FLAG_FEATS_EXTRA=<feats_ext đã tải> python run.py v1_train v2_complete
```

Code: [../../kaggle/flag_bilingual.py](../../kaggle/flag_bilingual.py) (thư viện), [run.py](run.py). Kết quả: `out/ext02_<src>.csv`, `out/ext02_summary.csv`.

## Kết quả

### v3 (châu Âu, En + German; 49–50 người, không có nhãn giới → chỉ cell `no_gender`) — 2026-09-26

5 fold × 3 model, 1500 + 1500 trial mỗi file. Δ = hiệu paired theo fold so với `v4` (âm = tốt hơn). [out/ext02_summary.csv](out/ext02_summary.csv)

| Arm | heard (En) | Δ heard | unheard (German) | Δ unheard | fold tốt hơn (h / u) | verdict |
|---|---:|---:|---:|---:|---|---|
| `v4` | 42.99 | – | 42.11 | – | – | control |
| `mix_en` | 36.33 | **−6.65** (sd 5.1) | 35.45 | **−6.65** (sd 4.5) | 5/5 · 5/5 | transfer |
| `mix_all` | 33.79 | −9.20 (5.2) | 30.35 | −11.76 (5.8) | 5/5 · 5/5 | transfer (unheard lạc quan) |
| `pt_en>ft` | 38.48 | −4.51 (4.6) | 39.11 | −3.00 (5.1) | 4/5 · 3/5 | transfer |
| `pt_all>ft` | 39.15 | −3.84 (5.7) | 37.28 | −4.83 (4.0) | 3/5 · 4/5 | transfer |

Đọc:
1. **Cầu nối học từ v4 gần như không dùng được cho người châu Âu** (~42–43 EER, gần ngẫu nhiên hơn nhiều so với
   ~24–35 trên dev v4) → khớp với EXT-01: ánh xạ mặt–giọng phụ thuộc quần thể.
2. **Thêm người cùng quần thể giúp mạnh và chuyển giao sang ngôn ngữ chưa nghe:** `mix_en` chưa từng thấy câu
   German nào mà German vẫn −6.65, 5/5 fold. Đây là bằng chứng có nhãn thật đầu tiên rằng *số người cùng quần thể*
   là đòn bẩy của cầu nối — và cũng giải thích vì sao v3 hại v4: nó dạy cầu nối cho quần thể khác.
3. **MIX > PT→FT** ở cả hai ngôn ngữ. ⚠️ So sánh này thiên về MIX: người held-out cùng quần thể với dữ liệu ngoài,
   còn PT→FT kết thúc bằng fine-tune trên v4 (khác quần thể) nên "quên" bớt. Với đích là v4 thì thiên lệch đảo chiều
   → quyết định MIX vs PT→FT cho bản nộp vẫn phải qua SEL-01b trên dev v4.
4. Phần hơn thêm của `mix_all` ở German (−11.76 so với −6.65) một phần do German đã được thấy trong train — không
   dùng để suy ra lợi ích cho Bangla (Bangla không có trong bất kỳ nguồn nào).

Bước tiếp: chạy y hệt trên `v1_complete` / `v2_complete` (Nam Á, có nhãn giới → thêm cell `gender`) khi feature
trích xong. Đó mới là phép thử quyết định cho v4: nếu `mix_en` cũng thắng rõ ở đó thì nộp MIX v1/v2 được chọn bằng SEL-01b.
