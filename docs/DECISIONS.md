# Báo cáo các quyết định của FourFusion — FLAG 2027 (21/09 – 29/09/2026)

> **Đồng bộ với nhật ký tới mục (75), ngày 29/09.** Thứ tự nguồn khi suy luận: [../Experiment/EXPERIMENT_LOG.md](../Experiment/EXPERIMENT_LOG.md) → tài liệu này → mọi kế hoạch / báo cáo nghiên cứu. Tài liệu cũ hơn (Google Docs 21–22/09, bản code trên GitHub) chỉ là bối cảnh lịch sử.

Số trong ngoặc, ví dụ (43), là số mục nhật ký chứa bằng chứng gốc.
CB = điểm CodaBench trên dev, EER %, trung bình 4 ô: ng/En · ng/Bn · g/En · g/Bn. Thấp hơn là tốt hơn.

## 0. Tóm tắt (29/09)

- **Cấu hình Evaluation: SET-13 qua pipeline [../BEST/v6_eval/](../BEST/v6_eval/README.md)** (`--mode auto`, 15 mạng mỗi thành phần). SET-13 là SET-02 thêm ImageBind + tuổi vào thành phần s007. CB **21.01** (11.31 · 20.91 · 23.22 · 28.61).
  - Chọn bằng validation độc lập, không bằng CB: v4 giữ ra, mức cụm −6.8 / −6.5 (5/5); VAL-70, mức người −7.4 (10/10 file ở mọi ô) (65, 66).
  - Toàn bộ quy trình đã chạy thử từ ảnh / giọng gốc tới zip. Đặc trưng từ Kaggle (FLAG_12) cho rank-corr 1.0000 với bản đã nộp (73).
- **BTC cho phép mọi thứ** (63): ImageBind, dữ liệu cặp ngoài, gom cụm lúc test. Người và giọng ở Evaluation không trùng train / dev.
- **Bốn bước nhảy lớn:**
  - bỏ FOP, chuyển sang CCA rồi InfoNCE: 42.17 → 31.34;
  - trộn theo hạng: → 28.69;
  - gom cụm lúc test (SET-02): → 21.68;
  - ImageBind + tuổi (SET-13): → 21.01 trên dev. Trên validation độc lập lợi lớn hơn nhiều (−5…−9), vì dev bị winner's curse (61).
- **Phần phụ thuộc test đáng giá duy nhất là gom cụm, khoảng 5.7 điểm.** Chấm từng cặp (INDEP-01) được 26.70 (68).
  - STRESS-01 (280 file mô phỏng, nhãn thật): gom cụm tốt hơn ở mọi dạng file đã thử, kể cả ngôn ngữ khác, 150 người mỗi file, 1–2 mẫu mỗi người (71, 73).
- **Đóng sau vòng 28–29/09:**
  - SetProto, kể cả khi thêm người (74);
  - SWA (75);
  - chỉnh ImageBind bằng adapter tuyến tính (70) hoặc bằng CCA (74);
  - LoRA cho ImageBind, với dữ liệu hiện có.
- **Còn mở:**
  - dữ liệu v1/v2 chỉ còn giúp bản production ở **English g** (74);
  - thêm người vẫn giúp cầu nối nhưng lợi giảm dần ở bản production, nên VoxCeleb (X1) chưa đạt cổng (74).

## 1. Các bản đã nộp và quyết định đằng sau mỗi bản

| Bản | CB | ng/En · ng/Bn · g/En · g/Bn | Quyết định nó thể hiện |
|---|---:|---|---|
| EXP-000c | 42.17 | 36.31 · 38.98 · 43.38 · 50.00 | Tái hiện baseline FOP; kiểm chứng chiều điểm (nộp −d² ra 57.83) |
| 000d / 000e | 35.79 / 34.46 | | CCA tuyến tính, không train sâu: 70 người thì mô hình ít tham số thắng |
| 000f | 33.35 | 28.97 · 29.02 · 34.42 · 41.01 | Ghép theo từng cell: mỗi file được chấm độc lập |
| 003b / 003c | 31.96 / 31.34 | 25.60 · 27.88 · 34.01 · 37.87 | InfoNCE thay CE + OPL; bỏ AS-norm (validation nội bộ đoán sai dấu) |
| 004 | 41.36 | | CORAL: thất bại, dẫn tới đóng hướng căn phân phối ở phía test |
| 007 | 30.90 | 25.60 · 29.02 · 30.96 · 38.01 | Trọng số CCA tường minh w = 0.25 |
| 010B | 31.71 | 23.81 · 30.87 · 32.38 · 39.78 | ECAPA 6144 đứng riêng: chỉ tốt ở ng/En → giữ làm thành phần |
| 011 | 30.13 | 23.81 · 27.88 · 30.96 · 37.87 | Ghép cell từ 3 hệ |
| FUSE-01 / 02 / 03 | 29.62 / 28.93 / 28.69 | 23.61 · 26.88 · 29.12 · 35.15 | Trộn theo hạng từng cell, chọn bằng pseudo-label SEL-01b |
| SET-02 | 21.68 | 12.50 · 21.34 · 28.31 · 24.59 | Gom cụm lúc test + trung bình khối |
| SET-08 | 22.30 | 12.30 · 21.34 · 28.72 · 26.84 | Trộn thêm CCA-16 vào s007 |
| SET-07 | 22.20 | 15.48 · 17.21 · 27.49 · 28.61 | s007 train thêm MAV-Celeb v1 / v2 (chưa có ImageBind) |
| HYB-01 | 20.40 | 12.30 · 17.21 · 27.49 · 24.59 | Ghép cell theo CB, chỉ cho bảng progress |
| SET-09 | 22.49 | 14.19 · 20.34 · 28.72 · 26.70 | Trộn stream FaRL vào s007 |
| SET-12 | 21.45 | 11.64 · 21.05 · 23.01 · 30.11 | s007 → ½ z(s007) + ½ z(ImageBind) (64) |
| **SET-13** | **21.01** | **11.31** · 20.91 · 23.22 · 28.61 | **SET-12 + 0.1 z(tuổi); cấu hình Evaluation** (67) |
| INDEP-01 | 26.70 | 19.05 · 26.32 · 27.09 · 34.33 | SET-13 chấm từng cặp, không thống kê file, không cụm: năng lực thật của mô hình (68) |

## 2. Quyết định theo chủ đề

### A. Đo lường và cách chọn hệ

| # | Quyết định | Vì sao / bằng chứng | Trạng thái |
|---|---|---|---|
| A1 | Nộp điểm theo chiều **lower = same** (+d²) | Cùng một checkpoint: −d² ra 57.83, +d² ra 42.17 (000c) | Còn hiệu lực. Evaluation: nếu lượt đầu ra EER > 50 thì đảo dấu ngay (scorer có thể khác) |
| A2 | Báo cáo và thay thế **theo từng cell** | Mỗi file được chấm độc lập (000f, (3)) | Còn hiệu lực. Chọn cell theo CB chỉ dùng cho progress |
| A3 | Đo cải tiến bằng **so sánh ghép cặp, ≥ 5 split / fold, đăng ký trước, có vùng hoà** | Xếp hạng từ 1 seed không dùng được (3); nhiễu OOF khoảng 1.5 EER (41); mạng đơn lẻ lệch 0.7–1.7 theo seed (75) | Còn hiệu lực |
| A4 | Chọn hệ bằng pseudo-label SEL-01b | Dựng từ danh sách trial (54) | **Đã thay.** Chỉ để cảnh báo (57) |
| A5 | **Cổng nhãn thật** (v4 giữ ra; người v1/v2 giữ ra với cặp dương khác video) | Đo embedding không nhờ cấu trúc tập test (57) | Còn hiệu lực |
| A6 | **Quyết định cho Evaluation dựa vào validation độc lập**; CB chỉ để phát hiện hỏng nặng | Chỉ đổi seed đã làm dev dịch 0.65–1.32 (61) | Còn hiệu lực |
| A7 | **Thước đo chính là mức người / mức cụm; mức mẫu là phụ** | Bản nộp gom cụm nên cái được chấm thực chất là mức người. Tuổi: mức mẫu −0.56 nhưng mức cụm −1.35 (62). Dữ liệu ngoài: mức mẫu thành phần trần +1.4…+5.5 nhưng mức cụm production khoảng 0 ở vài ô (74) | **Mới (29/09)**. Báo cáo cả ba mức; quyết định theo mức người / cụm của bản production |

### B. Cầu nối mặt ↔ giọng

| # | Quyết định | Vì sao / bằng chứng | Trạng thái |
|---|---|---|---|
| B1 | Dùng **CCA tuyến tính** trước khi train sâu | 70 người; 000d 35.79 đã vượt baseline của BTC (36.92) | CCA-4 vẫn là 25% của mọi thành phần |
| B2 | **InfoNCE thay CE + OPL** của FOP | EXP-003: InfoNCE thắng CCA ở cả hai protocol; FOP thua CCA | Còn hiệu lực |
| B3 | **Bỏ AS-norm** | Nội bộ nói giúp 2.4, CB nói ngược (003b → 003c) | Đóng |
| B4 | **Không bỏ dòng trùng giọng** (dedup = False) | 32.26 so với 33.02 (3, 4). Đội bạn (ALLPAIRS) xác nhận độc lập | Còn hiệu lực |
| B5 | **Trọng số CCA tường minh w = 0.25** | "Ensemble lớn hại" thực ra là CCA bị pha loãng (5) | Còn hiệu lực |
| B6 | Không dùng công thức hạng 1 FAME26 (linear + dropout 0.9 + AAM) | AAM ≈ InfoNCE; dropout 0.9 phá (14) | Đóng |
| B7 | **Không tăng dung lượng cầu nối** | ARCH-01: thêm dung lượng thì tệ đi (52). SCALE-ID: cầu nối cố định vẫn học tiếp tới 161 người, chưa thiếu dung lượng (74) | Đóng |
| B8 | **SetProto: đóng** | EXT-VAL70, cùng 1200 bước: SetProto ≈ 0, tương tác SetProto × thêm người ≈ 0; SCALE-ID ±0.3 ở mọi số người (74). Lợi của B3P ở EXT-03 đến từ thêm người và thêm bước tối ưu | **Đóng (29/09)** |
| B9 | **SWA / trung bình trọng số: đóng** | Epoch 31–40, 5 split × 10 seed: +0.03 EER, độ lệch theo seed không giảm (75) | **Đóng (29/09)** |
| B10 | **15 mạng mỗi thành phần cho Evaluation** | Cùng kỳ vọng với 5 mạng (69); tổ hợp 10 mạng tốt hơn mạng đơn 0.2–0.5 ở mức cụm v4 (75) | Còn hiệu lực (v6 mặc định) |

### C. Encoder và feature

| # | Quyết định | Vì sao / bằng chứng | Trạng thái |
|---|---|---|---|
| C1 | **Được trích lại feature từ ảnh / giọng gốc** | Luật + BTC cho phép (8, 63) | Còn hiệu lực |
| C2 | **Tái tạo đúng encoder của BTC** (VGG-Face fc7, `yangwang825/ecapa-tdnn-vox2`) | cos 1.0000 với CSV (34) | Nền cho mọi dữ liệu ngoài cùng không gian feature |
| C3 | **ECAPA 6144 chỉ làm thành phần** | Tốt nhất ở ng/En nhưng tệ hơn ở 3 cell còn lại (12, 13) | Còn trong SET-13 (English) |
| C4 | **Không dùng encoder nhận dạng mạnh cho cầu nối** (ArcFace, ReDimNet2, AdaFace) | Nhận dạng giỏi ≠ ghép chéo giỏi (20, 32, 62) | Đóng |
| C5 | **Chọn encoder theo vai trò**: ArcFace / ECAPA-192 để gom cụm, VGG / BTC cho cầu nối | EDA-002 (47); STRESS-01 B: ARI mặt 0.99, giọng 0.81–0.93 (73) | Còn hiệu lực |
| C6 | **Stream thuộc tính mềm** | Tuổi qua cổng và nằm trong SET-13 (62, 66); FaRL qua cổng ở MODEL-01 nhưng SET-14 (thêm FaRL) không hơn SET-13 ở VAL-70 (65); ViT-age làm stream trượt ở g (59) | Tuổi: dùng. FaRL: dự phòng không ImageBind (SET-11) |
| C7 | **ImageBind zero-shot được dùng** | BTC cho phép (63). Validation: −5…−9, 10/10 file (65). Cặp dương khác video vẫn thắng lớn (IB-01, VAL-70, IB-ADAPT, STRESS-01 B); cặp dương của dev không cùng buổi quay (SES-02) | **Trong cấu hình Evaluation (SET-13)** |
| C8 | **Không chỉnh ImageBind với dữ liệu hiện có** | Adapter tuyến tính: chỉ v4 thì tệ hơn (+0.6…+5.4), thêm v1/v2 thì lẫn lộn (70). CCA 4/16/64: tệ hơn zero-shot +2…+13, trộn thì ±2 (74) | **Đóng.** LoRA hoãn; chỉ xét lại khi có dữ liệu cỡ hàng nghìn người, sau khi chạy lại IB-ADAPT làm bước sàng lọc |

### D. Trộn hệ (fusion)

| # | Quyết định | Vì sao / bằng chứng | Trạng thái |
|---|---|---|---|
| D1 | **Trộn theo hạng từng cell**, danh sách ứng viên cố định trước | FUSE-01 / 02 (24–28) | Còn trong SET-13: English = s007′ + s010B, Bangla = r2mix + s007′ |
| D2 | **Tối đa 2 tham số fit trên dev** | Nhiễu OOF cỡ gain (29, 41) | Ngân sách đã dùng hết. Trọng số ImageBind 0.5 và tuổi 0.1 đăng ký trước, không fit trên dev |

### E. Xử lý lúc test (transductive)

| # | Quyết định | Vì sao / bằng chứng | Trạng thái |
|---|---|---|---|
| E1 | **Chỉ trừ mean theo file; không căn bậc 2 trở lên** | CORAL 41.36 (2, 004) | Còn hiệu lực |
| E2 | **Gom cụm mặt / giọng trong từng file, trung bình khối b = 0.99** | Oracle −7…−12 (43); CB −7.01 (44); INDEP-01 → SET-13: −5.7 (68) | Còn hiệu lực |
| E3 | **Không tinh chỉnh gom cụm thêm** | SET-03 … 06 (45–50); ngưỡng giọng tự thích nghi không hơn (71, 73); L2 cách trần danh tính thật 0.1–2 (73) | Đóng |
| E4 | **Không bao giờ dùng danh sách trial để chấm hay để chọn hệ** | LEAK-01 (54) | Còn hiệu lực. Không đi theo hai đội đầu |
| E5 | **Dự phòng khi cụm hỏng là `noclus`, không phải INDEP-01** | STRESS-01: L1 (thống kê file, không cụm) ≥ L0 (INDEP-01) ở gần hết kịch bản, nhất là file nhỏ (71) | Còn hiệu lực. INDEP-01 chỉ dùng nếu có quy định cấm dùng các mẫu test khác |
| E6 | **`auto` = gom cụm cho mọi file; chỉ cảnh báo khi cụm giọng / cụm mặt < 0.8** | Hai quy tắc tự chuyển tạm thời làm mất 2–6 EER (71); ở phần B không lần nào giọng bị gộp nhầm, tỉ lệ luôn 1.06–1.44 (73) | Còn hiệu lực (v6). Đổi từng file bằng `--file-mode <file>=noclus` |

### F. Dữ liệu ngoài

| # | Quyết định | Vì sao / bằng chứng | Trạng thái |
|---|---|---|---|
| F1 | **Được dùng nếu khai báo; không train trên tiếng Bengali** | BTC cho phép (63) | Còn hiệu lực. Kiểm trùng người với dữ liệu Evaluation khi có |
| F2 | Không dùng MAV-Celeb v3 (châu Âu) | Tệ hơn ở cả 4 cell (36) | Đóng |
| F3 | **v1 / v2 (Nam Á) trong cầu nối** | Cầu nối trần +1.4…+5.5. Nhưng ở bản production (ImageBind + tuổi) chỉ còn English g (cụm +2.33 [5/5]; SCALE-ID v4 g −4.55 [5/5]) và Hindi (+1.2 [3/5]); Urdu và English ng ≈ 0 (74) | **Ứng viên hẹp:** s007 train thêm v1/v2 chỉ cho các cell English. Cần một lượt kiểm chứng riêng; rủi ro ng/En (SET-07 +2.98 trên dev) |
| F4 | **VoxCeleb (X1)** | Hạng 1 và hạng 2 FAME26 dùng khoảng 6 000 người ngoài. SCALE-ID: bước 100 → 161 người ở bản production chỉ còn đạt ở v4 ng (74) | **Chưa đạt cổng.** Lợi dự kiến cho bản nộp ≤ 1–2 EER so với công trích lớn |

### G. Quy trình, hạ tầng, tuân thủ

| # | Quyết định | Trạng thái |
|---|---|---|
| G1 | Việc nặng chạy bằng notebook Kaggle, **smoke-test trên máy trước khi giao** | Bổ sung 28/09: notebook mới có **code đầy đủ trong cell**, không import file .py. Chạy thử bằng `run_notebook_local.py --jupyter` với **đúng phiên bản transformers của Kaggle (5.0.0)**; so với mảng thật (73) |
| G2 | Mỗi thí nghiệm một thư mục, ghi nhật ký ngay khi có số, một biến mỗi lần | Còn hiệu lực |
| G3 | **Chỉ nộp từ tài khoản của đội** (nguyennhan2006). Hai người cộng tác (anhdung1509, minhnhien2811) là đội độc lập: không dùng chung feature, bài nộp hay checkpoint | Còn hiệu lực |
| G4 | **Khai báo trong system description:** gom cụm không giám sát trong file test; trừ mean theo file; mọi encoder kèm licence (ImageBind CC-BY-NC, audeering CC-BY-NC-SA); dữ liệu ngoài và cách lọc; việc dùng pseudo-label trên dev để chọn hệ ở progress phase. Báo cả INDEP-01 (26.70) để tách phần của mô hình và phần của gom cụm | Còn hiệu lực |
| G5 | **Tái lập:** train CPU phụ thuộc số luồng (lệch khoảng 0.002 giữa 3 và 10 luồng). Mạng production được cache theo (thành phần, seed) trong `BEST/v6_eval/_members/`; train lại thì dùng 10 luồng | Mới (74) |

## 3. Những quyết định đã bị đảo ngược, và bài học

| Ban đầu | Sau đó | Bài học |
|---|---|---|
| Bỏ dòng trùng giọng (EDA-0) | Giữ lại tốt hơn (3) | Giả định từ EDA phải được kiểm bằng thí nghiệm |
| "ArcFace thua vì lỗi căn mặt" | ArcFace đã căn mặt vẫn thua (20) | Nhận dạng giỏi ≠ ghép chéo giỏi |
| "Dữ liệu ngoài vô ích" (v3) | v1 / v2 giúp trên validation (51) | Quần thể quan trọng hơn số lượng |
| Gate g/Bn là "hiệu ứng độ dài câu" | Chỉ là đổi trọng số (29, 41) | Kiểm bằng đối chứng xáo trộn trước khi đặt tên cơ chế |
| Pseudo-label là cổng tốt nhất (53) | Dựng từ danh sách trial (54) → chỉ để cảnh báo (57) | Công cụ đoán CB giỏi chưa chắc là thước đo tổng quát hoá |
| SET-07 / 08 / 09 "tệ hơn" vì CB nói vậy | Nằm trong nhiễu seed; dev bị winner's curse (61) | Phải biết mức nhiễu của thước đo |
| ImageBind "chỉ để probe" | BTC cho phép; thành phần chính của SET-13 (63–67) | Hỏi BTC sớm về những gì luật không nói rõ |
| INDEP-01 là dự phòng khi ít mẫu / một người chiếm file (68) | `noclus` ≥ INDEP-01 ở mọi dạng file đã thử (71) | Kiểm giả định về "độ an toàn" bằng mô phỏng có nhãn trước khi viết vào tài liệu |
| Quy tắc `auto` tự chuyển sang chấm từng cặp | Cả hai quy tắc làm mất 2–6 EER (71) | Quy tắc an toàn cũng phải qua validation |
| "SetProto càng có ích khi càng nhiều người" (B3P, EXT-03) | Cùng ngân sách: ≈ 0, tương tác ≈ 0 (74) | So sánh phải cân số bước tối ưu; lợi của B3P đến từ người và bước, không từ loss |
| Dữ liệu ngoài +3…+5 (EXT-03, thành phần trần) | Còn khoảng 0 ở nhiều ô khi đã có ImageBind (74) | Kiểm cải tiến trên **bản production**, không trên thành phần đứng riêng |

## 4. Đang chờ, và quyết định cần nhóm chốt

| Việc | Tình trạng | Ai / khi nào |
|---|---|---|
| **Cấu hình Evaluation** | **SET-13 qua `BEST/v6_eval`, `--mode auto`, n = 15.** Chạy thử toàn bộ quy trình xong (73): chấm điểm 16.6 phút khi mạng đã cache | **Nhóm xác nhận**; chốt trước 3/11 |
| FLAG_12 bản đã sửa (w2v2) | Đã kiểm chứng trên máy với transformers 5.0.0 ở chế độ Jupyter (73) | Chạy lại một lần trên Kaggle (dev) trước 11/11 |
| s007 + v1/v2 cho các cell English (F3) | Ứng viên duy nhất còn mở từ (74) | Dựng + kiểm chứng riêng (VAL-70 kiểu fold người + v4 giữ ra) nếu nhóm muốn |
| X1 VoxCeleb | Chưa đạt cổng (74) | Chỉ mở lại nếu có lý do mới |
| Kế hoạch 15 lượt Evaluation | Lượt 1: SET-13. Nếu EER > 50: đảo dấu. Nếu còn lượt: `noclus` để đo tác dụng của gom cụm trên dữ liệu mới | Chốt khi BTC trả lời câu 10 (lượt nào được tính) |
| Mốc thời gian | Progress kết thúc 10/11 · Evaluation 11–18/11 (15 lượt) · system description + link code 27/11 | |
