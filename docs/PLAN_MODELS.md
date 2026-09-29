# Kế hoạch model mới — hai hướng: không dữ liệu ngoài (S) / có dữ liệu ngoài (X)

Lập 2026-09-27. Câu hỏi: AdaFace hoặc các model SOTA mới có giúp tăng điểm không, và nên thử theo thứ tự nào.

> **Trạng thái 29/09 (nhật ký tới mục 75): kế hoạch này đã được làm gần hết.** Kết quả từng hạng mục ở bảng dưới. Phần thân tài liệu (từ §0) giữ nguyên để làm hồ sơ lý do. Nguồn hiện hành: [DECISIONS.md](DECISIONS.md) và [OPEN_DIRECTIONS.md](OPEN_DIRECTIONS.md); kế hoạch cho Evaluation: [PLAN_EVAL.md](PLAN_EVAL.md).
>
> | Hạng mục | Kết quả | Nguồn |
> |---|---|---|
> | S1 · MODEL-01 (stream mới cho cầu nối) | **ImageBind** −5…−9 → dùng. FaRL qua cổng nhưng không thêm gì khi đã có ImageBind (SET-14). ViT-age, SigLIP2, AdaFace trượt | (59, 62, 65, 66) |
> | So khớp tuổi tường minh | g cụm −1.35 [5/5] → dùng, w = 0.1 | (62) |
> | S2 · AdaFace cho gom cụm mặt | không cần: ArcFace đã cho ARI 0.99 trên người mới (STRESS-01 B) | (73) |
> | Lớp khác của BTC, TTA (S3 / S4 của OPEN_DIRECTIONS) | lợi < 0.5 → đóng | (62) |
> | X1 · VoxCeleb | **chưa đạt cổng**: SCALE-ID cho thấy ImageBind che phần lớn lợi của thêm người ở bản production | (74) |
> | X2 · encoder đa ngôn ngữ | không mở | — |
> | X3 · ImageBind (+ LoRA) | ImageBind zero-shot: dùng (BTC cho phép, 63). Chỉnh bằng adapter / CCA: tệ hơn → LoRA hoãn | (63, 70, 74) |
> | X4 · v1/v2 | ở bản production chỉ còn English g được lợi → ứng viên O1 trong OPEN_DIRECTIONS | (74) |
> | SetProto, SWA | không có tác dụng → đóng | (74, 75) |
>
> **Mốc hiện tại là SET-13: 21.01** (11.31 / 20.91 / 23.22 / 28.61), không còn là SET-02 như bảng dưới.

| | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---:|---:|---:|---:|---:|
| SET-02 (cấu hình Evaluation — **mốc phải thắng**) | 21.68 | 12.50 | 21.34 | **28.31** | 24.59 |
| HYB-01 (ghép per-cell 3 bản đã nộp, chọn theo CB — chỉ cho bảng progress) | 20.40 | 12.30 | 17.21 | 27.49 | 24.59 |
| pi-flag (hạng 3 bảng 27/09; hai đội đầu 3.22 / 4.60 dưới trần oracle, xem README) | 19.13 | 11.51 | 19.91 | **19.35** | 25.75 |
| g/En tốt nhất bảng: jeetu_sk / 24521254 | | | | **18.13 / 20.37** | |

HYB-01 không phải một pipeline, nên không dùng làm mốc so sánh: thành phần mới nào cũng phải thắng SET-02 trên cùng cách chấm.

## 0. Kết luận ngắn

1. **AdaFace không nên dùng để thay VGG ở cầu nối.** AdaFace, LVFace (ICCV 2025) và TopoFR cùng họ với ArcFace
   (loss margin, train để nhận dạng). ArcFace đã thử kỹ: dù có align, face↔face đạt 1.7 EER nhưng cross-modal 33.92
   so với VGG 26.16 (010-D); ở mức người, ArcFace với bất kỳ voice nào cho 33.3–36.7 / 38.0–40.6, còn VGG + BTC cho 23.9 / 32.1
   (EDA-002). Model nhận dạng mặt tốt hơn thì bỏ nhiều thuộc tính mềm hơn, mà thuộc tính mềm là thứ giọng nói dự đoán được.
   AdaFace chỉ còn hai chỗ dùng: một dòng đối chứng trong phép thử S1, và khâu gom cụm mặt. Khâu gom cụm có trần dưới 1 EER (CEIL, nhật ký 50).
2. **Phần còn dư địa là cầu nối (σ_p²).** Với cầu nối hiện tại, kể cả khi biết đúng danh tính (oracle) ta vẫn chỉ đạt khoảng
   **22.75 (g) / 13.87 (ng)** (ORA-01). Hạng 1 có g/En 19.35, thấp hơn cả oracle của ta. Vậy muốn g/En giảm xuống dưới khoảng 23
   thì bắt buộc phải có cầu nối tốt hơn. Đây là chỗ model mới và dữ liệu ngoài có thể tác động.
3. **Chọn model theo tiêu chí mới.** Ở protocol gender, giới tính đã bị khoá. Thứ còn lại là tuổi, vùng/sắc tộc, thể trạng
   và hình dáng mặt (liên quan tới cao độ và formant của giọng). Vì vậy ta chọn encoder **giàu thuộc tính mềm** và dùng nó
   làm **stream bổ sung** bên cạnh VGG/ECAPA-BTC, không thay thế. Xếp hạng nhận dạng (IJB-C, VoxCeleb EER) không dùng làm tiêu chí.
4. **Việc đầu tiên nên làm:** S1. Hệ hạng 1 FAME 2026 có một thành phần mà ta chưa bao giờ chạy: stream tuổi–giới cho cả mặt lẫn giọng.
   Code đã có sẵn ở [../kaggle/agegender.py](../kaggle/agegender.py) nhưng chưa trích lần nào.

## 0b. Nguyên tắc (27/09): tập trung vào embedding, giảm phụ thuộc vào tập test

Hai đội đầu bảng (3.22, 4.60) thấp hơn cả trần oracle của ghép mặt–giọng, tức là điểm của họ đến từ cấu trúc của tập test,
không phải từ việc nhận ra mặt và giọng cùng một người. Ta không đi theo hướng đó. Nỗ lực từ nay dồn vào **embedding / cầu nối**,
phần duy nhất mang sang được mọi định dạng dữ liệu Evaluation.

Các mức phụ thuộc vào tập test, và ta dùng tới đâu:

| Mức | Dùng gì của tập test | Ví dụ | Quyết định |
|---|---|---|---|
| 0 | chỉ cặp (mặt, giọng) đang chấm | cosine của embedding | **nơi đo tiến bộ** |
| 1 | thống kê không nhãn của cả file | trừ mean theo file (câu 1 email) | giữ, khai báo |
| 2 | các mẫu không nhãn khác trong file, **không** dùng danh sách trial | SET-02: gom cụm + trung bình khối (câu 2) | giữ làm lớp phía trên; điểm của (f, v) không đổi khi các trial khác bị ghép lại |
| 3 | danh sách trial để **chọn / trộn** hệ | pseudo-arc (SEL-01b), HYB-01 chọn cell theo CB | **dừng làm cổng**; pseudo-arc chỉ còn là cảnh báo |
| 4 | danh sách trial để **chấm điểm** | đếm trial nối cụm (LEAK-01), cách của hai đội đầu | không bao giờ |

Hệ quả:
- **Cổng chính = EER mức mẫu (mức 0) trên nhãn thật**: v4 giữ ra, Urdu/Hindi, cả ng lẫn g. EER mức cụm (mức 2) chỉ được yêu cầu là không tệ đi.
  Một cải thiện chỉ xuất hiện sau khi gộp cụm thì không tính là cải thiện embedding.
- **SET-02 vẫn là lớp phía trên cho bản nộp** (người dùng xác nhận 27/09: không vi phạm luật hiện hành, tái lập được trên dữ liệu Evaluation; −7 EER, khai báo đầy đủ). Luôn giữ song song một bản chỉ dùng mức 0–1.
  Nếu BTC trả lời câu 2 là không được phép, bản đó thành bản chính mà không phải làm lại gì.
- **Pseudo-arc không còn là cổng trước khi nộp.** Cái giá: ở nhật ký 53 nó đoán CB tốt hơn proxy nhãn thật (và đã chặn đúng SET-07 / SET-08).
  Từ nay CB là bước xác nhận cuối, mỗi ứng viên qua cổng nhãn thật được tối đa 1 lượt. Pseudo-arc chỉ dùng để cảnh báo khi nó ngược chiều mạnh.

**Cập nhật 28/09 (VAL-70, nhật ký 61):**
- Chỉ đổi seed của s007 đã làm dev dịch 0.65–1.32 EER tổng (g/Bn 2.5–4.8). Dev CodaBench không phân xử được chênh lệch cỡ 1 EER, và bị phồng cho SET-02 vì đã chọn trên chính nó.
- **Cổng cho quyết định Evaluation = validation độc lập:** v4 giữ ra (MODEL-01) + VAL-70 (thành phần train trên 70 người, file giống dev từ v1/v2). CodaBench chỉ để phát hiện hỏng nặng.

**Cập nhật luật 28/09 (nhật ký 63):** BTC cho phép tất cả: ImageBind, dữ liệu cặp ngoài, gom cụm lúc test. Ở Evaluation, người và giọng không trùng với train/dev.
Hướng S vẫn giữ làm bản dự phòng "sạch nhất", nhưng **ImageBind và dữ liệu ngoài giờ là hướng chính**.

## 1. Luật và định nghĩa hai hướng

Evaluation Plan FLAG 2027 ([arXiv 2609.17913](https://arxiv.org/abs/2609.17913)) chỉ có một luật về model:
*"A pretrained encoder for faces or voices is allowed."* Văn bản này **không** nói gì về dữ liệu cặp ngoài hay về tiếng Bengali
(khác với FAME 2026). Câu 6–8 trong [email_organizers.md](email_organizers.md) vẫn chưa có trả lời.

| | **Hướng S — an toàn** | **Hướng X — dữ liệu ngoài** |
|---|---|---|
| Train cầu nối | chỉ 70 người v4 | v4 + cặp mặt–giọng ngoài (MAV-Celeb v1/v2, VoxCeleb1/2, VoxBlink2) |
| Encoder mặt | mọi encoder pretrained công khai (mặt không liên quan tới ngôn ngữ) | như S |
| Encoder giọng | chỉ những model mà **dữ liệu pretrain và fine-tune đã kiểm là không có tiếng Bengali** (đọc model card trước khi trích) | thêm các model đa ngôn ngữ (Whisper, XLS-R, ReDimNet2 VoxBlink2) sau khi BTC trả lời câu 8 |
| Mô hình đã căn chỉnh mặt–giọng sẵn | không | ImageBind: chỉ khi BTC xác nhận |
| Validation | v4 giữ ra (nhãn thật); v1/v2 giữ ra **chỉ để đánh giá**, không đưa vào trọng số | như S |
| Vai trò ở Evaluation | **luôn giữ một bản S làm dự phòng** | bản chính nếu thắng trên validation nhãn thật (mức mẫu) và được CB xác nhận |

Mọi thứ đều khai báo trong system description. Dữ liệu ngoài: loại người trùng v4 (như [../Experiment/SET-07/overlap.csv](../Experiment/SET-07/overlap.csv))
và **lọc câu tiếng Bengali** bằng LID (speechbrain `lang-id-voxlingua107-ecapa`). Bước lọc này cũng khai báo.

## 2. Ứng viên và vai trò

**Mặt**

| Model | Loại | Thử ở vai trò | Kỳ vọng | Lý do |
|---|---|---|---|---|
| VGGFace fc7 (đang dùng) | softmax 2.6k người | tham chiếu | – | cầu nối tốt nhất tới nay |
| ViT age (`nateraw/vit-age-classifier`, 768-d CLS) | thuộc tính | stream cầu nối | **cao nhất** | hệ hạng 1 FAME26 dùng đúng kiểu này (ViT 768-d); code sẵn |
| FaRL ViT-B/16 (LAION-Face 20M, kiểu CLIP) | biểu diễn mặt tổng quát | stream cầu nối | trung bình–cao | mạnh ở thuộc tính (tuổi, giới, sắc tộc) |
| SigLIP2 / CLIP ViT-L/14 | ngữ nghĩa tổng quát | stream cầu nối | trung bình | nắm được tuổi, vùng, thể trạng |
| AdaFace IR-101 (WebFace12M, bản CVLFace), embedding cuối trên cùng crop 5 điểm với ArcFace | nhận dạng | đối chứng cầu nối + gom cụm | thấp / trần < 1 | cùng họ ArcFace |
| LVFace, TopoFR | nhận dạng SOTA | chỉ gom cụm, và chỉ khi AdaFace cho thấy gom cụm còn dư địa | thấp | như trên |

**Giọng**

| Model | Thử ở vai trò | Hướng | Ghi chú |
|---|---|---|---|
| ECAPA-BTC 192 / speechbrain 6144 (đang dùng) | tham chiếu | – | |
| `audeering/wav2vec2-large-robust-24-ft-age-gender` (1024-d) | stream cầu nối | S nếu kiểm ra phần Common Voice không có Bengali, ngược lại X | code sẵn; CC BY-NC-SA, phải khai báo |
| Vox-Profile ([arXiv 2505.14648](https://arxiv.org/abs/2505.14648)): tuổi/giới, accent (có nhóm South Asia), chất giọng | để sau | bản dựa trên WavLM: S; bản dựa trên Whisper: X | chưa đưa vào FLAG_10: code phụ thuộc nội bộ transformers 4.46; tuổi/giới train trên CommonVoice + TIMIT + VoxCeleb |
| ReDimNet2, WavLM/XLS-R pooling | – | – | đã đóng (NB-4, EDA-002) |

**Mô hình ghép chéo (head / loss / hệ hoàn chỉnh)**

| Hệ | Hướng | Đánh giá |
|---|---|---|
| Hạng 1 FAME26 ([2512.04814](https://arxiv.org/abs/2512.04814)): VGG 4096 + ViT tuổi–giới 768 ; ECAPA 6144 + ECAPA tuổi–giới 1536 ; linear → 192 ; AAM ; VoxCeleb2 (5 994 người) | phần tuổi–giới: S ; VoxCeleb2: X | Head linear + dropout 0.9 (đặt sau concat, trước linear) đã thua trên 70 người (EXP-009). Họ thử cross-attention: **28.92** so với 23.99. Họ train 3 bản VoxCeleb2 (đủ / bỏ English / bỏ German); fine-tune MavCeleb **hại heard** (24.93 → 25.30) nhưng **giúp unheard**. Bài **không có ablation bỏ stream tuổi–giới**, nên chưa biết phần nào tạo ra gain; con số kỳ vọng cho S1 là chưa định lượng được |
| Hạng 2 FAME26 ([2512.02759](https://arxiv.org/abs/2512.02759)): ImageBind + LoRA | X, chờ BTC | chỉ probe đóng băng |
| MSM, ICASSP 2026 ([2601.13651](https://arxiv.org/abs/2601.13651)): ma trận phân tách lớp tối đa cố định + CE + OPL, có code | X (chỉ khi có ≥ vài trăm người) | trên 70 người, dạng head/loss không phải nút thắt (ARCH-01, EXP-009) |
| RFOP (hạng 3, 33.1) | – | không |

## 3. Hướng S — không dữ liệu ngoài

### S1 · MODEL-01: phép thử stream cầu nối (ưu tiên 1)

- **Trích một lần trên Kaggle:** [../kaggle/FLAG_10_models.ipynb](../kaggle/FLAG_10_models.ipynb), sinh bằng
  [../kaggle/build_models_notebook.py](../kaggle/build_models_notebook.py); code encoder ở [../kaggle/flag_models.py](../kaggle/flag_models.py).
  Dữ liệu: v4 train + 4 file dev (11 349 mặt, 11 349 giọng), v1/v2 cho validation (theo đúng các dòng của `feats_ext_full`: 16.5k mặt, 40k giọng).
  Mặt: ViT-age, FaRL, SigLIP2, AdaFace (embedding cuối, cùng crop với ArcFace), ImageBind vision. Giọng: w2v2-age-gender, ImageBind audio.
  Notebook in thêm bảng probe: EER nhận dạng của từng stream, EER zero-shot mặt↔giọng của ImageBind (X3 go/no-go), kiểm tra thuộc tính.
  **Bỏ Vox-Profile:** code của nó thay các lớp WavLM bằng bản viết theo nội bộ transformers 4.46, rất dễ gãy trên Kaggle; stream tuổi
  của giọng đã có audeering. Không lấy feature map giữa mạng của AdaFace: AdaFace chỉ là đối chứng.
- **Ba cách ghép**, trên recipe EXP-007 và dùng harness của [../Experiment/ARCH-01/run.py](../Experiment/ARCH-01/run.py) (chấm như production: khối trên ma trận đầy đủ):
  - (a) thay thế: một cột đối chứng, dự kiến thua;
  - (b) **nối stream**: PCA-32/64 của stream mới ghép vào VGG/ECAPA trước MLP. Giữ số chiều nhỏ vì ARCH-01 cho thấy tăng dung lượng thì tệ đi;
  - (c) **cầu nối riêng rồi fusion theo hạng** với s007 (kiểu FUSE-01/02, trước đây đã có tác dụng).
- **Cổng đăng ký trước (§0b).** Trên v4 giữ ra 30 người, 5 split ghép cặp, **mức mẫu**: **g tốt hơn ≥ 0.5 ở ≥ 4/5 split** (g là cell đích), ng không tệ hơn quá 0.5;
  Urdu/Hindi mức mẫu không tệ hơn quá 0.5; mức cụm / người không tệ hơn quá 0.5. Qua cổng thì nộp 1 lượt CB (tối đa 3 hệ đăng ký trước);
  pseudo-arc chỉ để cảnh báo.
- **Đọc kết quả theo protocol.** Ở `gender`, phần "giới" của stream tuổi–giới vô dụng vì negative cùng giới; gain ở g chỉ có thể đến từ tuổi và các thuộc tính khác.
  Pilot ATTR-01 (pitch) cho tín hiệu yếu trong từng giới, nên nếu stream chỉ thắng ở ng thì đó là thông tin giới lặp lại, không phải thứ ta cần.
  Hạng 1 dùng các stream này với 5 994 người; với 70 người thì bắt buộc nén (b) hoặc tách cầu nối (c), không nối nguyên 768/1024 chiều.
- **Dừng khi:** không stream nào qua cổng ở cả (b) lẫn (c). Khi đó kết luận thuộc tính mềm từ encoder ngoài không bổ sung gì cho VGG fc7, và đóng luôn phần encoder của hướng S.

### S2 · CLU-02: AdaFace cho gom cụm mặt (ưu tiên 3, gần như miễn phí nếu S1 đã trích)

- Chỉ thay ArcFace bằng AdaFace ở khâu [1]; ngưỡng hiệu chỉnh lại trên người train. Tiêu chí: ARI và EER mức file trên v4 giữ ra.
- Trần lợi ích dưới 1 EER (CEIL). Chỉ nộp nếu tốt hơn ≥ 0.5 và cùng chiều trên cả hai bộ nhãn.

### Không làm trong S

AdaFace/LVFace/TopoFR thay VGG ở cầu nối. Head/loss mới (MSM, RFOP, AAM, cross-attention) trên 70 người. Các encoder giọng nhận dạng mạnh hơn.
Tất cả đều đã có bằng chứng âm tính: 010-D, EDA-002, ARCH-01, EXP-009, NB-4.

## 4. Hướng X — có dữ liệu ngoài

### X1 · VOX-01: VoxCeleb trong đúng không gian feature của BTC (ưu tiên 2)

**Vì sao.** Mọi triệu chứng đều trỏ về việc thiếu người: CCA k > 32 overfit, tăng dung lượng thì tệ đi, head không quan trọng.
Ta đã tái tạo chính xác encoder của BTC (VGGFace fc7 và `yangwang825/ecapa-tdnn-vox2`, cos 1.0000). Nhờ vậy dữ liệu ngoài
nằm cùng không gian với feature dev/eval mà không cần đổi encoder. Số người train có thể tăng từ 70 (hoặc khoảng 200 nếu tính v1/v2) lên hàng nghìn.

**Rủi ro đã thấy.** v3 (châu Âu) làm tệ mọi cell. v1/v2 thắng trên validation khi train với 40 người v4 nhưng không chuyển sang dev khi đủ 70 người (SET-07).
Vì vậy quy mô phải lớn hơn hẳn, và phải kiểm soát độ khớp quần thể.

**Các bước**

1. **X1a, khả thi (1 ngày).** Tìm nguồn có thể attach trên Kaggle Datasets trước, để không tốn băng thông. Thứ tự: VoxCeleb1 (1 251 người, có metadata quốc tịch và giới)
   → VoxCeleb2 dev (5 994 người). Mặt không cần lấy cùng video: có thể lấy ảnh theo danh tính (VGGFace2 chứa đúng các danh tính của VoxCeleb2).
   Đầu ra: số GB, số người lấy được, thời gian trích.
   **Kết quả X1a (27/09, tra cứu, chưa tải):**
   - VoxCeleb1 audio có trên Kaggle ([sabahesaraki/voxceleb-1-dataset](https://www.kaggle.com/datasets/sabahesaraki/voxceleb-1-dataset),
     [yosrahashem/voxceleb](https://www.kaggle.com/datasets/yosrahashem/voxceleb)), kể cả một bản chỉ người **Ấn Độ**
     ([gaurav41/voxceleb1-audio-wav-files-for-india-celebrity](https://www.kaggle.com/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity)), gần quần thể v4 nhất.
     `vox1_meta.csv` có giới và quốc tịch.
   - VGGFace2 có trên Kaggle ([hearfool/vggface2](https://www.kaggle.com/datasets/hearfool/vggface2),
     [yakhyokhuja/vggface2-112x112](https://www.kaggle.com/datasets/yakhyokhuja/vggface2-112x112): bản này đã căn 112 × 112, **khác** cách cắt ảnh
     mà VGG-Face của BTC nhận). Các danh tính của VoxCeleb2 nằm trong VGGFace2, nên có thể ghép giọng VoxCeleb2 với mặt VGGFace2 theo danh tính.
   - Audio VoxCeleb2 có trên Kaggle / Hugging Face ([bachng/voxceleb](https://www.kaggle.com/datasets/bachng/voxceleb),
     [Reverb/voxceleb2](https://huggingface.co/datasets/Reverb/voxceleb2)). Bản chính thức cần điền form để lấy mật khẩu.
   - **Rủi ro cần kiểm tra trước khi train:**
     - (1) VGG-Face của BTC được train trên VGGFace (bản 1). Danh tính của VoxCeleb1 lấy từ đó, nên fc7 của những người này là "đã thấy lúc train", lệch so với người lạ trong v4 → ưu tiên VoxCeleb2 + VGGFace2.
     - (2) Có người Ấn Độ / Bangladesh nói tiếng Bengali → lọc bằng LID.
     - (3) Licence: VoxCeleb CC BY 4.0, VGGFace2 chỉ cho nghiên cứu → khai báo.
   - **Việc tiếp theo:** mở một dataset (bản Ấn Độ của VoxCeleb1 trước, vì nhỏ) để xem cấu trúc thư mục, rồi mới viết notebook trích (dùng lại `ExtRows` / `run_voices` của FLAG_10 / FLAG_11).
2. **X1b, trích.** Mỗi người K câu (K khoảng 20) và vài ảnh mặt. Lọc tiếng Bengali bằng LID. Loại người trùng v4 bằng ArcFace + ECAPA.
3. **X1c, các nhánh** (một biến mỗi lần; chung split, seed và trial với EXT-03):
   - v4 (đối chứng);
   - mix: v4 + Vox ngẫu nhiên;
   - mix: v4 + Vox **khớp quần thể**, chọn người Vox gần tâm mặt ArcFace của v4 train + v1/v2. Không dùng dev để chọn;
   - PT Vox → FT v4, **chấm tách English / Bangla**: hạng 1 FAME26 thấy fine-tune hại ngôn ngữ đã nghe nhưng giúp ngôn ngữ chưa nghe.
     Vì mỗi cell là một file riêng, cho phép English ← bản không FT, Bangla ← bản FT (chuyên biệt hoá ở mức representation, bổ sung cho việc chọn hệ per-cell đang làm);
   - nhánh tốt nhất + v1/v2 + SetProto (B3P).
   Chỉ khi đã có ≥ vài trăm người mới thử thêm bridge lớn hơn và head MSM.
4. **Cổng.** Giống S1 (mức mẫu, nhãn thật). SET-07 cho thấy validation nhãn thật có thể thắng mà dev vẫn thua, nên mỗi nhánh qua cổng phải được CB xác nhận (1 lượt) trước khi vào cấu hình Evaluation.

### X2 · Encoder đa ngôn ngữ, X3 · ImageBind: chỉ sau khi BTC trả lời câu 7–8

- X2: Vox-Profile bản Whisper (accent), ReDimNet2 VoxBlink2, XLS-R. Mỗi model thử như một stream theo đúng giao thức S1.
- X3: ImageBind, chỉ probe đóng băng. Nếu gần ngẫu nhiên thì dừng, không fine-tune ([PLAN_V4 §2](PLAN_V4.md)).

### X4 · v1/v2 (đã có, SET-07 / HYB-01)

SET-07 thắng ng/Bn (−4.13) và g/En (−0.82) nhưng thua hai cell còn lại. Không thêm thí nghiệm riêng cho v1/v2.
Chúng đi vào X1c như một thành phần. Việc chọn per-cell cho Evaluation phải chốt từ dev trước ngày 11/11, không dựa vào lượt CB.

## 5. Lịch (progress phase kết thúc 10/11; Evaluation 11–18/11; system description 27/11)

| Thời gian | S | X |
|---|---|---|
| 28/9 – 1/10 | Build + smoke-test `FLAG_10_models`, chạy trên Kaggle | X1a khả thi |
| 2/10 – 6/10 | S1 probe (CPU) → cổng nhãn thật mức mẫu → ≤ 2 lượt CB; S2 nếu rảnh | X1b trích VoxCeleb1 |
| 6/10 – 20/10 | – | X1c trên VoxCeleb1; nếu có tín hiệu thì mở rộng sang VoxCeleb2 |
| 20/10 – 3/11 | Ghép thành phần đã thắng vào `BEST/v4_set02/pipeline.py` | như S; X2/X3 nếu BTC đã trả lời |
| trước 3/11 | **Chốt hai cấu hình Evaluation: bản S (dự phòng) và bản X**; chạy thử toàn bộ pipeline trên dev | |

Ngân sách: mỗi ứng viên qua cổng tốn ≤ 1 lượt CB. Còn khoảng 130 lượt ở progress phase, 15 lượt ở Evaluation.

## 6. Phải khai báo nếu dùng

Tên, checkpoint và licence của từng encoder mới (audeering là CC BY-NC-SA). Nguồn dữ liệu ngoài, số người và cách lọc (trùng danh tính, LID Bengali).
Việc dùng v1/v2 chỉ để validation trong hướng S.
