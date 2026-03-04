# 📖 HƯỚNG DẪN CHI TIẾT TASK 2.1: EVALUATION PIPELINE

> **Dành cho**: Lê Văn Hoàng — người chưa có nền tảng ML/DL, học qua thực hành
> **Triết lý**: Mỗi bước không chỉ hướng dẫn **làm gì** mà giải thích **tại sao làm vậy**
> **Thời gian**: ~2-3 ngày
> **Tiền đề**: Task 1.6 Baseline Training ✅ (Val AUC 0.9983, best.pt sẵn sàng)
> **Tham chiếu**: [TASK_2.1_EVALUATION_PIPELINE.md](../tasks/TASK_2.1_EVALUATION_PIPELINE.md) | [PROJECT_PLAN.md](../PROJECT_PLAN.md) Section 6
>
> **Output**: JSON report + confusion matrix + ROC curve + phân tích OOD failure

---

## 📋 Mục lục

- [Bức tranh tổng thể: Evaluation nằm ở đâu?](#bức-tranh-tổng-thể-evaluation-nằm-ở-đâu)
- [Tại sao cần Evaluation Pipeline riêng?](#tại-sao-cần-evaluation-pipeline-riêng)
- [Bài học từ Task 1.6: Vấn đề OOD Failure](#bài-học-từ-task-16-vấn-đề-ood-failure)
- [Kiến thức nền: Metrics cho Binary Classification](#kiến-thức-nền-metrics-cho-binary-classification)
- [Kiến thức nền: Confusion Matrix](#kiến-thức-nền-confusion-matrix)
- [Kiến thức nền: ROC Curve & AUC](#kiến-thức-nền-roc-curve--auc)
- [Kiến thức nền: In-Domain vs Out-of-Distribution](#kiến-thức-nền-in-domain-vs-out-of-distribution)
- [Tổng quan các bước](#tổng-quan-các-bước)
- [Bước 0: Chuẩn bị Git branch](#bước-0-chuẩn-bị-git-branch)
- [Bước 1: Mở rộng metrics module](#bước-1-mở-rộng-metrics-module)
- [Bước 2: Implement Evaluator class](#bước-2-implement-evaluator-class)
- [Bước 3: Implement visualization module](#bước-3-implement-visualization-module)
- [Bước 4: Implement scripts/test.py](#bước-4-implement-scriptstestpy)
- [Bước 5: Update configs/test.yaml](#bước-5-update-configstestyaml)
- [Bước 6: Chạy evaluation trên local](#bước-6-chạy-evaluation-trên-local)
- [Bước 7: Phân tích kết quả OOD](#bước-7-phân-tích-kết-quả-ood)
- [Bước 8: Unit tests](#bước-8-unit-tests)
- [Bước 9: Document results (CONTEXT.md)](#bước-9-document-results-contextmd)
- [Bước 10: Commit & PR](#bước-10-commit--pr)
- [Checklist hoàn thành](#checklist-hoàn-thành)
- [Troubleshooting](#troubleshooting)
- [Mối liên hệ với các Task tiếp theo](#mối-liên-hệ-với-các-task-tiếp-theo)

---

## Bức tranh tổng thể: Evaluation nằm ở đâu?

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        DỰ ÁN HOLMHZ — SPRINT 2                          │
│                                                                           │
│  Sprint 1 ✅ HOÀN TẤT                                                    │
│  ─────────────────────                                                    │
│  Task 1.1  Setup môi trường ✅                                            │
│  Task 1.2  Thu thập dữ liệu ✅ (27,680 ảnh)                              │
│  Task 1.3  Data Pipeline    ✅ (17/17 tests, 18,550 train)                │
│  Task 1.4  Model Architecture ✅ (30/30 tests, 4M params)                 │
│  Task 1.5  Training Pipeline ✅ (16/16 tests, dry run AUC 0.92)           │
│  Task 1.6  Baseline Training ✅ (Val AUC 0.9983, best.pt 48.5MB)          │
│                                                                           │
│  Sprint 2: Evaluation + XAI + Benchmark                                   │
│  ──────────────────────────────────────                                    │
│  ► Task 2.1  EVALUATION PIPELINE  ◄◄◄  BẠN ĐANG Ở ĐÂY                   │
│    │                                                                      │
│    │  Train xong rồi, nhưng model tốt cỡ nào?                            │
│    │  ⚠️ Smoke test cho thấy: Fake 5/5 ✅ nhưng Real 0/5 ❌               │
│    │  → Cần ĐÁNH GIÁ CHÍNH THỨC trên test sets (5,155 ảnh)               │
│    │                                                                      │
│    │  4 việc chính:                                                       │
│    │    1. Implement Evaluator class (tính metrics)                       │
│    │    2. Chạy eval trên ID test (3,975 ảnh) — kỳ vọng AUC ~0.99        │
│    │    3. Chạy eval trên OOD test (1,180 ảnh) — đo failure mức nào      │
│    │    4. Visualization: confusion matrix + ROC curve + JSON report      │
│    │                                                                      │
│    ├──► Task 2.2  Benchmark SOTA (so sánh 3 model khác)                   │
│    ├──► Task 2.3  Grad-CAM XAI (model nhìn vùng nào?)                    │
│    └──► Task 2.4  Model Export ONNX                                       │
│                                                                           │
│  Milestone 2: AUC ≥ 0.90 ID + ≥ 0.75 OOD + Bảng so sánh + XAI gallery  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Tại sao cần Evaluation Pipeline riêng?

### Training ≠ Evaluation

```
┌──────────────── HIỂU ĐƠN GIẢN ────────────────────────┐
│                                                          │
│  Training (Task 1.6):                                    │
│  ──────────────────                                      │
│  • Thi cuối kỳ: làm đề → chấm điểm (val_auc)           │
│  • "Đề thi" = val set (3,975 ảnh từ CÙNG phân phối)     │
│  • Kết quả: AUC 0.9983 → "Giỏi lắm!"                    │
│                                                          │
│  Evaluation (Task 2.1):                                  │
│  ──────────────────────                                  │
│  • Thi thực tế: đưa model ra đời thật                    │
│  • "Đề thi" gồm 2 phần:                                 │
│    1. ID test: cùng loại ảnh đã học → vẫn giỏi?         │
│    2. OOD test: ảnh lạ chưa bao giờ thấy → xử lý nổi?  │
│  • Kết quả: có thể rất khác val set!                     │
│                                                          │
│  TƯƠNG TỰ ĐỜI THỰC:                                     │
│  ─────────────────────                                   │
│  Bạn học Toán → thi Toán được 10 điểm (= val AUC 0.9983)│
│  Bây giờ thử thi:                                        │
│    • Toán khác (ID test): vẫn ~10 điểm ✅                 │
│    • Lý + Hóa (OOD test): có thể 3 điểm ❌               │
│                                                          │
│  → Evaluation cho biết model THỰC SỰ mạnh ở đâu,        │
│    yếu ở đâu, và giới hạn của nó.                        │
└──────────────────────────────────────────────────────────┘
```

### Training metrics vs Evaluation metrics

|               | Training (Task 1.6)          | Evaluation (Task 2.1)                            |
| ------------- | ---------------------------- | ------------------------------------------------ |
| Metrics dùng  | AUC, Accuracy (2 metrics)    | AUC, Accuracy, F1, Precision, Recall (5 metrics) |
| Data          | val set (cùng distribution)  | test_id + test_ood (khác distribution)           |
| Per-source    | ❌ Không có                  | ✅ Phân tích riêng mỗi nguồn                     |
| Visualization | W&B charts (training curves) | Confusion matrix, ROC curve (evaluation charts)  |
| Output        | Checkpoint (best.pt)         | JSON report + hình ảnh                           |
| Mục đích      | Tìm model tốt nhất           | Đánh giá & ghi nhận cho báo cáo                  |

---

## Bài học từ Task 1.6: Vấn đề OOD Failure

### Chuyện gì đã xảy ra?

```
┌──────────────── SMOKE TEST FAILURE ──────────────────────┐
│                                                            │
│  Sau khi train xong, bạn test trên ảnh cá nhân:           │
│                                                            │
│  $ python scripts/predict.py imgs/Fake_AI_generated/       │
│  ┌──────────────────────────────────────────┐               │
│  │  Gemini_Generated_Image...  FAKE  0.9971 │ ← ✅ Đúng    │
│  │  generation-07888...        FAKE  0.9998 │ ← ✅ Đúng    │
│  │  generation-9f6e6...        FAKE  1.0000 │ ← ✅ Đúng    │
│  │  generation-d6905...        FAKE  0.9993 │ ← ✅ Đúng    │
│  │  generation-e9eac...        FAKE  0.9993 │ ← ✅ Đúng    │
│  └──────────────────────────────────────────┘               │
│  → 5/5 FAKE ✅  Tuyệt vời!                                │
│                                                            │
│  $ python scripts/predict.py imgs/Real/                    │
│  ┌──────────────────────────────────────────┐               │
│  │  IMG_20211207_152750...     FAKE  0.9971 │ ← ❌ SAI!    │
│  │  IMG_20211207_153512...     FAKE  0.6556 │ ← ❌ SAI!    │
│  │  IMG_2344.jpg               FAKE  1.0000 │ ← ❌ SAI!    │
│  │  IMG_2365.jpg               FAKE  0.9605 │ ← ❌ SAI!    │
│  │  IMG_2369.jpg               FAKE  0.9924 │ ← ❌ SAI!    │
│  └──────────────────────────────────────────┘               │
│  → 5/5 FAKE ❌  Tất cả ảnh Real bị đoán FAKE!             │
│                                                            │
│  Val AUC = 0.9983 nhưng Real accuracy = 0%?!               │
│  → Đây là vấn đề OOD (Out-of-Distribution)                │
└────────────────────────────────────────────────────────────┘
```

### Tại sao model sai toàn bộ ảnh Real?

```
┌──────────────── ROOT CAUSE — DISTRIBUTION SHIFT ──────────┐
│                                                              │
│  TRAINING REAL IMAGES:                                       │
│  ────────────────────                                        │
│  • cifake (4,927 ảnh): 32×32 upscale → 224×224               │
│    → Mặt, vật thể đơn giản, chất lượng thấp                 │
│    → JPEG artifacts rõ ràng                                  │
│                                                              │
│  • ffhq (3,500 ảnh): 1024×1024 → crop 224×224                │
│    → CHỈ khuôn mặt (face dataset)                            │
│    → Chất lượng cao nhưng 1 loại content duy nhất            │
│                                                              │
│  TEST REAL IMAGES (imgs/Real/):                              │
│  ──────────────────────────────                              │
│  • iPhone photos: 948-4160px                                 │
│    → Cảnh đời thường (NOT faces)                             │
│    → High-resolution, different processing pipeline          │
│    → File size: 182-2,736 KB (vs 37-97 KB training)         │
│                                                              │
│  ┌──────────────────────────────────────────┐                │
│  │  Model đã học:                           │                │
│  │  "Ảnh 224×224 preprocessed = REAL"       │                │
│  │  "Ảnh khác = FAKE"                       │                │
│  │                                          │                │
│  │  → SHORTCUT LEARNING!                    │                │
│  │  Model học artifacts của preprocessing   │                │
│  │  chứ không học features thật sự          │                │
│  └──────────────────────────────────────────┘                │
│                                                              │
│  ĐÂY LÀ VẤN ĐỀ KINH ĐIỂN trong ML:                         │
│  "Model performs well on i.i.d. test but fails on OOD"       │
│  → Task 2.1 sẽ ĐO LƯỜNG CHÍNH XÁC mức độ failure           │
└──────────────────────────────────────────────────────────────┘
```

### Task 2.1 sẽ trả lời những câu hỏi gì?

```
1. In-domain (3,975 ảnh): Model vẫn giỏi trên data cùng loại?
   → Kỳ vọng: AUC ~0.99 (gần bằng val AUC 0.9983)

2. OOD tổng (1,180 ảnh): Model xử lý nổi ảnh lạ không?
   → Kỳ vọng: AUC thấp hơn nhiều (có thể 0.5-0.8)

3. OOD per-source: Nguồn nào yếu nhất?
   → real_camera (100 ảnh): Dự kiến THẤP NHẤT (giống imgs/Real/)
   → real_pexels (500 ảnh): Có thể thấp (ảnh stock, không phải face)
   → flux (80 ảnh Fake): Có thể cao (model bias FAKE → predict Fake đúng nhưng vì lý do sai)
   → tristanzhang_fake (500 ảnh): Tương tự flux

4. Error patterns: Lỗi chủ yếu là gì?
   → False Positive (Real → dự đoán FAKE)? hay
   → False Negative (Fake → dự đoán REAL)?
   → Từ smoke test: False Positive chiếm ưu thế (model bias FAKE)

5. Kết quả OOD (dù xấu) → đóng góp gì cho báo cáo?
   → Section "Limitations" — model chưa generalize
   → Gợi ý cải thiện: data augmentation, diverse training data
   → So sánh với SOTA ở Task 2.2: các model khác có cùng vấn đề?
```

---

## Kiến thức nền: Metrics cho Binary Classification

### 5 Metrics cần implement

```
┌───────────────── METRICS CHEAT SHEET ────────────────────┐
│                                                            │
│  Label convention: 0 = Real (Negative), 1 = Fake (Pos.)   │
│                                                            │
│  1. AUC (Area Under ROC Curve) — METRIC CHÍNH              │
│     → Đo khả năng phân biệt Real vs Fake                  │
│     → 1.0 = hoàn hảo, 0.5 = đoán ngẫu nhiên              │
│     → KHÔNG phụ thuộc threshold                            │
│     → ĐÃ IMPLEMENT: src/holmhz/metrics/auc.py ✅          │
│                                                            │
│  2. Accuracy — Tỷ lệ đoán đúng                            │
│     → (TP + TN) / Total                                    │
│     → ĐÃ IMPLEMENT: src/holmhz/metrics/accuracy.py ✅     │
│                                                            │
│  3. F1 Score — Cân bằng giữa Precision và Recall           │
│     → 2 × (Precision × Recall) / (Precision + Recall)     │
│     → CẦN IMPLEMENT ❌                                     │
│                                                            │
│  4. Precision — "Khi model nói FAKE, bao nhiêu % đúng?"   │
│     → TP / (TP + FP)                                       │
│     → Precision thấp = nhiều False Positive (Real → FAKE)  │
│     → CẦN IMPLEMENT ❌                                     │
│                                                            │
│  5. Recall — "Trong tất cả Fake thật, model tìm ra được   │
│     bao nhiêu %?"                                          │
│     → TP / (TP + FN)                                       │
│     → Recall thấp = nhiều False Negative (Fake → REAL)     │
│     → CẦN IMPLEMENT ❌                                     │
│                                                            │
│  Trong đó:                                                 │
│    TP = True Positive  = Fake thật, đoán FAKE ✅            │
│    TN = True Negative  = Real thật, đoán REAL ✅            │
│    FP = False Positive = Real thật, đoán FAKE ❌            │
│    FN = False Negative = Fake thật, đoán REAL ❌            │
└────────────────────────────────────────────────────────────┘
```

### Ví dụ trực quan

```
Giả sử model dự đoán 10 ảnh:

Ảnh 1:  Real  → đoán REAL  ✅ TN
Ảnh 2:  Real  → đoán REAL  ✅ TN
Ảnh 3:  Real  → đoán FAKE  ❌ FP  ← Lỗi từ smoke test!
Ảnh 4:  Real  → đoán FAKE  ❌ FP  ← Lỗi từ smoke test!
Ảnh 5:  Real  → đoán FAKE  ❌ FP  ← Lỗi từ smoke test!
Ảnh 6:  Fake  → đoán FAKE  ✅ TP
Ảnh 7:  Fake  → đoán FAKE  ✅ TP
Ảnh 8:  Fake  → đoán FAKE  ✅ TP
Ảnh 9:  Fake  → đoán FAKE  ✅ TP
Ảnh 10: Fake  → đoán REAL  ❌ FN

Tính metrics:
  TP = 4, TN = 2, FP = 3, FN = 1

  Accuracy  = (4+2)/(4+2+3+1) = 0.60 (60%)
  Precision = 4/(4+3) = 0.571 (57.1%)  ← THẤP vì nhiều FP
  Recall    = 4/(4+1) = 0.800 (80%)    ← Khá (tìm được 4/5 Fake)
  F1        = 2×(0.571×0.800)/(0.571+0.800) = 0.667

→ Precision thấp nhất = vấn đề chính là FALSE POSITIVE
  (Real bị đoán FAKE) — ĐÚNG VỚI SMOKE TEST TASK 1.6!
```

---

## Kiến thức nền: Confusion Matrix

```
┌──────────────── CONFUSION MATRIX ──────────────────────┐
│                                                          │
│  Ma trận 2×2 hiển thị phân bố dự đoán:                  │
│                                                          │
│                   PREDICTED                              │
│              ┌──────────┬──────────┐                     │
│              │   REAL    │   FAKE   │                     │
│  ───────┬────┼──────────┼──────────┤                     │
│  ACTUAL │ R  │  TN: 2   │  FP: 3   │  ← 3 Real bị sai   │
│         │ F  │  FN: 1   │  TP: 4   │  ← 1 Fake bị bỏ    │
│  ───────┴────┴──────────┴──────────┘                     │
│                                                          │
│  ĐỌC NHƯ THẾ NÀO:                                       │
│  ─────────────────                                       │
│  • Hàng = label thật (ground truth)                      │
│  • Cột = model dự đoán                                   │
│  • Đường chéo chính (TN, TP) = đoán đúng ✅              │
│  • Ngoài đường chéo (FP, FN) = đoán sai ❌               │
│                                                          │
│  MODEL HOÀN HẢO:                                         │
│  ─────────────────                                       │
│              ┌──────────┬──────────┐                     │
│              │   REAL    │   FAKE   │                     │
│  ───────┬────┼──────────┼──────────┤                     │
│  ACTUAL │ R  │  TN: 5   │  FP: 0   │  ← Không sai Real  │
│         │ F  │  FN: 0   │  TP: 5   │  ← Không sai Fake  │
│  ───────┴────┴──────────┴──────────┘                     │
│                                                          │
│  VÌ SAO CẦN CONFUSION MATRIX?                            │
│  ─────────────────────────────                           │
│  Accuracy = 60% nói "sai 40%" nhưng KHÔNG nói sai loại  │
│  nào. Confusion matrix cho thấy: FP (3) >> FN (1)       │
│  → VẤN ĐỀ CHÍNH: Model bias dự đoán FAKE                │
│    (Real images bị đoán nhầm thành FAKE)                 │
│                                                          │
│  Trong Task 2.1: confusion matrix sẽ CHỨNG MINH         │
│  vấn đề OOD mà smoke test đã phát hiện.                 │
└──────────────────────────────────────────────────────────┘
```

---

## Kiến thức nền: ROC Curve & AUC

```
┌──────────────── ROC CURVE ────────────────────────────┐
│                                                        │
│  ROC = Receiver Operating Characteristic               │
│                                                        │
│  Trục X: False Positive Rate = FP / (FP + TN)         │
│    → "Bao nhiêu % Real bị đoán FAKE?"                  │
│                                                        │
│  Trục Y: True Positive Rate = TP / (TP + FN)          │
│    → "Bao nhiêu % Fake được tìm ra?"                   │
│                                                        │
│  ROC vẽ TPR vs FPR khi THAY ĐỔI threshold:            │
│                                                        │
│  TPR                                                   │
│  1.0 ┤                ╭────────────                    │
│      │              ╭╯                                 │
│  0.8 ┤            ╭╯         ← Model tốt              │
│      │           ╭╯            (cong về góc trên trái) │
│  0.6 ┤         ╭╯                                     │
│      │       ╭╯                                       │
│  0.4 ┤     ╭╯                                        │
│      │   ╭╯        ╱╱╱╱╱ ← Đường chéo                │
│  0.2 ┤ ╭╯         ╱        = đoán random              │
│      │╯          ╱           AUC = 0.5                 │
│  0.0 ┼────────┬────────┬───                           │
│      0.0      0.5      1.0  FPR                       │
│                                                        │
│  AUC = diện tích dưới đường cong                       │
│  AUC = 1.0: phân biệt hoàn hảo                        │
│  AUC = 0.5: đoán ngẫu nhiên                           │
│  AUC < 0.5: model dự đoán ngược (đổi label sẽ tốt hơn)│
│                                                        │
│  TRONG TASK 2.1:                                       │
│  → Vẽ 2 ROC curves: ID test + OOD test                 │
│  → AUC ID: kỳ vọng ~0.99                              │
│  → AUC OOD: kỳ vọng thấp hơn (có thể 0.5-0.8)        │
│  → So sánh 2 curves → thấy rõ "gap" ID vs OOD         │
└────────────────────────────────────────────────────────┘
```

---

## Kiến thức nền: In-Domain vs Out-of-Distribution

```
┌──────────── ID vs OOD — HIỂU ĐƠN GIẢN ──────────────────┐
│                                                              │
│  IN-DOMAIN (ID):                                             │
│  ──────────────                                              │
│  Ảnh TEST có CÙNG LOẠI với ảnh TRAINING                      │
│                                                              │
│  Training Real: cifake (upscaled 32→224) + ffhq (faces)      │
│  Training Fake: stable-diffusion-1.5 + stylegan              │
│                                                              │
│  ID Test (test_id.json = 3,975 ảnh):                         │
│  ┌──────────────┬───────┬──────────────────────────────┐     │
│  │ Nguồn        │ N     │ Vì sao ID?                   │     │
│  ├──────────────┼───────┼──────────────────────────────┤     │
│  │ cifake       │ 2,100 │ Cùng nguồn với train (split) │     │
│  │ ffhq         │ 750   │ Cùng nguồn với train (split) │     │
│  │ stylegan     │ 750   │ Cùng loại fake với train      │     │
│  │ sd15         │ 375   │ Cùng model diffusion với train│     │
│  └──────────────┴───────┴──────────────────────────────┘     │
│  → Kỳ vọng: AUC ~0.99 (model đã "thấy" loại ảnh này rồi)   │
│                                                              │
│  OUT-OF-DISTRIBUTION (OOD):                                  │
│  ──────────────────────────                                  │
│  Ảnh TEST KHÁC LOẠI hoàn toàn với ảnh TRAINING               │
│                                                              │
│  OOD Test (test_ood.json = 1,180 ảnh):                       │
│  ┌────────────────┬─────┬──────────────────────────────┐     │
│  │ Nguồn          │ N   │ Vì sao OOD?                  │     │
│  ├────────────────┼─────┼──────────────────────────────┤     │
│  │ real_pexels    │ 500 │ Ảnh stock — NOT faces!       │     │
│  │ real_camera    │ 100 │ Ảnh camera — NOT faces!      │     │
│  │ tristanzhang   │ 500 │ Fake từ model LẠ             │     │
│  │ flux           │ 80  │ Flux model — CHƯA THẤY BAO GIỜ│   │
│  └────────────────┴─────┴──────────────────────────────┘     │
│  → Kỳ vọng: AUC thấp hơn (0.5-0.8), đặc biệt               │
│    real_pexels và real_camera sẽ bị đoán sai nhiều           │
│                                                              │
│  TƯƠNG TỰ ĐỜI THỰC:                                         │
│  ─────────────────────                                       │
│  Bạn học tiếng Anh → thi tiếng Anh giỏi (ID)                │
│  Nhưng bắt thi tiếng Pháp (OOD) → sẽ kém                   │
│  → Không phải lỗi, mà là GIỚI HẠN!                          │
│  → Task 2.1 đo giới hạn → báo cáo → đề xuất cải thiện       │
└──────────────────────────────────────────────────────────────┘
```

---

## Tổng quan các bước

```
                                          Thời gian ước tính
                                          ──────────────────
Bước 0:  Git branch ──────────────────     5 phút
Bước 1:  Mở rộng metrics module ──────    30 phút (code F1, Precision, Recall)
Bước 2:  Implement Evaluator class ────   1 giờ (code + test)
Bước 3:  Implement visualization ──────   45 phút (confusion matrix + ROC)
Bước 4:  Implement scripts/test.py ────   45 phút (CLI end-to-end)
Bước 5:  Update configs/test.yaml ─────   10 phút
Bước 6:  Chạy evaluation trên local ──   15 phút (~3 min inference)
Bước 7:  Phân tích kết quả OOD ───────   1 giờ (quan trọng nhất!)
Bước 8:  Unit tests ──────────────────    1 giờ
Bước 9:  Document results ────────────    30 phút
Bước 10: Commit & PR ─────────────────    15 phút
                                   Tổng: ~2-3 ngày
```

> **Tất cả chạy LOCAL** — RTX 3050 đủ sức.
> Evaluation chỉ inference (no gradient) → VRAM thấp.
> 5,155 ảnh × batch=32 → ~161 batches → ~3 phút tổng.

---

## Bước 0: Chuẩn bị Git branch

```bash
# Từ branch main (hoặc feat/s1/baseline-training nếu chưa merge)
git checkout main
git pull origin main

# Tạo branch Sprint 2
git checkout -b feat/s2/evaluation-pipeline

# Verify
git branch
# * feat/s2/evaluation-pipeline
```

> **Tại sao branch mới?**
> Sprint 2 là phase mới → branch riêng.
> Nếu chưa merge Task 1.6 branch:
>
> ```bash
> git checkout feat/s1/baseline-training
> git checkout -b feat/s2/evaluation-pipeline
> ```

---

## Bước 1: Mở rộng metrics module

### 1.1 Hiện trạng metrics

```
src/holmhz/metrics/
├── __init__.py       ← exports compute_accuracy, compute_auc
├── accuracy.py       ← compute_accuracy(logits, labels, threshold) ✅
└── auc.py            ← compute_auc(logits, labels) ✅

CẦN THÊM:
├── f1.py             ← compute_f1(logits, labels, threshold) ❌
├── precision.py      ← compute_precision(logits, labels, threshold) ❌
└── recall.py         ← compute_recall(logits, labels, threshold) ❌
```

### 1.2 Tạo `src/holmhz/metrics/f1.py`

```python
"""
F1 Score — Harmonic mean của Precision và Recall.

F1 cân bằng giữa "đoán đúng khi nói FAKE" (Precision) và
"tìm ra được bao nhiêu FAKE" (Recall).

F1 = 2 × (Precision × Recall) / (Precision + Recall)

F1 tốt khi CẢ Precision và Recall đều cao.
Nếu 1 trong 2 rất thấp → F1 cũng thấp (penalize imbalance).

Ví dụ:
  Precision=0.90, Recall=0.90 → F1=0.90 (tốt)
  Precision=0.99, Recall=0.10 → F1=0.18 (tệ! Recall quá thấp)
"""

import torch


def compute_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Tính F1 score từ logits và labels.

    Args:
        logits: [N] hoặc [N, 1] — raw logits từ model.
        labels: [N] — ground truth (0.0 = Real, 1.0 = Fake).
        threshold: Ngưỡng phân loại (default 0.5).

    Returns:
        f1: float ∈ [0.0, 1.0].

    Edge cases:
        - Không có TP → F1 = 0.0 (model không tìm được Fake nào).
        - Không có Positive predictions → F1 = 0.0.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs >= threshold).float()

        # True Positives, False Positives, False Negatives
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)

        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)

        # F1 = 2 × (Precision × Recall) / (Precision + Recall)
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = torch.tensor(0.0)

        return float(f1)
```

### 1.3 Tạo `src/holmhz/metrics/precision.py`

```python
"""
Precision — "Khi model nói FAKE, nó đúng bao nhiêu %?"

Precision = TP / (TP + FP)

Precision THẤP → nhiều False Positive (Real bị đoán FAKE).
→ Đây chính là vấn đề phát hiện ở smoke test Task 1.6!

Ví dụ:
  Model đoán 10 ảnh FAKE, nhưng 4 trong đó thực ra là Real.
  Precision = 6/10 = 0.60 ← "40% dự đoán FAKE là sai"
"""

import torch


def compute_precision(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Tính Precision từ logits và labels.

    Args:
        logits: [N] hoặc [N, 1] — raw logits từ model.
        labels: [N] — ground truth (0.0 = Real, 1.0 = Fake).
        threshold: Ngưỡng phân loại (default 0.5).

    Returns:
        precision: float ∈ [0.0, 1.0].

    Edge cases:
        - Model không dự đoán FAKE nào → Precision = 0.0
          (division by zero protection)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs >= threshold).float()

        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()

        if (tp + fp) > 0:
            return float(tp / (tp + fp))
        return 0.0
```

### 1.4 Tạo `src/holmhz/metrics/recall.py`

```python
"""
Recall — "Trong tất cả Fake thật, model tìm ra được bao nhiêu %?"

Recall = TP / (TP + FN)

Recall THẤP → nhiều False Negative (Fake bị bỏ sót, đoán thành Real).
→ Nguy hiểm nếu mục đích là "catch all fakes".

Ví dụ:
  Có 10 ảnh Fake thật, model chỉ tìm ra 7.
  Recall = 7/10 = 0.70 ← "bỏ sót 30% Fake"
"""

import torch


def compute_recall(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Tính Recall từ logits và labels.

    Args:
        logits: [N] hoặc [N, 1] — raw logits từ model.
        labels: [N] — ground truth (0.0 = Real, 1.0 = Fake).
        threshold: Ngưỡng phân loại (default 0.5).

    Returns:
        recall: float ∈ [0.0, 1.0].

    Edge cases:
        - Không có Fake nào trong data → Recall = 0.0
          (không có TP hay FN)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs >= threshold).float()

        tp = ((preds == 1) & (labels == 1)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()

        if (tp + fn) > 0:
            return float(tp / (tp + fn))
        return 0.0
```

### 1.5 Update `src/holmhz/metrics/__init__.py`

```python
"""Metrics module — đo lường performance của model."""

from .accuracy import compute_accuracy
from .auc import compute_auc
from .f1 import compute_f1
from .precision import compute_precision
from .recall import compute_recall

__all__ = [
    "compute_accuracy",
    "compute_auc",
    "compute_f1",
    "compute_precision",
    "compute_recall",
]
```

> **Tại sao tách riêng file thay vì viết chung?**
>
> 1. Follow codebase pattern (accuracy.py, auc.py riêng)
> 2. Dễ test riêng từng metric
> 3. Dễ đọc — mỗi file có docstring giải thích ý nghĩa
> 4. Import rõ ràng: `from holmhz.metrics import compute_f1`

---

## Bước 2: Implement Evaluator class

### 2.1 Thiết kế Evaluator

```
┌──────────────── EVALUATOR CLASS ──────────────────────┐
│                                                         │
│  Input:                                                 │
│  ──────                                                 │
│  • model (nn.Module): Trained model                     │
│  • dataloader (DataLoader): Test data                   │
│  • device (torch.device): CPU/GPU                       │
│  • threshold (float): Classification threshold (0.5)    │
│                                                         │
│  Flow:                                                  │
│  ──────                                                 │
│  1. Inference: model(images) → logits                   │
│  2. Collect: all logits + labels + sources + paths      │
│  3. Overall metrics: AUC, Acc, F1, Prec, Rec            │
│  4. Per-source breakdown: metrics cho mỗi nguồn         │
│  5. Return: dict với tất cả kết quả                     │
│                                                         │
│  Output:                                                │
│  ──────                                                 │
│  {                                                      │
│    "overall": { "auc": 0.99, "accuracy": 0.98, ... },  │
│    "per_source": {                                      │
│      "cifake": { "accuracy": 0.99, "n": 2100, ... },   │
│      "ffhq": { "accuracy": 0.97, "n": 750, ... },     │
│      ...                                                │
│    },                                                   │
│    "all_logits": tensor,  ← cho ROC curve               │
│    "all_labels": tensor,  ← cho confusion matrix        │
│    "all_sources": list,   ← cho per-source              │
│  }                                                      │
│                                                         │
│  Tại sao lưu all_logits/labels?                         │
│  → Visualization module cần raw data để vẽ              │
│    confusion matrix và ROC curve.                       │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Code `src/holmhz/evaluation/evaluator.py`

```python
"""
Evaluator — đánh giá model trên test set.

Evaluator chạy inference trên toàn bộ dataloader, thu thập
predictions, và tính metrics (overall + per-source breakdown).

Usage:
    evaluator = Evaluator(model, dataloader, device)
    results = evaluator.evaluate()
    print(results["overall"]["auc"])        # 0.9983
    print(results["per_source"]["cifake"])  # {"accuracy": 0.99, ...}
"""

from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from holmhz.metrics import (
    compute_accuracy,
    compute_auc,
    compute_f1,
    compute_precision,
    compute_recall,
)
from holmhz.utils.logger import get_logger

logger = get_logger("evaluator")


class Evaluator:
    """Đánh giá model trên 1 test set.

    Attributes:
        model: Trained model (nn.Module).
        dataloader: Test DataLoader (ImageDataset format).
        device: torch.device.
        threshold: Classification threshold (default 0.5).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        threshold: float = 0.5,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Chạy evaluation trên toàn bộ dataloader.

        Returns:
            dict chứa:
                - "overall": dict metrics tổng (auc, accuracy, f1, precision, recall)
                - "per_source": dict[source_name] → dict metrics
                - "all_logits": tensor [N] — raw logits
                - "all_labels": tensor [N] — ground truth
                - "all_sources": list[str] — source tag cho mỗi sample
                - "total": int — tổng số samples
        """
        self.model.eval()
        self.model.to(self.device)

        all_logits = []
        all_labels = []
        all_sources = []

        logger.info(
            f"Evaluating {len(self.dataloader.dataset)} samples "
            f"({len(self.dataloader)} batches)..."
        )

        for batch in tqdm(self.dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"]
            sources = batch["source"]

            # Inference
            logits = self.model(images).squeeze(-1)  # [B, 1] → [B]

            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_sources.extend(sources)

        # Concatenate all batches
        all_logits = torch.cat(all_logits)   # [N]
        all_labels = torch.cat(all_labels)   # [N]

        # ─── Overall metrics ───
        overall = self._compute_metrics(all_logits, all_labels)
        logger.info(
            f"Overall — AUC: {overall['auc']:.4f}, "
            f"Acc: {overall['accuracy']:.4f}, "
            f"F1: {overall['f1']:.4f}"
        )

        # ─── Per-source metrics ───
        per_source = self._compute_per_source(all_logits, all_labels, all_sources)
        for source, metrics in per_source.items():
            logger.info(
                f"  {source:20s} — "
                f"Acc: {metrics['accuracy']:.4f}, "
                f"N: {metrics['n']}"
            )

        return {
            "overall": overall,
            "per_source": per_source,
            "all_logits": all_logits,
            "all_labels": all_labels,
            "all_sources": all_sources,
            "total": len(all_logits),
        }

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Tính tất cả 5 metrics cho 1 tập logits/labels."""
        return {
            "auc": compute_auc(logits, labels),
            "accuracy": compute_accuracy(logits, labels, self.threshold),
            "f1": compute_f1(logits, labels, self.threshold),
            "precision": compute_precision(logits, labels, self.threshold),
            "recall": compute_recall(logits, labels, self.threshold),
        }

    def _compute_per_source(
        self,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor,
        all_sources: list,
    ) -> dict:
        """Tính metrics riêng cho mỗi nguồn dữ liệu.

        Nhóm samples theo source tag, tính metrics cho mỗi nhóm.
        """
        # Nhóm indices theo source
        source_indices = defaultdict(list)
        for i, src in enumerate(all_sources):
            source_indices[src].append(i)

        per_source = {}
        for source, indices in sorted(source_indices.items()):
            idx = torch.tensor(indices)
            src_logits = all_logits[idx]
            src_labels = all_labels[idx]

            metrics = self._compute_metrics(src_logits, src_labels)
            metrics["n"] = len(indices)

            # Thêm label distribution info
            n_real = int((src_labels == 0).sum())
            n_fake = int((src_labels == 1).sum())
            metrics["n_real"] = n_real
            metrics["n_fake"] = n_fake

            per_source[source] = metrics

        return per_source
```

### 2.3 Update `src/holmhz/evaluation/__init__.py`

```python
"""Evaluation module — đánh giá model performance."""

from .evaluator import Evaluator

__all__ = ["Evaluator"]
```

> **Tại sao Evaluator là class thay vì function?**
>
> 1. Giữ state (model, dataloader, device, threshold) — không cần truyền lại mỗi lần
> 2. Dễ mở rộng: thêm method `evaluate_with_tta()` (test-time augmentation) sau
> 3. Dễ test: mock model/dataloader riêng
> 4. Follow pattern trong codebase (Trainer cũng là class)

---

## Bước 3: Implement visualization module

### 3.1 Tại sao cần visualization?

```
┌──────────────── VISUALIZATION TRONG ML ──────────────────┐
│                                                            │
│  Số liệu đơn thuần: "AUC = 0.72" → Không đủ hiểu         │
│                                                            │
│  Confusion Matrix hình ảnh → thấy NGAY:                    │
│    • Ô nào lớn nhất? (loại lỗi chính)                     │
│    • Balance giữa FP và FN                                 │
│                                                            │
│  ROC Curve → thấy NGAY:                                   │
│    • Curve cong hay thẳng? (performance overall)           │
│    • ID vs OOD curve gap?                                  │
│    • Ở threshold nào tối ưu?                               │
│                                                            │
│  Trong BÁO CÁO + BẢO VỆ:                                  │
│    • Hội đồng nhìn hình dễ hơn bảng số                     │
│    • Confusion matrix + ROC curve = "must have"            │
│    • Professional presentation                              │
│                                                            │
│  Output:                                                   │
│    outputs/evaluation/                                     │
│    ├── confusion_matrix_id.png                             │
│    ├── confusion_matrix_ood.png                            │
│    ├── roc_curve.png (2 curves chồng lên)                  │
│    └── per_source_accuracy.png (bar chart)                 │
└────────────────────────────────────────────────────────────┘
```

### 3.2 Code `src/holmhz/utils/visualization.py`

```python
"""
Visualization module — vẽ confusion matrix, ROC curve, bar chart.

Sử dụng matplotlib (không cần GUI — save trực tiếp file PNG).
Dùng backend 'Agg' để tránh lỗi Tkinter trên server/notebook.

Usage:
    from holmhz.utils.visualization import plot_confusion_matrix, plot_roc_curve
    plot_confusion_matrix(labels, preds, save_path="outputs/evaluation/cm.png")
    plot_roc_curve(labels, probs, save_path="outputs/evaluation/roc.png")
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

from holmhz.utils.logger import get_logger

logger = get_logger("visualization")


def plot_confusion_matrix(
    labels: torch.Tensor | np.ndarray,
    logits: torch.Tensor | np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix",
    threshold: float = 0.5,
) -> str:
    """Vẽ confusion matrix và save thành PNG.

    Args:
        labels: [N] ground truth (0=Real, 1=Fake).
        logits: [N] raw logits (sẽ qua sigmoid → threshold).
        save_path: Đường dẫn file PNG output.
        title: Tiêu đề biểu đồ.
        threshold: Ngưỡng phân loại.

    Returns:
        save_path: Đường dẫn file đã save.
    """
    # Convert to numpy
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits).cpu().numpy()
    else:
        probs = 1 / (1 + np.exp(-logits))  # sigmoid numpy

    preds = (probs >= threshold).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real", "Fake"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Thêm annotation: tổng số, tỷ lệ sai
    total = cm.sum()
    correct = cm.diagonal().sum()
    accuracy = correct / total if total > 0 else 0
    fp = cm[0, 1]  # Real → predicted Fake
    fn = cm[1, 0]  # Fake → predicted Real
    ax.set_xlabel(
        f"Predicted Label\n\n"
        f"Total: {total} | Accuracy: {accuracy:.1%} | "
        f"FP (Real→Fake): {fp} | FN (Fake→Real): {fn}",
        fontsize=10,
    )

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Confusion matrix saved: {save_path}")
    return save_path


def plot_roc_curve(
    results_dict: dict,
    save_path: str,
    title: str = "ROC Curve — ID vs OOD",
) -> str:
    """Vẽ ROC curves cho nhiều test set trên cùng 1 biểu đồ.

    Args:
        results_dict: Dict[name → {"all_logits": tensor, "all_labels": tensor}]
            Ví dụ: {"In-Domain": id_results, "OOD": ood_results}
        save_path: Đường dẫn file PNG output.
        title: Tiêu đề biểu đồ.

    Returns:
        save_path: Đường dẫn file đã save.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800"]

    for i, (name, results) in enumerate(results_dict.items()):
        logits = results["all_logits"]
        labels = results["all_labels"]

        if isinstance(logits, torch.Tensor):
            probs = torch.sigmoid(logits).cpu().numpy()
        else:
            probs = 1 / (1 + np.exp(-logits))
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels

        # Kiểm tra có ít nhất 2 class
        if len(np.unique(labels_np)) < 2:
            logger.warning(f"Skipping {name}: only 1 class in data")
            continue

        fpr, tpr, _ = roc_curve(labels_np, probs)
        roc_auc = auc(fpr, tpr)

        color = colors[i % len(colors)]
        ax.plot(
            fpr, tpr,
            color=color,
            lw=2,
            label=f"{name} (AUC = {roc_auc:.4f})",
        )

    # Đường chéo (random baseline)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"ROC curve saved: {save_path}")
    return save_path


def plot_per_source_accuracy(
    per_source: dict,
    save_path: str,
    title: str = "Per-Source Accuracy",
) -> str:
    """Vẽ bar chart accuracy cho mỗi source.

    Args:
        per_source: Dict[source → {"accuracy": float, "n": int, ...}]
        save_path: Đường dẫn file PNG output.
        title: Tiêu đề biểu đồ.

    Returns:
        save_path: Đường dẫn file đã save.
    """
    sources = list(per_source.keys())
    accuracies = [per_source[s]["accuracy"] for s in sources]
    counts = [per_source[s]["n"] for s in sources]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color: xanh nếu accuracy >= 0.8, cam nếu >= 0.5, đỏ nếu < 0.5
    colors = []
    for acc in accuracies:
        if acc >= 0.8:
            colors.append("#4CAF50")   # Green
        elif acc >= 0.5:
            colors.append("#FF9800")   # Orange
        else:
            colors.append("#F44336")   # Red

    bars = ax.bar(sources, accuracies, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate mỗi bar: accuracy + count
    for bar, acc, n in zip(bars, accuracies, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.1%}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.3, label="Good threshold")
    ax.legend(fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Per-source accuracy chart saved: {save_path}")
    return save_path
```

> **Giải thích thiết kế:**
>
> 1. `matplotlib.use("Agg")` — không cần GUI, save file trực tiếp
> 2. `plot_roc_curve()` nhận dict → vẽ nhiều curves chồng lên (ID + OOD)
> 3. `plot_per_source_accuracy()` dùng màu: xanh (≥0.8), cam (≥0.5), đỏ (<0.5)
>    → Thấy ngay nguồn nào yếu
> 4. DPI=150 — đủ rõ cho báo cáo, không quá nặng

---

## Bước 4: Implement scripts/test.py

### 4.1 Thiết kế test.py

```
┌──────────────── test.py FLOW ──────────────────────────┐
│                                                          │
│  1. Load config (configs/test.yaml)                      │
│  2. Load model + checkpoint (best.pt)                    │
│  3. Tạo 2 dataloaders: ID + OOD                         │
│  4. Evaluator → chạy ID test                            │
│  5. Evaluator → chạy OOD test                           │
│  6. Visualization: confusion matrix + ROC curve          │
│  7. Save JSON report                                     │
│  8. Print summary table                                  │
│                                                          │
│  Usage:                                                  │
│    python scripts/test.py                                │
│    python scripts/test.py --checkpoint best.pt           │
│    python scripts/test.py configs/test.yaml              │
│                                                          │
│  Output:                                                 │
│    outputs/evaluation/                                   │
│    ├── eval_report.json                                  │
│    ├── confusion_matrix_id.png                           │
│    ├── confusion_matrix_ood.png                          │
│    ├── roc_curve.png                                     │
│    └── per_source_accuracy.png                           │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Code `scripts/test.py`

```python
"""
HolmHz Evaluation Script — đánh giá model trên ID + OOD test set.

Chạy inference trên cả 2 test sets, tính metrics đa chiều,
vẽ confusion matrix + ROC curve, lưu JSON report.

Usage:
    # Default config
    python scripts/test.py

    # Custom checkpoint
    python scripts/test.py model.checkpoint=outputs/checkpoints/best.pt

    # Adjust for local machine (RTX 3050)
    python scripts/test.py data.batch_size=32 data.num_workers=0

Example output:
    outputs/evaluation/
    ├── eval_report.json
    ├── confusion_matrix_id.png
    ├── confusion_matrix_ood.png
    ├── roc_curve.png
    └── per_source_accuracy.png
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()

import holmhz.detectors  # noqa: E402, F401
from holmhz.data import create_dataloader
from holmhz.evaluation import Evaluator
from holmhz.utils.logger import get_logger
from holmhz.utils.registry import DETECTOR_REGISTRY
from holmhz.utils.visualization import (
    plot_confusion_matrix,
    plot_per_source_accuracy,
    plot_roc_curve,
)

logger = get_logger("test")


def main():
    """Main evaluation entry point."""
    # ─── Load config ───
    config_path = "configs/test.yaml"

    if (
        len(sys.argv) > 1
        and not sys.argv[1].startswith("--")
        and "=" not in sys.argv[1]
        and sys.argv[1].endswith(".yaml")
    ):
        config_path = sys.argv[1]

    config = OmegaConf.load(config_path)

    # CLI overrides
    cli_args = [a for a in sys.argv[1:] if a != config_path]
    if cli_args:
        cli_overrides = OmegaConf.from_cli(cli_args)
        config = OmegaConf.merge(config, cli_overrides)

    logger.info(f"Config: {config_path}")

    # ─── Device ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ─── Load model ───
    checkpoint_path = config.model.checkpoint
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Train first: python scripts/train.py")
        sys.exit(1)

    model = DETECTOR_REGISTRY.build(
        config.model.name,
        pretrained=False,
        dropout=config.model.get("dropout", 0.3),
        freeze_backbone=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "N/A")
    best_auc = checkpoint.get("best_metric", "N/A")
    logger.info(f"Loaded: {checkpoint_path} (epoch {epoch}, val_auc {best_auc})")

    # ─── Dataloaders ───
    threshold = config.evaluation.get("threshold", 0.5)
    batch_size = config.data.get("batch_size", 64)
    num_workers = config.data.get("num_workers", 4)

    id_loader = create_dataloader(
        manifest_path=config.data.test_manifest,
        batch_size=batch_size,
        image_size=config.data.image_size,
        is_training=False,
        num_workers=num_workers,
    )

    ood_loader = create_dataloader(
        manifest_path=config.data.ood_manifest,
        batch_size=batch_size,
        image_size=config.data.image_size,
        is_training=False,
        num_workers=num_workers,
    )

    logger.info(f"ID test:  {len(id_loader.dataset)} samples")
    logger.info(f"OOD test: {len(ood_loader.dataset)} samples")

    # ─── Evaluate ID ───
    print("\n" + "=" * 60)
    print("📊 IN-DOMAIN EVALUATION")
    print("=" * 60)

    id_evaluator = Evaluator(model, id_loader, device, threshold)
    id_results = id_evaluator.evaluate()

    # ─── Evaluate OOD ───
    print("\n" + "=" * 60)
    print("📊 OOD EVALUATION")
    print("=" * 60)

    ood_evaluator = Evaluator(model, ood_loader, device, threshold)
    ood_results = ood_evaluator.evaluate()

    # ─── Output dir ───
    output_dir = Path(config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Visualization ───
    print("\n" + "=" * 60)
    print("📈 GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Confusion matrices
    plot_confusion_matrix(
        id_results["all_labels"],
        id_results["all_logits"],
        save_path=str(output_dir / "confusion_matrix_id.png"),
        title="Confusion Matrix — In-Domain Test",
        threshold=threshold,
    )

    plot_confusion_matrix(
        ood_results["all_labels"],
        ood_results["all_logits"],
        save_path=str(output_dir / "confusion_matrix_ood.png"),
        title="Confusion Matrix — OOD Test",
        threshold=threshold,
    )

    # ROC curve (both ID and OOD on same plot)
    plot_roc_curve(
        {
            "In-Domain": id_results,
            "Out-of-Distribution": ood_results,
        },
        save_path=str(output_dir / "roc_curve.png"),
        title="ROC Curve — In-Domain vs OOD",
    )

    # Per-source accuracy (combine ID + OOD)
    all_per_source = {}
    for src, metrics in id_results["per_source"].items():
        all_per_source[f"ID: {src}"] = metrics
    for src, metrics in ood_results["per_source"].items():
        all_per_source[f"OOD: {src}"] = metrics

    plot_per_source_accuracy(
        all_per_source,
        save_path=str(output_dir / "per_source_accuracy.png"),
        title="Per-Source Accuracy — All Test Sets",
    )

    # ─── JSON Report ───
    report = {
        "model": config.model.name,
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": epoch,
        "checkpoint_val_auc": float(best_auc) if isinstance(best_auc, (int, float)) else best_auc,
        "threshold": threshold,
        "timestamp": datetime.now().isoformat(),
        "in_domain": {
            "manifest": config.data.test_manifest,
            "total": id_results["total"],
            "overall": id_results["overall"],
            "per_source": {
                src: {k: v for k, v in metrics.items()}
                for src, metrics in id_results["per_source"].items()
            },
        },
        "ood": {
            "manifest": config.data.ood_manifest,
            "total": ood_results["total"],
            "overall": ood_results["overall"],
            "per_source": {
                src: {k: v for k, v in metrics.items()}
                for src, metrics in ood_results["per_source"].items()
            },
        },
    }

    report_path = output_dir / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Report saved: {report_path}")

    # ─── Summary table ───
    print("\n" + "=" * 70)
    print("📋 EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nModel: {config.model.name}")
    print(f"Checkpoint: {checkpoint_path} (epoch {epoch})")
    print(f"Threshold: {threshold}")

    print(f"\n{'Set':<25} {'AUC':<8} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'N':<6}")
    print("-" * 70)

    for name, results in [("In-Domain", id_results), ("OOD", ood_results)]:
        o = results["overall"]
        print(
            f"{name:<25} "
            f"{o['auc']:<8.4f} "
            f"{o['accuracy']:<8.4f} "
            f"{o['f1']:<8.4f} "
            f"{o['precision']:<8.4f} "
            f"{o['recall']:<8.4f} "
            f"{results['total']:<6}"
        )

    print(f"\n{'Source':<25} {'Acc':<8} {'AUC':<8} {'N':<6} {'Real':<6} {'Fake':<6}")
    print("-" * 70)

    for src, metrics in sorted(id_results["per_source"].items()):
        print(
            f"  ID/{src:<22} "
            f"{metrics['accuracy']:<8.4f} "
            f"{metrics['auc']:<8.4f} "
            f"{metrics['n']:<6} "
            f"{metrics['n_real']:<6} "
            f"{metrics['n_fake']:<6}"
        )

    for src, metrics in sorted(ood_results["per_source"].items()):
        print(
            f"  OOD/{src:<21} "
            f"{metrics['accuracy']:<8.4f} "
            f"{metrics['auc']:<8.4f} "
            f"{metrics['n']:<6} "
            f"{metrics['n_real']:<6} "
            f"{metrics['n_fake']:<6}"
        )

    print("=" * 70)

    # ─── OOD Analysis ───
    print("\n🔍 OOD FAILURE ANALYSIS")
    print("-" * 50)

    ood_overall = ood_results["overall"]
    id_overall = id_results["overall"]
    gap = id_overall["auc"] - ood_overall["auc"]

    print(f"ID AUC:  {id_overall['auc']:.4f}")
    print(f"OOD AUC: {ood_overall['auc']:.4f}")
    print(f"Gap:     {gap:.4f} ({'⚠️ LARGE' if gap > 0.15 else '✅ acceptable'})")

    # Tìm source yếu nhất
    worst_source = min(
        ood_results["per_source"].items(),
        key=lambda x: x[1]["accuracy"],
    )
    print(f"\nWeakest OOD source: {worst_source[0]} "
          f"(Acc: {worst_source[1]['accuracy']:.4f}, N: {worst_source[1]['n']})")

    # False Positive analysis (Real → FAKE)
    ood_precision = ood_overall["precision"]
    ood_recall = ood_overall["recall"]
    if ood_precision < ood_recall:
        print("\n⚠️ False Positive dominant: Model bias dự đoán FAKE")
        print("   → Nhiều ảnh Real OOD bị đoán nhầm thành FAKE")
        print("   → Khớp với smoke test Task 1.6 (5/5 Real → FAKE)")
    else:
        print("\n⚠️ False Negative dominant: Model bỏ sót nhiều Fake")
        print("   → Nhiều ảnh Fake OOD bị đoán thành Real")

    print(f"\n📁 All outputs saved to: {output_dir}/")
    print("Done! ✅")


if __name__ == "__main__":
    main()
```

> **Giải thích quan trọng:**
>
> - `test.py` pattern giống `train.py`: OmegaConf config, CLI overrides, same model loading
> - Chạy 2 evaluator riêng: ID và OOD → so sánh
> - ROC curve vẽ cả 2 trên cùng 1 biểu đồ → thấy gap
> - OOD failure analysis: tự động phân tích FP vs FN dominant
> - JSON report: lưu tất cả metrics → dùng lại trong Task 2.2, 4.1

---

## Bước 5: Update configs/test.yaml

### 5.1 Sửa `configs/test.yaml`

File hiện tại gần đúng rồi, chỉ cần sửa vài chỗ:

```yaml
# ============================================
# HolmHz Test / Evaluation Configuration
# ============================================

model:
  name: efficientnet_b0
  checkpoint: outputs/checkpoints/best.pt # ← SỬA: best.pt (not best_model.pt)
  dropout: 0.3

data:
  test_manifest: data/manifests/test_id.json
  ood_manifest: data/manifests/test_ood.json
  image_size: 224
  batch_size: 32 # ← SỬA: 32 (RTX 3050 đủ vì no gradient)
  num_workers: 0 # ← SỬA: 0 cho Windows

evaluation:
  metrics: [auc, accuracy, f1, precision, recall]
  threshold: 0.5
  save_predictions: true
  output_dir: outputs/evaluation

wandb:
  project: holmhz
  log_results: true
```

> **Thay đổi so với file cũ:**
>
> | Field       | Cũ            | Mới     | Lý do                                     |
> | ----------- | ------------- | ------- | ----------------------------------------- |
> | checkpoint  | best_model.pt | best.pt | Tên file thực tế từ training              |
> | dropout     | (không có)    | 0.3     | Cần cho DETECTOR_REGISTRY.build()         |
> | batch_size  | 64            | 32      | RTX 3050 chỉ 4GB (inference vẫn cần VRAM) |
> | num_workers | 4             | 0       | Windows safe                              |

---

## Bước 6: Chạy evaluation trên local

### 6.1 Kiểm tra trước khi chạy

```bash
# 1. Checkpoint tồn tại?
ls outputs/checkpoints/best.pt
# → outputs/checkpoints/best.pt  (48.5 MB)

# 2. Test manifests tồn tại?
python -c "
import json
for name in ['test_id', 'test_ood']:
    d = json.load(open(f'data/manifests/{name}.json'))
    labels = [x['label'] for x in d]
    print(f'{name}: {len(d)} samples, {sum(1 for l in labels if l==0)} real, {sum(1 for l in labels if l==1)} fake')
"
# test_id: 3975 samples, 1797 real, 2178 fake
# test_ood: 1180 samples, 600 real, 580 fake

# 3. Processed images tồn tại?
python -c "
import json, os
d = json.load(open('data/manifests/test_id.json'))
exists = sum(1 for x in d if os.path.exists(x['path']))
print(f'test_id: {exists}/{len(d)} files exist')
"
# test_id: 3975/3975 files exist

# (Tương tự cho test_ood)
python -c "
import json, os
d = json.load(open('data/manifests/test_ood.json'))
exists = sum(1 for x in d if os.path.exists(x['path']))
print(f'test_ood: {exists}/{len(d)} files exist')
"
```

### 6.2 Chạy evaluation

```bash
# Chạy evaluation (Windows local, RTX 3050)
python scripts/test.py data.num_workers=0 data.batch_size=32
```

**Nếu OOM (out of memory):**

```bash
# Giảm batch size
python scripts/test.py data.num_workers=0 data.batch_size=16

# Nếu vẫn OOM, dùng CPU (chậm hơn nhưng chắc chắn chạy)
# CPU inference ~5,155 ảnh → ~5-10 phút
CUDA_VISIBLE_DEVICES="" python scripts/test.py data.num_workers=0 data.batch_size=32
```

### 6.3 Kết quả kỳ vọng

```
============================================================
📊 IN-DOMAIN EVALUATION
============================================================
Evaluating 3975 samples (125 batches)...
Overall — AUC: 0.998x, Acc: 0.98xx, F1: 0.98xx

============================================================
📊 OOD EVALUATION
============================================================
Evaluating 1180 samples (37 batches)...
Overall — AUC: 0.6xxx, Acc: 0.5xxx, F1: 0.6xxx  ← THẤP (expected)

============================================================
📋 EVALUATION SUMMARY
============================================================

Model: efficientnet_b0
Checkpoint: outputs/checkpoints/best.pt (epoch 11)
Threshold: 0.5

Set                      AUC      Acc      F1       Prec     Rec      N
----------------------------------------------------------------------
In-Domain                0.998x   0.98xx   0.98xx   0.98xx   0.98xx   3975
OOD                      0.6xxx   0.5xxx   0.6xxx   0.5xxx   0.9xxx   1180

Source                   Acc      AUC      N      Real   Fake
----------------------------------------------------------------------
  ID/cifake              0.99xx   0.99xx   2100   1050   1050
  ID/ffhq                0.99xx   0.5000   750    750    0     ← real only
  ID/sd15                0.99xx   0.5000   375    0      375   ← fake only
  ID/stylegan            0.99xx   0.5000   750    0      750   ← fake only
  OOD/flux               0.99xx   0.5000   80     0      80    ← fake only
  OOD/real_camera        0.0xxx   0.5000   100    100    0     ← ⚠️ GẦN 0%
  OOD/real_pexels        0.0xxx   0.5000   500    500    0     ← ⚠️ GẦN 0%
  OOD/tristanzhang_fake  0.99xx   0.5000   500    0      500   ← fake only
======================================================================

🔍 OOD FAILURE ANALYSIS
--------------------------------------------------
ID AUC:  0.998x
OOD AUC: 0.6xxx
Gap:     0.3xxx (⚠️ LARGE)

Weakest OOD source: real_camera (Acc: 0.0xxx, N: 100)

⚠️ False Positive dominant: Model bias dự đoán FAKE
   → Nhiều ảnh Real OOD bị đoán nhầm thành FAKE
   → Khớp với smoke test Task 1.6 (5/5 Real → FAKE)
```

> **LƯU Ý QUAN TRỌNG:**
>
> Một số source chỉ có 1 label (ffhq chỉ Real, stylegan chỉ Fake).
> → AUC không tính được (cần ≥ 2 class) → hiện 0.5.
> → Dùng Accuracy thay vì AUC cho per-source đơn class.

---

## Bước 7: Phân tích kết quả OOD

### 7.1 Đọc JSON report

```bash
# Xem report
python -m json.tool outputs/evaluation/eval_report.json
```

### 7.2 Phân tích chi tiết

```
┌──────────── KẾT QUẢ DỰ KIẾN — PHÂN TÍCH ──────────────┐
│                                                           │
│  IN-DOMAIN (3,975 ảnh):                                  │
│  ─────────────────────                                   │
│  AUC: ~0.998 → Gần bằng val AUC 0.9983                  │
│  → ✅ Model rất tốt trên data cùng distribution          │
│  → Milestone 2 target AUC ≥ 0.90 → VƯỢT                 │
│                                                           │
│  OOD (1,180 ảnh):                                        │
│  ──────────────                                          │
│  AUC: ~0.60-0.70 → THẤP hơn nhiều so với ID              │
│                                                           │
│  Per-source breakdown:                                   │
│  ┌────────────────┬────────┬──────────────────────────┐  │
│  │ Source          │ Acc    │ Giải thích               │  │
│  ├────────────────┼────────┼──────────────────────────┤  │
│  │ flux           │ ~99%   │ Fake → đoán FAKE ✅      │  │
│  │                │        │ (model BIAS fake anyway)  │  │
│  │ tristanzhang   │ ~99%   │ Fake → đoán FAKE ✅      │  │
│  │                │        │ (may detect artifacts)    │  │
│  │ real_pexels    │ ~5%    │ Real → đoán FAKE ❌      │  │
│  │                │        │ Stock photos, non-face    │  │
│  │ real_camera    │ ~2%    │ Real → đoán FAKE ❌      │  │
│  │                │        │ Camera photos, giống      │  │
│  │                │        │ smoke test imgs/Real/     │  │
│  └────────────────┴────────┴──────────────────────────┘  │
│                                                           │
│  ⚠️ PATTERN RÕ RÀNG:                                     │
│  → Fake OOD: accuracy CAO (model bias FAKE → đúng)       │
│  → Real OOD: accuracy CỰC THẤP (model bias FAKE → sai!)  │
│                                                           │
│  → Model không thực sự "phát hiện Fake"                  │
│  → Model học "ảnh nào giống training = Real, khác = Fake" │
│  → Đây là SHORTCUT LEARNING                              │
│                                                           │
│  KẾT LUẬN CHO BÁO CÁO:                                   │
│  ─────────────────────                                    │
│  1. In-domain: Xuất sắc (AUC ~0.998)                    │
│  2. OOD: Kém (AUC ~0.65)                                │
│  3. Root cause: Shortcut learning + domain bias          │
│  4. Limitations: Chưa generalize ngoài training domain   │
│  5. Đề xuất: Diverse training data, augmentation,        │
│     adversarial training, domain adaptation               │
│                                                           │
│  → Kết quả xấu CŨNG là đóng góp khoa học!               │
│  → Hội đồng đánh giá cao phân tích sâu hơn AUC đẹp     │
└───────────────────────────────────────────────────────────┘
```

### 7.3 So sánh với target milestones

```
┌──────── MILESTONE 2 TARGETS ────────────────┐
│                                               │
│  Target         │ Kết quả     │ Status       │
│  ───────────────┼─────────────┼────────────  │
│  ID AUC ≥ 0.90  │ ~0.998      │ ✅ VƯỢT      │
│  OOD AUC ≥ 0.75 │ ~0.65 (?)   │ ❌ CHƯA ĐẠT  │
│  Bảng so sánh   │ (Task 2.2)  │ ⬜           │
│  XAI gallery    │ (Task 2.3)  │ ⬜           │
│                                               │
│  → ID đạt nhưng OOD chưa → EXPECTED!         │
│  → Task 2.2 sẽ so sánh: SOTA có tốt hơn?    │
│  → Task 2.3 sẽ giải thích: model nhìn gì?   │
└───────────────────────────────────────────────┘
```

---

## Bước 8: Unit tests

### 8.1 Test structure

```
tests/
├── test_metrics.py        ← Test F1, Precision, Recall (MỚI)
├── test_evaluator.py      ← Test Evaluator class (MỚI)
└── test_visualization.py  ← Test visualization functions (MỚI)
```

### 8.2 Code `tests/test_metrics.py`

```python
"""Tests cho metrics module — F1, Precision, Recall."""

import pytest
import torch

from holmhz.metrics import compute_accuracy, compute_auc
from holmhz.metrics.f1 import compute_f1
from holmhz.metrics.precision import compute_precision
from holmhz.metrics.recall import compute_recall


class TestPrecision:
    """Test compute_precision."""

    def test_perfect_precision(self):
        """Tất cả dự đoán FAKE đều đúng → precision = 1.0."""
        logits = torch.tensor([3.0, -3.0, 3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_precision(logits, labels) == pytest.approx(1.0, abs=0.01)

    def test_low_precision(self):
        """Nhiều False Positive → precision thấp."""
        # 2 TP + 2 FP → precision = 2/4 = 0.5
        logits = torch.tensor([3.0, 3.0, 3.0, 3.0])  # All predict FAKE
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])   # But 2 are Real
        assert compute_precision(logits, labels) == pytest.approx(0.5, abs=0.01)

    def test_no_positive_predictions(self):
        """Model không dự đoán FAKE nào → precision = 0.0."""
        logits = torch.tensor([-3.0, -3.0, -3.0])
        labels = torch.tensor([1.0, 1.0, 0.0])
        assert compute_precision(logits, labels) == 0.0

    def test_2d_logits(self):
        """Logits shape [N, 1] cũng hoạt động."""
        logits = torch.tensor([[3.0], [-3.0]])
        labels = torch.tensor([1.0, 0.0])
        assert compute_precision(logits, labels) == pytest.approx(1.0, abs=0.01)


class TestRecall:
    """Test compute_recall."""

    def test_perfect_recall(self):
        """Tìm ra tất cả Fake → recall = 1.0."""
        logits = torch.tensor([3.0, -3.0, 3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_recall(logits, labels) == pytest.approx(1.0, abs=0.01)

    def test_low_recall(self):
        """Bỏ sót nhiều Fake → recall thấp."""
        # 1 TP + 1 FN → recall = 1/2 = 0.5
        logits = torch.tensor([3.0, -3.0])
        labels = torch.tensor([1.0, 1.0])  # Both Fake, but only 1 detected
        assert compute_recall(logits, labels) == pytest.approx(0.5, abs=0.01)

    def test_no_actual_fakes(self):
        """Không có Fake nào trong data → recall = 0.0."""
        logits = torch.tensor([3.0, -3.0])
        labels = torch.tensor([0.0, 0.0])  # All Real
        assert compute_recall(logits, labels) == 0.0


class TestF1:
    """Test compute_f1."""

    def test_perfect_f1(self):
        """Precision và Recall đều = 1.0 → F1 = 1.0."""
        logits = torch.tensor([3.0, -3.0, 3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_f1(logits, labels) == pytest.approx(1.0, abs=0.01)

    def test_zero_f1(self):
        """Prediction hoàn toàn sai → F1 = 0.0."""
        logits = torch.tensor([-3.0, -3.0])  # All predict Real
        labels = torch.tensor([1.0, 1.0])     # All actually Fake
        assert compute_f1(logits, labels) == 0.0

    def test_balanced_f1(self):
        """Precision=Recall=0.5 → F1=0.5."""
        # 1 TP, 1 FP, 1 FN → Prec=0.5, Rec=0.5, F1=0.5
        logits = torch.tensor([3.0, 3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0])
        assert compute_f1(logits, labels) == pytest.approx(0.5, abs=0.05)

    def test_consistency_with_precision_recall(self):
        """F1 = 2 * (P * R) / (P + R)."""
        logits = torch.tensor([3.0, 3.0, -3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        p = compute_precision(logits, labels)
        r = compute_recall(logits, labels)
        expected_f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        assert compute_f1(logits, labels) == pytest.approx(expected_f1, abs=0.01)
```

### 8.3 Code `tests/test_evaluator.py`

```python
"""Tests cho Evaluator class."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from holmhz.evaluation import Evaluator


class FakeModel(torch.nn.Module):
    """Model giả — trả logits cố định cho testing."""

    def __init__(self, predictions: torch.Tensor):
        super().__init__()
        self._predictions = predictions
        self._idx = 0

    def forward(self, x):
        batch_size = x.shape[0]
        logits = self._predictions[self._idx : self._idx + batch_size]
        self._idx += batch_size
        return logits.unsqueeze(-1)


class FakeDataset(torch.utils.data.Dataset):
    """Dataset giả — trả dict giống ImageDataset."""

    def __init__(self, images, labels, sources):
        self.images = images
        self.labels = labels
        self.sources = sources

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
            "source": self.sources[idx],
            "path": f"test_{idx}.jpg",
        }


class TestEvaluator:
    """Test Evaluator class."""

    def _make_evaluator(self, logits, labels, sources):
        """Helper: tạo Evaluator với data giả."""
        images = torch.randn(len(labels), 3, 224, 224)
        dataset = FakeDataset(images, labels, sources)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        model = FakeModel(logits)
        return Evaluator(model, loader, torch.device("cpu"))

    def test_perfect_predictions(self):
        """Model dự đoán hoàn hảo → AUC = 1.0."""
        logits = torch.tensor([3.0, -3.0, 3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        sources = ["src_a", "src_a", "src_b", "src_b"]

        evaluator = self._make_evaluator(logits, labels, sources)
        results = evaluator.evaluate()

        assert results["overall"]["auc"] == pytest.approx(1.0, abs=0.01)
        assert results["overall"]["accuracy"] == pytest.approx(1.0, abs=0.01)
        assert results["total"] == 4

    def test_per_source_breakdown(self):
        """Per-source metrics được tính riêng."""
        logits = torch.tensor([3.0, -3.0, 3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        sources = ["cifake", "cifake", "ffhq", "ffhq"]

        evaluator = self._make_evaluator(logits, labels, sources)
        results = evaluator.evaluate()

        assert "cifake" in results["per_source"]
        assert "ffhq" in results["per_source"]
        assert results["per_source"]["cifake"]["n"] == 2
        assert results["per_source"]["ffhq"]["n"] == 2

    def test_returns_raw_data(self):
        """Evaluator trả về raw logits/labels/sources cho visualization."""
        logits = torch.tensor([1.0, -1.0])
        labels = torch.tensor([1.0, 0.0])
        sources = ["src_a", "src_b"]

        evaluator = self._make_evaluator(logits, labels, sources)
        results = evaluator.evaluate()

        assert "all_logits" in results
        assert "all_labels" in results
        assert "all_sources" in results
        assert len(results["all_logits"]) == 2

    def test_single_class_source(self):
        """Source chỉ có 1 class → AUC = 0.5 (edge case)."""
        logits = torch.tensor([3.0, 3.0, -3.0, -3.0])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        sources = ["fake_only", "fake_only", "real_only", "real_only"]

        evaluator = self._make_evaluator(logits, labels, sources)
        results = evaluator.evaluate()

        # fake_only source: chỉ label=1 → AUC = 0.5
        assert results["per_source"]["fake_only"]["auc"] == 0.5
        assert results["per_source"]["real_only"]["auc"] == 0.5
```

### 8.4 Code `tests/test_visualization.py`

```python
"""Tests cho visualization module."""

import os
from pathlib import Path

import pytest
import numpy as np
import torch

from holmhz.utils.visualization import (
    plot_confusion_matrix,
    plot_per_source_accuracy,
    plot_roc_curve,
)


@pytest.fixture
def tmp_dir(tmp_path):
    """Thư mục tạm cho output."""
    return str(tmp_path)


class TestConfusionMatrix:
    """Test plot_confusion_matrix."""

    def test_saves_file(self, tmp_dir):
        """Tạo file PNG thành công."""
        labels = torch.tensor([0, 0, 1, 1])
        logits = torch.tensor([-3.0, -3.0, 3.0, 3.0])
        path = os.path.join(tmp_dir, "cm.png")

        result = plot_confusion_matrix(labels, logits, path)

        assert os.path.exists(result)
        assert result.endswith(".png")

    def test_creates_directory(self, tmp_dir):
        """Tự tạo thư mục nếu chưa có."""
        labels = torch.tensor([0, 1])
        logits = torch.tensor([-3.0, 3.0])
        path = os.path.join(tmp_dir, "subdir", "cm.png")

        result = plot_confusion_matrix(labels, logits, path)
        assert os.path.exists(result)


class TestROCCurve:
    """Test plot_roc_curve."""

    def test_saves_file(self, tmp_dir):
        """Tạo file PNG thành công."""
        results = {
            "Test": {
                "all_logits": torch.tensor([3.0, -3.0, 2.0, -2.0]),
                "all_labels": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            }
        }
        path = os.path.join(tmp_dir, "roc.png")

        result = plot_roc_curve(results, path)
        assert os.path.exists(result)

    def test_multiple_curves(self, tmp_dir):
        """Vẽ nhiều curves chồng lên."""
        results = {
            "ID": {
                "all_logits": torch.tensor([3.0, -3.0, 2.0, -2.0]),
                "all_labels": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            },
            "OOD": {
                "all_logits": torch.tensor([1.0, 1.0, -1.0, -1.0]),
                "all_labels": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            },
        }
        path = os.path.join(tmp_dir, "roc_multi.png")

        result = plot_roc_curve(results, path)
        assert os.path.exists(result)


class TestPerSourceAccuracy:
    """Test plot_per_source_accuracy."""

    def test_saves_file(self, tmp_dir):
        """Tạo file PNG thành công."""
        per_source = {
            "cifake": {"accuracy": 0.99, "n": 2100},
            "real_camera": {"accuracy": 0.05, "n": 100},
        }
        path = os.path.join(tmp_dir, "per_source.png")

        result = plot_per_source_accuracy(per_source, path)
        assert os.path.exists(result)
```

> **Chạy tests:**
>
> ```bash
> pytest tests/test_metrics.py tests/test_evaluator.py tests/test_visualization.py -v
> ```

---

## Bước 9: Document results (CONTEXT.md)

Sau khi chạy evaluation, thêm vào `docs/CONTEXT.md`:

```markdown
## 15. Evaluation Pipeline (Task 2.1) — ✅ COMPLETED DD/MM/2026

### In-Domain Test (test_id.json)

| Metric    | Value |
| --------- | ----- |
| AUC       | ?     |
| Accuracy  | ?     |
| F1        | ?     |
| Precision | ?     |
| Recall    | ?     |
| Total     | 3,975 |

### OOD Test (test_ood.json)

| Metric    | Value |
| --------- | ----- |
| AUC       | ?     |
| Accuracy  | ?     |
| F1        | ?     |
| Precision | ?     |
| Recall    | ?     |
| Total     | 1,180 |

### Per-Source Breakdown

| Source       | Accuracy | N     | Type | Notes               |
| ------------ | -------- | ----- | ---- | ------------------- |
| cifake       | ?        | 2,100 | ID   |                     |
| ffhq         | ?        | 750   | ID   | Real only           |
| sd15         | ?        | 375   | ID   | Fake only           |
| stylegan     | ?        | 750   | ID   | Fake only           |
| flux         | ?        | 80    | OOD  | Fake only           |
| tristanzhang | ?        | 500   | OOD  | Fake only           |
| real_pexels  | ?        | 500   | OOD  | Real only, non-face |
| real_camera  | ?        | 100   | OOD  | Real only, camera   |

### OOD Failure Analysis

- **ID-OOD AUC Gap**: ? (⚠️ LARGE if > 0.15)
- **Weakest source**: real_camera (Acc: ?, N: 100)
- **Error type**: False Positive dominant (Real → FAKE)
- **Root cause**: Shortcut learning — model learned preprocessing artifacts
- **Limitations**: Model does not generalize outside cifake/ffhq domain

### Artifacts

- `outputs/evaluation/eval_report.json`
- `outputs/evaluation/confusion_matrix_id.png`
- `outputs/evaluation/confusion_matrix_ood.png`
- `outputs/evaluation/roc_curve.png`
- `outputs/evaluation/per_source_accuracy.png`
```

---

## Bước 10: Commit & PR

```bash
# Stage new files
git add src/holmhz/metrics/f1.py
git add src/holmhz/metrics/precision.py
git add src/holmhz/metrics/recall.py
git add src/holmhz/metrics/__init__.py
git add src/holmhz/evaluation/evaluator.py
git add src/holmhz/evaluation/__init__.py
git add src/holmhz/utils/visualization.py
git add scripts/test.py
git add configs/test.yaml
git add tests/test_metrics.py
git add tests/test_evaluator.py
git add tests/test_visualization.py
git add docs/CONTEXT.md
git add docs/tasks/TASK_2.1_EVALUATION_PIPELINE.md
git add docs/guides/GUIDE_TASK_2.1_EVALUATION_PIPELINE.md

# DON'T add outputs/ (large files, not needed in git)
# .gitignore should already exclude outputs/

# Commit
git commit -m "feat(s2): evaluation pipeline — ID + OOD eval with per-source breakdown

- Add metrics: compute_f1, compute_precision, compute_recall
- Implement Evaluator class with per-source breakdown
- Implement visualization: confusion_matrix, roc_curve, per_source_accuracy
- Implement scripts/test.py — end-to-end evaluation CLI
- Update configs/test.yaml
- In-Domain AUC: ??? (target ≥ 0.90 ✅)
- OOD AUC: ??? (target ≥ 0.75 ???)
- OOD failure analysis: FP dominant (shortcut learning)
- Add tests: test_metrics, test_evaluator, test_visualization"

# Push
git push -u origin feat/s2/evaluation-pipeline
```

---

## Checklist hoàn thành

```
TASK 2.1 — EVALUATION PIPELINE

Subtask 2.1.1: Mở rộng metrics module
  [ ] compute_f1() implemented + tested
  [ ] compute_precision() implemented + tested
  [ ] compute_recall() implemented + tested
  [ ] __init__.py updated

Subtask 2.1.2: Evaluator class
  [ ] src/holmhz/evaluation/evaluator.py implemented
  [ ] Overall metrics computed
  [ ] Per-source breakdown works
  [ ] Returns raw logits/labels for visualization

Subtask 2.1.3: Visualization
  [ ] plot_confusion_matrix() → PNG
  [ ] plot_roc_curve() → PNG (ID + OOD overlay)
  [ ] plot_per_source_accuracy() → PNG (bar chart with colors)

Subtask 2.1.4: scripts/test.py
  [ ] Loads config + model + checkpoint
  [ ] Evaluates ID test set (3,975 ảnh)
  [ ] Evaluates OOD test set (1,180 ảnh)
  [ ] Generates all visualizations
  [ ] Saves JSON report
  [ ] Prints summary table + OOD analysis

Subtask 2.1.5: configs/test.yaml
  [ ] Checkpoint path corrected (best.pt)
  [ ] batch_size, num_workers adjusted for local

Subtask 2.1.6: Run evaluation
  [ ] ID AUC ≥ 0.99 (kỳ vọng)
  [ ] OOD AUC measured (even if low)
  [ ] All visualizations generated
  [ ] JSON report saved

Subtask 2.1.7: OOD analysis
  [ ] Per-source breakdown documented
  [ ] Weakest source identified (real_camera)
  [ ] Error type analyzed (FP dominant)
  [ ] Root cause documented (shortcut learning)
  [ ] CONTEXT.md updated

Subtask 2.1.8: Unit tests
  [ ] test_metrics.py: ≥ 8 tests
  [ ] test_evaluator.py: ≥ 4 tests
  [ ] test_visualization.py: ≥ 4 tests
  [ ] All tests pass

Branch & PR
  [ ] Branch: feat/s2/evaluation-pipeline
  [ ] All changes committed
  [ ] PR created with evaluation results
```

---

## Troubleshooting

### FileNotFoundError: checkpoint not found

```
Triệu chứng: Checkpoint not found: outputs/checkpoints/best.pt

Fix:
1. Kiểm tra tên file: ls outputs/checkpoints/
   → Có thể tên khác: hp_lr1e4_best.pt, phase2_best.pt
2. Copy đúng tên:
   cp outputs/checkpoints/hp_lr1e4_best.pt outputs/checkpoints/best.pt
3. Hoặc chỉ định qua CLI:
   python scripts/test.py model.checkpoint=outputs/checkpoints/hp_lr1e4_best.pt
```

### FileNotFoundError: test images not found

```
Triệu chứng: FileNotFoundError khi load test images

Fix:
1. Kiểm tra paths trong manifest:
   python -c "import json; d=json.load(open('data/manifests/test_id.json'));
   print(d[0]['path'])"

2. Kiểm tra file tồn tại:
   python -c "import json, os; d=json.load(open('data/manifests/test_id.json'));
   print(os.path.exists(d[0]['path']))"

3. Nếu path sai → update manifest hoặc tạo symlink:
   ln -s /actual/path data/processed
```

### OOM (CUDA Out of Memory)

```
Triệu chứng: RuntimeError: CUDA out of memory

Fix:
1. Giảm batch:
   python scripts/test.py data.batch_size=16

2. Giảm tiếp:
   python scripts/test.py data.batch_size=8

3. Dùng CPU (chậm nhưng chắc chắn):
   set CUDA_VISIBLE_DEVICES=  (Windows)
   python scripts/test.py data.batch_size=32
```

### ModuleNotFoundError: No module named 'holmhz.evaluation'

```
Triệu chứng: ModuleNotFoundError khi import Evaluator

Fix:
1. Kiểm tra __init__.py tồn tại:
   ls src/holmhz/evaluation/__init__.py

2. Cài lại package:
   pip install -e .

3. Hoặc thêm sys.path:
   import sys; sys.path.insert(0, ".")
```

### matplotlib error: Tkinter not found

```
Triệu chứng: _tkinter.TclError hoặc ImportError: tkinter

Fix:
→ Đã handle bằng matplotlib.use("Agg") trong visualization.py
→ Nếu vẫn lỗi: kiểm tra import order (Agg phải set TRƯỚC import pyplot)
```

### AUC per-source = 0.5 cho single-class source

```
Triệu chứng: ffhq AUC = 0.5, stylegan AUC = 0.5

Giải thích: KHÔNG PHẢI LỖI!
→ ffhq chỉ có Real images (label=0)
→ stylegan chỉ có Fake images (label=1)
→ AUC cần ≥ 2 classes → trả 0.5 (undefined)
→ Dùng Accuracy thay vì AUC cho single-class source
→ Code đã handle edge case này (trong compute_auc)
```

### JSON report values are 'NaN'

```
Triệu chứng: "auc": NaN trong eval_report.json

Fix:
1. NaN xuất hiện khi division by zero (e.g., 0 TP + 0 FP)
2. Kiểm tra source data: có ít nhất 1 sample mỗi class?
3. compute_precision() đã có edge case handling (trả 0.0)
→ Nếu vẫn NaN: thêm float() cast trước khi JSON dump
```

---

## Mối liên hệ với các Task tiếp theo

```
┌──────────────── FROM TASK 2.1 → NEXT TASKS ───────────────┐
│                                                              │
│  Task 2.1 OUTPUT                 │ Dùng ở đâu?              │
│  ────────────────────────────────┼──────────────────────────│
│  eval_report.json                │ Task 2.2: Benchmark so   │
│                                  │ sánh với 3 SOTA models   │
│                                  │ Task 4.1: Báo cáo        │
│                                  │                          │
│  confusion_matrix_*.png          │ Task 4.1: Hình trong     │
│  roc_curve.png                   │ báo cáo chapter "Results"│
│  per_source_accuracy.png         │ Task 4.2: Slide bảo vệ  │
│                                  │                          │
│  OOD failure analysis            │ Task 4.1: Limitations    │
│                                  │ section trong báo cáo    │
│                                  │                          │
│  Evaluator class                 │ Task 2.2: Dùng lại cho   │
│                                  │ benchmark SOTA models    │
│                                  │ Task 2.3: Kết hợp với    │
│                                  │ Grad-CAM evaluation      │
│                                  │                          │
│  Per-source breakdown            │ Task 2.3: Xem Grad-CAM   │
│                                  │ cho source yếu nhất      │
│                                  │ → model nhìn gì sai?     │
│                                  │                          │
│  ⚡ Task 2.1 là NỀN TẢNG cho     │                          │
│  toàn bộ Sprint 2 + Sprint 4!    │                          │
└──────────────────────────────────┴──────────────────────────┘
```

### Câu hỏi Task 2.1 trả lời vs Task tiếp theo

```
Task 2.1: "Model tốt cỡ nào? Yếu ở đâu?"
  → ID AUC ~0.998 ✅, OOD AUC ~0.65 ❌

Task 2.2: "Các model SOTA có tốt hơn OOD không?"
  → So sánh CNNDetection, UniversalFakeDetect, DeepfakeBench

Task 2.3: "Model nhìn vào đâu khi dự đoán sai?"
  → Grad-CAM trên real_camera images → model nhìn vùng nào?
  → Nếu nhìn background = shortcut ✅ (confirm hypothesis)

Task 4.1: "Viết gì trong báo cáo?"
  → Results: bảng metrics + hình visualization
  → Limitations: OOD failure, shortcut learning
  → Future work: diverse data, domain adaptation
```

---

> **🎯 Key takeaway:**
>
> Task 2.1 không chỉ là "chạy model trên test set". Nó là bước **chứng minh khoa học**:
>
> 1. Model mạnh ở đâu? (ID AUC ~0.998)
> 2. Model yếu ở đâu? (OOD — real_camera, real_pexels)
> 3. Tại sao yếu? (Shortcut learning, domain bias)
> 4. Cải thiện thế nào? (Diverse data, augmentation — đề xuất cho Future Work)
>
> Kết quả xấu trên OOD **KHÔNG PHẢI thất bại** — nó là đóng góp khoa học
> quan trọng cho báo cáo. Hội đồng đánh giá cao sinh viên hiểu và phân tích
> limitations thay vì chỉ report con số đẹp.
