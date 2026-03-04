# 📖 HƯỚNG DẪN CHI TIẾT TASK 1.4: MODEL ARCHITECTURE

> **Dành cho**: Lê Văn Hoàng — người chưa có nền tảng ML/DL, học qua thực hành  
> **Triết lý**: Mỗi bước không chỉ hướng dẫn **làm gì** mà giải thích **tại sao làm vậy**  
> **Thời gian**: ~3-4 ngày (có thể làm song song với Task 1.3)  
> **Tiền đề**: Task 1.1 Environment Setup ✅ | Task 1.3 Data Pipeline ✅ (optional nhưng đã xong)  
> **Tham chiếu**: [TASK_1.4_MODEL_ARCHITECTURE.md](../tasks/TASK_1.4_MODEL_ARCHITECTURE.md) | [PROJECT_PLAN.md](../PROJECT_PLAN.md) Section 3
>
> **Output**: Model EfficientNet-B0 detector nhận `[B, 3, 224, 224]` → trả về logits `[B, 1]`

---

## 📋 Mục lục

- [Bức tranh tổng thể: Model Architecture nằm ở đâu?](#bức-tranh-tổng-thể-model-architecture-nằm-ở-đâu)
- [Tại sao chọn EfficientNet-B0?](#tại-sao-chọn-efficientnet-b0)
- [Kiến thức nền: Transfer Learning](#kiến-thức-nền-transfer-learning)
- [Kiến thức nền: EfficientNet Architecture](#kiến-thức-nền-efficientnet-architecture)
- [Kiến thức nền: Backbone + Head Pattern](#kiến-thức-nền-backbone--head-pattern)
- [Kiến thức nền: Registry Pattern](#kiến-thức-nền-registry-pattern)
- [Kiến thức nền: Logits vs Probabilities vs BCEWithLogitsLoss](#kiến-thức-nền-logits-vs-probabilities-vs-bcewithlogitsloss)
- [Tổng quan các bước](#tổng-quan-các-bước)
- [Bước 0: Chuẩn bị Git branch](#bước-0-chuẩn-bị-git-branch)
- [Bước 1: Implement BaseBackbone (abstract class)](#bước-1-implement-basebackbone-abstract-class)
- [Bước 2: Implement EfficientNetBackbone](#bước-2-implement-efficientnetbackbone)
- [Bước 3: Implement BaseDetector (abstract class)](#bước-3-implement-basedetector-abstract-class)
- [Bước 4: Implement EfficientNetDetector](#bước-4-implement-efficientnetdetector)
- [Bước 5: Implement Registry Pattern](#bước-5-implement-registry-pattern)
- [Bước 6: Kết nối Registry với Detector](#bước-6-kết-nối-registry-với-detector)
- [Bước 7: Unit tests](#bước-7-unit-tests)
- [Bước 8: Kiểm tra tích hợp với Data Pipeline](#bước-8-kiểm-tra-tích-hợp-với-data-pipeline)
- [Bước 9: Commit & PR](#bước-9-commit--pr)
- [Checklist hoàn thành](#checklist-hoàn-thành)
- [Troubleshooting](#troubleshooting)

---

## Bức tranh tổng thể: Model Architecture nằm ở đâu?

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        DỰ ÁN HOLMHZ — SPRINT 1                          │
│                                                                           │
│  Task 1.1  Setup môi trường ✅ DONE                                      │
│  Task 1.2  Thu thập dữ liệu ✅ DONE (27,680 ảnh)                        │
│  Task 1.3  Data Pipeline    ✅ DONE (17/17 tests pass)                   │
│    │                                                                      │
│    │  Task 1.3 đã tạo ra:                                                │
│    │  • ImageDataset class → load ảnh thành tensor [B, 3, 224, 224]      │
│    │  • DataLoader → gom batch, shuffle, parallel loading                │
│    │  • 4 manifest JSON (train/val/test_id/test_ood)                     │
│    │                                                                      │
│  ► Task 1.4  MODEL ARCHITECTURE  ◄◄◄  BẠN ĐANG Ở ĐÂY                   │
│    │                                                                      │
│    │  Đây là "bộ não" nhận ảnh và trả lời: Real hay Fake?                │
│    │  Input: tensor [B, 3, 224, 224] (từ DataLoader)                     │
│    │  Output: logits [B, 1] (raw score, chưa sigmoid)                    │
│    │                                                                      │
│    │  4 việc chính:                                                       │
│    │    1. Backbone (EfficientNet-B0) — trích xuất features              │
│    │    2. Head (Dropout + Linear) — phân loại Real/Fake                 │
│    │    3. Registry — factory pattern để dễ đổi model                    │
│    │    4. Unit tests — kiểm tra shape, params, forward pass             │
│    │                                                                      │
│    │  Assignee: Hoàng                                                     │
│    │  Target:   07/03/2026                                                │
│    │                                                                      │
│    └──► Task 1.5  Training Pipeline (cần cả 1.3 + 1.4 xong)             │
│              │                                                            │
│              │  Trainer class: DataLoader + Model → train loop            │
│              │  batch["image"] → model(x) → BCEWithLogitsLoss → backward │
│              ▼                                                            │
│         Task 1.6  Baseline Training                                       │
│                                                                           │
│  ⚡ Task 1.3 đã xong → giờ chỉ tập trung vào 1.4                       │
└───────────────────────────────────────────────────────────────────────────┘
```

### Mối quan hệ với Task 1.3 (Data Pipeline)

Task 1.3 đã tạo ra DataLoader trả về batch:

```python
# Từ Task 1.3 — đã implement xong
batch = {
    "image": tensor [B, 3, 224, 224],  # float32, normalized ImageNet
    "label": tensor [B],                # float32, 0.0 (Real) hoặc 1.0 (Fake)
    "source": list[str],               # ["cifake", "stylegan", ...]
    "path": list[str],                  # ["data/processed/...", ...]
}

# Task 1.4 cần tạo Model nhận batch["image"] → trả về logits
model = EfficientNetDetector(pretrained=True, freeze_backbone=True)
logits = model(batch["image"])  # [B, 1] — raw scores

# Task 1.5 sẽ ghép: loss = BCEWithLogitsLoss(logits, batch["label"])
```

---

## Tại sao chọn EfficientNet-B0?

Bạn đã chạy thử 3 models SOTA từ benchmark. Đây là kết quả:

```
┌─────────────────────────────────────────────────────────────────────────┐
│               SO SÁNH CÁC LỰA CHỌN BACKBONE                          │
│                                                                         │
│  Model                   │ Params  │ Ưu điểm          │ Nhược điểm     │
│  ────────────────────────│─────────│───────────────────│────────────────│
│  ResNet-50 (CNNDetect)   │ 25M     │ Đơn giản, nhanh  │ Cũ, không sâu  │
│  CLIP ViT-L/14 (UFD)    │ 427M    │ Generalize tốt   │ QUÁ NẶNG       │
│  EfficientNet-B4 (DFB)  │ 19M     │ Mạnh, cân bằng   │ Vẫn lớn        │
│  ════════════════════════│═════════│═══════════════════│════════════════│
│  EfficientNet-B0 (HolmHz│ ~4M ★   │ NHẸ NHẤT, đủ     │ Cần data tốt   │
│                          │         │ < 2s trên CPU     │ (đã có 27K!)   │
│                                                                         │
│  ⟹ HolmHz cần: nhẹ (demo web ≤2s/CPU) + đủ mạnh → B0 là lựa chọn    │
│  ⟹ Nếu B0 không đạt OOD AUC → fallback sang CLIP ViT (đã test)       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Verified bằng `timm` v1.0.24:**

```python
import timm
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
# Backbone params: 4,007,548
# Features dim: 1280
# Thêm Linear(1280, 1) → Total: 4,008,829 (~4M)
```

**4 lý do chính:**

1. **Nhẹ (4M params)**: Chạy được trên CPU laptop ≤ 2 giây — đúng KPI web demo
2. **Hiệu quả**: EfficientNet "scale" thông minh (compound scaling — đồng đều chiều rộng, sâu, resolution)
3. **Pre-trained ImageNet**: Đã "thấy" 1.2 triệu ảnh → biết phân biệt cấu trúc ảnh thật
4. **`timm` library**: Một dòng code = load weights + model kiến trúc hoàn chỉnh

---

## Kiến thức nền: Transfer Learning

### "Chuyển ngành" cho AI — Tại sao không train từ đầu?

**Transfer Learning** = lấy kiến thức model đã học từ bài toán A (ImageNet) để áp dụng cho bài toán B (deepfake detection).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRANSFER LEARNING — TỔNG QUAN                        │
│                                                                         │
│  BÀI TOÁN A (ImageNet — 1.2 triệu ảnh, 1000 loại):                    │
│                                                                         │
│    Ảnh chó, mèo, xe, hoa...  →  EfficientNet  →  "Đây là con mèo"     │
│                                  (đã train)                             │
│                                                                         │
│    Sau khi train, model BIẾT CÁCH NHÌN:                                 │
│    • Layer đầu: phát hiện cạnh, đường thẳng, góc                       │
│    • Layer giữa: phát hiện texture, pattern, hình dạng                 │
│    • Layer cuối: phát hiện vật thể hoàn chỉnh (khuôn mặt, mắt...)     │
│                                                                         │
│    → Kiến thức "nhìn" này là TỔNG QUÁT, dùng cho bài toán khác!       │
│                                                                         │
│  BÀI TOÁN B (Deepfake Detection — bài của chúng ta):                   │
│                                                                         │
│    25K ảnh real/fake  →  EfficientNet (copy từ A)  →  "Real hay Fake?" │
│                          + Head mới (Linear)                            │
│                                                                         │
│    → KHÔNG cần train "cách nhìn" từ đầu (đã có rồi!)                  │
│    → CHỈ CẦN train "cách phán đoán" Real/Fake (lớp head)              │
│    → Tiết kiệm 100x thời gian + data                                  │
│                                                                         │
│  TƯƠNG TỰ ĐỜI THỰC:                                                   │
│    Bác sĩ Mắt → chuyển sang Da liễu                                   │
│    Không cần học lại 6 năm đại học (kiến thức nền đã có)               │
│    Chỉ cần 1-2 năm chuyên sâu da liễu                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Freeze vs Unfreeze — 2 giai đoạn training

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    EfficientNet-B0 (Pretrained ImageNet)                 │
│                                                                          │
│  ┌──────────────────────────────────┐  ┌────────────────────────┐        │
│  │  BACKBONE (Layers 1-16)          │  │  HEAD (2 layers)       │        │
│  │  ────────────────────            │  │  ────────────────────  │        │
│  │  Conv → Conv → ... → Global Pool │  │  Dropout(0.3)         │        │
│  │                                   │  │  Linear(1280 → 1)    │        │
│  │  "MẮT" — biết nhìn texture,     │  │                        │        │
│  │   cạnh, pattern, khuôn mặt...   │  │  "NÃO" — phán đoán    │        │
│  │                                   │  │  Real hay Fake?       │        │
│  │  Params: 4,007,548               │  │  Params: 1,281        │        │
│  └──────────────────────────────────┘  └────────────────────────┘        │
│                                                                          │
│  ⭐ PHASE 1: FREEZE BACKBONE                                            │
│     Backbone: ĐÓng BĂNG (requires_grad = False)                         │
│     Head:     TRAIN (requires_grad = True)                               │
│     Trainable: 1,281 params → NHANH (3 phút/epoch)                      │
│     Mục đích: Kiểm tra pipeline chạy đúng, Head học phán đoán cơ bản   │
│                                                                          │
│  ⭐ PHASE 2: UNFREEZE BACKBONE (Task 1.6)                               │
│     Backbone: MỞ KHÓA (requires_grad = True)                            │
│     Head:     Tiếp tục train                                             │
│     Trainable: 4,008,829 params → CHẬM hơn nhưng CHÍNH XÁC hơn        │
│     Mục đích: Tinh chỉnh "cách nhìn" cho deepfake detection             │
│     LR rất nhỏ: backbone dùng LR thấp hơn head 10x (tránh phá)        │
└──────────────────────────────────────────────────────────────────────────┘
```

**Tại sao freeze trước?**

- Backbone đã biết nhìn ảnh rất tốt rồi (ImageNet trained). Nếu mở khóa ngay + learning rate cao → **phá hỏng kiến thức cũ** (catastrophic forgetting)
- Phase 1 chỉ train 1,281 params (head) → nhanh, dùng để verify pipeline chạy đúng
- Phase 2 tinh chỉnh backbone → model học cách nhìn "deepfake-specific features" (texture da, artifact mắt, tóc...)

---

## Kiến thức nền: EfficientNet Architecture

### Compound Scaling — "Scale" thông minh

Trước EfficientNet, khi muốn model mạnh hơn:

- ResNet: thêm layers (ResNet-18 → 50 → 152) → chỉ scale chiều sâu
- WideResNet: tăng số filters → chỉ scale chiều rộng
- Chỉ tăng 1 chiều → hiệu quả giảm dần (diminishing returns)

EfficientNet "scale" 3 chiều ĐỒNG ĐỀU:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPOUND SCALING (EfficientNet)                       │
│                                                                         │
│  3 chiều scaling:                                                       │
│                                                                         │
│  1. DEPTH  (Chiều sâu)  — Số layers                                    │
│     B0: 16 layers  →  B7: 66 layers                                    │
│                                                                         │
│  2. WIDTH  (Chiều rộng) — Số filters mỗi layer                         │
│     B0: 32 filters →  B7: 64 filters                                   │
│                                                                         │
│  3. RESOLUTION (Độ phân giải) — Input size                              │
│     B0: 224×224   →   B7: 600×600                                       │
│                                                                         │
│  EfficientNet scale cả 3 chiều với tỷ lệ cố định (α, β, γ)            │
│  → Hiệu quả hơn nhiều so với chỉ tăng 1 chiều                         │
│                                                                         │
│  EfficientNet-B0  (baseline):  4M params, 224×224, 77.1% ImageNet      │
│  EfficientNet-B4  (lớn hơn):  19M params, 380×380, 82.9% ImageNet     │
│  EfficientNet-B7  (lớn nhất): 66M params, 600×600, 84.3% ImageNet     │
│                                                                         │
│  HolmHz dùng B0: nhẹ nhất, input 224×224 (match ảnh đã resize Task 1.2)│
└─────────────────────────────────────────────────────────────────────────┘
```

### MBConv Block — Building block chính

EfficientNet xây từ nhiều MBConv (Mobile Inverted Bottleneck Convolution) blocks:

```
Input [C_in channels]
  │
  ├─ Expand (Conv 1×1): C_in → C_in * ratio  (tăng channels)
  │
  ├─ Depthwise Conv (3×3 hoặc 5×5): tính trên từng channel riêng
  │   → Nhẹ hơn nhiều so với Conv thường: params giảm ~9-25 lần
  │
  ├─ Squeeze-and-Excitation: "chú ý" channel quan trọng
  │   (như Attention nhưng trên channels thay vì spatial)
  │
  ├─ Project (Conv 1×1): C_in * ratio → C_out  (giảm channels)
  │
  └─ Skip Connection: cộng với input (nếu same shape)
```

> **Bạn KHÔNG cần code MBConv.** `timm` đã implement sẵn. Chỉ cần hiểu tổng quan để debug/explain khi cần.

---

## Kiến thức nền: Backbone + Head Pattern

### Tách model thành 2 phần — Tại sao?

Mọi model phân loại ảnh đều gồm 2 phần:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  BACKBONE + HEAD PATTERN                                │
│                                                                         │
│  Input [B, 3, 224, 224]                                                 │
│      │                                                                  │
│      ▼                                                                  │
│  ┌───────────────────────────────────────────────┐                      │
│  │  BACKBONE (Feature Extractor)                  │                      │
│  │  ─────────────────────────────                 │                      │
│  │  "Transforms ảnh thành vector đặc trưng"       │                      │
│  │                                                 │                      │
│  │  EfficientNet-B0:                               │                      │
│  │    Input [B, 3, 224, 224]                       │                      │
│  │    → 16 MBConv layers                           │                      │
│  │    → Global Average Pooling                     │                      │
│  │    → Output [B, 1280]                           │                      │
│  │                                                 │                      │
│  │  Mỗi ảnh 224×224×3 = 150,528 pixels             │                      │
│  │  → nén thành 1 vector 1280 chiều                │                      │
│  │  Vector này chứa "bản chất" của ảnh             │                      │
│  └───────────────────────────────────────────────┘                      │
│      │ features [B, 1280]                                               │
│      ▼                                                                  │
│  ┌───────────────────────────────────────────────┐                      │
│  │  HEAD (Classification Layers)                  │                      │
│  │  ─────────────────────────────                 │                      │
│  │  "Dựa vào features → phán đoán Real/Fake"     │                      │
│  │                                                 │                      │
│  │  Dropout(0.3)  — tắt 30% neuron (chống overfit) │                     │
│  │  Linear(1280, 1) — 1 output = score Real/Fake  │                      │
│  │  → Output [B, 1]  — raw logits                  │                      │
│  └───────────────────────────────────────────────┘                      │
│      │ logits [B, 1]                                                    │
│      ▼                                                                  │
│  Khi training: BCEWithLogitsLoss(logits, labels)                        │
│  Khi inference: probs = sigmoid(logits) → P(Fake)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

**Tại sao tách?**

1. **Swap backbone dễ**: Đổi EfficientNet → CLIP chỉ cần thay backbone class, head giữ nguyên
2. **Freeze riêng part one**: Phase 1 freeze backbone, chỉ train head
3. **Grad-CAM cần backbone**: XAI (Task 2.3) cần truy cập layer cuối backbone để tạo heatmap
4. **OOP tốt**: Single Responsibility — backbone lo extract, head lo classify

### Abstract Base Class — Tại sao?

```python
# Giả sử mai sau muốn thử CLIP backbone:
class CLIPBackbone(BaseBackbone):
    def extract_features(self, x):  # PHẢI implement (abstract)
        return self.clip_model(x)   # [B, 768]

    def get_features_dim(self):     # PHẢI implement (abstract)
        return 768

# Code ở Detector KHÔNG CẦN SỬA gì:
# self.backbone.extract_features(x)  ← vẫn hoạt động dù là CLIP hay EfficientNet
# self.backbone.get_features_dim()   ← trả về 768 hay 1280 tùy backbone
```

Đây gọi là **Open/Closed Principle**: code mở cho extension (thêm backbone mới), đóng cho modification (không sửa code cũ).

---

## Kiến thức nền: Registry Pattern

### Factory Pattern — Tạo model theo tên

Khi project có nhiều detector (EfficientNet, CLIP, ResNet...), code **KHÔNG NÊN** viết:

```python
# ❌ BAD — thêm model mới phải sửa if/elif
if name == "efficientnet_b0":
    model = EfficientNetDetector(...)
elif name == "clip_vit":
    model = CLIPDetector(...)
elif name == "resnet50":
    model = ResNetDetector(...)
```

Thay vào đó, dùng **Registry**:

```python
# ✅ GOOD — Registry pattern (học từ DeepfakeBench)
model = DETECTOR_REGISTRY.build("efficientnet_b0", pretrained=True)

# Thêm model mới? Chỉ cần @register, KHÔNG sửa code cũ:
@DETECTOR_REGISTRY.register("clip_vit")
class CLIPDetector(BaseDetector):
    ...
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     REGISTRY PATTERN                                    │
│                                                                         │
│  Registry = "Danh bạ" mapping tên → class                              │
│                                                                         │
│  ┌────────────────────────────────────────────┐                         │
│  │  DETECTOR_REGISTRY                          │                         │
│  │  ──────────────────                         │                         │
│  │  "efficientnet_b0"  →  EfficientNetDetector │                         │
│  │  "clip_vit"         →  CLIPDetector    (sau) │                         │
│  │  "resnet50"         →  ResNetDetector  (sau) │                         │
│  └────────────────────────────────────────────┘                         │
│                                                                         │
│  Đăng ký: @DETECTOR_REGISTRY.register("efficientnet_b0")               │
│  Tạo:     model = DETECTOR_REGISTRY.build("efficientnet_b0")           │
│  List:    DETECTOR_REGISTRY.list() → ["efficientnet_b0"]               │
│                                                                         │
│  Config YAML chỉ cần đổi 1 dòng:                                      │
│    model.name: efficientnet_b0  →  model.name: clip_vit                │
│  Code training KHÔNG đổi gì.                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kiến thức nền: Logits vs Probabilities vs BCEWithLogitsLoss

### Luồng từ model → loss → prediction

Đây là phần HAY NHẦM nhất khi mới học:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                LOGITS vs PROBABILITIES — HIỂU RÕ                       │
│                                                                         │
│  Model forward():                                                       │
│    Input [B, 3, 224, 224]                                               │
│    → Backbone → Head                                                    │
│    → Output: LOGITS [B, 1]     ← Con số bất kỳ: -3.2, 0.5, 4.7...    │
│                                                                         │
│  Logits = "raw scores", chưa ép về [0,1]                               │
│    - Logit = -3.2 → "rất có khả năng Real"                             │
│    - Logit = 0.0  → "50/50 không chắc"                                 │
│    - Logit = +4.7 → "rất có khả năng Fake"                             │
│                                                                         │
│  ═══════════════════════════════════════════════════════                 │
│                                                                         │
│  Khi TRAINING:                                                          │
│    loss = BCEWithLogitsLoss(logits, labels)                             │
│    │                                                                    │
│    │  BCEWithLogitsLoss NỘI BỘ:                                        │
│    │  1. Tính sigmoid(logits) → probabilities                          │
│    │  2. Tính BCE(probs, labels) → loss value                          │
│    │  → GỘP 2 bước → numerical stability tốt hơn                      │
│    │  → KHÔNG cần sigmoid trong model!                                  │
│    │                                                                    │
│    ❌ SAI:  model → sigmoid → BCELoss (2 bước, mất precision)          │
│    ✅ ĐÚNG: model → logits  → BCEWithLogitsLoss (gộp, ổn định)        │
│                                                                         │
│  ═══════════════════════════════════════════════════════                 │
│                                                                         │
│  Khi INFERENCE (predict):                                               │
│    probs = torch.sigmoid(logits)    ← Ép về [0,1]                     │
│    label = (probs > 0.5).long()     ← 1 = Fake, 0 = Real              │
│                                                                         │
│    Ví dụ:                                                               │
│    logits = [4.7]  → sigmoid → probs = [0.991] → label = 1 (Fake)     │
│    logits = [-3.2] → sigmoid → probs = [0.039] → label = 0 (Real)     │
│    logits = [0.0]  → sigmoid → probs = [0.500] → label = 0 (tie→Real) │
└─────────────────────────────────────────────────────────────────────────┘
```

**Tóm tắt:** Model chỉ trả logits. Sigmoid chỉ dùng khi predict. Training dùng BCEWithLogitsLoss (gộp sigmoid + loss cho ổn định). Config hiện tại đã đúng: `loss.name: bce_with_logits`.

**Labels từ Task 1.3:** `batch["label"]` có dtype `float32`, giá trị `0.0` (Real) hoặc `1.0` (Fake) — tương thích trực tiếp với BCEWithLogitsLoss.

---

## Tổng quan các bước

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TASK 1.4 — ROADMAP                                    │
│                                                                         │
│  Bước 0  Chuẩn bị Git branch                              (~5 phút)    │
│  Bước 1  Implement BaseBackbone (abstract class)           (~15 phút)   │
│  Bước 2  Implement EfficientNetBackbone (timm wrapper)     (~20 phút)   │
│  Bước 3  Implement BaseDetector (abstract class)           (~15 phút)   │
│  Bước 4  Implement EfficientNetDetector (backbone + head)  (~30 phút)   │
│  Bước 5  Implement Registry Pattern (factory)              (~20 phút)   │
│  Bước 6  Kết nối Registry với Detector                     (~10 phút)   │
│  Bước 7  Unit tests (backbones + detectors)                (~45 phút)   │
│  Bước 8  Kiểm tra tích hợp với Data Pipeline               (~15 phút)   │
│  Bước 9  Commit & PR                                      (~10 phút)   │
│                                                                         │
│  Tổng ước tính: ~3-4 giờ (có thể chia ra 2 ngày)                       │
│                                                                         │
│  File sẽ tạo/sửa:                                                      │
│    ✏️  src/holmhz/backbones/base.py          (abstract class)           │
│    ✏️  src/holmhz/backbones/efficientnet.py  (timm wrapper)             │
│    ✏️  src/holmhz/backbones/__init__.py      (exports)                  │
│    ✏️  src/holmhz/detectors/base.py          (abstract class)           │
│    ✏️  src/holmhz/detectors/efficientnet_detector.py  (main model)     │
│    ✏️  src/holmhz/detectors/__init__.py      (exports + registry)      │
│    ✏️  src/holmhz/utils/registry.py          (factory pattern)         │
│    ✏️  src/holmhz/utils/__init__.py          (exports)                  │
│    ✏️  tests/test_backbones.py               (backbone tests)           │
│    ✏️  tests/test_detectors.py               (detector tests)           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Bước 0: Chuẩn bị Git branch

```bash
# Đảm bảo đang ở thư mục project
cd R:/_Projects/Eurus_Workspace/HolmHz

# Kích hoạt venv
.venv\Scripts\activate

# Chuyển về main và pull mới nhất
git checkout main
git pull origin main

# Tạo branch mới cho model architecture
git checkout -b feat/s1/model-architecture
```

> **Tại sao branch riêng?** Task 1.3 có branch `feat/s1/data-pipeline`. Task 1.4 có branch `feat/s1/model-architecture`. Mỗi task 1 branch → review dễ, revert dễ, merge riêng.

---

## Bước 1: Implement BaseBackbone (abstract class)

### Tại sao cần Abstract Base Class?

Nếu mai sau bạn muốn thử CLIP-ViT làm backbone, bạn chỉ cần:

```python
class CLIPBackbone(BaseBackbone):      # Kế thừa base
    def extract_features(self, x): ... # Implement abstract method
    def get_features_dim(self): ...    # Implement abstract method
```

Code Detector, Trainer... **KHÔNG CẦN SỬA** vì chúng chỉ gọi `backbone.extract_features()`.

### Code: `src/holmhz/backbones/base.py`

```python
"""
Base class cho tất cả Backbones.

Backbone = phần "mắt" của model — biết nhìn ảnh, trích xuất đặc trưng.

Tại sao cần Abstract Base Class?
→ Định nghĩa "hợp đồng": mọi backbone PHẢI có:
  - extract_features(x) → vector features
  - get_features_dim() → số chiều features
→ Đổi backbone: code khác KHÔNG cần sửa (Open/Closed Principle).

Pattern từ DeepfakeBench: AbstractDetector định nghĩa interface chung.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(ABC, nn.Module):
    """Abstract base class cho tất cả backbones.

    Mọi backbone kế thừa class này phải implement:
    - extract_features(): trích xuất feature vector từ ảnh
    - get_features_dim(): trả về số chiều của feature vector

    Cung cấp sẵn:
    - freeze(): đóng băng tất cả params
    - unfreeze(): mở khóa tất cả params
    - forward(): alias cho extract_features (nn.Module compatibility)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất features từ ảnh input.

        Args:
            x: Tensor [B, 3, H, W] — batch ảnh đã normalize

        Returns:
            features: Tensor [B, features_dim] — vector đặc trưng
        """

    @abstractmethod
    def get_features_dim(self) -> int:
        """Trả về kích thước vector features.

        Ví dụ: 1280 cho EfficientNet-B0, 768 cho CLIP ViT-B.
        """

    def freeze(self) -> None:
        """Đóng băng tất cả parameters — không cho gradient chạy qua.

        Dùng trong Phase 1 Transfer Learning:
        backbone.freeze() → chỉ train head.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Mở khóa tất cả parameters — cho phép training.

        Dùng trong Phase 2 Fine-tuning:
        backbone.unfreeze() → train toàn bộ model.
        """
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass = extract_features.

        nn.Module yêu cầu forward(), nhưng logic thực sự ở extract_features().
        Giữ forward() để có thể dùng backbone(x) thay vì backbone.extract_features(x).
        """
        return self.extract_features(x)
```

> **Kiểm tra nhanh sau khi viết:**
>
> ```bash
> ruff check src/holmhz/backbones/base.py
> # Expected: All checks passed!
> ```

---

## Bước 2: Implement EfficientNetBackbone

### timm API — Cách dùng

```python
import timm

# Load model đầy đủ (với classification head)
model_full = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1000)
# → Output: [B, 1000] — 1000 class ImageNet

# Load model KHÔNG có classification head (chỉ backbone)
model_backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
# → Output: [B, 1280] — feature vector (đã qua Global Average Pooling)

# num_classes=0 là KEY:
# - Bỏ lớp Linear cuối (classification head của ImageNet)
# - Giữ toàn bộ feature extractor + Global Average Pooling
# - Output là vector 1280 chiều cho mỗi ảnh
```

### Code: `src/holmhz/backbones/efficientnet.py`

```python
"""
EfficientNet-B0 Backbone sử dụng thư viện timm.

timm (PyTorch Image Models): thư viện chứa 700+ pre-trained models.
Thay vì tự code kiến trúc EfficientNet (rất phức tạp), ta import từ timm.

Verified với timm v1.0.24:
- Output shape: [B, 1280] khi num_classes=0
- Backbone params: 4,007,548
- Bao gồm Global Average Pooling (không cần thêm)
"""

import timm
import torch

from .base import BaseBackbone


class EfficientNetBackbone(BaseBackbone):
    """EfficientNet-B0 feature extractor.

    Kiến trúc:
        Input [B, 3, 224, 224]
        → EfficientNet MBConv Layers (pretrained ImageNet)
        → Global Average Pooling
        → Output [B, 1280]  ← vector features

    Args:
        pretrained: Load pretrained ImageNet weights (default: True)

    Params: 4,007,548 (backbone only, không tính head)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load EfficientNet-B0 từ timm
        # num_classes=0 → bỏ lớp classification cuối
        # → chỉ lấy phần feature extractor + global pool
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # Bỏ FC cuối — ta tự thêm head ở Detector
        )

        self._features_dim = 1280  # EfficientNet-B0 output dimension

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất 1280-dim feature vector từ ảnh.

        Args:
            x: [B, 3, 224, 224] — batch ảnh đã normalize (ImageNet stats)

        Returns:
            [B, 1280] — feature vector cho mỗi ảnh

        Note:
            timm model đã bao gồm Global Average Pooling.
            Output đã được flatten thành 1D vector.
        """
        return self.model(x)

    def get_features_dim(self) -> int:
        """Trả về 1280 — feature dimension của EfficientNet-B0."""
        return self._features_dim
```

### Cập nhật `__init__.py`

```python
# src/holmhz/backbones/__init__.py
"""Backbone modules — CNN feature extractors."""

from .base import BaseBackbone
from .efficientnet import EfficientNetBackbone

__all__ = ["BaseBackbone", "EfficientNetBackbone"]
```

> **Kiểm tra nhanh:**
>
> ```bash
> # Lint check
> ruff check src/holmhz/backbones/
>
> # Quick test (trong Python)
> python -c "
> import torch
> from holmhz.backbones import EfficientNetBackbone
> backbone = EfficientNetBackbone(pretrained=False)  # False để nhanh
> x = torch.randn(2, 3, 224, 224)
> features = backbone.extract_features(x)
> print(f'Features shape: {features.shape}')  # [2, 1280]
> print(f'Features dim: {backbone.get_features_dim()}')  # 1280
> "
> ```

---

## Bước 3: Implement BaseDetector (abstract class)

### Code: `src/holmhz/detectors/base.py`

```python
"""
Base class cho tất cả Detectors.

Detector = Backbone + Head (classification layers).
- Backbone "nhìn" ảnh → trích xuất đặc trưng (features)
- Head "phán đoán" → Real hay Fake

Tại sao tách Backbone và Head?
1. Swap backbone dễ (EfficientNet → CLIP)
2. Freeze/unfreeze backbone riêng
3. Grad-CAM cần truy cập backbone layers
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseDetector(ABC, nn.Module):
    """Abstract base class cho tất cả detectors.

    Mọi detector phải implement:
    - forward(x): ảnh → logits (raw scores, CHƯA sigmoid)

    Cung cấp sẵn:
    - predict(x): ảnh → labels (0 hoặc 1, đã qua sigmoid + threshold)
    - predict_proba(x): ảnh → probabilities (đã qua sigmoid)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ảnh → logits.

        Args:
            x: [B, 3, H, W] — batch ảnh đã normalize

        Returns:
            logits: [B, 1] — raw scores (CHƯA qua Sigmoid)
                Logit > 0 → nghiêng về Fake
                Logit < 0 → nghiêng về Real
                Logit = 0 → 50/50

        QUAN TRỌNG: Output là LOGITS, KHÔNG phải probabilities.
        - Training: dùng BCEWithLogitsLoss(logits, labels)
        - Inference: dùng predict_proba() để lấy P(Fake) ∈ [0,1]
        """

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Dự đoán nhãn (0 hoặc 1).

        Args:
            x: [B, 3, H, W] — batch ảnh
            threshold: ngưỡng phân loại (mặc định 0.5)

        Returns:
            labels: [B, 1] — 0 (Real) hoặc 1 (Fake)
        """
        probs = self.predict_proba(x)
        return (probs > threshold).long()

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về probability P(Fake) ∈ [0, 1].

        Đã qua Sigmoid. Dùng khi inference (không phải training).

        Args:
            x: [B, 3, H, W] — batch ảnh

        Returns:
            probs: [B, 1] — P(Fake) cho mỗi ảnh
                0.0 = chắc chắn Real
                1.0 = chắc chắn Fake
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
```

> **Lưu ý:**
>
> - `forward()` trả về **logits** (raw), KHÔNG sigmoid
> - `predict_proba()` trả về **probabilities** (đã sigmoid) — dùng khi demo web
> - `predict()` trả về **labels** (0/1) — dùng khi evaluate metrics

---

## Bước 4: Implement EfficientNetDetector

### Đây là model chính của HolmHz!

```python
# Luồng dữ liệu qua EfficientNetDetector:

Input [B, 3, 224, 224]          # Ảnh đã normalize (từ DataLoader Task 1.3)
    │
    ▼
EfficientNetBackbone            # timm pretrained, 4M params
    │ features [B, 1280]
    ▼
Dropout(0.3)                    # Tắt 30% neuron random (chống overfit)
    │ [B, 1280]
    ▼
Linear(1280, 1)                 # Fully connected: 1280 inputs → 1 output
    │ logits [B, 1]
    ▼
Output                          # Raw logits → BCEWithLogitsLoss (training)
                                # → sigmoid → P(Fake) (inference)
```

### Code: `src/holmhz/detectors/efficientnet_detector.py`

```python
"""
EfficientNet-B0 Deepfake Detector — Model chính của HolmHz.

Kiến trúc tổng thể:
    Input [B, 3, 224, 224]
    → EfficientNet-B0 Backbone [B, 1280]  (Pretrained ImageNet, 4M params)
    → Dropout(0.3)                         (Chống overfitting)
    → Linear(1280, 1)                      (1 output = raw logit)
    → Output [B, 1]                        (Logits — chưa sigmoid)

Training: loss = BCEWithLogitsLoss(logits, labels)
Inference: probs = sigmoid(logits) → P(Fake)

Tại sao KHÔNG có Sigmoid trong forward()?
→ BCEWithLogitsLoss tự tính Sigmoid bên trong
→ Numerical stability tốt hơn (tránh log(0) và saturated gradients)
→ Pattern chuẩn trong PyTorch deep learning

Params breakdown:
    Backbone: 4,007,548
    Head:     1,281 (Linear: 1280*1 + 1 bias)
    Total:    4,008,829 (~4M, well under 6M limit)
"""

import torch
import torch.nn as nn

from ..backbones.efficientnet import EfficientNetBackbone
from .base import BaseDetector


class EfficientNetDetector(BaseDetector):
    """Detector sử dụng EfficientNet-B0 backbone.

    Args:
        pretrained: Load pretrained ImageNet weights cho backbone
        dropout: Tỷ lệ dropout (0.3 = tắt 30% neuron ngẫu nhiên khi train)
        freeze_backbone: Đóng băng backbone (Phase 1 transfer learning)

    Example:
        >>> model = EfficientNetDetector(pretrained=True, freeze_backbone=True)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> logits = model(x)  # [4, 1]
        >>> probs = model.predict_proba(x)  # [4, 1] — P(Fake) ∈ [0, 1]
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Backbone: EfficientNet-B0 feature extractor
        self.backbone = EfficientNetBackbone(pretrained=pretrained)

        # Head: Classification layers
        # Dropout → Linear, KHÔNG có Sigmoid (BCEWithLogitsLoss xử lý)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.get_features_dim(), 1),
        )

        # Freeze backbone nếu được yêu cầu (Phase 1)
        if freeze_backbone:
            self.backbone.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ảnh → logits.

        Args:
            x: [B, 3, 224, 224] — batch ảnh đã normalize (ImageNet stats)

        Returns:
            logits: [B, 1] — raw scores

        QUAN TRỌNG: Output là LOGITS (chưa sigmoid).
        - Training: BCEWithLogitsLoss(logits, labels) — loss tự sigmoid
        - Inference: dùng predict_proba(x) để lấy P(Fake) ∈ [0,1]
        """
        features = self.backbone.extract_features(x)  # [B, 1280]
        logits = self.head(features)                    # [B, 1]
        return logits

    def get_feature_layer(self) -> nn.Module:
        """Trả về layer cuối của backbone — dùng cho Grad-CAM (Task 2.3).

        Grad-CAM cần "nhìn vào" layer convolution cuối cùng
        để tạo heatmap giải thích model đang nhìn vùng nào.

        Returns:
            nn.Module: conv_head layer của EfficientNet-B0
        """
        return self.backbone.model.conv_head
```

### Cập nhật `__init__.py`

```python
# src/holmhz/detectors/__init__.py
"""Detector modules — Backbone + Head models for deepfake detection."""

from .base import BaseDetector
from .efficientnet_detector import EfficientNetDetector

__all__ = ["BaseDetector", "EfficientNetDetector"]
```

> **Kiểm tra nhanh:**
>
> ```bash
> ruff check src/holmhz/detectors/
>
> python -c "
> import torch
> from holmhz.detectors import EfficientNetDetector
> model = EfficientNetDetector(pretrained=False, freeze_backbone=True)
> x = torch.randn(2, 3, 224, 224)
> logits = model(x)
> print(f'Logits shape: {logits.shape}')  # [2, 1]
> print(f'Logits values: {logits.squeeze().tolist()}')  # [x.xx, x.xx]
>
> total = sum(p.numel() for p in model.parameters())
> trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
> print(f'Total params: {total:,}')       # 4,008,829
> print(f'Trainable params: {trainable:,}')  # 1,281 (chỉ head)
> "
> ```

---

## Bước 5: Implement Registry Pattern

### Code: `src/holmhz/utils/registry.py`

```python
"""
Registry Pattern — Factory cho Detectors và Backbones.

Tại sao cần Registry?
→ Tạo model theo tên (string) thay vì import trực tiếp
→ Config YAML chỉ đổi model.name → code không đổi
→ Thêm model mới: chỉ cần @register, code cũ không sửa

Pattern từ DeepfakeBench: @DETECTOR.register_module()

Usage:
    # Đăng ký
    @DETECTOR_REGISTRY.register("efficientnet_b0")
    class EfficientNetDetector(BaseDetector):
        ...

    # Tạo
    model = DETECTOR_REGISTRY.build("efficientnet_b0", pretrained=True)

    # Liệt kê
    print(DETECTOR_REGISTRY.list())  # ["efficientnet_b0"]
"""

from __future__ import annotations

from typing import Any


class Registry:
    """Registry quản lý mapping: tên (str) → class.

    Arguments:
        name: Tên registry (cho error messages), ví dụ "detector", "backbone"
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, type] = {}

    def register(self, name: str):
        """Decorator đăng ký class vào registry.

        Args:
            name: Tên dùng để lookup, ví dụ "efficientnet_b0"

        Returns:
            Decorator function

        Raises:
            ValueError: Nếu tên đã được đăng ký (tránh ghi đè nhầm)
        """
        def decorator(cls):
            if name in self._registry:
                raise ValueError(
                    f"'{name}' already registered in {self.name} registry. "
                    f"Existing: {self._registry[name].__name__}"
                )
            self._registry[name] = cls
            return cls
        return decorator

    def build(self, name: str, **kwargs) -> Any:
        """Tạo instance từ tên đã đăng ký.

        Args:
            name: Tên đã register, ví dụ "efficientnet_b0"
            **kwargs: Arguments truyền vào constructor của class

        Returns:
            Instance của class đã đăng ký

        Raises:
            KeyError: Nếu tên chưa được đăng ký
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[name](**kwargs)

    def get(self, name: str) -> type:
        """Lấy class (không tạo instance) từ tên.

        Hữu ích khi muốn kiểm tra class trước khi tạo instance.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[name]

    def list(self) -> list[str]:
        """Liệt kê tất cả tên đã đăng ký."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Kiểm tra tên đã đăng ký chưa: 'efficientnet_b0' in registry."""
        return name in self._registry

    def __len__(self) -> int:
        """Số lượng classes đã đăng ký."""
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', items={self.list()})"


# === Global Registries ===
# Sử dụng trong toàn project

BACKBONE_REGISTRY = Registry("backbone")
DETECTOR_REGISTRY = Registry("detector")
```

### Cập nhật `src/holmhz/utils/__init__.py`

```python
# src/holmhz/utils/__init__.py
"""Utility modules — Registry, logging, I/O helpers."""

from .registry import BACKBONE_REGISTRY, DETECTOR_REGISTRY, Registry

__all__ = ["Registry", "BACKBONE_REGISTRY", "DETECTOR_REGISTRY"]
```

---

## Bước 6: Kết nối Registry với Detector

Sau khi có Registry và Detector, cần đăng ký:

### Cập nhật `src/holmhz/detectors/__init__.py`

```python
# src/holmhz/detectors/__init__.py
"""Detector modules — Backbone + Head models for deepfake detection."""

from ..utils.registry import DETECTOR_REGISTRY
from .base import BaseDetector
from .efficientnet_detector import EfficientNetDetector

# Đăng ký detector vào registry
# Sau này thêm CLIP: import CLIPDetector + register
DETECTOR_REGISTRY.register("efficientnet_b0")(EfficientNetDetector)

__all__ = ["BaseDetector", "EfficientNetDetector", "DETECTOR_REGISTRY"]
```

### Cập nhật `src/holmhz/backbones/__init__.py`

```python
# src/holmhz/backbones/__init__.py
"""Backbone modules — CNN feature extractors."""

from ..utils.registry import BACKBONE_REGISTRY
from .base import BaseBackbone
from .efficientnet import EfficientNetBackbone

# Đăng ký backbone vào registry
BACKBONE_REGISTRY.register("efficientnet_b0")(EfficientNetBackbone)

__all__ = ["BaseBackbone", "EfficientNetBackbone", "BACKBONE_REGISTRY"]
```

> **Kiểm tra nhanh:**
>
> ```bash
> python -c "
> from holmhz.utils.registry import DETECTOR_REGISTRY, BACKBONE_REGISTRY
>
> # Import để trigger registration
> import holmhz.detectors
> import holmhz.backbones
>
> print(f'Detectors: {DETECTOR_REGISTRY.list()}')   # ['efficientnet_b0']
> print(f'Backbones: {BACKBONE_REGISTRY.list()}')    # ['efficientnet_b0']
>
> # Build qua registry
> model = DETECTOR_REGISTRY.build('efficientnet_b0', pretrained=False)
> print(f'Model type: {type(model).__name__}')  # EfficientNetDetector
> "
> ```

---

## Bước 7: Unit tests

### File: `tests/test_backbones.py`

```python
"""Unit tests cho backbone modules.

Kiểm tra:
- EfficientNetBackbone tạo được, extract features đúng shape
- Freeze/unfreeze hoạt động đúng
- Features dimension đúng (1280)
- Forward pass = extract_features
"""

import pytest
import torch

from holmhz.backbones import EfficientNetBackbone
from holmhz.backbones.base import BaseBackbone


class TestBaseBackbone:
    """Test abstract base class."""

    def test_cannot_instantiate_base(self):
        """BaseBackbone là abstract — không thể tạo instance trực tiếp."""
        with pytest.raises(TypeError):
            BaseBackbone()

    def test_efficientnet_is_subclass(self):
        """EfficientNetBackbone kế thừa BaseBackbone."""
        assert issubclass(EfficientNetBackbone, BaseBackbone)


class TestEfficientNetBackbone:
    """Test EfficientNet-B0 backbone."""

    @pytest.fixture
    def backbone(self):
        """Tạo backbone không pretrained (nhanh hơn cho test)."""
        return EfficientNetBackbone(pretrained=False)

    @pytest.fixture
    def dummy_input(self):
        """Batch 4 ảnh giả 224×224."""
        return torch.randn(4, 3, 224, 224)

    def test_features_dim(self, backbone):
        """Feature dimension phải là 1280."""
        assert backbone.get_features_dim() == 1280

    def test_extract_features_shape(self, backbone, dummy_input):
        """extract_features phải trả về [B, 1280]."""
        features = backbone.extract_features(dummy_input)
        assert features.shape == (4, 1280)

    def test_forward_equals_extract_features(self, backbone, dummy_input):
        """forward() phải cho kết quả giống extract_features()."""
        backbone.eval()
        with torch.no_grad():
            f1 = backbone.forward(dummy_input)
            f2 = backbone.extract_features(dummy_input)
        assert torch.allclose(f1, f2)

    def test_forward_call_syntax(self, backbone, dummy_input):
        """Có thể gọi backbone(x) thay vì backbone.forward(x)."""
        backbone.eval()
        with torch.no_grad():
            features = backbone(dummy_input)
        assert features.shape == (4, 1280)

    def test_freeze(self, backbone):
        """Freeze phải tắt requires_grad cho tất cả params."""
        backbone.freeze()
        for param in backbone.parameters():
            assert not param.requires_grad

    def test_unfreeze(self, backbone):
        """Unfreeze phải bật requires_grad cho tất cả params."""
        backbone.freeze()  # Freeze trước
        backbone.unfreeze()  # Rồi unfreeze
        for param in backbone.parameters():
            assert param.requires_grad

    def test_param_count(self, backbone):
        """Backbone params phải ~ 4M (EfficientNet-B0)."""
        total = sum(p.numel() for p in backbone.parameters())
        # EfficientNet-B0 backbone: 4,007,548 params
        assert 3_500_000 < total < 5_000_000, f"Unexpected param count: {total:,}"

    def test_output_dtype(self, backbone, dummy_input):
        """Output phải là float32."""
        features = backbone.extract_features(dummy_input)
        assert features.dtype == torch.float32

    def test_single_image(self, backbone):
        """Phải hoạt động với batch_size=1."""
        x = torch.randn(1, 3, 224, 224)
        features = backbone.extract_features(x)
        assert features.shape == (1, 1280)
```

### File: `tests/test_detectors.py`

```python
"""Unit tests cho detector modules.

Kiểm tra:
- EfficientNetDetector tạo được, forward pass đúng shape
- Freeze backbone: chỉ head trainable
- Unfreeze backbone: toàn bộ trainable
- predict_proba trả về [0, 1]
- predict trả về 0 hoặc 1
- Registry hoạt động
- Tích hợp với DataLoader (batch shape)
"""

import pytest
import torch

from holmhz.detectors import EfficientNetDetector
from holmhz.detectors.base import BaseDetector


class TestBaseDetector:
    """Test abstract base class."""

    def test_cannot_instantiate_base(self):
        """BaseDetector là abstract — không thể tạo instance trực tiếp."""
        with pytest.raises(TypeError):
            BaseDetector()

    def test_efficientnet_is_subclass(self):
        """EfficientNetDetector kế thừa BaseDetector."""
        assert issubclass(EfficientNetDetector, BaseDetector)


class TestEfficientNetDetector:
    """Test EfficientNet-B0 detector."""

    @pytest.fixture
    def model_frozen(self):
        """Detector với backbone frozen (Phase 1)."""
        return EfficientNetDetector(
            pretrained=False,
            dropout=0.3,
            freeze_backbone=True,
        )

    @pytest.fixture
    def model_unfrozen(self):
        """Detector với backbone unfrozen (Phase 2)."""
        return EfficientNetDetector(
            pretrained=False,
            dropout=0.3,
            freeze_backbone=False,
        )

    @pytest.fixture
    def dummy_input(self):
        """Batch 4 ảnh giả 224×224."""
        return torch.randn(4, 3, 224, 224)

    # --- Forward pass ---

    def test_forward_shape(self, model_frozen, dummy_input):
        """Forward phải trả về [B, 1]."""
        model_frozen.eval()
        with torch.no_grad():
            logits = model_frozen(dummy_input)
        assert logits.shape == (4, 1)

    def test_forward_dtype(self, model_frozen, dummy_input):
        """Output phải là float32."""
        model_frozen.eval()
        with torch.no_grad():
            logits = model_frozen(dummy_input)
        assert logits.dtype == torch.float32

    def test_forward_returns_logits(self, model_frozen, dummy_input):
        """Forward trả về logits (có thể âm hoặc dương, không ép [0,1])."""
        model_frozen.eval()
        with torch.no_grad():
            logits = model_frozen(dummy_input)
        # Logits là raw scores — có thể nằm ngoài [0, 1]
        # Không kiểm tra range cụ thể vì random input
        assert logits.shape == (4, 1)

    def test_single_image(self, model_frozen):
        """Phải hoạt động với batch_size=1."""
        model_frozen.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = model_frozen(x)
        assert logits.shape == (1, 1)

    # --- Predict methods ---

    def test_predict_proba_range(self, model_frozen, dummy_input):
        """predict_proba phải trả về giá trị trong [0, 1]."""
        probs = model_frozen.predict_proba(dummy_input)
        assert probs.shape == (4, 1)
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()

    def test_predict_labels(self, model_frozen, dummy_input):
        """predict phải trả về 0 hoặc 1."""
        labels = model_frozen.predict(dummy_input)
        assert labels.shape == (4, 1)
        assert set(labels.flatten().tolist()).issubset({0, 1})

    # --- Freeze / Unfreeze ---

    def test_frozen_backbone_trainable_params(self, model_frozen):
        """Khi freeze backbone, chỉ head params là trainable."""
        trainable = sum(
            p.numel() for p in model_frozen.parameters() if p.requires_grad
        )
        # Head = Linear(1280, 1) + bias = 1280 + 1 = 1,281
        assert trainable == 1281, f"Expected 1281 trainable params, got {trainable}"

    def test_unfrozen_all_trainable(self, model_unfrozen):
        """Khi unfreeze, tất cả params là trainable."""
        total = sum(p.numel() for p in model_unfrozen.parameters())
        trainable = sum(
            p.numel() for p in model_unfrozen.parameters() if p.requires_grad
        )
        assert total == trainable

    def test_total_params(self, model_frozen):
        """Total params phải ~4M (backbone 4M + head 1.3K)."""
        total = sum(p.numel() for p in model_frozen.parameters())
        # EfficientNet-B0 + head: 4,008,829
        assert 3_500_000 < total < 5_000_000, f"Unexpected total: {total:,}"
        assert total <= 6_000_000, "Exceeds 6M param limit from AC"

    def test_unfreeze_backbone(self, model_frozen):
        """Có thể unfreeze backbone sau khi tạo model."""
        # Kiểm tra frozen
        trainable_before = sum(
            p.numel() for p in model_frozen.parameters() if p.requires_grad
        )
        assert trainable_before == 1281

        # Unfreeze
        model_frozen.backbone.unfreeze()
        trainable_after = sum(
            p.numel() for p in model_frozen.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in model_frozen.parameters())
        assert trainable_after == total

    # --- Grad-CAM layer ---

    def test_get_feature_layer(self, model_frozen):
        """get_feature_layer phải trả về nn.Module (cho Grad-CAM)."""
        layer = model_frozen.get_feature_layer()
        assert isinstance(layer, torch.nn.Module)

    # --- Gradient flow ---

    def test_gradient_flows_through_head(self, model_frozen, dummy_input):
        """Gradient phải chạy qua head khi backbone frozen."""
        logits = model_frozen(dummy_input)
        loss = logits.sum()
        loss.backward()

        # Head params phải có gradient
        for name, param in model_frozen.head.named_parameters():
            assert param.grad is not None, f"No gradient for head param: {name}"

    def test_no_gradient_frozen_backbone(self, model_frozen, dummy_input):
        """Backbone frozen → không có gradient cho backbone params."""
        logits = model_frozen(dummy_input)
        loss = logits.sum()
        loss.backward()

        # Backbone params KHÔNG có gradient
        for param in model_frozen.backbone.parameters():
            assert param.grad is None


class TestDetectorRegistry:
    """Test Registry pattern cho detectors."""

    def test_registry_build(self):
        """Phải tạo được model qua registry."""
        from holmhz.utils.registry import DETECTOR_REGISTRY

        # Ensure registration happened
        import holmhz.detectors  # noqa: F401

        model = DETECTOR_REGISTRY.build(
            "efficientnet_b0", pretrained=False, freeze_backbone=True
        )
        assert isinstance(model, EfficientNetDetector)

    def test_registry_list(self):
        """Registry phải list được các detectors đã đăng ký."""
        from holmhz.utils.registry import DETECTOR_REGISTRY

        import holmhz.detectors  # noqa: F401

        detectors = DETECTOR_REGISTRY.list()
        assert "efficientnet_b0" in detectors

    def test_registry_unknown_raises(self):
        """Tên không tồn tại phải raise KeyError."""
        from holmhz.utils.registry import DETECTOR_REGISTRY

        with pytest.raises(KeyError, match="not_a_real_model"):
            DETECTOR_REGISTRY.build("not_a_real_model")

    def test_registry_contains(self):
        """Registry hỗ trợ 'in' operator."""
        from holmhz.utils.registry import DETECTOR_REGISTRY

        import holmhz.detectors  # noqa: F401

        assert "efficientnet_b0" in DETECTOR_REGISTRY
```

> **Chạy tests:**
>
> ```bash
> # Chạy tất cả tests mới
> pytest tests/test_backbones.py tests/test_detectors.py -v
>
> # Mong đợi: tất cả PASSED
> # Nếu có FAILED, đọc error message và fix theo Troubleshooting bên dưới
> ```

---

## Bước 8: Kiểm tra tích hợp với Data Pipeline

Task 1.3 đã tạo DataLoader. Kiểm tra model nhận được batch đúng:

```python
# Chạy bằng: python -c "..." hoặc trong Python REPL

import torch
from holmhz.data import create_dataloader
from holmhz.detectors import EfficientNetDetector

# 1. Tạo DataLoader (từ Task 1.3)
val_loader = create_dataloader(
    manifest_path="data/manifests/val.json",
    batch_size=4,
    shuffle=False,
    num_workers=0,  # Windows safe
)

# 2. Tạo Model (Task 1.4)
model = EfficientNetDetector(pretrained=False, freeze_backbone=True)
model.eval()

# 3. Lấy 1 batch
batch = next(iter(val_loader))
images = batch["image"]   # [4, 3, 224, 224]
labels = batch["label"]   # [4]

print(f"Images shape: {images.shape}")  # [4, 3, 224, 224]
print(f"Labels: {labels.tolist()}")      # [0.0, 1.0, ...]

# 4. Forward pass
with torch.no_grad():
    logits = model(images)  # [4, 1]
    probs = torch.sigmoid(logits)

print(f"Logits shape: {logits.shape}")     # [4, 1]
print(f"Logits: {logits.squeeze().tolist()}")
print(f"Probs: {probs.squeeze().tolist()}")   # [0.xx, ...]

# 5. Simulate loss (BCEWithLogitsLoss — Task 1.5 sẽ dùng chính thức)
loss_fn = torch.nn.BCEWithLogitsLoss()
loss = loss_fn(logits.squeeze(), labels)
print(f"Loss: {loss.item():.4f}")  # Some number > 0
print("✅ Integration test passed!")
```

> **Nếu tất cả print đúng shape → model + data pipeline tương thích!**
> Task 1.5 (Training Pipeline) sẽ ghép chính xác như trên thành training loop.

---

## Bước 9: Commit & PR

```bash
# 1. Kiểm tra lint trước khi commit
ruff check src/holmhz/backbones/ src/holmhz/detectors/ src/holmhz/utils/
ruff check tests/test_backbones.py tests/test_detectors.py

# 2. Chạy toàn bộ test suite (cả data tests lẫn model tests)
pytest tests/ -v

# 3. Stage files
git add src/holmhz/backbones/
git add src/holmhz/detectors/
git add src/holmhz/utils/
git add tests/test_backbones.py tests/test_detectors.py

# 4. Commit
git commit -m "feat(model): implement EfficientNet-B0 detector with registry

- BaseBackbone abstract class with freeze/unfreeze
- EfficientNetBackbone wrapping timm (1280-dim features)
- BaseDetector with predict/predict_proba
- EfficientNetDetector: backbone + Dropout + Linear (4M params)
- Registry pattern for detector/backbone lookup
- Unit tests: test_backbones.py + test_detectors.py"

# 5. Push
git push origin feat/s1/model-architecture

# 6. Tạo PR trên GitHub (nếu dùng GitHub)
```

---

## Checklist hoàn thành

Trước khi đánh dấu Task 1.4 ✅ DONE:

### Code implementation

- [ ] `src/holmhz/backbones/base.py` — `BaseBackbone` abstract class
- [ ] `src/holmhz/backbones/efficientnet.py` — `EfficientNetBackbone(timm)`, 1280-dim
- [ ] `src/holmhz/backbones/__init__.py` — exports + registry registration
- [ ] `src/holmhz/detectors/base.py` — `BaseDetector` abstract class
- [ ] `src/holmhz/detectors/efficientnet_detector.py` — `EfficientNetDetector`
- [ ] `src/holmhz/detectors/__init__.py` — exports + registry registration
- [ ] `src/holmhz/utils/registry.py` — `Registry` class + global registries
- [ ] `src/holmhz/utils/__init__.py` — exports

### Shapes & Params

- [ ] `EfficientNetBackbone.extract_features(x)` → shape `[B, 1280]`
- [ ] `EfficientNetDetector.forward(x)` → shape `[B, 1]` (logits, KHÔNG sigmoid)
- [ ] `EfficientNetDetector.predict_proba(x)` → shape `[B, 1]`, range `[0, 1]`
- [ ] Freeze backbone → trainable params = 1,281 (chỉ head)
- [ ] Total params ≤ 6M (actual: ~4,008,829)

### Registry

- [ ] `DETECTOR_REGISTRY.build("efficientnet_b0")` → `EfficientNetDetector`
- [ ] `BACKBONE_REGISTRY.build("efficientnet_b0")` → `EfficientNetBackbone`
- [ ] Unknown name → `KeyError` với helpful message

### Tests

- [ ] `pytest tests/test_backbones.py -v` → tất cả PASSED
- [ ] `pytest tests/test_detectors.py -v` → tất cả PASSED
- [ ] `ruff check src/holmhz/ tests/` → clean

### Git

- [ ] Branch: `feat/s1/model-architecture`
- [ ] Code committed
- [ ] PR Created trên GitHub

---

## Troubleshooting

### Q: `timm.create_model()` rất chậm / tải weights lâu

**A**: Lần đầu `pretrained=True` sẽ download weights (~20MB) từ internet. Lần sau cached ở `~/.cache/huggingface/hub/`. Trong tests, dùng `pretrained=False` để nhanh.

### Q: `ImportError: No module named 'timm'`

**A**: Đảm bảo venv active và timm đã cài:

```bash
.venv\Scripts\activate
pip install timm
# Verify
python -c "import timm; print(timm.__version__)"  # 1.0.24
```

### Q: `TypeError: Can't instantiate abstract class`

**A**: Bạn đang cố tạo instance của `BaseBackbone` hoặc `BaseDetector` (abstract class). Phải dùng concrete class: `EfficientNetBackbone()` hoặc `EfficientNetDetector()`.

### Q: Trainable params không đúng 1,281 khi freeze

**A**: Kiểm tra:

1. `freeze_backbone=True` trong constructor
2. `backbone.freeze()` thực sự gọi `requires_grad = False` cho tất cả backbone params
3. Head chỉ có `Linear(1280, 1)` = 1280 weights + 1 bias = 1,281

```python
for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"Trainable: {name} — {p.numel()} params")
# Mong đợi chỉ 2 dòng: head.1.weight (1280) + head.1.bias (1)
```

### Q: `DETECTOR_REGISTRY.build()` raise KeyError

**A**: Registration chỉ xảy ra khi module được import. Đảm bảo:

```python
import holmhz.detectors  # Trigger registration
model = DETECTOR_REGISTRY.build("efficientnet_b0")
```

### Q: Forward pass trả về giá trị lạ (NaN, Inf)

**A**: Kiểm tra input:

- Ảnh phải đã normalize (ImageNet stats: mean=[0.485, 0.456, 0.406])
- Shape phải đúng [B, 3, 224, 224]
- Dtype phải là float32

```python
x = torch.randn(1, 3, 224, 224)  # Random đúng format
assert not torch.isnan(model(x)).any()
assert not torch.isinf(model(x)).any()
```

### Q: OOM khi forward pass

**A**: RTX 3050 4GB VRAM — batch_size lớn có thể OOM. Giảm batch:

```python
# Test với batch nhỏ
x = torch.randn(2, 3, 224, 224)  # batch=2 thay vì 32
```

Trong training (Task 1.5), sẽ dùng batch=8-16 trên local, batch=32 trên Colab/Kaggle.

### Q: `model.get_feature_layer()` trả về None hoặc lỗi

**A**: `conv_head` là attribute cụ thể của EfficientNet trong timm. Kiểm tra:

```python
print(type(model.backbone.model.conv_head))  # <class 'torch.nn.Conv2d'>
```

Nếu timm version khác → attribute name có thể khác. Xem `model.backbone.model` để tìm layer phù hợp.

---

## Mối liên hệ với các Task tiếp theo

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SAU TASK 1.4 — CÁC BƯỚC TIẾP THEO                   │
│                                                                         │
│  Task 1.3 (✅ xong) tạo ra:                                            │
│  • ImageDataset + DataLoader                                            │
│  • batch["image"] [B, 3, 224, 224]                                     │
│  • batch["label"] float32                                               │
│                                                                         │
│  Task 1.4 (✅ xong) tạo ra:                                            │
│  • EfficientNetDetector                                                 │
│  • model(x) → logits [B, 1]                                            │
│  • model.predict_proba(x) → P(Fake) [B, 1]                            │
│                                                                         │
│  Task 1.5 GHÉP cả hai:                                                 │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Training Loop (1 epoch):                                         │  │
│  │                                                                    │  │
│  │  for batch in train_loader:           # Từ Task 1.3              │  │
│  │      images = batch["image"]          # [B, 3, 224, 224]         │  │
│  │      labels = batch["label"]          # [B] float32              │  │
│  │                                                                    │  │
│  │      logits = model(images)           # Từ Task 1.4 → [B, 1]    │  │
│  │      loss = BCEWithLogitsLoss(        # Gộp sigmoid + BCE        │  │
│  │          logits.squeeze(), labels     # squeeze [B,1] → [B]      │  │
│  │      )                                                             │  │
│  │                                                                    │  │
│  │      loss.backward()                  # Tính gradient             │  │
│  │      optimizer.step()                 # Cập nhật weights          │  │
│  │      optimizer.zero_grad()            # Reset gradient            │  │
│  │                                                                    │  │
│  │  → WandB logging, early stopping, checkpoint save                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Task 1.6 CHẠY:                                                         │
│  • Train trên toàn bộ 18,550 train ảnh                                 │
│  • Evaluate AUC trên val (3,975) + OOD (1,180)                        │
│  • Save best checkpoint → dùng cho Task 2.x (eval, Grad-CAM, export)  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**Last Updated**: 25/02/2026  
**Author**: Generated by GitHub Copilot for Lê Văn Hoàng  
**Version**: 1.0 (aligned with Task 1.3 completed, timm v1.0.24 verified)
