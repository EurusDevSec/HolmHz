# 📖 HƯỚNG DẪN CHI TIẾT TASK 2.3: GRAD-CAM XAI

> **Dành cho**: Lê Văn Hoàng — người chưa có nền tảng ML/DL, học qua thực hành  
> **Triết lý**: Mỗi bước không chỉ hướng dẫn **làm gì** mà giải thích **tại sao làm vậy**  
> **Thời gian**: ~1 ngày (code đã có sẵn, chỉ cần chạy + hiểu)  
> **Tiền đề**: Task 2.2b Multi-Arch ✅ DONE, Task 1.6 Baseline ✅ DONE  
> **Tham chiếu**: [CONTEXT.md](../CONTEXT.md) Section 22 | [PROJECT_PLAN.md](../PROJECT_PLAN.md)  
>
> **Output**: 61 Grad-CAM heatmap images trong `outputs/xai_gallery/`

---

## 📋 Mục lục

- [Bức tranh tổng thể](#bức-tranh-tổng-thể)
- [Tại sao cần Grad-CAM?](#tại-sao-cần-grad-cam)
- [Grad-CAM giải thích dễ hiểu](#grad-cam-giải-thích-dễ-hiểu)
- [Kiến trúc code đã implement](#kiến-trúc-code-đã-implement)
- [Cách chạy](#cách-chạy)
- [Đọc kết quả heatmap](#đọc-kết-quả-heatmap)
- [Gallery composition](#gallery-composition)
- [Kết luận cho báo cáo](#kết-luận-cho-báo-cáo)
- [Checklist hoàn thành](#checklist-hoàn-thành)
- [Troubleshooting](#troubleshooting)

---

## Bức tranh tổng thể

```
┌───────────────────────────────────────────────────────────┐
│                  DỰ ÁN HOLMHZ — SPRINT 2                  │
│                                                             │
│  Task 2.1  Evaluation Pipeline ✅                          │
│  Task 2.2  Benchmark SOTA ✅                               │
│  Task 2.2b Multi-Arch Benchmark ✅                         │
│                                                             │
│  ► Task 2.3  GRAD-CAM XAI  ✅ DONE                        │
│    │                                                        │
│    │  MỤC ĐÍCH:  Giải thích cho hội đồng                   │
│    │  "Model của bạn nhìn vào ĐÂU để phán đoán fake?"     │
│    │                                                        │
│    │  OUTPUT: 61 heatmap images                            │
│    │                                                        │
│    └──► Task 2.4  Model Export ONNX ✅                     │
└───────────────────────────────────────────────────────────┘
```

---

## Tại sao cần Grad-CAM?

```
┌──────────────── TẠI SAO CẦN GRAD-CAM? ───────────────────┐
│                                                            │
│  Hội đồng sẽ hỏi:                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Q: "Model detect fake dựa trên CÁI GÌ?"          │   │
│  │  A: "Model tập trung vào texture, edge, và noise   │   │
│  │      residuals — được minh chứng bằng heatmap"      │   │
│  │  → THUYẾT PHỤC vì CÓ BẰNG CHỨNG TRỰC QUAN!       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                            │
│  KHÔNG CÓ Grad-CAM:                                       │
│  • "Model predict fake" → BLACK BOX → Không tin           │
│                                                            │
│  CÓ Grad-CAM:                                             │
│  • "Model nhìn vào vùng mắt + tóc → detect noise"        │
│  • → GIẢI THÍCH ĐƯỢC → Tin tưởng + Điểm đánh giá cao     │
└────────────────────────────────────────────────────────────┘
```

> **XAI = Explainable AI**: Lĩnh vực nghiên cứu giúp con người hiểu tại sao AI ra quyết định.
> Grad-CAM là một trong những phương pháp XAI phổ biến nhất cho Vision models.

---

## Grad-CAM giải thích dễ hiểu

### Ý tưởng cốt lõi

```
Input Image → CNN → Feature Maps → Grad-CAM → Heatmap
     ↓              ↓                    ↓
  Ảnh cần         Các "đặc trưng"     Vùng nào ẢNH HƯỞNG
  phân tích       model đã học        NHIỀU NHẤT đến kết quả?
```

### Chi tiết từng bước

1. **Forward pass**: Đưa ảnh qua model → ra kết quả "Fake 87%"
2. **Backward pass**: Tính gradient (đạo hàm) → biết feature nào ảnh hưởng nhiều nhất
3. **Weight × Feature Map**: Nhân gradient với feature map → heatmap
4. **ReLU**: Chỉ giữ phần dương (vùng model "chú ý")
5. **Resize**: Scale heatmap về kích thước ảnh gốc → overlay

### Cách đọc heatmap

```
 🔴 ĐỎ      = Model chú ý NHIỀU vào vùng này → quyết định chính
 🟡 VÀNG    = Model chú ý VỪA
 🔵 XANH    = Model KHÔNG chú ý → không ảnh hưởng kết quả
 ⬛ TỐI     = Model bỏ qua hoàn toàn
```

### Ví dụ thực tế

```
Ảnh FAKE (SD-generated face):
┌─────────────────┐
│     (forehead)   │  ← xanh (bỏ qua)
│   🔴 (eyes) 🔴   │  ← ĐỎ: Model phát hiện noise ở mắt
│     (nose)       │  ← vàng
│   🟡 (mouth) 🟡  │  ← Vàng: texture bất thường
│     (chin)       │  ← xanh
└─────────────────┘
→ Model predict FAKE vì vùng mắt + miệng có artifacts

Ảnh REAL (camera photo):
┌─────────────────┐
│   🔵 (random)    │  ← Heatmap phân tán
│   🔵 (random)    │  ← Không tập trung ở đâu cả
│   🔵 (random)    │  ← Model "không chắc" → predict Real
└─────────────────┘
→ Heatmap nhạt, phân tán = Model không tìm thấy artifacts → Real
```

---

## Kiến trúc code đã implement

### File structure

```
src/holmhz/xai/
├── __init__.py          # Exports GradCAMExplainer, load_image_for_gradcam
├── gradcam.py           # GradCAMExplainer class
└── utils.py             # load_image_for_gradcam(), create_comparison_grid()

scripts/
├── explain.py           # CLI: single image hoặc folder batch
└── generate_xai_gallery.py  # Gallery generator (50 OOD + manual)
```

### GradCAMExplainer class

```python
from holmhz.xai.gradcam import GradCAMExplainer

# 1. Khởi tạo (tự động lấy target layer từ model)
explainer = GradCAMExplainer(model, device="cpu")

# 2. Generate heatmap
heatmap = explainer.explain(image_tensor)       # [H, W] float32 0-1

# 3. Overlay heatmap lên ảnh gốc
overlay = explainer.overlay(image_tensor, rgb_image)  # [H, W, 3] uint8

# 4. Save trực tiếp
explainer.save(image_tensor, rgb_image, "output.png")
```

> **Target layer**: Lấy tự động từ `model.get_feature_layer()`:
> - EfficientNet-B0: `backbone.model.conv_head` (layer cuối trước FC)
> - ResNet-18: `backbone.model.layer4` (block cuối)
> - ViT-Small: `backbone.model.blocks[-1].norm1` (attention block cuối)
> - Swin-T: `backbone.model.layers[-1].blocks[-1].norm1` (window attention cuối)

### Cách `load_image_for_gradcam()` hoạt động

```python
from holmhz.xai.utils import load_image_for_gradcam

tensor, rgb_image = load_image_for_gradcam("path/to/image.jpg")
# tensor:    [1, 3, 224, 224] — normalized (ImageNet mean/std), cho model
# rgb_image: [224, 224, 3]    — float32 [0, 1], cho heatmap overlay
```

---

## Cách chạy

### Bước 0: Activate venv

```bash
cd R:/_Projects/Eurus_Workspace/HolmHz
source .venv/Scripts/activate   # ← BẮT BUỘC trước mọi lệnh Python!
```

### Bước 1: Single image

```bash
python scripts/explain.py \
  --image imgs/Fake_AI_generated/Gemini_Generated_Image_h2x4b6h2x4b6h2x4b6h2x4b6.jpg \
  --model efficientnet_b0 \
  --checkpoint outputs/checkpoints/best_v4.pt \
  --output outputs/xai_gallery/ \
  --device cpu
```

### Bước 2: Batch folder

```bash
python scripts/explain.py \
  --image-dir data/processed/ood_test/flux/ \
  --model efficientnet_b0 \
  --checkpoint outputs/checkpoints/best_v4.pt \
  --output outputs/xai_gallery/ \
  --device cpu
```

### Bước 3: Full gallery (50 OOD samples)

```bash
python scripts/generate_xai_gallery.py
```

> Script này random chọn samples từ 5 sources, export 50 heatmaps.
> Combined với 11 ảnh manual → total 61 heatmaps.

---

## Đọc kết quả heatmap

### Patterns cho FAKE images

| Source | Heatmap Pattern | Giải thích |
| --- | --- | --- |
| **flux** (FLUX.1-schnell) | Tập trung ở **mắt, tóc, skin texture** | Diffusion để lại noise ở high-frequency regions |
| **tristanzhang** (SD/MJ/DALLE mix) | Tập trung ở **facial contours, background edges** | Multi-generator → diverse artifacts |
| **sd15** (Stable Diffusion v1.5) | Rộng hơn, **background + face borders** | GAN-like artifacts ở transitions |

### Patterns cho REAL images

| Source | Heatmap Pattern | Giải thích |
| --- | --- | --- |
| **real_pexels** (stock photos) | **Phân tán, nhạt** | Model không tìm thấy artifacts → confident Real |
| **real_camera** (phone photos) | **Phân tán nhưng đôi khi ở edges** | Compression artifacts gây nhầm lẫn (known weakness) |

---

## Gallery composition

```
outputs/xai_gallery/              ← 61 ảnh tổng cộng
├── gradcam_Gemini_*.png          ← 5 ảnh Fake manual test
├── gradcam_IMG_*.png             ← 6 ảnh Real manual test
├── gradcam_flux_hf_*.png         ← 10 ảnh Flux OOD
├── gradcam_tristanzhang_*.png    ← 15 ảnh Tristanzhang OOD
├── gradcam_real_pexels_*.png     ← 10 ảnh Real Pexels OOD
├── gradcam_real_camera_*.png     ← 10 ảnh Real Camera OOD
└── gradcam_sd15_fake_*.png       ← 5 ảnh SD v1.5 train
```

---

## Kết luận cho báo cáo

> **Viết vào Chapter 4 (Kết quả thực nghiệm):**
>
> "Grad-CAM analysis reveals that EfficientNet-B0 focuses on **local texture and noise patterns** when classifying synthetic images, particularly at **facial features (eyes, hair boundary)** and **background transitions**. For real images, attention is **diffuse**, indicating the model detects the absence of generation artifacts rather than specific 'real' features. This supports our hypothesis that CNN's local receptive field is more effective than Transformer's global attention for detecting Diffusion artifacts."

---

## Checklist hoàn thành

- [x] `src/holmhz/xai/gradcam.py` — GradCAMExplainer class ✅
- [x] `src/holmhz/xai/utils.py` — load_image_for_gradcam() + comparison grid ✅
- [x] `scripts/explain.py` — CLI (single + batch) ✅
- [x] `scripts/generate_xai_gallery.py` — gallery generator ✅
- [x] 61 heatmap images in `outputs/xai_gallery/` ✅
- [x] CONTEXT.md Section 22 updated ✅
- [x] PROJECT_PLAN.md Task 2.3 → ✅ ✅

---

## Troubleshooting

### `ModuleNotFoundError: pytorch_grad_cam`

```bash
pip install grad-cam
```

### `AttributeError: model has no get_feature_layer`

Chỉ EfficientNetDetector và TimmDetector có method này. Nếu dùng model khác, cần implement `get_feature_layer()`.

### Heatmap toàn xanh (không có vùng đỏ)

Model rất tự tin kết quả → gradient nhỏ → heatmap nhạt.
Đây là hành vi ĐÚNG cho ảnh Real (model không tìm thấy artifacts).

### Heatmap toàn đỏ

Thường xảy ra với ảnh bị distorted nặng hoặc resolution quá thấp.
Thử ảnh resolution cao hơn (≥ 512px).
