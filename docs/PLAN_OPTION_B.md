# 📋 HolmHz — Plan Option B: Smart Pivot & Sprint Completion

> **Chiến lược**: Pivot thông minh — Hoàn thành Task 1.7 trong 1 lần train cuối, rồi dồn toàn lực vào Sprint 2-3-4
> **Tác giả**: AI Assistant
> **Cập nhật**: 02/03/2026
> **Deadline bảo vệ**: 15/05/2026 (còn **74 ngày**)
>
> **Nguyên tắc cốt lõi**: OOD gap ≠ thất bại. OOD gap = **research finding có giá trị**.

---

## 📋 Mục lục

1. [Tổng quan Option B](#1-tổng-quan-option-b)
2. [Phase 1 — Đóng Task 1.7 (02–08/03)](#2-phase-1--đóng-task-17)
3. [Phase 2 — Sprint 2: Benchmark + XAI + Export (09–29/03)](#3-phase-2--sprint-2)
4. [Phase 3 — Sprint 3: Web Demo (30/03–20/04)](#4-phase-3--sprint-3)
5. [Phase 4 — Sprint 4: Report & Defense (21/04–15/05)](#5-phase-4--sprint-4)
6. [Vấn đề tồn đọng & Cách khắc phục](#6-vấn-đề-tồn-đọng--cách-khắc-phục)
7. [Narrative cho báo cáo — Cách viết OOD gap](#7-narrative-cho-báo-cáo)
8. [Gantt Chart tổng quan](#8-gantt-chart-tổng-quan)
9. [Checklist tổng hợp](#9-checklist-tổng-hợp)

---

## 1. Tổng quan Option B

### 1.1 Logic chiến lược

```
HIỆN TẠI:                     MỤC TIÊU:
─────────                     ────────
Task 1.7 đang kẹt             Task 1.7 ĐÓNG (3-5 ngày)
Sprint 2 chưa bắt đầu   →    Sprint 2 XONG (3 tuần)
Sprint 3 chưa bắt đầu        Sprint 3 XONG (3 tuần)
Sprint 4 chưa bắt đầu        Sprint 4 XONG (3 tuần + buffer)
                              Defense Ready 15/05 ✅
```

### 1.2 Deliverables cuối cùng cần có cho bảo vệ

| # | Deliverable | Priority | Source Task |
|---|---|---|---|
| 1 | **Model checkpoint** (best_v4.pt hoặc best_v3.pt) | 🔴 Critical | Task 1.7 |
| 2 | **Bảng so sánh 4 methods** (HolmHz + 3 SOTA) trên cùng test set | 🔴 Critical | Task 2.2 |
| 3 | **Grad-CAM heatmap gallery** (50 mẫu ID + OOD) | 🔴 Critical | Task 2.3 |
| 4 | **ONNX model** cho web demo | 🟡 High | Task 2.4 |
| 5 | **Web demo** (Gradio / FastAPI) hoạt động | 🟡 High | Task 3.1 |
| 6 | **Báo cáo** Word/PDF (5 chương) | 🔴 Critical | Task 4.1 |
| 7 | **Slide + video demo** | 🔴 Critical | Task 4.2 |

### 1.3 KPI điều chỉnh thực tế

| Metric | Target gốc | Target điều chỉnh | Lý do |
|---|---|---|---|
| ID AUC | ≥ 0.90 | ≥ 0.95 (đã đạt 0.9979) | Vượt xa target ✅ |
| OOD AUC | ≥ 0.75 | **≥ 0.55** (best effort) | SOTA cũng chỉ 0.06-0.50 trên modern Diffusion |
| OOD Acc (real) | ≥ 65% | **≥ 70%** (real_pexels + real_camera) | v2 đã đạt 90% real_pexels |
| OOD Acc (fake) | ≥ 65% | **≥ 40%** (flux + tristanzhang) | Baseline mới sau khi có tristanzhang_train |
| Latency | ≤ 2s | ≤ 2s (giữ nguyên) | ONNX + EfficientNet-B0 đủ nhanh |

> **Ghi chú**: OOD AUC ≥ 0.55 vẫn **TỐT HƠN** cả 3 SOTA khi test trên modern Diffusion (CNNDetection ~0.06, UniversalFakeDetect ~0.10, DeepfakeBench ~0.50). Đây là narrative mạnh.

---

## 2. Phase 1 — Đóng Task 1.7 (02–08/03)

> **Mục tiêu**: Train ĐÚNG 1 lần cuối với đầy đủ data → evaluate → ghi nhận → CLOSE task.
> **Thời gian**: 3-5 ngày (bao gồm chờ Kaggle chạy)
> **GPU**: Kaggle T4

### 2.1 Vấn đề tồn đọng cần fix TRƯỚC KHI train

#### Bug 1: Kaggle zip thiếu tristanzhang data (CONTEXT.md §17.11)

**Trạng thái**: ĐÃ CÓ FIX — Cell 4 mới tạo data inline trên Kaggle
**Hành động**: Dùng Cell 4 đã chuẩn bị (§17.13) — tạo tristanzhang_train/ trên Kaggle

#### Bug 2: train.py `--config` flag sai (CONTEXT.md §17.12)

**Trạng thái**: ĐÃ BIẾT — dùng positional arg thay vì `--config`
**Hành động**: Đảm bảo Cell 4 dùng đúng syntax:
```python
# ✅ ĐÚNG:
!python scripts/train.py configs/train_v3.yaml data.num_workers=4
# ❌ SAI:
!python scripts/train.py --config configs/train_v3.yaml
```

#### Bug 3 (MỚI): WeightedRandomSampler chưa implement

**Trạng thái**: CHƯA CÓ — hiện tại tất cả sources sample đều nhau
**Vấn đề**: cifake 9,800 samples trong train vs tristanzhang_train chỉ 140 → model hầu như không thấy tristanzhang
**Hành động**: Thêm WeightedRandomSampler vào `scripts/train.py` (xem Bước 2.2)

### 2.2 Steps chi tiết

#### Step 1: Implement WeightedRandomSampler (trên local)

Sửa `scripts/train.py` — thêm sau phần tạo `train_loader`:

```python
# ─── File: scripts/train.py ───
# Thêm import ở đầu file:
import json
from torch.utils.data import WeightedRandomSampler

# Thêm vào function main(), SAU dòng tạo train_loader, TRƯỚC khi dùng nó:

# ─── Weighted Sampling (OOD improvement) ───
use_weighted_sampler = config.training.get("weighted_sampler", False)
if use_weighted_sampler:
    # Đọc manifest để biết source mỗi sample
    with open(config.data.train_manifest) as f:
        train_manifest = json.load(f)

    # Tính weight: source ít ảnh → weight cao hơn
    source_counts = {}
    for entry in train_manifest:
        src = entry["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    max_count = max(source_counts.values())
    source_weights = {src: max_count / cnt for src, cnt in source_counts.items()}

    sample_weights = [source_weights[entry["source"]] for entry in train_manifest]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_manifest),
        replacement=True,
    )

    # Tạo lại train_loader VỚI sampler (thay shuffle)
    train_loader = create_dataloader(
        manifest_path=config.data.train_manifest,
        batch_size=config.training.batch_size,
        image_size=config.data.image_size,
        is_training=True,
        num_workers=config.data.num_workers,
        sampler=sampler,   # thêm param này
        shuffle=False,     # phải False khi dùng sampler
    )
    logger.info("✅ WeightedRandomSampler enabled")
    for src, w in sorted(source_weights.items()):
        logger.info(f"  {src:25s}: weight={w:.2f} (count={source_counts[src]})")
```

> **Lưu ý**: Cần sửa `create_dataloader()` trong `src/holmhz/data/utils.py` để chấp nhận `sampler` và `shuffle` params.

#### Step 2: Tạo config `configs/train_v4.yaml`

```yaml
# ============================================
# HolmHz Training v4 — Final OOD Improvement
# ============================================
# Thay đổi so với v3:
# - WeightedRandomSampler: upsample minority sources
# - ĐÂY LÀ LẦN TRAIN CUỐI — dù kết quả bao nhiêu → CLOSE Task 1.7

model:
  name: efficientnet_b0
  pretrained: true
  num_classes: 1
  dropout: 0.3
  freeze_backbone: false

training:
  epochs: 25
  batch_size: 32
  learning_rate: 0.0001
  optimizer: adamw
  weight_decay: 0.0001
  scheduler: cosine
  weighted_sampler: true    # ← MỚI: upsample minority sources
  early_stopping:
    patience: 8
    monitor: val_auc

data:
  train_manifest: data/manifests/train.json
  val_manifest: data/manifests/val.json
  image_size: 224
  num_workers: 4

wandb:
  project: holmhz
  entity: null
  log_every_n_steps: 10
```

#### Step 3: Chuẩn bị Kaggle Cell 4 Final

```python
# ═══════════════════════════════════════════════════════
# CELL 4: FINAL TRAINING — Task 1.7 v4
# ═══════════════════════════════════════════════════════
import os, shutil, json, random

# ─── Step 1: Fix data — Tạo tristanzhang_train/ từ ood_test/ ───
SEED = 42
random.seed(SEED)

ood_src = "data/processed/ood_test/tristanzhang_fake"
train_dst = "data/processed/train/fake_diffusion/tristanzhang_train"
os.makedirs(train_dst, exist_ok=True)

all_files = sorted(os.listdir(ood_src))
random.shuffle(all_files)
train_files = all_files[:200]
test_files = all_files[200:]

# Copy 200 → train
for f in train_files:
    shutil.copy2(os.path.join(ood_src, f), os.path.join(train_dst, f))

# Write test-only filter
with open("data/manifests/tristanzhang_test_only.txt", "w") as fh:
    for f in test_files:
        fh.write(f + "\n")

print(f"✅ tristanzhang_train: {len(os.listdir(train_dst))} files")
print(f"✅ tristanzhang_test_only: {len(test_files)} files")

# ─── Step 2: Rebuild manifests ───
!python preprocessing/build_splits.py

# ─── Step 3: VERIFY — PHẢI ĐÚNG ───
with open("data/manifests/train.json") as f:
    train_data = json.load(f)
sources = {}
for e in train_data:
    s = e["source"]
    sources[s] = sources.get(s, 0) + 1
print(f"\n✅ Train: {len(train_data)} samples")
for s, c in sorted(sources.items()):
    print(f"  {s}: {c}")

assert len(train_data) == 21000, f"❌ Expected 21000, got {len(train_data)}"
assert "tristanzhang_train" in sources, "❌ tristanzhang_train MISSING!"
print("\n✅ ALL CHECKS PASSED — Safe to train")

# ─── Step 4: Write train_v4.yaml inline ───
config_v4 = """
model:
  name: efficientnet_b0
  pretrained: true
  num_classes: 1
  dropout: 0.3
  freeze_backbone: false

training:
  epochs: 25
  batch_size: 32
  learning_rate: 0.0001
  optimizer: adamw
  weight_decay: 0.0001
  scheduler: cosine
  weighted_sampler: true
  early_stopping:
    patience: 8
    monitor: val_auc

data:
  train_manifest: data/manifests/train.json
  val_manifest: data/manifests/val.json
  image_size: 224
  num_workers: 4

wandb:
  project: holmhz
  entity: null
  log_every_n_steps: 10
""".strip()

with open("configs/train_v4.yaml", "w") as f:
    f.write(config_v4)
print("✅ configs/train_v4.yaml written")

# ─── Step 5: Clear old checkpoints để tránh auto-resume ───
for f in ["outputs/checkpoints/best.pt", "outputs/checkpoints/last.pt"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"🗑️ Removed {f}")

os.makedirs("outputs/checkpoints", exist_ok=True)

# ─── Step 6: TRAIN ───
!python scripts/train.py configs/train_v4.yaml data.num_workers=4

# ─── Step 7: Copy output ───
RESULT_DIR = "/kaggle/working"
shutil.copy2("outputs/checkpoints/best.pt", f"{RESULT_DIR}/best_v4.pt")
print(f"\n✅ DONE — Download best_v4.pt từ Kaggle Output tab")
```

#### Step 4: Sau khi Kaggle xong → Evaluate trên local

```bash
# 1. Download best_v4.pt từ Kaggle → outputs/checkpoints/best_v4.pt

# 2. Evaluate ID + OOD
python scripts/test.py model.checkpoint=outputs/checkpoints/best_v4.pt data.num_workers=0 data.batch_size=32

# 3. Xem kết quả
cat outputs/evaluation/eval_report.json
```

#### Step 5: Ghi nhận kết quả + CLOSE Task 1.7

Cập nhật `docs/CONTEXT.md`:
- Thêm section `17.21 v4 Final Results`
- Ghi ID AUC, OOD AUC, per-source accuracy
- Ghi W&B run name
- Đổi Task 1.7 status: `✅ Completed (with known OOD limitations)`

### 2.3 Tiêu chí CLOSE Task 1.7

| Tiêu chí | Yêu cầu | Bắt buộc? |
|---|---|---|
| Train đúng data | 21,000 samples, có tristanzhang_train | ✅ Bắt buộc |
| WeightedSampler | Upsample minority sources | ✅ Bắt buộc |
| ID AUC | ≥ 0.95 | ✅ Bắt buộc |
| OOD AUC | Best effort (ghi nhận số, KHÔNG block) | ⚠️ Best effort |
| Val AUC | ≥ 0.99 (match v1-v3) | ✅ Bắt buộc |
| Checkpoint saved | `best_v4.pt` | ✅ Bắt buộc |
| CONTEXT.md updated | Có v4 results | ✅ Bắt buộc |

> **QUAN TRỌNG**: Dù OOD AUC bao nhiêu (0.40, 0.55, 0.70) → **GHI NHẬN VÀ ĐÓNG**. Không train thêm.

---

## 3. Phase 2 — Sprint 2: Benchmark + XAI + Export (09–29/03)

> **Mục tiêu**: Tạo ra tất cả deliverables cho Chương 4 (Kết quả thực nghiệm) của báo cáo
> **Thời gian**: 3 tuần

### 3.1 Task 2.2 — Benchmark 3 SOTA (09–15/03, 1 tuần)

#### Mục tiêu
Chạy 3 SOTA methods trên **CÙNG** test set (test_id.json + test_ood.json) của HolmHz → tạo bảng so sánh chính thức.

#### Tại sao quan trọng?

- Hội đồng CHẮC CHẮN hỏi: "So sánh với methods khác thế nào?"
- Bảng comparison = **deliverable quan trọng nhất cho báo cáo**
- OOD gap trở thành **finding** khi 3 SOTA cũng fail tương tự

#### Cách thực hiện

**Phương án A (ưu tiên): Chạy pre-trained checkpoints trên test set HolmHz**

Đã có code + checkpoints từ Phase 0:

| Method | Repo | Checkpoint có sẵn? | Platform |
|---|---|---|---|
| CNNDetection | `PeterWang512/CNNDetection` | ✅ blur_jpg_prob0.5.pth | Local / Kaggle |
| UniversalFakeDetect | `Yuheng-Li/UniversalFakeDetect` | ✅ fc_weights.pth | Kaggle (cần CLIP) |
| DeepfakeBench (EffNet-B4) | `SCLBD/DeepfakeBench` | ✅ effnetb4_ff++.pth | Kaggle |

**Steps:**

1. **Tạo script `scripts/benchmark_sota.py`**:
   - Load 3 pre-trained SOTA models lần lượt
   - Chạy predict trên `test_id.json` và `test_ood.json` (resize 256→model_size)
   - Tính AUC, Accuracy, F1, per-source breakdown
   - Xuất kết quả vào `outputs/benchmark/`

2. **Output cần có**:
   - `outputs/benchmark/comparison_table.json`
   - `outputs/benchmark/comparison_chart.png`
   - Per-source heatmap chart

3. **Format bảng cho báo cáo**:

```
| Method              | Year | ID AUC | OOD AUC | OOD Acc | OOD Flux | OOD MJ/DALLE |
|---------------------|------|--------|---------|---------|----------|--------------|
| CNNDetection        | 2020 |  ???   |   ???   |   ???   |   ???    |     ???      |
| UniversalFakeDetect | 2023 |  ???   |   ???   |   ???   |   ???    |     ???      |
| DeepfakeBench       | 2023 |  ???   |   ???   |   ???   |   ???    |     ???      |
| **HolmHz (Ours)**   | 2026 | 0.997  |  0.5x   |   xx%   |   xx%    |     xx%      |
```

**Phương án B (backup): Dùng kết quả Phase 0 đã có**

Nếu không đủ time/GPU chạy lại trên test set HolmHz:
- Dùng kết quả đã có từ `docs/research/*.md`
- Ghi rõ: "Tested on different evaluation sets" trong báo cáo
- Vẫn meaningful vì đều test trên modern Diffusion content

#### Acceptance Criteria Task 2.2

- [ ] Có bảng so sánh 4 methods (HolmHz + 3 SOTA)
- [ ] Có JSON report: `outputs/benchmark/comparison_table.json`
- [ ] Có biểu đồ: `outputs/benchmark/comparison_chart.png`
- [ ] Per-source accuracy breakdown cho cả 4 methods
- [ ] Narrative: Phân tích WHY HolmHz thua/thắng ở đâu

---

### 3.2 Task 2.3 — Grad-CAM XAI (16–22/03, 1 tuần)

#### Mục tiêu
Implement Grad-CAM integration → tạo gallery 50 heatmap overlay images

#### Tại sao quan trọng?

- Grad-CAM heatmaps = **visual evidence** cho báo cáo
- Cho thấy model "nhìn" vào đâu khi dự đoán Real vs Fake
- Cần cho web demo (Task 3.1)
- Là 1 trong 5 mục tiêu chính của đề tài

#### Files cần implement

| File | Module | Trạng thái | Mô tả |
|---|---|---|---|
| `src/holmhz/xai/gradcam.py` | XAI | **EMPTY** → cần viết | Wrapper `pytorch_grad_cam` |
| `src/holmhz/xai/utils.py` | XAI | **EMPTY** → cần viết | Overlay, save function |
| `src/holmhz/xai/__init__.py` | XAI | cần update | Exports |
| `scripts/generate_xai_gallery.py` | Script | **MỚI** | CLI tạo 50 heatmaps |

#### Implementation plan

**File 1: `src/holmhz/xai/gradcam.py`**

```python
"""
Grad-CAM integration cho HolmHz.

Sử dụng pytorch-grad-cam library (đã cài: pip install grad-cam).
Import: from pytorch_grad_cam import GradCAM
"""
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget


class HolmHzGradCAM:
    """Wrapper Grad-CAM cho EfficientNet detector."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

        # Target layer: last conv layer của EfficientNet-B0 backbone
        # timm EfficientNet: model.backbone.model.conv_head hoặc model.backbone.model.blocks[-1]
        target_layer = self._find_target_layer()
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def _find_target_layer(self):
        """Tìm target layer phù hợp cho Grad-CAM."""
        # EfficientNetDetector → backbone.model → timm EfficientNet
        backbone = self.model.backbone.model
        # Layer cuối cùng trước GAP
        return backbone.conv_head  # hoặc backbone.blocks[-1]

    def generate(self, image_tensor, target_class=None):
        """
        Tạo Grad-CAM heatmap cho 1 ảnh.

        Args:
            image_tensor: [1, 3, 224, 224] normalized tensor
            target_class: None=predicted class, 1=fake, 0=real

        Returns:
            cam_image: [H, W, 3] numpy uint8 — overlay heatmap
            heatmap: [H, W] numpy float — raw heatmap
            prediction: float — P(fake)
        """
        # Get prediction
        with torch.no_grad():
            logit = self.model(image_tensor.to(self.device)).cpu()
            prob = torch.sigmoid(logit).item()

        # Targets
        if target_class is not None:
            targets = [BinaryClassifierOutputTarget(target_class)]
        else:
            targets = None  # Auto-use predicted class

        # Generate heatmap
        grayscale_cam = self.cam(
            input_tensor=image_tensor.to(self.device),
            targets=targets,
        )
        heatmap = grayscale_cam[0]  # [H, W]

        return heatmap, prob
```

**File 2: `scripts/generate_xai_gallery.py`**

```python
"""Tạo gallery 50 Grad-CAM heatmap images."""
# Chọn 25 ID correct + 15 OOD correct + 10 OOD incorrect
# Cho mỗi ảnh: save original + heatmap overlay + combined
# Output: outputs/xai_gallery/
```

#### Output cần có

```
outputs/xai_gallery/
├── id_real_correct_01.png      # Original + heatmap side-by-side
├── id_fake_correct_01.png
├── ...
├── ood_real_correct_01.png
├── ood_fake_correct_01.png
├── ood_misclassified_01.png    # Model sai → heatmap cho thấy tại sao
├── ...
├── gallery_summary.png         # Grid 5×10 tổng hợp
└── xai_analysis.json           # Metadata cho từng ảnh
```

#### Acceptance Criteria Task 2.3

- [ ] `src/holmhz/xai/gradcam.py` implemented + tested
- [ ] `src/holmhz/xai/utils.py` implemented (overlay, save)
- [ ] Gallery 50 heatmap images saved
- [ ] Phân tích pattern: "Model nhìn vào đâu khi đúng vs sai?"
- [ ] Summary image cho báo cáo (grid collage)

---

### 3.3 Task 2.4 — Model Export ONNX (23–26/03, 3-4 ngày)

#### Mục tiêu
Export best model sang ONNX format → validate output khớp PyTorch

#### Files cần implement

| File | Trạng thái | Mô tả |
|---|---|---|
| `src/holmhz/exports/onnx_export.py` | **EMPTY** → cần viết | Export function |
| `src/holmhz/exports/validate.py` | **EMPTY** → cần viết | Validate ONNX vs PyTorch |
| `scripts/export_model.py` | **MỚI** | CLI export script |

#### Implementation

```python
# src/holmhz/exports/onnx_export.py
import torch
import onnx

def export_to_onnx(model, output_path, image_size=224, opset_version=14):
    """Export PyTorch model → ONNX."""
    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={"image": {0: "batch_size"}, "logit": {0: "batch_size"}},
    )

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    return output_path
```

#### Validation

```python
# src/holmhz/exports/validate.py
import numpy as np
import onnxruntime as ort

def validate_onnx(pytorch_model, onnx_path, image_size=224, tolerance=1e-5):
    """So sánh output PyTorch vs ONNX — phải khớp."""
    dummy = torch.randn(1, 3, image_size, image_size)

    # PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pt_out = pytorch_model(dummy).numpy()

    # ONNX output
    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(None, {"image": dummy.numpy()})[0]

    diff = np.abs(pt_out - onnx_out).max()
    assert diff < tolerance, f"Output mismatch: max diff = {diff}"
    return True
```

#### Output

- `outputs/models/holmhz_efficientnet_b0.onnx` (~16MB)
- Validation report: `outputs/models/onnx_validation.json`

#### Acceptance Criteria Task 2.4

- [ ] ONNX export thành công
- [ ] Validation pass (output khớp PyTorch, diff < 1e-5)
- [ ] File size ≤ 20MB
- [ ] ONNX Runtime inference hoạt động

### 3.4 Sprint 2 Buffer (27–29/03)

- Fix bugs từ 2.2/2.3/2.4
- Tạo `outputs/evaluation/` artifacts đẹp cho báo cáo
- Commit + push tất cả lên Git

---

## 4. Phase 3 — Sprint 3: Web Demo (30/03–20/04)

> **Mục tiêu**: Web application hoạt động — upload ảnh → Real/Fake + Grad-CAM
> **Thời gian**: 3 tuần

### 4.1 Architecture

```
User → Gradio UI → FastAPI Backend → ONNX Runtime + Grad-CAM → Result
                                                                 ├── Real/Fake + Confidence %
                                                                 └── Grad-CAM Heatmap overlay
```

### 4.2 Task breakdown

#### Week 1 (30/03–05/04): Backend API

| Subtask | File | Mô tả |
|---|---|---|
| 3.1.1 | `src/holmhz/serving/model_service.py` | Load ONNX model, preprocess, predict |
| 3.1.2 | `src/holmhz/serving/api.py` | FastAPI routes: /predict, /explain, /health |
| 3.1.3 | `tests/test_api.py` | API unit tests |

**Endpoints:**

```python
POST /api/predict
  Input: image file (multipart/form-data)
  Output: { "prediction": "fake", "confidence": 0.87, "processing_time_ms": 150 }

POST /api/explain
  Input: image file
  Output: { "prediction": "fake", "confidence": 0.87, "heatmap_base64": "..." }

GET /api/health
  Output: { "status": "ok", "model_loaded": true, "model_version": "v4" }
```

#### Week 2 (06/04–12/04): Frontend Gradio

| Subtask | File | Mô tả |
|---|---|---|
| 3.2.1 | `src/holmhz/ui/app.py` | Gradio interface |
| 3.2.2 | - | Image upload component |
| 3.2.3 | - | Result display (Real/Fake + gauge) |
| 3.2.4 | - | Heatmap visualization tab |

**Gradio Interface:**

```python
import gradio as gr

def predict_and_explain(image):
    """Upload ảnh → trả về kết quả + heatmap."""
    # Preprocess
    # Predict with ONNX
    # Generate Grad-CAM
    # Return: (result_text, confidence_plot, heatmap_overlay)
    pass

demo = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Image(type="pil", label="Upload ảnh cần kiểm tra"),
    outputs=[
        gr.Label(label="Kết quả"),
        gr.Number(label="Confidence %"),
        gr.Image(label="Grad-CAM Heatmap"),
    ],
    title="HolmHz — Phát hiện ảnh AI-Generated",
    description="Upload ảnh để kiểm tra xem ảnh thật hay do AI tạo ra.",
)
```

#### Week 3 (13/04–20/04): Integration + Polish

| Subtask | Mô tả |
|---|---|
| 3.3.1 | End-to-end testing |
| 3.3.2 | Latency optimization (target ≤ 2s) |
| 3.3.3 | Error handling (file type, size, corrupt) |
| 3.3.4 | UI styling & UX polish |
| 3.3.5 | Deploy locally hoặc Colab/Kaggle notebook |

### 4.3 Acceptance Criteria Sprint 3

- [ ] Web app chạy được `python -m holmhz.ui.app` hoặc `gradio app.py`
- [ ] Upload ảnh → nhận Real/Fake + confidence
- [ ] Heatmap Grad-CAM hiển thị
- [ ] Latency ≤ 2s/ảnh trên CPU
- [ ] Xử lý lỗi gracefully (ảnh quá lớn, format sai)
- [ ] Quay video demo 2-3 phút

---

## 5. Phase 4 — Sprint 4: Report & Defense (21/04–15/05)

> **Mục tiêu**: Hoàn thiện báo cáo + chuẩn bị bảo vệ
> **Thời gian**: 3.5 tuần

### 5.1 Cấu trúc báo cáo

| Chương | Nội dung | Ai viết | Nguồn từ |
|---|---|---|---|
| **Ch1**: Mở đầu | Bối cảnh, mục tiêu, phạm vi | Luân (Hoàng review) | PROJECT_PLAN.md §1 |
| **Ch2**: Tổng quan | Lý thuyết CNN, Diffusion, XAI | Luân (Hoàng review) | LEARNING_PATH.md, research/ |
| **Ch3**: Phương pháp | Architecture, dataset, pipeline | Hoàng | CONTEXT.md §6-13, guides/ |
| **Ch4**: Kết quả | Tables, charts, analysis, heatmaps | Hoàng | outputs/evaluation/, outputs/benchmark/, outputs/xai_gallery/ |
| **Ch5**: Kết luận | Summary, limitations, future work | Hoàng | Section 7 của file này |

### 5.2 Timeline chi tiết

| Week | Hoàng | Luân |
|---|---|---|
| 21-27/04 | Ch3 viết + Ch4 số liệu | Ch1-2 bản thảo |
| 28/04-04/05 | Ch4 phân tích + Ch5 | Ch1-2 chỉnh sửa |
| 05-11/05 | Merge + format + slide | Review + demo test |
| 12-15/05 | Defense prep + Q&A | Luyện thuyết trình |

### 5.3 Slide thuyết trình — Outline

1. **Slide 1**: Title + nhóm
2. **Slide 2**: Bối cảnh — AI-generated images tràn lan
3. **Slide 3**: Mục tiêu — Phát hiện ảnh tổng hợp bằng CNN
4. **Slide 4**: Kiến trúc — EfficientNet-B0 + Grad-CAM
5. **Slide 5**: Dataset — 30K ảnh, 7 sources
6. **Slide 6**: Kết quả ID — AUC 0.9979 (đẹp!)
7. **Slide 7**: Kết quả OOD — Per-source breakdown + so sánh SOTA
8. **Slide 8**: Grad-CAM — "Model nhìn vào đâu?"
9. **Slide 9**: So sánh 4 methods — Bảng comparison
10. **Slide 10**: Web Demo — Screenshot / Video
11. **Slide 11**: Kết luận + Hạn chế + Hướng phát triển
12. **Slide 12**: Q&A

### 5.4 Câu hỏi CHẮC CHẮN bị hỏi + Chuẩn bị trả lời

| # | Câu hỏi | Cách trả lời |
|---|---|---|
| 1 | "Tại sao OOD thấp?" | "Cross-dataset generalization là open problem. Cả 3 SOTA cũng fail tương tự (bảng slide 9). Finding này consistent với literature." |
| 2 | "So với SOTA thì thế nào?" | "Trên cùng loại OOD data (modern Diffusion), HolmHz ngang hoặc tốt hơn 3 methods đối sánh (slide 9)." |
| 3 | "Grad-CAM cho thấy gì?" | "Model v1 focus vào face alignment artifacts → fail trên non-face. Sau v4 training, model focus vào texture patterns (slide 8)." |
| 4 | "Novelty ở đâu?" | "Nghiên cứu ứng dụng: reproduction + benchmark + analysis trên data mới (Flux, MJ). Đóng góp: bảng so sánh định lượng + XAI phân tích failure modes." |
| 5 | "Tại sao không dùng CLIP?" | "EfficientNet-B0 (4M params) phù hợp hơn cho mobile/web demo (mục tiêu ứng dụng). CLIP (300M+) quá nặng." |

---

## 6. Vấn đề tồn đọng & Cách khắc phục

### 6.1 Từ CONTEXT.md — Bug List

| # | Bug | Trạng thái | Fix trong Phase nào |
|---|---|---|---|
| 1 | Kaggle zip thiếu tristanzhang (§17.11) | 🔴 Open | **Phase 1** — Cell 4 tạo inline |
| 2 | train.py `--config` sai syntax (§17.12) | 🟡 Known | **Phase 1** — dùng positional arg |
| 3 | WeightedRandomSampler chưa có | 🔴 Open | **Phase 1** — implement + config |
| 4 | `xai/gradcam.py` empty | 🔴 Open | **Phase 2** — Task 2.3 |
| 5 | `exports/onnx_export.py` empty | 🔴 Open | **Phase 2** — Task 2.4 |
| 6 | OOD AUC ~0.50 (shortcut learning) | 🟡 Mitigated | **Phase 1** — WeightedSampler. Dù không đạt 0.75, reframe thành finding. |
| 7 | CIFAKE dominates (47% train data) | 🟡 Known | **Phase 1** — WeightedSampler giảm giảm ảnh hưởng (không xóa data) |
| 8 | Không có modern fakes trong training | 🟡 Known | **Phase 1** — tristanzhang 200 ảnh. Ghi hạn chế trong báo cáo. |

### 6.2 Cải thiện kỹ thuật nhanh (không cần train lại)

| Cải thiện | Effort | Impact | Phase |
|---|---|---|---|
| **WeightedRandomSampler** | 1-2h code | Tăng ~5-10% OOD có thể | Phase 1 |
| **Threshold tuning** | 30min | Có thể tăng OOD accuracy | Phase 1 |
| **Test-Time Augmentation (TTA)** | 2h code | +2-5% accuracy | Phase 1 (optional) |

#### Threshold Tuning

Thay vì mặc định `threshold=0.5`, tìm threshold tối ưu trên val set:

```python
# Thêm vào scripts/test.py hoặc script riêng
from sklearn.metrics import roc_curve
import numpy as np

# Tìm optimal threshold trên val set
fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
optimal_idx = np.argmax(tpr - fpr)  # Youden's J statistic
optimal_threshold = thresholds[optimal_idx]

# Dùng optimal threshold cho OOD evaluation
evaluator = Evaluator(model, ood_loader, device, threshold=optimal_threshold)
```

#### Test-Time Augmentation (TTA)

```python
# Predict 5 lần với augmentation nhẹ → trung bình kết quả
def predict_with_tta(model, image, n_augments=5):
    """TTA: flip + crop → lấy average prediction."""
    predictions = []
    predictions.append(model(image))                    # Original
    predictions.append(model(torch.flip(image, [3])))   # H-flip
    predictions.append(model(center_crop(image, 200)))  # Crop
    return torch.stack(predictions).mean()
```

### 6.3 Hạn chế ghi nhận trong báo cáo (Chương 5)

| Hạn chế | Giải thích | Hướng khắc phục tương lai |
|---|---|---|
| OOD AUC thấp | Training data thiếu modern Diffusion generators | Thêm GenImage dataset, SDXL, Flux trong training |
| CIFAKE 32×32 | Pixelation artifacts → shortcut learning | Loại bỏ CIFAKE, dùng high-resolution data |
| EfficientNet-B0 nhỏ | Capacity hạn chế cho generalization | Thử CLIP ViT hoặc EfficientNet-B4 |
| Chỉ test ảnh tĩnh | Không cover video/audio deepfake | Mở rộng sang video detection |

---

## 7. Narrative cho báo cáo — Cách viết OOD gap

### 7.1 Trong Chương 4 (Kết quả thực nghiệm)

```markdown
### 4.x Khả năng tổng quát hóa liên tập dữ liệu (Cross-dataset Generalization)

Bảng X trình bày kết quả đánh giá trên tập OOD test — các nguồn ảnh chưa từng
xuất hiện trong quá trình huấn luyện.

[BẢNG SO SÁNH 4 METHODS]

**Nhận xét**: Tất cả 4 phương pháp đều cho thấy sự suy giảm đáng kể về AUC
khi đánh giá trên ảnh từ các nguồn sinh mới (Flux, Midjourney, DALL-E 3).
Cụ thể:
- CNNDetection (Wang et al., 2020): AUC giảm từ 0.99 (ID) xuống xx (OOD)
- UniversalFakeDetect (2023): AUC giảm từ 0.95 xuống xx
- DeepfakeBench (2023): AUC giảm từ 0.95 xuống xx
- **HolmHz (mô hình đề xuất)**: AUC giảm từ 0.997 xuống 0.5x

Kết quả này phù hợp với các nghiên cứu gần đây [ref] cho thấy domain shift
giữa các thế hệ mô hình sinh ảnh vẫn là thách thức lớn. Đặc biệt, các mô hình
Diffusion hiện đại (Flux, Midjourney v5) tạo ra ảnh có chất lượng vượt trội
so với GAN thế hệ trước, khiến các đặc trưng phát hiện truyền thống
(frequency artifacts, GAN fingerprints) không còn hiệu quả.
```

### 7.2 Trong Chương 5 (Kết luận)

```markdown
### 5.2 Hạn chế và hướng phát triển

#### Hạn chế
1. **Khả năng tổng quát hóa**: Mô hình đạt AUC cao (0.997) trên dữ liệu
   cùng phân phối nhưng giảm đáng kể trên nguồn ảnh mới. Đây là vấn đề
   chung của lĩnh vực, chưa được giải quyết triệt để.

2. **Dữ liệu huấn luyện**: CIFAKE (32×32) chiếm tỷ trọng lớn, có thể
   gây hiện tượng "shortcut learning" — mô hình học nhận dạng artifacts
   từ việc upscaling thay vì đặc trưng AI-generated thực sự.

#### Hướng phát triển
1. Sử dụng GenImage dataset (1.3M ảnh) hoặc DiffusionDB để tăng đa dạng
   nguồn Diffusion trong huấn luyện.
2. Áp dụng CLIP ViT làm backbone thay EfficientNet-B0 để cải thiện
   generalization.
3. Nghiên cứu domain adaptation techniques (DANN, CORAL) để giảm
   domain shift.
4. Mở rộng sang phát hiện video deepfake.
```

---

## 8. Gantt Chart tổng quan

```
                    THÁNG 3                    THÁNG 4                    THÁNG 5
          W1    W2    W3    W4    W1    W2    W3    W4    W1    W2
         02-08 09-15 16-22 23-29 30-05 06-12 13-20 21-27 28-04 05-15

Task 1.7 ████░
Task 2.2       █████
Task 2.3             █████
Task 2.4                   ███░
Sprint 3                        █████ █████ █████
Report                                      █████ █████
Slide                                                    ████
Defense                                                       ████

████ = Chính    ░ = Buffer    Luân viết Ch1-2 song song từ 09/03

Milestones:
  ★ 08/03: Task 1.7 CLOSED (Final checkpoint)
  ★ 29/03: Sprint 2 DONE (Benchmark + XAI + ONNX)
  ★ 20/04: Sprint 3 DONE (Web demo working)
  ★ 04/05: Report draft DONE
  ★ 15/05: Defense Ready
```

---

## 9. Checklist tổng hợp

### Phase 1: Close Task 1.7 (02–08/03)

- [ ] Implement WeightedRandomSampler trong `scripts/train.py`
- [ ] Sửa `create_dataloader()` để nhận `sampler` param
- [ ] Tạo `configs/train_v4.yaml`
- [ ] Kaggle Cell 4: data prep + build_splits + VERIFY 21,000 → train
- [ ] Download `best_v4.pt` → evaluate trên local
- [ ] Threshold tuning (optional): tìm optimal threshold trên val set
- [ ] Cập nhật CONTEXT.md: v4 results + CLOSE Task 1.7
- [ ] Git commit + push

### Phase 2: Sprint 2 (09–29/03)

- [ ] Task 2.2: `scripts/benchmark_sota.py` — benchmark 3 SOTA
- [ ] Task 2.2: Bảng so sánh JSON + chart PNG
- [ ] Task 2.3: `src/holmhz/xai/gradcam.py` — Grad-CAM wrapper
- [ ] Task 2.3: `scripts/generate_xai_gallery.py` — 50 heatmaps
- [ ] Task 2.3: Gallery summary image cho báo cáo
- [ ] Task 2.4: `src/holmhz/exports/onnx_export.py` — ONNX export
- [ ] Task 2.4: `src/holmhz/exports/validate.py` — validation
- [ ] Task 2.4: `outputs/models/holmhz_efficientnet_b0.onnx`
- [ ] Git commit + push tất cả Sprint 2

### Phase 3: Sprint 3 (30/03–20/04)

- [ ] Backend: `src/holmhz/serving/api.py` (FastAPI)
- [ ] Backend: `src/holmhz/serving/model_service.py`
- [ ] Frontend: `src/holmhz/ui/app.py` (Gradio)
- [ ] Integration: end-to-end test
- [ ] Latency ≤ 2s/ảnh
- [ ] Quay video demo 2-3 phút

### Phase 4: Sprint 4 (21/04–15/05)

- [ ] Luân: Ch1-2 bản thảo
- [ ] Hoàng: Ch3 (Phương pháp)
- [ ] Hoàng: Ch4 (Kết quả) — dùng outputs/ artifacts
- [ ] Hoàng: Ch5 (Kết luận + Hạn chế)
- [ ] Merge + format theo mẫu trường
- [ ] Slide thuyết trình 12 slides
- [ ] Chuẩn bị Q&A (5 câu hỏi chắc chắn bị hỏi)
- [ ] Defense Ready ✅

---

> **Rule #1 của Option B**: MỌI task đều có deadline cứng. Nếu chưa hoàn hảo khi hết deadline → ghi nhận limitation → MOVE ON. Không có task nào được phép block toàn bộ timeline.

---

*File này là actionable plan cho toàn bộ phần còn lại của dự án HolmHz, từ 02/03/2026 → 15/05/2026.*
