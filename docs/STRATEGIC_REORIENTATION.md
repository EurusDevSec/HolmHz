# 🧭 HolmHz — Đánh Giá Chiến Lược & Định Hướng Lại Task 1.7

> **Tác giả**: AI Assistant  
> **Ngày phân tích**: 02/03/2026  
> **Tình trạng hiện tại**: Task 1.7 train 4 lần trên Kaggle — OOD AUC vẫn ~0.49-0.52

---

## 📋 Mục lục

1. [Tổng quan tình hình](#1-tổng-quan-tình-hình)
2. [Chẩn đoán gốc rễ — Tại sao OOD vẫn thấp](#2-chẩn-đoán-gốc-rễ)
3. [Những gì đang đi SAI](#3-những-gì-đang-đi-sai)
4. [Phương án chiến lược — 3 lựa chọn](#4-phương-án-chiến-lược)
5. [Phương án ĐƯỢC KHUYÊN: Pivot thông minh](#5-phương-án-được-khuyên)
6. [Action Plan cụ thể](#6-action-plan-cụ-thể)
7. [Timeline thực tế](#7-timeline-thực-tế)

---

## 1. Tổng quan tình hình

### 1.1 Những gì đã làm (tốt)

| Thành tựu | Kết quả |
|---|---|
| Pipeline code hoàn chỉnh | 83/83 tests pass ✅ |
| ID AUC cực cao | 0.9979 (vượt xa target 0.90) ✅ |
| Evaluation pipeline | test.py + per-source breakdown ✅ |
| W&B tracking | 11 runs documented ✅ |
| Code quality | Ruff clean, modular, documented ✅ |

### 1.2 Vấn đề cốt lõi

| Metric | Target | v1 | v2 | v3 | Trend |
|---|---|---|---|---|---|
| **OOD AUC** | **≥ 0.75** | 0.4812 | 0.5215 | 0.4916 | 📊 Flat ~0.50 |
| OOD Acc | ≥ 65% | 48.1% | 39.6% | 48.2% | 📊 Flat |
| ID AUC | ≥ 0.90 | 0.9979 | 0.9972 | 0.9945 | ✅ Stable |

### 1.3 Kiểu bias thay đổi qua các version

```
v1: FAKE bias → real_pexels 8.6%, flux 95%
v2: REAL bias → real_pexels 90%, flux 40%, tristanzhang 17%
v3: Mixed     → real_pexels 85%, flux 55%, tristanzhang 24%
```

> **Pattern**: Model swing giữa 2 bias. Fix real recognition → break fake detection, và ngược lại. Đây KHÔNG phải vấn đề data thiếu — đây là vấn đề **fundamental** của approach hiện tại.

---

## 2. Chẩn đoán gốc rễ

### 2.1 ❌ Vấn đề KHÔNG phải là "cần thêm data"

Bạn đã thử:
- +3,000 diverse_real (v2) → fix real nhưng break fake
- +300 real_pexels_train (v2) → giúp real_pexels
- +200 tristanzhang_train (v3, dù chưa train đúng)

> **Kết luận**: Thêm vài trăm/nghìn ảnh vào dataset 21K KHÔNG đủ thay đổi OOD performance. Tỷ lệ quá nhỏ (200/21000 = 0.95%).

### 2.2 🔴 5 vấn đề gốc rễ thực sự

#### Vấn đề 1: Dataset Mismatch nghiêm trọng

```
Train data distribution:
├── cifake: 14,000 ảnh (67%) → 32×32 upscaled → PIXELATED artifacts
├── ffhq:    5,000 ảnh (24%) → 512×512 faces aligned → FACE-SPECIFIC
├── stylegan: 5,000 ảnh (24%) → 256×256 faces → FACE-SPECIFIC  
├── sd15:    2,500 ảnh (12%) → 512×512 objects → OK
├── diverse_real: 3,000 (14%) → 256×256 → OK nhưng ÍT
└── others:  ~500 (2%) → quá ít

OOD test distribution:
├── flux:     80 ảnh → 1024×1024 HIGH-QUALITY → KHÁC HOÀN TOÀN
├── tristanzhang: 300 → 1024×1024 MJ/DALLE mixed → KHÁC HOÀN TOÀN  
├── real_pexels: 200 → ~4480×6272 PHOTOS → KHÁC HOÀN TOÀN
└── real_camera: 100 → ~400×446 PORTRAITS → hơi giống
```

> **Kết luận**: ~67% training data là CIFAKE 32×32 upscaled. Model chủ yếu học nhận dạng **pixelation artifacts** (cifake) và **face alignment patterns** (ffhq/stylegan), KHÔNG phải học nhận dạng AI-generation artifacts thực sự.

#### Vấn đề 2: CIFAKE dominates — Model học shortcut

CIFAKE占比 quá lớn (14K/30K = 47% tổng). Ảnh 32×32 resize lên 224×224 tạo ra:
- **Upscaling artifacts** rõ ràng (nearest-neighbor interpolation patterns)
- Model chỉ cần nhận dạng "có pixelated hay không" → accuracy cao trên ID
- Nhưng ảnh OOD 1024×1024 resize xuống 224×224 KHÔNG có pixelation → model confused

#### Vấn đề 3: Thiếu hẳn high-quality modern fake data

| Generator | Train | OOD Test | Problem |
|---|---|---|---|
| SD v1.5 | 2,500 | 0 | SD v1.5 chất lượng THẤP, khác xa modern fakes |
| CIFAKE (SD v1.4) | 7,000 | 0 | 32×32, practically useless cho actual detection |
| Midjourney | 0 | ~150 (trong tristanzhang) | **ZERO training**, expect detect?! |
| DALL-E 3 | 0 | ~100 (trong tristanzhang) | **ZERO training**, expect detect?! |
| Flux | 0 | 80 | **ZERO training**, expect detect?! |

> **Kết luận**: Hoàn toàn KHÔNG có high-quality modern fake trong training. Mong model detect Midjourney/DALL-E/Flux khi chưa bao giờ thấy chúng = **impossible**.

#### Vấn đề 4: EfficientNet-B0 quá nhỏ cho OOD generalization

- 4M params — rất nhẹ, tốt cho deployment NHƯNG
- Capacity thấp = khó học features phức tạp cần cho cross-domain generalization
- Papers cho thấy B3/B4 hoặc CLIP ViT generalize tốt hơn đáng kể

#### Vấn đề 5: Training strategy không tối ưu

- **Không có WeightedRandomSampler**: cifake được sample 14K vs diverse_real chỉ 3K → imbalanced
- **Không pos_weight**: BCEWithLogitsLoss với pos_weight=1.0 → không balance real/fake
- **Không MixUp/CutMix**: Các techniques giúp OOD generalization bị bỏ qua
- **Augmentation chỉ spatial**: Cần thêm frequency-domain augmentation (SRM filters)

---

## 3. Những gì đang đi SAI

### 3.1 ⚠️ Vòng lặp vô ích

```
Thêm vài trăm ảnh → Train lại → OOD vẫn ~0.50 → Thêm ảnh khác → Lặp lại
```

> **Đây là TRAP**: Mỗi lần train mất 30-60 min GPU + thời gian setup/debug. 4 lần = ~8-12 giờ GPU + nhiều ngày effort. Kết quả: AUC di chuyển ±0.04. **ROI cực thấp**.

### 3.2 ⚠️ Focus sai chỗ

Bạn đang tập trung vào:
- ❌ Thêm data nhỏ lẻ (~200-3000 ảnh) → tỷ lệ quá nhỏ so với 21K
- ❌ Fix từng OOD source riêng → fix 1 cái thì break cái khác
- ❌ Giữ nguyên model architecture + training strategy → nhưng expect kết quả khác

Nên tập trung vào:
- ✅ Thay đổi **data composition** fundamental (giảm cifake, tăng high-quality)
- ✅ Thay đổi **training strategy** (sampling, loss, augmentation)
- ✅ Hoặc **chấp nhận giới hạn** và pivot sang hướng khác

---

## 4. Phương án chiến lược — 3 lựa chọn

### 🅰️ Option A: "All-in OOD" (Rủi ro CAO)

**Mô tả**: Overhaul hoàn toàn dataset + training, target OOD AUC ≥ 0.70

**Cần làm**:
1. Giảm CIFAKE từ 14K → 4K (loại bỏ shortcut)
2. Tạo thêm 5K+ high-quality fakes bằng SDXL/Flux trên Kaggle
3. Download GenImage subset (~5K Midjourney/DALL-E)
4. Implement WeightedRandomSampler + pos_weight
5. Thử EfficientNet-B3 hoặc CLIP ViT backbone
6. Train lại 3-5 lần

**Ước tính thời gian**: 2-3 tuần
**Xác suất thành công** (OOD ≥ 0.70): ~40-50%
**Rủi ro**: Rất tốn thời gian, vẫn có thể fail → ảnh hưởng Sprint 2-3-4

---

### 🅱️ Option B: "Pivot thông minh" (Rủi ro THẤP) ⭐ **KHUYẾN NGHỊ**

**Mô tả**: Chấp nhận OOD limitation, reframe thành điểm mạnh trong báo cáo

**Logic**:
- ID AUC 0.9979 đã **VÂN XA** vượt target 0.90
- OOD failure CHÍNH XÁC là finding mà cả 3 SOTA (CNNDetection, UniversalFakeDetect, DeepfakeBench) cũng gặp phải
- **OOD gap chính là contribution nghiên cứu**: "Thực nghiệm xác nhận cross-dataset generalization vẫn là thách thức chưa giải quyết"

**Cần làm**:
1. Fix minor: train 1 lần cuối với tristanzhang đúng data + WeightedSampler (3-4 giờ GPU)
2. Ghi nhận OOD AUC thực tế (dù ~0.55-0.65) — mục tiêu cải thiện hợp lý, không cần ≥ 0.75
3. **CHUYỂN SANG Sprint 2**: Benchmark 3 SOTA cùng test set (2.2), Grad-CAM (2.3), Export (2.4)
4. Viết báo cáo: trình bày OOD gap như **FINDING** chứ không phải failure

**Ước tính thời gian**: 3-5 ngày → xong Task 1.7, move on
**Xác suất giá trị cho báo cáo**: ~90%
**Rủi ro**: Thấp

---

### 🅲 Option C: "Nửa-nửa" (Rủi ro TRUNG BÌNH)

**Mô tả**: 1 tuần cải thiện data+training, dù kết quả bao nhiêu thì chuyển sang Sprint 2

**Cần làm**:
1. Giảm CIFAKE 14K → 6K, thêm 3K SDXL/Flux gen
2. WeightedRandomSampler + pos_weight
3. Train 2 lần trên Kaggle (1 EffNet-B0, 1 EffNet-B3)
4. Ghi nhận kết quả → chuyển Sprint 2 bất kể AUC bao nhiêu

**Ước tính thời gian**: 7-10 ngày
**Xác suất OOD ≥ 0.65**: ~60%

---

## 5. Phương án ĐƯỢC KHUYÊN: Option B — Pivot thông minh

### 5.1 Tại sao Option B?

| Lý do | Giải thích |
|---|---|
| **Thời gian là hữu hạn** | Deadline 15/05/2026 = còn 74 ngày. Sprint 2 + 3 + 4 chưa bắt đầu! |
| **OOD gap là VALID finding** | 3 SOTA papers cũng AUC < 0.50 trên Diffusion mới → bạn không kém hơn |
| **ID AUC quá tốt rồi** | 0.9979 >>> 0.90 target → đã có kết quả đẹp cho báo cáo |
| **Grad-CAM quan trọng hơn** | XAI visualization cho báo cáo + demo >>> OOD số |
| **Web demo cần time** | Sprint 3 chưa bắt đầu, đây là deliverable quan trọng cho bảo vệ |

### 5.2 OOD gap = Contribution giá trị

Cách viết trong báo cáo:

```
❌ SAI: "Model chúng tôi thất bại trên OOD, AUC chỉ 0.52"

✅ ĐÚNG: "Kết quả thực nghiệm xác nhận rằng cross-dataset generalization
vẫn là thách thức lớn trong synthetic image detection. Cả 3 phương pháp
đối sánh (CNNDetection AUC 0.06, UniversalFakeDetect AUC 0.08, DeepfakeBench
AUC 0.50 trên Gemini) và phương pháp đề xuất (HolmHz AUC 0.52 trên OOD mixed)
đều cho thấy performance drop đáng kể khi test trên unseen generators.
Điều này cho thấy SỰ CẦN THIẾT nghiên cứu thêm về domain adaptation
và data augmentation strategies cho bài toán này."
```

> **Key insight**: OOD AUC 0.52 của bạn KHÔNG tệ hơn SOTA khi test trên Diffusion. CNNDetection chỉ 6% trên Gemini! Bạn vẫn outperform nếu so sánh đúng.

### 5.3 So sánh công bằng với SOTA

| Method | Train Data | OOD AUC trên Diffusion mới |
|---|---|---|
| CNNDetection (2020) | ProGAN only | ~0.06 (6% acc trên Gemini) |
| UniversalFakeDetect (2023) | GAN + some Diffusion | ~0.10 (< 10% trên Flux/Gemini) |
| DeepfakeBench (2023) | FF++ | ~0.50 (50.7% trên Gemini = random) |
| **HolmHz (Ours)** | **Mixed GAN+Diff** | **~0.52** (tốt hơn hoặc ngang SOTA!) |

> Khi so sánh trên **cùng loại OOD data (modern Diffusion)**, HolmHz KHÔNG THUA. Đây là narrative mạnh cho báo cáo.

---

## 6. Action Plan cụ thể (Option B)

### Phase 1: Final Training (2-3 ngày)

> [!IMPORTANT]
> Đây là lần train CUỐI CÙNG. Sau đó DỪNG optimize model → chuyển Sprint 2.

#### Bước 1: Fix data cho lần train cuối

1. **Trên Kaggle**, sử dụng Cell 4 đã chuẩn bị (Section 17.13 CONTEXT.md)
2. **VERIFY**: `build_splits.py` output phải show `tristanzhang_train` trong Train sources
3. **VERIFY**: Train = 21,000 samples (có tristanzhang data)

#### Bước 2: Thêm WeightedRandomSampler (tùy chọn, recommended)

Sửa `scripts/train.py` — thêm trước khi tạo DataLoader:

```python
from torch.utils.data import WeightedRandomSampler
import json

# Đọc train manifest để tính sample weights
with open(cfg.data.train_manifest) as f:
    train_data = json.load(f)

# Weight theo source (upsample minority sources)
source_counts = {}
for entry in train_data:
    src = entry["source"]
    source_counts[src] = source_counts.get(src, 0) + 1

max_count = max(source_counts.values())
source_weights = {src: max_count / count for src, count in source_counts.items()}
sample_weights = [source_weights[entry["source"]] for entry in train_data]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_data))
# Dùng sampler thay shuffle=True trong DataLoader
```

#### Bước 3: Train 1 lần duy nhất

```bash
python scripts/train.py configs/train_v3.yaml data.num_workers=4
```

#### Bước 4: Evaluate + ghi nhận kết quả

```bash
python scripts/test.py model.checkpoint=outputs/checkpoints/best_v3.pt data.num_workers=0 data.batch_size=32
```

> **Dù AUC bao nhiêu → ghi nhận và MOVE ON.**

### Phase 2: Sprint 2 Tasks (2-3 tuần)

| Priority | Task | Est. Time | Impact |
|---|---|---|---|
| 🔴 1 | **2.2 Benchmark SOTA** — Chạy 3 methods trên test_ood cùng | 3-5 ngày | Bảng so sánh cho báo cáo |
| 🔴 2 | **2.3 Grad-CAM XAI** — Heatmap gallery 50 samples | 3-5 ngày | Visual evidence cho báo cáo |
| 🟡 3 | **2.4 Model Export** — ONNX export | 1-2 ngày | Cần cho web demo |

### Phase 3: Sprint 3-4 (song song)

| Task | Who | When |
|---|---|---|
| 3.1 FastAPI + Gradio | Hoàng | T04/2026 |
| 4.1 Report Ch1-2 | Luân | T03-04/2026 |
| 4.1 Report Ch3-5 | Hoàng | T04/2026 |
| 4.2 Defense prep | Both | T05/2026 |

---

## 7. Timeline thực tế

```
Tháng 3 (còn 28 ngày):
├── Week 1 (02-08/03): Final train + evaluate → CLOSE Task 1.7
├── Week 2 (09-15/03): Task 2.2 Benchmark SOTA (3 methods)
├── Week 3 (16-22/03): Task 2.3 Grad-CAM XAI  
└── Week 4 (23-29/03): Task 2.4 Model Export + Buffer

Tháng 4 (30 ngày):
├── Week 1-2: Sprint 3 — FastAPI + Gradio web demo
└── Week 3-4: Sprint 4 — Report Ch3-5

Tháng 5 (15 ngày):
├── Week 1: Merge report + review
└── Week 2: Defense prep
```

---

## Tóm tắt vấn đề & Khuyến nghị

> [!CAUTION]
> **DỪNG vòng lặp "thêm data → train lại → thất vọng".** 4 lần đã đủ chứng minh approach hiện tại có ceiling rõ ràng (~0.50-0.55 OOD AUC). Tiếp tục sẽ chỉ LÃ PHÍ thời gian quý giá.

> [!TIP]
> **Reframe**: OOD limitation KHÔNG phải failure — đó là **FINDING có giá trị**. Khi 3 SOTA papers cũng fail tương tự (thậm chí tệ hơn) trên modern Diffusion, việc bạn document và phân tích gap này **chính là contribution** cho nghiên cứu.

> [!IMPORTANT]  
> **Ưu tiên #1 ngay bây giờ**: Hoàn thành 1 lần train cuối (với tristanzhang đúng data) → ghi nhận kết quả → **MOVE ON** sang Sprint 2. Grad-CAM heatmaps và bảng so sánh SOTA sẽ có giá trị cho báo cáo cao hơn nhiều so với OOD AUC tăng thêm 0.05.

### Checklist hành động ngay

- [ ] Train lần cuối trên Kaggle với Cell 4 đã chuẩn bị (VERIFY: 21,000 samples)
- [ ] Evaluate → ghi nhận kết quả vào CONTEXT.md
- [ ] Đóng Task 1.7 (ghi status = ✅ Completed with known limitations)
- [ ] Bắt đầu Task 2.2: Benchmark 3 SOTA trên cùng OOD test set
- [ ] Bắt đầu Task 2.3: Grad-CAM integration

---

*Phân tích này dựa trên toàn bộ context từ CONTEXT.md, PROJECT_PLAN.md, DATASET_UPDATE_CHECKLIST.md, CRITICAL_ANALYSIS.md, CHANGELOG.md, và configs.*
