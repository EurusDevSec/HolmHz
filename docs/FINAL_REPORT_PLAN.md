# HolmHz — Final Report Preparation Plan

> **Mục tiêu**: Hoàn thành benchmark công bằng giữa 7 models, viết báo cáo nghiên cứu cấp đại học.
> **Cập nhật**: 27/03/2026
> **Deadline**: Tháng 05/2026 (defense prep)

---

## 1. Tình trạng hiện tại

### 1.1 Models đã train

| Model              | Params | Dataset train | Val AUC | OOD AUC | Trạng thái     |
| ------------------ | ------ | ------------- | ------- | ------- | -------------- |
| EfficientNet-B0 v4 | 4M     | **raw_v2**    | 0.9972  | 0.7838  | ✅ Fair        |
| ResNet-18          | 11M    | raw_v1 (cũ)   | 0.9907  | 0.6596  | ❌ Cần retrain |
| ViT-Small/16       | 22M    | raw_v1 (cũ)   | 0.9942  | 0.6860  | ❌ Cần retrain |
| Swin-T             | 28M    | raw_v1 (cũ)   | 0.9966  | 0.6932  | ❌ Cần retrain |

### 1.2 Vấn đề

- **Không công bằng**: 3 models (ResNet-18, ViT-Small, Swin-T) train trên raw_v1 (dataset lộn xộn, chưa chuẩn hóa). EfficientNet-B0 v4 train trên raw_v2 (dataset chuẩn hóa). So sánh OOD như vậy biased.
- **Thiếu external test set**: Tất cả evaluation dùng OOD set tự xây (1,180 ảnh) — có thể confirmation bias.
- **3 research models (CNNDetection, UniversalFakeDetect, DeepfakeBench)**: Đã benchmark nhưng chưa trên external test set.

### 1.3 Dataset raw_v2 (chuẩn, dùng cho benchmark)

```
data/processed/train/     — 30,300 ảnh (224×224 PNG)
  real/cifake               7,000
  real/ffhq                 5,000
  real/diverse_real         3,000
  real/real_pexels_train      300
  real/real_camera_train      300
  fake_gan/stylegan         5,000
  fake_diffusion/cifake     7,000
  fake_diffusion/sd15       2,500
  fake_diffusion/tristanzhang_train  200

data/processed/ood_test/  — 1,180 ảnh
  tristanzhang_fake           500
  flux                         80
  real_pexels                 500
  real_camera                 100
```

Manifests: `data/manifests/train.json`, `val.json`, `test_id.json`, `test_ood.json`
Split: train 21,000 / val 4,500 / test_id 4,500 / OOD 1,180 (seed=42, stratified by source)

---

## 2. Kế hoạch thực hiện

### Phase 1: Retrain 3 models trên raw_v2 (Kaggle T4)

**Thời gian ước tính**: 1-2 ngày (mỗi model ~30-55 phút trên Kaggle T4)

#### 2.1 Chuẩn bị

- [x] Config files đã có sẵn, tham chiếu đúng manifests:
  - `configs/train_resnet18.yaml` (batch_size=32)
  - `configs/train_vit_small.yaml` (batch_size=16)
  - `configs/train_swin_tiny.yaml` (batch_size=16)
- [x] Hyperparameters GIỐNG v4 (AdamW, lr=1e-4, cosine, pos_weight=1.2, 30 epochs)
- [ ] Upload dataset raw_v2 lên Kaggle (hoặc dùng dataset đã upload sẵn)
- [ ] Tạo/cập nhật Kaggle notebook cho 3 models

#### 2.2 Training

| Model        | Config                         | Expected Time | Checkpoint output      |
| ------------ | ------------------------------ | ------------- | ---------------------- |
| ResNet-18    | `configs/train_resnet18.yaml`  | ~23 min       | `best_resnet18_v2.pt`  |
| ViT-Small/16 | `configs/train_vit_small.yaml` | ~40 min       | `best_vit_small_v2.pt` |
| Swin-T       | `configs/train_swin_tiny.yaml` | ~55 min       | `best_swin_tiny_v2.pt` |

> **Lưu ý**: Naming convention `_v2` để phân biệt với checkpoint cũ (raw_v1).
> Sau khi có kết quả, rename → `best_resnet18.pt`, v.v. (thay thế cũ).

#### 2.3 Evaluation ngay trên Kaggle

Sau mỗi lần train, chạy evaluation ngay trên notebook:

```python
# ID test
python scripts/test.py --config configs/test.yaml --checkpoint <path> --split test_id
# OOD test
python scripts/test.py --config configs/test.yaml --checkpoint <path> --split test_ood
```

Ghi nhận: Val AUC, ID AUC, ID Acc, OOD AUC, OOD Acc, OOD F1, per-source OOD accuracy.

---

### Phase 2: External OOD Test Set (Single Point of Truth)

**Mục tiêu**: Tìm 1 bộ test hoàn toàn ngoài (không trùng source nào với train/val/test), dùng làm "single point of truth" để tránh confirmation bias.

#### 2.4 Yêu cầu external test set

- **Hoàn toàn disjoint** với training data (không trùng nguồn: CIFAKE, FFHQ, StyleGAN, SD v1.5, tristanzhang, Flux, Pexels, Unsplash)
- **Đủ đa dạng**: Cả Real lẫn Fake, từ multiple AI generators
- **Đủ lớn**: Tối thiểu 500-1,000 ảnh để có statistical significance
- **Có ground truth labels**: Real/Fake đã labeled sẵn

#### 2.5 Nguồn đề xuất từ Kaggle

| Dataset candidate               | Real   | Fake   | Generators covered            | Ghi chú                                  |
| ------------------------------- | ------ | ------ | ----------------------------- | ---------------------------------------- |
| **AI vs Real Images (AIRD)**    | ~1,000 | ~1,000 | MJ, DALL-E 2, SD              | Kaggle, clean labels                     |
| **Deepfake and Real Images**    | 1,000  | 1,000  | Mixed GANs + Diffusion        | Kaggle, 256×256                          |
| **Real vs AI-Generated Faces**  | ~5,000 | ~5,000 | Thispersondoesnotexist + more | Kaggle, face-only, good for our use case |
| **ArtiFact (artifact-dataset)** | ~1,400 | ~1,400 | Multiple GANs + Diffusion     | Academic, diverse generators             |

> **Recommendation**: Chọn 1-2 datasets, KHÔNG overlap với training sources. Priority: có modern Diffusion generators (MJ, DALL-E), face images, với ít nhất 500+ samples mỗi class.

#### 2.6 Workflow

1. Download external test set từ Kaggle
2. Resize 224×224, tạo manifest `data/manifests/test_external.json`
3. Chạy ALL 7 models trên cùng external set (4 HolmHz + 3 research)
4. Ghi bảng so sánh → đây là "ground truth" benchmark cho báo cáo

---

### Phase 3: Full Benchmark (7 Models × 3 Test Sets)

#### 2.7 Benchmark matrix

| Test Set     | Mô tả                               | Size    | Purpose                 |
| ------------ | ----------------------------------- | ------- | ----------------------- |
| **Test ID**  | In-distribution (same sources)      | 4,500   | Đánh giá khả năng học   |
| **Test OOD** | Out-of-distribution (different gen) | 1,180   | Đánh giá generalization |
| **External** | Hoàn toàn ngoài (Kaggle)            | ~1,000+ | Single point of truth   |

#### 2.8 Models to benchmark

| #   | Model                  | Type        | Params | Checkpoint                      |
| --- | ---------------------- | ----------- | ------ | ------------------------------- |
| 1   | **EfficientNet-B0 v4** | CNN         | 4M     | `best_v4.pt` (raw_v2)           |
| 2   | **ResNet-18**          | CNN         | 11M    | `best_resnet18_v2.pt` (raw_v2)  |
| 3   | **ViT-Small/16**       | Transformer | 22M    | `best_vit_small_v2.pt` (raw_v2) |
| 4   | **Swin-T**             | Swin Trans. | 28M    | `best_swin_tiny_v2.pt` (raw_v2) |
| 5   | CNNDetection           | CNN         | 25M    | `blur_jpg_prob0.5.pth`          |
| 6   | UniversalFakeDetect    | CLIP        | 300M   | `fc_weights.pth`                |
| 7   | DeepfakeBench          | CNN         | 19M    | `effnb4_best.pth`               |

#### 2.9 Metrics cần thu thập

Cho mỗi model × mỗi test set:

- **AUC** (primary metric)
- **Accuracy** @ optimal threshold
- **F1-Score**
- **Per-source accuracy** (cho OOD và External)

Output: `outputs/benchmark/final_benchmark_results.json`

#### 2.10 Script chạy benchmark

Dùng `scripts/benchmark_sota.py` (đã có sẵn) cho 4 loại models:

```bash
python scripts/benchmark_sota.py --model holmhz --checkpoint <path> --test-set <manifest>
python scripts/benchmark_sota.py --model cnndetection --test-set <manifest>
python scripts/benchmark_sota.py --model universalfake --test-set <manifest>
python scripts/benchmark_sota.py --model deepfakebench --test-set <manifest>
```

Cần mở rộng script để:

- Hỗ trợ `--checkpoint` flag cho multiple HolmHz models
- Hỗ trợ `--test-set` flag cho external test set
- Output per-source breakdown

---

### Phase 4: ONNX Export & Grad-CAM (Nếu đủ thời gian)

#### 2.11 ONNX Re-export

Sau khi có 3 checkpoints mới (raw_v2), re-export ONNX:

```bash
set CUDA_VISIBLE_DEVICES=
python scripts/export_all_onnx.py
```

#### 2.12 Grad-CAM Gallery

Tạo so sánh heatmap giữa 4 models trên cùng 1 set ảnh:

- 10 ảnh OOD Fake (flux + tristanzhang)
- 10 ảnh OOD Real (real_pexels + real_camera)
- 5 ảnh External test

→ Dùng cho phần XAI Analysis trong báo cáo.

---

### Phase 5: Viết báo cáo

#### 2.13 Cấu trúc báo cáo dự kiến

| Chương | Nội dung                                      | Người viết | Deadline   |
| ------ | --------------------------------------------- | ---------- | ---------- |
| Ch 1   | Giới thiệu, tính cấp thiết, mục tiêu          | Luân       | 15/04/2026 |
| Ch 2   | Tổng quan nghiên cứu (Related Work)           | Luân       | 15/04/2026 |
| Ch 3   | Phương pháp (Dataset, Architecture, Training) | Hoàng      | 22/04/2026 |
| Ch 4   | Thực nghiệm & Kết quả (Benchmark, Analysis)   | Hoàng      | 29/04/2026 |
| Ch 5   | Kết luận, hạn chế, hướng mở rộng              | Hoàng+Luân | 05/05/2026 |

#### 2.14 Key results cho báo cáo (sẽ cập nhật sau retrain)

**Bảng chính Chapter 4**: Full 7-Model Benchmark trên 3 test sets

- Table 4.1: ID Test Results
- Table 4.2: OOD Test Results
- Table 4.3: External Test Results (single point of truth)
- Table 4.4: Per-Source OOD Analysis
- Figure 4.1: ROC Curves (7 models overlay)
- Figure 4.2: Grad-CAM comparison grid (4 HolmHz models)
- Figure 4.3: Model size vs OOD performance scatter plot

---

## 3. Priority & Timeline

| #   | Task                                  | Priority | Effort   | Deadline   |
| --- | ------------------------------------- | -------- | -------- | ---------- |
| 1   | Retrain 3 models trên raw_v2 (Kaggle) | **P0**   | 1-2 ngày | 30/03/2026 |
| 2   | Download & prep external test set     | **P0**   | 0.5 ngày | 31/03/2026 |
| 3   | Full benchmark (7 × 3 test sets)      | **P0**   | 1 ngày   | 02/04/2026 |
| 4   | ONNX re-export (3 models mới)         | P1       | 0.5 ngày | 03/04/2026 |
| 5   | Grad-CAM comparison gallery           | P1       | 0.5 ngày | 04/04/2026 |
| 6   | Ch 3+4 báo cáo                        | **P0**   | 1-2 tuần | 29/04/2026 |
| 7   | Ch 1+2 (Luân)                         | P1       | 2 tuần   | 15/04/2026 |
| 8   | Defense prep (slide + demo)           | P1       | 3-5 ngày | 15/05/2026 |

---

## 4. Lưu ý kỹ thuật

### 4.1 CUDA hang fix

Trên Windows, `import torch` có thể hang do CUDA context lock từ killed Python processes.
**Fix**: Set environment variable trước khi chạy script local:

```cmd
set CUDA_VISIBLE_DEVICES=
python scripts/export_all_onnx.py
```

### 4.2 Kaggle training tips

- Upload `data/processed/` + `data/manifests/` + `src/holmhz/` + `configs/` lên Kaggle dataset
- Dùng T4 GPU (16GB VRAM), 30h/tuần
- Train 3 models liên tiếp trong 1 session (~2h total)
- Download checkpoints ngay sau khi train xong

### 4.3 Benchmark reproducibility

- Tất cả evaluation dùng **cùng 1 threshold** (optimal trên val set) hoặc report AUC (threshold-free)
- Seed cố định: 42
- Report cả mean ± std nếu chạy nhiều lần (optional cho university-level)

---

## 5. Deliverables Checklist

- [ ] 3 checkpoints mới (ResNet-18, ViT-Small, Swin-T) trained on raw_v2
- [ ] External test set downloaded, preprocessed, labeled
- [ ] Full benchmark table (7 models × 3 test sets × 4 metrics)
- [ ] ROC curve plots (7 models overlay)
- [ ] Per-source OOD breakdown table
- [ ] Grad-CAM comparison grid (4 HolmHz models, optional)
- [ ] ONNX files re-exported (optional)
- [ ] Báo cáo Ch 3-4 (Phương pháp + Kết quả)
- [ ] Báo cáo hoàn chỉnh (5 chương)
- [ ] Slide thuyết trình + video demo

---

## 6. Quyết định đã chốt

1. **DỪNG cải tiến**: Không thêm features mới (CLIP, FFT, EXIF, ensemble). Chỉ giữ 4 models cơ bản cho benchmark.
2. **Model chính**: EfficientNet-B0 v4 — best OOD performance, nhẹ nhất, dùng cho web demo.
3. **Dataset chuẩn**: raw_v2 (30,300 train images) — tất cả 4 models phải train trên dataset này.
4. **External test**: Bắt buộc — single point of truth, tránh confirmation bias.
5. **3 research models**: Chạy inference only (weights có sẵn) — không retrain.
6. **Scope báo cáo**: Applied research, không claim novelty. Focus: benchmark + analysis + XAI demo.
