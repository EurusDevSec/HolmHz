## 💡 Context

> **Task ID**: S1-003  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 1 - Data + Baseline Training  
> **Status**: ✅ COMPLETED  
> **Created**: 10/02/2026  
> **Completed**: 25/02/2026  
> **Target**: ~~28/02/2026~~ → **07/03/2026** (hoàn thành sớm)  
> **Assignee**: Hoàng  
> **Blocked by**: S1-001 (env) ✅, S1-002 (data) ✅  
> **Blocks**: S1-005 (Training cần data pipeline)

> Xây dựng data pipeline: Dataset class, augmentation, train/val/test split.
> Pattern học từ: CNNDetection `data/datasets.py`, DeepfakeBench `dataset/abstract_dataset.py`

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Engineer**, I want to **build a robust data pipeline with proper augmentation and train/val/test-OOD splitting** so that **the model trains on diverse data, validates correctly, and gets tested on unseen sources (Flux, tristanzhang).**

**Acceptance Criteria:**

- [x] `ImageDataset(torch.utils.data.Dataset)` class hoạt động — load ảnh từ manifest JSON
- [x] Augmentation pipeline dùng Albumentations (JPEG compression, blur, noise, flip)
- [x] Normalization dùng ImageNet stats (match EfficientNet-B0 pretrained)
- [x] Train/Val/Test-OOD split tạo ra 4 manifest files JSON (train, val, test_id, test_ood)
- [x] DataLoader chạy được, batch = 32 không OOM trên 8GB RAM
- [x] OOD test set tách riêng (Flux, tristanzhang, real_pexels, real_camera) không lẫn vào train
- [x] Unit test: load 1 batch, kiểm tra shape, dtype, value range

---

## 🛠️ Implementation

### Subtasks

- [x] 1.3.1 Implement `src/holmhz/data/image_dataset.py` (load từ JSON manifest)
- [x] 1.3.2 Implement `src/holmhz/data/transforms.py` (Albumentations pipeline)
- [x] 1.3.3 Script `preprocessing/build_splits.py` — tạo train/val/test JSON manifests

### Branch & PR

- [x] Branch: `feat/s1/data-pipeline`
- [ ] PR Created
- [x] Unit test `tests/test_data.py` passed (17/17 tests, 0 warnings)
- [x] Sample batch visualization (notebook + script)

### Kết quả thực tế

#### Data Splits (seed=42, stratified by source)

| Split       | Total  | Real  | Fake   | File                           |
| ----------- | ------ | ----- | ------ | ------------------------------ |
| **Train**   | 18,550 | 8,427 | 10,123 | `data/manifests/train.json`    |
| **Val**     | 3,975  | 1,776 | 2,199  | `data/manifests/val.json`      |
| **Test ID** | 3,975  | 1,797 | 2,178  | `data/manifests/test_id.json`  |
| **OOD**     | 1,180  | 600   | 580    | `data/manifests/test_ood.json` |

#### Files đã implement

| File                                  | Mô tả                                                       |
| ------------------------------------- | ----------------------------------------------------------- |
| `preprocessing/build_splits.py`       | Script tạo 4 JSON manifests (stratified split)              |
| `src/holmhz/data/transforms.py`       | `get_train_transforms()`, `get_val_transforms()`            |
| `src/holmhz/data/image_dataset.py`    | `ImageDataset` class (cv2 + Albumentations)                 |
| `src/holmhz/data/utils.py`            | `create_dataloader()`, `get_dataset_info()`                 |
| `src/holmhz/data/__init__.py`         | Exports all public API                                      |
| `tests/test_data.py`                  | 17 tests (TestTransforms, TestImageDataset, TestDataLoader) |
| `scripts/verify_pipeline.py`          | Standalone terminal verification script                     |
| `notebooks/01_data_exploration.ipynb` | 6-cell interactive exploration notebook                     |

#### Test Results

```
pytest tests/test_data.py -v
17 passed, 0 warnings in 7.7s
```

---

## 📝 Notes

> **Patterns từ benchmark:**
>
> ```python
> # CNNDetection pattern: đơn giản, hiệu quả
> class ImageDataset(Dataset):
>     def __init__(self, manifest_path, transform=None):
>         self.data = json.load(open(manifest_path))  # [{"path": ..., "label": 0/1, "source": ...}]
>         self.transform = transform
>
> # DeepfakeBench pattern: abstract base + nhiều specialized dataset
> class AbstractDataset(Dataset):  # base class
> class PairDataset(AbstractDataset):  # pair real/fake
> ```

> **Manifest JSON format (thực tế):**
>
> ```json
> [
>   {
>     "path": "data/processed/train/real/cifake/00001.png",
>     "label": 0,
>     "source": "cifake",
>     "category": "real"
>   },
>   {
>     "path": "data/processed/train/fake_gan/stylegan/00001.png",
>     "label": 1,
>     "source": "stylegan",
>     "category": "fake_gan"
>   }
> ]
> ```

> **Lưu ý augmentation:**
>
> - Normalization dùng ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
> - JPEG compression augmentation: `quality_range=(60, 100)` (Albumentations v2.0.8 API)
> - Gaussian noise: `std_range=(0.01, 0.03)` (Albumentations v2.0.8 API — thay `var_limit` deprecated)
> - Không augment trên val/test — chỉ Resize + Normalize + ToTensorV2

> **Interface cho Task 1.4/1.5:**
>
> ```python
> # DataLoader trả về batch dict:
> batch = {
>     "image": tensor [B, 3, 224, 224],  # float32, normalized ImageNet
>     "label": tensor [B],                # float32, 0.0 hoặc 1.0
>     "source": list[str],               # ["cifake", "stylegan", ...]
>     "path": list[str],                  # ["data/processed/...", ...]
> }
> # Model nhận batch["image"], loss dùng batch["label"]
> ```
