## 💡 Context

> **Task ID**: S1-003  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 1 - Data + Baseline Training  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: ~~28/02/2026~~ → **07/03/2026**  
> **Assignee**: Hoàng  
> **Blocked by**: S1-001 (env), S1-002 (data)  
> **Blocks**: S1-005 (Training cần data pipeline)

> Xây dựng data pipeline: Dataset class, augmentation, train/val/test split.
> Pattern học từ: CNNDetection `data/datasets.py`, DeepfakeBench `dataset/abstract_dataset.py`

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Engineer**, I want to **build a robust data pipeline with proper augmentation and train/val/test-OOD splitting** so that **the model trains on diverse data, validates correctly, and gets tested on unseen sources (Gemini, Flux).**

**Acceptance Criteria:**

- [ ] `ImageDataset(torch.utils.data.Dataset)` class hoạt động — load ảnh từ manifest JSON
- [ ] Augmentation pipeline dùng Albumentations (JPEG compression, blur, noise, flip)
- [ ] Normalization hỗ trợ cả ImageNet stats và custom stats
- [ ] Train/Val/Test-OOD split tạo ra 3 manifest files JSON
- [ ] DataLoader chạy được, batch = 32 không OOM trên 8GB RAM
- [ ] OOD test set tách riêng (Gemini, Flux) không lẫn vào train
- [ ] Unit test: load 1 batch, kiểm tra shape, dtype, value range

---

## 🛠️ Implementation

### Subtasks

- [ ] 1.3.1 Implement `src/holmhz/data/image_dataset.py` (load từ JSON manifest)
- [ ] 1.3.2 Implement `src/holmhz/data/transforms.py` (Albumentations pipeline)
- [ ] 1.3.3 Script `preprocessing/build_splits.py` — tạo train/val/test JSON manifests

### Branch & PR

- [ ] Branch: `feat/s1/data-pipeline`
- [ ] PR Created
- [ ] Unit test `tests/test_data.py` passed
- [ ] Sample batch visualization (notebook)

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

> **Manifest JSON format:**
>
> ```json
> [
>   {
>     "path": "data/processed/train/real/ffhq_00001.png",
>     "label": 0,
>     "source": "ffhq"
>   },
>   {
>     "path": "data/processed/train/fake/genimage_00001.png",
>     "label": 1,
>     "source": "genimage_sd15"
>   }
> ]
> ```

> **Lưu ý augmentation:**
>
> - Normalization phải match backbone (ImageNet cho EfficientNet, CLIP stats cho CLIP)
> - JPEG compression augmentation rất quan trọng (từ CNNDetection paper)
> - Không augment trên val/test
