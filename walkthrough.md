# Walkthrough: Data Reset v6 — Kết quả & So sánh

## Training v6 (Kaggle T4)

- **Best Val AUC: 0.9964** (epoch 13, early stop epoch 20)
- Training time: ~15 phút (20 epochs × 44s)
- Model: EfficientNet-B0, full fine-tune, LR=0.0003

---

## So sánh v4 vs v6

### In-Domain Test

| Metric | v4 | v6 | Thay đổi |
|--------|-----|-----|---------|
| Val AUC | 0.9700 | **0.9958** | +2.6% |
| Val Accuracy | ~95% | **97.4%** | +2.4% |
| All sources >95% | ❌ | ✅ | Fixed |

### OOD Test (ảnh ngoài tập huấn luyện)

| Metric | v4 | v6 | Thay đổi |
|--------|-----|-----|---------|
| Real images (camera) | ~40% | **98.3%** | 🟢 **+58.3% — FIXED** |
| AI images (camera) | ~100% | 2.7% | 🟡 Opposite bias |

### Sample Images (12 ảnh test)

| | v4 (CIFAKE data) | v6 (Clean data) |
|---|---|---|
| Real → REAL ✅ | 2/5 (40%) | **6/6 (100%)** |
| Fake → FAKE ✅ | 5/5 (100%) | 0/6 (0%) |
| **Overall** | **7/10 (70%)** | **6/12 (50%)** |

---

## Phân tích

### ✅ Đã fix được
- **Real-image false-positive bias** — V4 nhận 3/5 ảnh thật là FAKE. V6 nhận 6/6 ảnh thật là REAL.
- **CIFAKE shortcut learning** — Không còn học từ upscaling artifacts.
- **In-domain performance** — AUC 0.9958 trên dữ liệu cùng distribution.

### ❌ Chưa fix (OOD gap)
- V6 giờ **quá conservative** — không dám gọi ảnh là FAKE.
- Nguyên nhân: Fake trong training (rvf10k + ciplab) chủ yếu là **GAN-generated faces** (StyleGAN, ProGAN). Camera_vs_ai fakes là **text-to-image** (DALL-E, Midjourney style) — domain gap lớn.
- V6 học rất giỏi phân biệt GAN faces vs real faces, nhưng chưa thấy diffusion-model fakes.

### Giải pháp tiếp theo

> [!IMPORTANT]
> Cần bổ sung fake images từ **diffusion models** (SDXL, FLUX, DALL-E 3, Midjourney) vào training data.

Có 2 lựa chọn:
1. **Nhanh**: Generate 2K-5K ảnh từ SDXL/FLUX (API free hoặc local) → thêm vào manifests_v2 → retrain
2. **Chuẩn**: Download GenImage full (Google Drive, ~50GB) → lấy subset SD + MJ + ADM → retrain

---

## Files đã tạo/thay đổi

| File | Thay đổi |
|------|----------|
| [configs/train_v6.yaml](file:///r:/_Projects/Eurus_Workspace/HolmHz/configs/train_v6.yaml) | Config mới (manifests_v2, LR=0.0003) |
| [configs/test_v6.yaml](file:///r:/_Projects/Eurus_Workspace/HolmHz/configs/test_v6.yaml) | Test config cho manifests_v2 |
| [web/config.py](file:///r:/_Projects/Eurus_Workspace/HolmHz/web/config.py) | ONNX v6 + threshold 0.5 |
| [docs/KAGGLE_V6_TRAINING.md](file:///r:/_Projects/Eurus_Workspace/HolmHz/docs/KAGGLE_V6_TRAINING.md) | Fixed Cell 7 eval |
| [scripts/prepare_data_v2.py](file:///r:/_Projects/Eurus_Workspace/HolmHz/scripts/prepare_data_v2.py) | Build manifests từ raw data |
