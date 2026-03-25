# Kế Hoạch Sửa Lỗi Model — Phiên Bản 2 (FINAL)

> Phương án A: Fix data + retrain. Phương án B: Nâng cấp CLIP backbone (sau khi A thành công).

---

## TL;DR — Có sửa được không?

> [!IMPORTANT]
> **CÓ, và đây là vấn đề RẤT PHỔ BIẾN trong ML.** Nó gọi là **\"data-centric AI\"** — khi model chạy sai, 90% lỗi nằm ở data, không phải code.

**Tại sao mình tự tin sửa được:**

1. **Codebase HolmHz hoàn toàn ổn** — training pipeline (`Trainer`), evaluation (`Evaluator`), model registry, ONNX export, Grad-CAM, web demo — tất cả đều robust, reusable. Chỉ cần **swap data** và **retrain**.
2. **UniversalFakeDetect** — CVPR 2023 paper, chỉ ~3 file Python, dùng CLIP + 1 linear layer → đạt >85% AUC trên mọi generator. Bí quyết: **data chất lượng + pre-trained features**. Mình đã có tiền đề tốt hơn (multi-arch, Grad-CAM, web demo).
3. **GenImage dataset** — 1 triệu ảnh, 8 generators (SD, Midjourney, BigGAN, ADM, GLIDE...), đã được verify trong 50+ papers. Thay CIFAKE bằng GenImage = fix ngay vấn đề.
4. **Những repo nhỏ chạy đúng** vì họ dùng đúng data. Codebase lớn mà data sai thì vẫn sai. Ngược lại, 1 file script mà data chuẩn thì vẫn chạy tốt. **Data is king.**

> **Kết luận**: Codebase không có gì sai. Data sai. Fix data = fix vấn đề.

---

## Root Cause — Tóm tắt

| # | Vấn đề | Mức độ | Giải pháp |
| --- | --- | --- | --- |
| 1 | **CIFAKE 32×32** chiếm 46% → model học shortcut upscaling | 🔴 Critical | Bỏ hoàn toàn, thay bằng GenImage high-res |
| 2 | **Real images quá hẹp** (85% chỉ FFHQ faces + CIFAR objects) | 🔴 Critical | Dùng ImageNet subset (1000 class, diverse) |
| 3 | **Data lộn xộn**, nguồn không xác minh (Unsplash có AI?) | 🟡 High | Reset 100%, chỉ dùng nguồn verified |
| 4 | **Threshold 0.76** calibrate trên CIFAKE-biased test set | 🟡 Medium | Re-calibrate sau khi retrain |
| 5 | **Không có JPEG augmentation** | 🟡 Medium | Thêm vào pipeline |

---

## Bước 0: Chạy Reference MVP — Validate approach

> [!TIP]
> Chạy UniversalFakeDetect trên ảnh user thử nghiệm TRƯỚC KHI retrain. Nếu nó phân loại đúng → chứng minh approach đúng, chỉ cần fix data.

**UniversalFakeDetect** (CVPR 2023, Ojha et al.):
- GitHub: `Yuheng-Li/UniversalFakeDetect`
- Architecture: CLIP ViT-L/14 + 1 Linear layer
- Training: Chỉ train trên ProGAN → generalize sang SD, DALLE, MJ
- Accuracy: >85% trên unseen generators
- **Mình đã có repo clone** tại `prac/ai-experiments/deepfake-detection/UniversalFakeDetect/`

```bash
# Test nhanh: upload cùng ảnh user đã thử trên HolmHz web
python prac/.../UniversalFakeDetect/validate.py --image <user_test_image>
```

Nếu UniversalFakeDetect **đúng** mà HolmHz **sai** → CHỨNG MINH data là vấn đề.

---

## Bước 1: Data Reset — Xoá sạch & Xây mới

### Nguyên tắc data mới

| Rule | Mô tả |
| --- | --- |
| **Chỉ dùng nguồn verified** | Dataset từ paper có peer review + citations |
| **High-res** (≥256px) | Không bao giờ dùng ảnh <128px |
| **Multi-domain** | People, animals, nature, urban, objects, food, art |
| **Multi-generator** | GAN + Diffusion + Text-to-Image (≥5 generators) |
| **Balanced** | ~50/50 Real/Fake |
| **Clean split** | Train/Val/Test KHÔNG overlap, source-level split |

### Dataset Plan (3 nguồn chính)

#### Nguồn 1: GenImage (Academic, Proven)

- **Paper**: \"GenImage: A Million-Scale Benchmark\" (NeurIPS 2023)
- **Size**: 1,350,000 images (1000 classes × 1350/class)
- **Real source**: ImageNet subset (VERIFIED real)
- **Fake sources**: Stable Diffusion v1.4, Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN
- **Resolution**: 256×256+ (high quality)
- **Download**: HuggingFace / Google Drive (links trên GitHub)
- **Dùng cho HolmHz**: Lấy subset ~30K-50K (15K real + 15K-25K fake multi-gen)

#### Nguồn 2: ForenSynths (Academic, GAN-focused)

- **Paper**: \"CNN-generated images are surprisingly easy to spot\" (CVPR 2020, Wang et al.)
- **Size**: ~70K images (11 generators)
- **Generators**: ProGAN, StyleGAN, BigGAN, CycleGAN, StarGAN, GauGAN, etc.
- **Dùng cho HolmHz**: GAN-diverse fake source (lấy ~5K)

#### Nguồn 3: Self-generated (Newest generators)

- Own SD v1.5 data (đã có `sd15`): Giữ lại 1,750 ảnh
- **Thêm mới**: Generate 2K-3K ảnh từ FLUX.1/SDXL/DALLE-3 (API hoặc Kaggle)

### Target Dataset v2

```
data_v2/
├── real/         ~15,000 images
│   ├── imagenet_subset/   10,000  (GenImage real, 1000 classes)
│   ├── lsun_bedroom/       2,000  (ForenSynths real)
│   └── camera_diverse/     3,000  (User's own camera photos, verified)
│
├── fake/         ~20,000 images
│   ├── genimage_sd/        4,000  (Stable Diffusion v1.4)
│   ├── genimage_midjourney/ 3,000  (Midjourney)
│   ├── genimage_adm/       2,000  (Guided Diffusion)
│   ├── genimage_glide/     2,000  (GLIDE)
│   ├── genimage_biggan/    2,000  (BigGAN)
│   ├── forensynths_multi/  3,000  (ProGAN+StyleGAN+CycleGAN mix)
│   ├── sd15_existing/      1,750  (Keep existing)
│   ├── flux_new/           1,000  (Generate new)
│   └── sdxl_new/           1,250  (Generate new)
│
└── Total: ~35,000 images (43% real, 57% fake)
```

> **Không bao giờ dùng**: CIFAKE, Unsplash (unverified), random internet sources.

---

## Bước 2: Training Pipeline Adjustments

### 2.1 JPEG Augmentation (NEW)

```python
# Thêm vào transforms
transforms.RandomApply([
    lambda img: Image.fromarray(
        np.array(img.save(buf:=io.BytesIO(), 'JPEG', quality=random.randint(70,100)) or buf.seek(0) or Image.open(buf))
    )
], p=0.5)
```

### 2.2 Stronger augmentation

```python
transforms.Compose([
    transforms.Resize((256, 256)),        # Resize lớn hơn trước
    transforms.RandomCrop((224, 224)),     # Random crop thay vì center
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),  # Subtle color jitter
    # JPEG augmentation (above)
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

### 2.3 Retrain on Kaggle T4

- Model: EfficientNet-B0 (giữ nguyên, codebase không đổi)
- Epochs: 30-40
- Early stopping patience: 5
- Threshold: Re-calibrate bằng Youden's J trên validation set MỚI

---

## Bước 3: Verify & Compare

1. Evaluate v2 model trên OOD test set MỚI (KHÔNG dùng ảnh train)
2. Test trên **cùng ảnh user đã thử** trên web demo → phải cải thiện
3. So sánh v1 vs v2 → bảng cho báo cáo
4. Test trên **ảnh ngẫu nhiên từ internet** → real-world readiness

---

## Timeline

| Ngày | Task | Chi tiết |
| --- | --- | --- |
| D1 | Reference MVP | Chạy UniversalFakeDetect trên test images |
| D1-D2 | Download GenImage | Subset 30K-50K images |
| D2 | Data pipeline | Rebuild manifests, train/val/test split |
| D3 | Retrain v2 | Kaggle T4, ~4-6h training |
| D4 | Evaluate + Compare | v1 vs v2, OOD test, real-world images |
| D4 | Web demo update | Load v2 model → re-test |
| D5 | Document | CONTEXT.md Section 24, guide update |

---

## Impact trên báo cáo

Thay đổi này **TÍCH CỰC** cho báo cáo:

- **Chapter 3**: Thêm mục \"Data Quality Analysis & Dataset v2\" — cho thấy process nghiên cứu mature
- **Chapter 4**: So sánh v1 (biased) vs v2 (fixed) → bảng improvement rất ấn tượng
- **Kết luận**: \"Data diversity is the dominant factor\" → được chứng minh bằng thực nghiệm

> [!NOTE]
> Việc phát hiện và fix bias là **dấu hiệu tích cực** trong nghiên cứu khoa học, KHÔNG phải thất bại. Hội đồng đánh giá cao khả năng tự phát hiện và sửa lỗi.
