# 📖 HƯỚNG DẪN CHI TIẾT TASK 1.2: DATA COLLECTION

> **Dành cho**: Lê Văn Hoàng — người chưa có nền tảng ML/DL, học qua thực hành  
> **Triết lý**: Mỗi bước không chỉ hướng dẫn **làm gì** mà giải thích **tại sao làm vậy**  
> **Thời gian**: ~1 tuần (24/02/2026 → 02/03/2026)  
> **Tiền đề**: Task 1.1 Environment Setup ✅ DONE  
> **Tham chiếu**: [TASK_1.2_DATA_COLLECTION.md](../tasks/TASK_1.2_DATA_COLLECTION.md) | [PROJECT_PLAN.md](../PROJECT_PLAN.md) Section 3

---

## 📋 Mục lục

- [Bức tranh tổng thể: Data Collection nằm ở đâu?](#bức-tranh-tổng-thể-data-collection-nằm-ở-đâu)
- [Tại sao Data lại quan trọng nhất?](#tại-sao-data-lại-quan-trọng-nhất)
- [Kiến thức nền: GAN vs Diffusion](#kiến-thức-nền-gan-vs-diffusion)
- [Kiến thức nền: In-Domain vs OOD](#kiến-thức-nền-in-domain-vs-ood)
- [Chiến lược dữ liệu của HolmHz](#chiến-lược-dữ-liệu-của-holmhz)
- [Bước 0: Chuẩn bị Git branch](#bước-0-chuẩn-bị-git-branch)
- [Bước 1: Download CIFAKE (Ưu tiên #1)](#bước-1-download-cifake-ưu-tiên-1)
- [Bước 2: Download FFHQ subset (Real faces)](#bước-2-download-ffhq-subset-real-faces)
- [Bước 3: Download StyleGAN faces (GAN Fake)](#bước-3-download-stylegan-faces-gan-fake)
- [Bước 4: Self-generate SD v1.5 (Diffusion Fake)](#bước-4-self-generate-sd-v15-diffusion-fake)
- [Bước 5: Chuẩn bị OOD Test Set](#bước-5-chuẩn-bị-ood-test-set)
- [Bước 6: Resize và tổ chức folder](#bước-6-resize-và-tổ-chức-folder)
- [Bước 7: Tạo dataset_stats.json](#bước-7-tạo-dataset_statsjson)
- [Bước 8: Validation & Data Integrity](#bước-8-validation--data-integrity)
- [Bước 9: Commit & PR](#bước-9-commit--pr)
- [Checklist hoàn thành](#checklist-hoàn-thành)
- [Troubleshooting](#troubleshooting)
- [Phân công Luân](#phân-công-luân)

---

## Bức tranh tổng thể: Data Collection nằm ở đâu?

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        DỰ ÁN HOLMHZ — SPRINT 1                          │
│                                                                           │
│  Task 1.1  Setup môi trường ✅ DONE                                      │
│                                                                           │
│  ► Task 1.2  THU THẬP DỮ LIỆU  ◄◄◄  BẠN ĐANG Ở ĐÂY                    │
│    │                                                                      │
│    │  Đây là "nguyên liệu đầu vào" cho toàn bộ dự án.                   │
│    │  Không có data → không có gì để train → không có kết quả.           │
│    │  Data sai / thiếu → model sai / yếu → fail hội đồng.               │
│    │                                                                      │
│    │  Assignee: Hoàng (chính) + Luân (hỗ trợ download)                   │
│    │  Target:   02/03/2026 (1 tuần)                                      │
│    │  Budget:   $0 (tất cả nguồn miễn phí)                               │
│    │                                                                      │
│    ├───► Task 1.3  Data Pipeline (code đọc & xử lý ảnh)                  │
│    │     Task 1.4  Model Architecture (song song với 1.3)                 │
│    │         │                                                            │
│    │         ▼                                                            │
│    │     Task 1.5  Training Pipeline                                      │
│    │         │                                                            │
│    │         ▼                                                            │
│    └──► Task 1.6  Baseline Training                                       │
│                                                                           │
│  ⚡ Song song: Hoàng làm 1.2.3 + 1.2.4, Luân làm 1.2.1 + 1.2.2         │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Tại sao Data lại quan trọng nhất?

Đây là **bài học số 1** bạn đã rút ra từ Phase 0 (chạy 3 SOTA projects):

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BÀI HỌC #1 TỪ BENCHMARK (Bạn đã chứng kiến trực tiếp):               │
│                                                                         │
│  CNNDetection    → Train trên ProGAN    → Fail ảnh Gemini (6% !!!)     │
│  UniversalFakeDetect → Train trên GAN   → Fail ảnh Flux (<10%)        │
│  DeepfakeBench   → Train trên FF++      → Đoán mò ảnh Gemini (50.7%)  │
│                                                                         │
│  ⟹ NGUYÊN NHÂN: Training data chỉ chứa ảnh GAN cũ                    │
│  ⟹ KẾT LUẬN: Training data QUAN TRỌNG HƠN kiến trúc model            │
│                                                                         │
│  → HolmHz BẮT BUỘC phải có ảnh Diffusion trong training data          │
└─────────────────────────────────────────────────────────────────────────┘
```

Nói đơn giản: **model chỉ phát hiện được những gì nó đã "thấy" khi học**. Nếu bạn chỉ cho model xem ảnh GAN (StyleGAN, ProGAN) — nó sẽ giỏi bắt GAN nhưng hoàn toàn mù trước Diffusion (Stable Diffusion, Gemini, Flux). Giống như dạy học sinh chỉ giải toán cộng, rồi hỏi bài nhân vậy.

---

## Kiến thức nền: GAN vs Diffusion

### GAN (Generative Adversarial Network) — "Thế hệ cũ" (2014-2022)

```
Cơ chế: 2 mạng đánh nhau
┌──────────────┐     ┌──────────────┐
│  Generator   │ ──► │ Discriminator │
│  (Tạo ảnh)   │ ◄── │ (Bắt lỗi)    │
└──────────────┘     └──────────────┘
Generator cố tạo ảnh giống thật,
Discriminator cố phân biệt thật/giả.
Hai bên "chạy đua vũ trang" → ảnh ngày càng thật.
```

**Dấu hiệu ảnh GAN** (model AI dễ phát hiện):

- Vết lưới (grid artifacts) do upsampling
- Phổ tần số bất thường (high-frequency peaks)
- Đối xứng bất thường ở khuôn mặt

**Đại diện**: StyleGAN, ProGAN, StarGAN

### Diffusion — "Thế hệ mới" (2022-nay)

```
Cơ chế: Thêm nhiễu rồi khử nhiễu
Ảnh thật → +nhiễu → +nhiễu → ... → Nhiễu hoàn toàn (noise)
                                         ↓
Ảnh mới  ← -nhiễu ← -nhiễu ← ... ← Bắt đầu khử nhiễu
```

**Dấu hiệu ảnh Diffusion** (model AI khó phát hiện hơn):

- KHÔNG có grid artifacts (khác GAN)
- Phổ tần số rất giống ảnh thật
- Chi tiết nhỏ (lông mi, tóc) vẫn có thể hơi "mượt" bất thường

**Đại diện**: Stable Diffusion, Midjourney, DALL-E, Gemini, Flux

> 💡 **Kết luận cho HolmHz**: Vì Diffusion KHÁC GAN hoàn toàn về cách sinh ảnh → dấu vết cũng khác → model PHẢI học cả hai loại.

---

## Kiến thức nền: In-Domain vs OOD

Đây là khái niệm **cực kỳ quan trọng** trong ML, và là tiêu chí đánh giá của hội đồng:

| Thuật ngữ                     | Ý nghĩa                                | Ví dụ                                         |
| ----------------------------- | -------------------------------------- | --------------------------------------------- |
| **In-Domain (ID)**            | Dữ liệu cùng loại với training         | Train trên CIFAKE → Test trên CIFAKE held-out |
| **Out-of-Distribution (OOD)** | Dữ liệu KHÁC loại, chưa thấy khi train | Train trên CIFAKE → Test trên **Gemini/Flux** |

**Tại sao OOD quan trọng?** Trong thế giới thực, kẻ xấu sẽ dùng công cụ mới nhất để tạo ảnh giả. Model của bạn không thể biết trước họ dùng công cụ gì. Nếu model chỉ đạt điểm cao trên ID mà thất bại trên OOD → vô dụng ngoài thực tế.

> **KPI từ PROJECT_PLAN.md (điều chỉnh 24/02/2026)**:
>
> - AUC ≥ 0.85 (In-Domain) — mức đạt
> - AUC ≥ 0.65 (OOD) — mức đạt
> - OOD test trên Gemini/Flux = **thước đo QUAN TRỌNG NHẤT cho hội đồng**

---

## Chiến lược dữ liệu của HolmHz

### Tổng quan các nguồn (Revised 24/02/2026)

```
┌────────────────────────────────────────────────────────────────────────┐
│              DATASET STRATEGY — HolmHz (Revised 24/02/2026)            │
│              "Tiết kiệm + thực tế" cho SV năm 4                       │
├─────────────────────┬──────────┬─────────────┬─────────────────────────┤
│ Nguồn               │ Số ảnh   │ Loại        │ Cách lấy               │
├─────────────────────┼──────────┼─────────────┼─────────────────────────┤
│ ⭐ CIFAKE Real      │ 60,000   │ Real        │ Kaggle 1-click (~500MB) │
│ ⭐ CIFAKE Fake      │ 60,000   │ Diffusion   │ Cùng package CIFAKE     │
│ FFHQ subset         │ 3-5k     │ Real faces  │ Kaggle mirror           │
│ StyleGAN faces      │ 3-5k     │ GAN         │ Kaggle / scrape         │
│ SD v1.5 self-gen    │ 2-3k     │ Diffusion   │ Colab free (diffusers)  │
│ Gemini (OOD)        │ 100-200  │ Diffusion   │ gemini.google.com free  │
│ Flux (OOD)          │ 100-200  │ Diffusion   │ replicate.com free tier │
│ Real camera (OOD)   │ 200      │ Real        │ Chụp thật / Internet    │
├─────────────────────┼──────────┼─────────────┼─────────────────────────┤
│ TỔNG TRAINING       │ ~15-20k  │ Mixed       │ Budget: $0              │
│ TỔNG OOD TEST       │ ~500     │ Unseen      │ Thước đo chính          │
└─────────────────────┴──────────┴─────────────┴─────────────────────────┘
```

### Tại sao chọn CIFAKE làm backbone dataset?

| Tiêu chí            | GenImage (kế hoạch cũ) | CIFAKE (kế hoạch mới) |
| ------------------- | ---------------------- | --------------------- |
| **Dung lượng**      | ~50GB                  | ~500MB                |
| **Download**        | Nhiều link, hay chết   | Kaggle 1-click        |
| **Cần xin access?** | Không nhưng link chết  | Không                 |
| **Số ảnh**          | 1.3M (quá nhiều)       | 120K (vừa đủ)         |
| **Phân loại sẵn**   | Cần tự chia            | ✅ Đã chia Real/Fake  |
| **Có Diffusion?**   | ✅ Có                  | ✅ Có (AI-generated)  |
| **Resolution**      | 256×256+               | 32×32 (nhỏ)           |

> ⚠️ **Nhược điểm CIFAKE**: Ảnh 32×32, rất nhỏ. Khi resize lên 224×224 sẽ bị mờ/pixelated. Tuy nhiên:
>
> - EfficientNet-B0 vẫn học được low-level features (texture, noise pattern) ngay cả từ ảnh nhỏ
> - Đây là **research** — minh chứng model hoạt động quan trọng hơn chất lượng ảnh tối đa
> - Nếu sau AUC thấp → bổ sung FFHQ (1024×1024) + tăng SD v1.5 generation
>
> **Quyết định**: CIFAKE là "80% work for 20% effort" — đủ để có baseline nhanh nhất.

### Cách chia Train/Val/Test

```
┌──────────────────────────────────────────────────────────────────────┐
│              DATASET SPLIT STRATEGY (từ PROJECT_PLAN.md)             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TRAIN (70%)               VAL (15%)            TEST ID (15%)        │
│  ─────────────             ─────────            ─────────            │
│  Real:                     Real:                Real:                │
│  • CIFAKE Real (5k)        • CIFAKE Real (1k)   • CIFAKE Real (1k)   │
│  • FFHQ subset (3k)        • FFHQ (500)                              │
│                                                                      │
│  Fake (GAN):               Fake (GAN):          Fake (GAN):          │
│  • StyleGAN faces (3k)     • ProGAN (500)       • StyleGAN (500)     │
│                                                                      │
│  Fake (Diffusion):         Fake (Diffusion):    Fake (Diff):         │
│  • CIFAKE Fake (5k)        • CIFAKE Fake (1k)   • CIFAKE Fake (1k)   │
│  • SD v1.5 self-gen (2k)                                              │
│                                                                      │
│  Total: ~18k               Total: ~3k           Total: ~2.5k         │
│                                                                      │
│  OOD TEST (riêng — QUAN TRỌNG NHẤT):                                │
│  • Gemini-generated (100-200)                                        │
│  • Flux-generated (100-200)                                          │
│  • Real camera (200)                                                 │
│  ➜ Đây là thước đo QUAN TRỌNG NHẤT cho hội đồng                     │
└──────────────────────────────────────────────────────────────────────┘
```

**Giải thích logic chia**:

- **Train**: Model học từ đây. Cần ĐA DẠNG nhất có thể (nhiều nguồn).
- **Validation (Val)**: Kiểm tra model đang học tốt không TRONG quá trình train. Dùng nguồn TƯƠNG TỰ train nhưng KHÁC ảnh.
- **Test ID**: Đánh giá cuối cùng trên ảnh cùng nguồn nhưng chưa thấy.
- **Test OOD**: Đánh giá trên nguồn **model CHƯA HỀ THẤY** (Gemini, Flux). Đây là thước đo thực tế nhất.

> **Tại sao ProGAN nằm ở Val mà không ở Train?** Vì ProGAN và StyleGAN cùng "họ" (đều là GAN), nhưng khác nguồn. Nếu model train trên StyleGAN mà val tốt trên ProGAN → chứng tỏ nó đang học "phát hiện GAN nói chung" chứ không chỉ "nhớ mặt StyleGAN".

> ⚠️ **Lưu ý**: Bạn CHƯA CẦN chia train/val/test ở Task 1.2 này. Chỉ cần download RAW data đúng folder. Việc chia sẽ tự động hóa ở Task 1.3 (Data Pipeline). Task 1.2 chỉ lo **thu thập đủ nguyên liệu**.

---

## Bước 0: Chuẩn bị Git branch

Trước khi bắt tay làm, tạo branch riêng cho task này:

```bash
# Đảm bảo đang ở thư mục project
cd R:/_Projects/Eurus_Workspace/HolmHz

# Kích hoạt venv
.venv\Scripts\activate

# Chuyển về main và pull mới nhất
git checkout main
git pull origin main

# Tạo branch mới cho data collection
git checkout -b feat/s1/data-collection

# Tạo cấu trúc folder data
mkdir -p data/raw/real/cifake
mkdir -p data/raw/real/ffhq
mkdir -p data/raw/fake_gan/stylegan
mkdir -p data/raw/fake_diffusion/cifake
mkdir -p data/raw/fake_diffusion/sd15
mkdir -p data/raw/ood_test/gemini
mkdir -p data/raw/ood_test/flux
mkdir -p data/raw/ood_test/real_camera
mkdir -p data/manifests
```

> **Tại sao chia folder theo source?** Vì khi train, bạn cần biết rõ mỗi ảnh đến từ đâu để:
>
> - Cân bằng tỷ lệ (balancing) giữa Real/GAN/Diffusion
> - Đánh giá per-source (model giỏi GAN nhưng dở Diffusion → biết cần thêm data Diffusion)
> - Debug: nếu model fail → biết fail ở nguồn nào

---

## Bước 1: Download CIFAKE (Ưu tiên #1)

### Tại sao CIFAKE là ưu tiên số 1?

CIFAKE = **CIFAR-10 + Fake** — bộ dataset 120K ảnh (60K thật từ CIFAR-10, 60K giả sinh bởi Stable Diffusion v1.4). Đây là lựa chọn tối ưu nhất vì:

1. **1-click download trên Kaggle**: ~500MB, không cần xin access
2. **Đã phân loại sẵn**: Folder `REAL/` và `FAKE/` — không cần xử lý thêm
3. **Có Diffusion data**: Ảnh fake sinh bởi Stable Diffusion (chính xác loại data HolmHz cần)
4. **Nhỏ gọn**: Fit vào Google Drive / Kaggle storage dễ dàng

### Cách download

#### Cách 1: Download trực tiếp qua Kaggle web (LUÂN LÀM)

1. Vào: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
2. Click nút **Download** (cần đăng nhập Kaggle — miễn phí)
3. File `.zip` sẽ download (~500MB)
4. Giải nén vào `data/raw/`:

```bash
# Sau khi giải nén, cấu trúc sẽ là:
# cifake-real-and-ai-generated-synthetic-images/
#   ├── test/
#   │   ├── FAKE/  (10,000 ảnh)
#   │   └── REAL/  (10,000 ảnh)
#   └── train/
#       ├── FAKE/  (50,000 ảnh)
#       └── REAL/  (50,000 ảnh)

# Copy vào đúng folder structure của HolmHz:
# CIFAKE Real → data/raw/real/cifake/
# CIFAKE Fake → data/raw/fake_diffusion/cifake/

# Windows PowerShell:
Copy-Item -Recurse "path\to\cifake\train\REAL\*" "data\raw\real\cifake\"
Copy-Item -Recurse "path\to\cifake\test\REAL\*"  "data\raw\real\cifake\"
Copy-Item -Recurse "path\to\cifake\train\FAKE\*" "data\raw\fake_diffusion\cifake\"
Copy-Item -Recurse "path\to\cifake\test\FAKE\*"  "data\raw\fake_diffusion\cifake\"

# Hoặc Git Bash:
cp -r path/to/cifake/train/REAL/* data/raw/real/cifake/
cp -r path/to/cifake/test/REAL/*  data/raw/real/cifake/
cp -r path/to/cifake/train/FAKE/* data/raw/fake_diffusion/cifake/
cp -r path/to/cifake/test/FAKE/*  data/raw/fake_diffusion/cifake/
```

#### Cách 2: Download qua Kaggle API (HOÀNG LÀM nếu quen CLI)

```bash
# 1. Setup Kaggle API (chạy 1 lần)
pip install kaggle
# Vào https://www.kaggle.com/settings → API → Create New Token
# File kaggle.json sẽ download
# Windows: Copy vào C:\Users\<tên_bạn>\.kaggle\kaggle.json

# 2. Download dataset
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data/raw/

# 3. Giải nén
cd data/raw/
unzip cifake-real-and-ai-generated-synthetic-images.zip -d cifake_download
cd ../..

# 4. Copy vào đúng folder
cp -r data/raw/cifake_download/train/REAL/* data/raw/real/cifake/
cp -r data/raw/cifake_download/test/REAL/*  data/raw/real/cifake/
cp -r data/raw/cifake_download/train/FAKE/* data/raw/fake_diffusion/cifake/
cp -r data/raw/cifake_download/test/FAKE/*  data/raw/fake_diffusion/cifake/

# 5. Dọn file zip + folder tạm (tiết kiệm ổ cứng)
rm -rf data/raw/cifake_download data/raw/cifake-real-and-ai-generated-synthetic-images.zip
```

### Kiểm tra sau download

```bash
# Đếm số ảnh
find data/raw/real/cifake -type f | wc -l        # Kỳ vọng: ~60,000
find data/raw/fake_diffusion/cifake -type f | wc -l  # Kỳ vọng: ~60,000

# Hoặc PowerShell:
(Get-ChildItem data\raw\real\cifake -File).Count
(Get-ChildItem data\raw\fake_diffusion\cifake -File).Count
```

### Subset (chỉ lấy đủ dùng)

Không cần dùng cả 60K ảnh. Để tiết kiệm thời gian xử lý + storage, lấy subset:

```python
# scripts/subset_cifake.py — Chọn random subset từ CIFAKE
"""
Lấy random subset từ CIFAKE dataset.
CIFAKE có 60K real + 60K fake, nhưng ta chỉ cần 7K mỗi loại cho training.
- 5K train + 1K val + 1K test = 7K ảnh mỗi loại
"""
import shutil
import random
from pathlib import Path

random.seed(42)  # Seed cố định để reproducible

def subset_folder(src_dir: str, dst_dir: str, count: int):
    """Lấy random `count` ảnh từ src_dir, copy sang dst_dir."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    all_images = sorted(list(src.glob("*.png")) + list(src.glob("*.jpg")))
    selected = random.sample(all_images, min(count, len(all_images)))

    for img_path in selected:
        shutil.copy2(img_path, dst / img_path.name)

    print(f"Copied {len(selected)} images: {src} → {dst}")
    return len(selected)

if __name__ == "__main__":
    # CIFAKE Real: lấy 7K có sẵn (thêm FFHQ sau)
    subset_folder("data/raw/real/cifake", "data/raw/real/cifake_subset", 7000)

    # CIFAKE Fake (Diffusion): lấy 7K
    subset_folder("data/raw/fake_diffusion/cifake", "data/raw/fake_diffusion/cifake_subset", 7000)

    print("\n✅ Subset done! Dùng folder *_subset cho pipeline.")
    print("Nếu muốn tăng data sau → chạy lại với count lớn hơn.")
```

```bash
# Chạy subset script
python scripts/subset_cifake.py
```

> 💡 **Tại sao seed=42?** Bất kỳ số nào cũng được, nhưng `42` là convention (nó nổi tiếng từ "The Hitchhiker's Guide to the Galaxy"). Quan trọng là dùng seed cố định để **reproducible** — chạy lại bao nhiêu lần cũng cho cùng kết quả. Điều này rất quan trọng trong khoa học.

---

## Bước 2: Download FFHQ subset (Real faces)

### Tại sao cần FFHQ khi đã có CIFAKE Real?

CIFAKE Real là ảnh từ CIFAR-10 — chủ yếu là ảnh đồ vật, phong cảnh, động vật ở **32×32 pixel**. HolmHz tập trung vào **ảnh khuôn mặt người** (face detection), nên cần thêm ảnh faces chất lượng cao.

**FFHQ** (Flickr-Faces-HQ) là bộ dataset chuẩn 70K ảnh khuôn mặt thật resolution 1024×1024. Toàn bộ cộng đồng AI dùng nó, nên kết quả sẽ tương đương paper quốc tế.

### Cách download (LUÂN LÀM)

#### Cách 1: Kaggle mirror (khuyến nghị)

1. Tìm trên Kaggle: https://www.kaggle.com/datasets — search "FFHQ"
2. Có nhiều mirror nhỏ, chọn một trong:
   - `arnaud58/flickrfaceshq-dataset-ffhq` (~10GB cho 70K ảnh 128×128)
   - `greatgamedota/ffhq-face-data-set` (thumbnail 128×128)
3. Download toàn bộ hoặc subset 5K

> ⚠️ **Lưu ý cho Luân**: Nếu file quá lớn (>5GB), download trên máy có WiFi mạnh. Hoặc download trực tiếp trên Google Drive để khỏi tốn bandwidth.

```bash
# Nếu dùng Kaggle API:
kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq -p data/raw/real/

# Giải nén + lấy subset 5K:
python scripts/subset_ffhq.py
```

#### Script subset_ffhq.py

```python
# scripts/subset_ffhq.py — Chọn random 5K ảnh khuôn mặt từ FFHQ
import shutil
import random
from pathlib import Path

random.seed(42)

src = Path("data/raw/real/ffhq_full")  # Folder chứa FFHQ đã giải nén
dst = Path("data/raw/real/ffhq")
dst.mkdir(parents=True, exist_ok=True)

all_images = sorted(
    list(src.glob("**/*.png")) + list(src.glob("**/*.jpg"))
)

if len(all_images) == 0:
    print(f"❌ Không tìm thấy ảnh trong {src}")
    print("Kiểm tra lại: giải nén FFHQ vào data/raw/real/ffhq_full/")
    exit(1)

count = min(5000, len(all_images))
selected = random.sample(all_images, count)

for img_path in selected:
    shutil.copy2(img_path, dst / img_path.name)

print(f"✅ Copied {count} FFHQ images to {dst}")
```

### Kiểm tra

```bash
# Đếm ảnh FFHQ
find data/raw/real/ffhq -type f | wc -l  # Kỳ vọng: 3000-5000

# Xem sample ảnh (mở 1 ảnh để kiểm tra)
# Windows: explorer data\raw\real\ffhq\
```

---

## Bước 3: Download StyleGAN faces (GAN Fake)

### Tại sao cần GAN data khi focus là Diffusion?

Vì mục tiêu của HolmHz là phát hiện **mọi loại** ảnh AI-generated, không chỉ Diffusion. Mặc dù GAN đang dần bị thay thế, nó vẫn được sử dụng trong nhiều ứng dụng. Training trên cả GAN + Diffusion giúp model robust hơn.

Ngoài ra, khi benchmark với 3 SOTA (đều train trên GAN), bạn cần model cũng xử lý tốt GAN → chứng minh HolmHz **không chỉ giỏi Diffusion mà còn không mất khả năng phát hiện GAN**.

### Cách download

#### Cách 1: Kaggle dataset (nhanh nhất)

Tìm trên Kaggle: search "fake faces" hoặc "stylegan faces":

- `xhlulu/140k-real-and-fake-faces` — 140K ảnh (70K thật + 70K StyleGAN)
- `ciplab/real-and-fake-face-detection` — Real + Fake faces

```bash
# Ví dụ với dataset 140k-real-and-fake-faces:
kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/raw/

# Giải nén, lấy folder fake/
# Copy 3-5K ảnh fake vào data/raw/fake_gan/stylegan/
```

#### Cách 2: Scrape từ thispersondoesnotexist.com

Mỗi lần truy cập = 1 ảnh StyleGAN mới. Viết script tự động:

```python
# scripts/download_stylegan_faces.py
"""
Download StyleGAN-generated faces từ thispersondoesnotexist.com.
Mỗi request = 1 ảnh 1024×1024 (~200KB).
3000 ảnh ≈ 600MB, ~1 giờ (rate limit 1 req/giây).
"""
import requests
from pathlib import Path
import time

dst = Path("data/raw/fake_gan/stylegan")
dst.mkdir(parents=True, exist_ok=True)

TARGET = 3000  # Số ảnh cần download
existing = len(list(dst.glob("*.jpg")))
print(f"Đã có {existing} ảnh, cần thêm {max(0, TARGET - existing)}")

for i in range(existing, TARGET):
    try:
        resp = requests.get(
            "https://thispersondoesnotexist.com",
            headers={"User-Agent": "HolmHz-Research/1.0"},
            timeout=15,
        )
        if resp.status_code == 200:
            (dst / f"stylegan_{i:05d}.jpg").write_bytes(resp.content)
            if i % 100 == 0:
                print(f"✅ [{i}/{TARGET}] Downloaded")
        else:
            print(f"⚠️ [{i}] HTTP {resp.status_code}")

        time.sleep(1.0)  # Rate limit: 1 request/giây — KHÔNG giảm, sẽ bị block
    except Exception as e:
        print(f"❌ [{i}] Error: {e}")
        time.sleep(5)  # Chờ lâu hơn nếu lỗi

print(f"\n✅ Done! Total: {len(list(dst.glob('*.jpg')))} ảnh")
```

```bash
# Chạy script (sẽ mất ~1 giờ cho 3000 ảnh)
python scripts/download_stylegan_faces.py

# Tip: chạy trong background nếu dùng Linux/Mac:
# nohup python scripts/download_stylegan_faces.py &
```

> ⚠️ **Lưu ý**: Nếu bị block (HTTP 429), tăng `time.sleep()` lên 2-3 giây. Hoặc chuyển sang Cách 1 (Kaggle) cho nhanh.

### Kiểm tra

```bash
find data/raw/fake_gan/stylegan -type f | wc -l  # Kỳ vọng: 3000-5000
```

---

## Bước 4: Self-generate SD v1.5 (Diffusion Fake)

### Tại sao tự generate khi đã có CIFAKE?

CIFAKE được sinh bởi Stable Diffusion **v1.4**. Nếu HolmHz chỉ train trên SD v1.4 → có thể chỉ "nhớ mặt" SD v1.4 mà fail trên SD v1.5, SDXL, hoặcGemini. Tự generate bằng SD v1.5 tạo **diversity** — model học pattern chung của Diffusion thay vì nhớ pattern riêng của 1 model.

### Cách generate trên Colab/Kaggle (miễn phí)

#### Bước 4.1: Tạo notebook trên Google Colab

Mở https://colab.research.google.com → New Notebook → đặt Runtime = **T4 GPU**

#### Bước 4.2: Code generate ảnh

Paste các cell sau vào Colab notebook:

**Cell 1: Cài đặt thư viện**

```python
!pip install diffusers transformers accelerate torch -q
```

**Cell 2: Load model**

```python
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

# Load Stable Diffusion v1.5 (sẽ tải ~5GB lần đầu)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,  # Float16 tiết kiệm VRAM
    safety_checker=None,        # Tắt safety check (cho research)
).to("cuda")

# Tối ưu VRAM
pipe.enable_attention_slicing()

print("✅ Model loaded!")
```

**Cell 3: Định nghĩa prompts**

```python
# Prompts đa dạng — tạo nhiều loại khuôn mặt khác nhau
face_prompts = [
    "a portrait photo of a young woman with brown hair, natural lighting, DSLR quality",
    "a headshot of a middle-aged man wearing glasses, studio lighting",
    "a close-up portrait of an elderly woman smiling, soft lighting",
    "a professional headshot of a young man in a suit, corporate style",
    "a candid portrait of a teenager, outdoor natural light",
    "a passport photo of a woman, neutral expression, white background",
    "a portrait of a man with beard, dramatic lighting, high quality",
    "a close-up face photo of a child, happy expression, bright daylight",
    "a selfie of a young woman, smartphone camera quality",
    "a formal portrait of an older man, black and white background",
]

# Prompts phong cảnh/đồ vật (tương tự CIFAR-10 categories)
object_prompts = [
    "a realistic photo of a red sports car on a highway",
    "a photo of a golden retriever dog in a park",
    "a photo of a tabby cat sitting on a windowsill",
    "a landscape photo of a mountain lake at sunset",
    "a photo of a commercial airplane taking off",
    "a photo of a white horse in a green meadow",
    "a photo of a cargo ship at sea",
    "a realistic photo of a deer in a forest",
    "a photo of a red fire truck on a city street",
    "a photo of a bouquet of colorful flowers",
]

all_prompts = face_prompts + object_prompts
print(f"✅ {len(all_prompts)} base prompts defined")
```

**Cell 4: Generate ảnh (2-3K)**

```python
from pathlib import Path
import random

# Mount Google Drive để save
from google.colab import drive
drive.mount('/content/drive')

output_dir = Path("/content/drive/MyDrive/HolmHz/data/sd15_generated")
output_dir.mkdir(parents=True, exist_ok=True)

TARGET = 2500  # Số ảnh cần generate
existing = len(list(output_dir.glob("*.png")))
print(f"Đã có {existing} ảnh, sẽ generate thêm {TARGET - existing}")

random.seed(42)

for i in range(existing, TARGET):
    prompt = random.choice(all_prompts)

    # Thêm variation bằng random seed
    generator = torch.Generator("cuda").manual_seed(i)

    image = pipe(
        prompt,
        num_inference_steps=30,     # 30 steps = chất lượng ok, nhanh
        guidance_scale=7.5,         # Mặc định SD v1.5
        generator=generator,
    ).images[0]

    image.save(output_dir / f"sd15_{i:05d}.png")

    if i % 100 == 0:
        print(f"✅ [{i}/{TARGET}] Generated: {prompt[:50]}...")

print(f"\n🎉 Done! Total: {len(list(output_dir.glob('*.png')))} images in {output_dir}")
```

> **Ước tính thời gian**: ~2 giây/ảnh trên T4 → 2500 ảnh ≈ **1.5 giờ**. Vừa đủ 1 session Colab free.

> **Nếu Colab disconnect**: Ảnh đã save vào Google Drive → chạy lại, script tự skip ảnh đã có (nhờ biến `existing`).

**Cell 5: Verify ảnh**

```python
from PIL import Image
import os

sample_images = list(output_dir.glob("*.png"))[:5]
for img_path in sample_images:
    img = Image.open(img_path)
    size_kb = os.path.getsize(img_path) / 1024
    print(f"{img_path.name}: {img.size} | {size_kb:.0f} KB")
```

#### Bước 4.3: Copy ảnh từ Drive về local

Sau khi generate xong trên Colab, download folder từ Google Drive:

```bash
# Cách 1: Download toàn bộ folder từ Google Drive web
# Vào Drive → HolmHz/data/sd15_generated/ → chuột phải → Download

# Cách 2: Sync bằng rclone (nếu quen CLI)
# rclone copy "drive:HolmHz/data/sd15_generated" data/raw/fake_diffusion/sd15/

# Copy vào đúng folder
# Giải nén nếu Drive export thành .zip
cp -r path/to/sd15_generated/* data/raw/fake_diffusion/sd15/
```

### Kiểm tra

```bash
find data/raw/fake_diffusion/sd15 -type f | wc -l  # Kỳ vọng: 2000-3000
```

---

## Bước 5: Chuẩn bị OOD Test Set

### Tại sao OOD Test lại là thước đo QUAN TRỌNG NHẤT?

```
┌──────────────────────────────────────────────────────────────────────┐
│   IN-DOMAIN TEST:                                                     │
│   Train trên CIFAKE + StyleGAN + SD v1.5                             │
│   Test trên CIFAKE held-out + StyleGAN held-out                       │
│   → Kỳ vọng: AUC ≥ 0.85 (khá dễ đạt)                               │
│                                                                       │
│   OUT-OF-DISTRIBUTION TEST:                                           │
│   Train trên CIFAKE + StyleGAN + SD v1.5                             │
│   Test trên GEMINI + FLUX (model CHƯA HỀ THẤY)                      │
│   → Kỳ vọng: AUC ≥ 0.65 (RẤT KHÓ — đây là thách thức chính)       │
│                                                                       │
│   Hội đồng sẽ HỎI: "Model hoạt động thế nào trên Gemini/Flux?"     │
│   Nếu AUC OOD < 0.60: Model vô dụng ngoài thực tế                  │
│   Nếu AUC OOD ≥ 0.65: Tốt hơn 3 SOTA (đều fail trên Diffusion)    │
│   Nếu AUC OOD ≥ 0.70: Rất tốt cho nghiên cứu SV                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.1: Ảnh Gemini (100-200 ảnh)

Vào https://gemini.google.com → nhập prompt tạo ảnh. Mỗi ảnh save vào `data/raw/ood_test/gemini/`.

**Danh sách prompt mẫu** (mỗi prompt tạo 1-4 ảnh):

```
Portrait prompts (ưu tiên - vì HolmHz focus faces):
- "Generate a realistic portrait photo of a young Vietnamese woman"
- "Create a headshot of a middle-aged Asian man with glasses"
- "Generate a passport-style photo of an elderly woman"
- "Create a selfie of a young man at a cafe"
- "Generate a professional corporate headshot"

Object/Scene prompts (đa dạng):
- "Generate a realistic photo of a golden retriever"
- "Create a photo of a red sports car"
- "Generate a landscape photo of Da Lat city"
- "Create a photo of Vietnamese street food"
- "Generate a photo of a tropical beach sunset"
```

> 💡 **Tip**: Nhập cùng kiểu prompt đã dùng cho SD v1.5 ở Bước 4. Điều này giúp so sánh: cùng concept → Gemini sinh khác SD v1.5 thế nào → model có phân biệt được không?

**Cách save ảnh từ Gemini**:

1. Gemini generate ảnh → chuột phải → Save Image As
2. Đặt tên: `gemini_001.png`, `gemini_002.png`, ...
3. Save vào `data/raw/ood_test/gemini/`

> ⚠️ Công đoạn này **thủ công** — không có API miễn phí. Mỗi ảnh mất ~30 giây. 200 ảnh ≈ **1.5 giờ**. Có thể chia cho Luân làm cùng.

### 5.2: Ảnh Flux (100-200 ảnh)

Flux là model Diffusion mới nhất (2024) từ Black Forest Labs. Có thể dùng miễn phí qua:

#### Cách 1: flux1.ai (free web)

1. Vào https://flux1.ai hoặc https://replicate.com/black-forest-labs/flux-schnell
2. Nhập prompt → Generate → Download
3. Save vào `data/raw/ood_test/flux/`
4. Đặt tên: `flux_001.png`, `flux_002.png`, ...

#### Cách 2: Replicate API (free tier)

```python
# Nếu tạo tài khoản Replicate.com (free tier)
# scripts/generate_flux_ood.py
import replicate
from pathlib import Path
import urllib.request

output_dir = Path("data/raw/ood_test/flux")
output_dir.mkdir(parents=True, exist_ok=True)

prompts = [
    "a realistic portrait photo of a young Asian woman",
    "a headshot of a middle-aged man in business attire",
    "a close-up photo of an elderly person smiling",
    # ... thêm prompts
]

for i, prompt in enumerate(prompts):
    try:
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": prompt}
        )
        # output là URL → download
        if output:
            url = output[0] if isinstance(output, list) else str(output)
            urllib.request.urlretrieve(url, output_dir / f"flux_{i:03d}.png")
            print(f"✅ [{i}] {prompt[:40]}...")
    except Exception as e:
        print(f"❌ [{i}] {e}")
```

> 💡 **Free tier Replicate**: Thường cho ~50-100 runs miễn phí. Đủ cho OOD test set.

### 5.3: Ảnh Real camera (200 ảnh)

Ảnh thật chụp bằng camera thật — để đối chứng OOD test:

```bash
# Copy ảnh đã có trong imgs/ folder
cp -r imgs/Real/* data/raw/ood_test/real_camera/

# Thêm: tự chụp bằng điện thoại hoặc download từ Unsplash (free)
# Unsplash: https://unsplash.com/s/photos/portrait
# Pexels: https://www.pexels.com/search/portrait/
```

> **Tạo sao cần ảnh thật trong OOD?** Vì nếu chỉ test Gemini/Flux (toàn ảnh giả) → không biết model có phân biệt đúng ảnh thật hay không. Cần **cả hai**: ảnh thật (kỳ vọng: predict "Real") + ảnh giả (kỳ vọng: predict "Fake").

### 5.4: Copy ảnh Gemini/Flux đã có trong imgs/

Project đã có sẵn một số ảnh mẫu:

```bash
# Kiểm tra ảnh đã có
ls imgs/Fake_AI_generated/
ls imgs/Real/

# Copy vào OOD test set
cp imgs/Fake_AI_generated/* data/raw/ood_test/gemini/
cp imgs/Real/* data/raw/ood_test/real_camera/
```

### Kiểm tra OOD

```bash
find data/raw/ood_test/gemini -type f | wc -l       # Kỳ vọng: 100-200
find data/raw/ood_test/flux -type f | wc -l          # Kỳ vọng: 100-200
find data/raw/ood_test/real_camera -type f | wc -l   # Kỳ vọng: ≥200
```

---

## Bước 6: Resize và tổ chức folder

### Tại sao phải resize?

Ảnh từ các nguồn khác nhau có kích thước khác nhau:

| Nguồn    | Resolution gốc | Sau resize          |
| -------- | -------------- | ------------------- |
| CIFAKE   | 32×32          | 224×224 (upscale)   |
| FFHQ     | 1024×1024      | 224×224 (downscale) |
| StyleGAN | 1024×1024      | 224×224 (downscale) |
| SD v1.5  | 512×512        | 224×224 (downscale) |
| Gemini   | Varies         | 224×224             |
| Flux     | Varies         | 224×224             |

EfficientNet-B0 nhận input 224×224. Nếu không resize → lỗi shape mismatch khi train.

> ⚠️ **CIFAKE 32→224 sẽ bị mờ**: Đúng, nhưng model vẫn học được texture patterns. Nếu AUC thấp quá → dùng CIFAKE ít hơn, tăng FFHQ + SD v1.5 (resolution cao hơn).

### Script resize toàn bộ

```python
# scripts/resize_all.py
"""
Resize tất cả ảnh trong data/raw/ về 224×224 → save vào data/processed/.
Cấu trúc folder giữ nguyên:
  data/raw/real/cifake/ → data/processed/real/cifake/
  data/raw/fake_gan/stylegan/ → data/processed/fake_gan/stylegan/
  ...
"""
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

TARGET_SIZE = (224, 224)
RAW_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")

stats = {}  # Ghi lại số ảnh mỗi folder


def resize_folder(src_dir: Path, dst_dir: Path) -> int:
    """Resize tất cả ảnh trong folder. Trả về số ảnh đã xử lý."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(src_dir.glob("*.jpg"))
        + list(src_dir.glob("*.jpeg"))
        + list(src_dir.glob("*.png"))
        + list(src_dir.glob("*.webp"))
    )

    success = 0
    errors = 0

    for img_path in tqdm(images, desc=f"Resizing {src_dir.name}"):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            # Save tất cả thành PNG (lossless, consistent)
            img.save(dst_dir / f"{img_path.stem}.png")
            success += 1
        except Exception as e:
            errors += 1
            if errors <= 5:  # Chỉ print 5 lỗi đầu
                print(f"  ⚠️ Error: {img_path.name} — {e}")

    print(f"  ✅ {success} resized, {errors} errors")
    return success


def main():
    folders_to_process = [
        # (src_relative, dst_relative)
        ("real/cifake_subset", "real/cifake"),
        ("real/ffhq", "real/ffhq"),
        ("fake_gan/stylegan", "fake_gan/stylegan"),
        ("fake_diffusion/cifake_subset", "fake_diffusion/cifake"),
        ("fake_diffusion/sd15", "fake_diffusion/sd15"),
        ("ood_test/gemini", "ood_test/gemini"),
        ("ood_test/flux", "ood_test/flux"),
        ("ood_test/real_camera", "ood_test/real_camera"),
    ]

    total = 0
    for src_rel, dst_rel in folders_to_process:
        src = RAW_ROOT / src_rel
        dst = PROCESSED_ROOT / dst_rel

        if not src.exists():
            print(f"⏭️ Skip (not found): {src}")
            continue

        count = resize_folder(src, dst)
        stats[dst_rel] = count
        total += count

    # Save stats
    stats["total"] = total
    stats_path = Path("data/manifests/dataset_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"📊 TỔNG: {total} ảnh đã resize về {TARGET_SIZE}")
    print(f"📄 Stats saved: {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
```

```bash
# Chạy resize
python scripts/resize_all.py
```

> 💡 **Chạy mất bao lâu?** ~20K ảnh × ~0.01 giây/ảnh ≈ **3-4 phút**. Rất nhanh trên local.

### Cấu trúc folder sau resize

```
data/
├── raw/                         # Ảnh gốc (giữ nguyên, backup)
│   ├── real/
│   │   ├── cifake/              # 60K ảnh CIFAKE Real (32×32)
│   │   ├── cifake_subset/       # 7K subset
│   │   └── ffhq/               # 3-5K FFHQ (1024×1024)
│   ├── fake_gan/
│   │   └── stylegan/            # 3-5K StyleGAN (1024×1024)
│   ├── fake_diffusion/
│   │   ├── cifake/              # 60K CIFAKE Fake (32×32)
│   │   ├── cifake_subset/       # 7K subset
│   │   └── sd15/                # 2-3K SD v1.5 (512×512)
│   └── ood_test/
│       ├── gemini/              # 100-200 Gemini
│       ├── flux/                # 100-200 Flux
│       └── real_camera/         # 200 ảnh thật
│
├── processed/                   # ẢNH ĐÃ RESIZE 224×224 ← Model đọc từ đây
│   ├── real/
│   │   ├── cifake/
│   │   └── ffhq/
│   ├── fake_gan/
│   │   └── stylegan/
│   ├── fake_diffusion/
│   │   ├── cifake/
│   │   └── sd15/
│   └── ood_test/
│       ├── gemini/
│       ├── flux/
│       └── real_camera/
│
└── manifests/
    └── dataset_stats.json       # Thống kê số ảnh
```

> ⚠️ **Quan trọng**: `data/raw/` và `data/processed/` đều đã có trong `.gitignore` → KHÔNG được push lên GitHub. Ảnh chỉ lưu local + Google Drive (backup).

---

## Bước 7: Tạo dataset_stats.json

Script `resize_all.py` ở Bước 6 đã tự tạo file này. Nhưng nếu cần tạo/cập nhật riêng:

```python
# scripts/dataset_stats.py
"""
Tạo/cập nhật file data/manifests/dataset_stats.json.
Đếm số ảnh trong mỗi folder của data/processed/.
"""
import json
from pathlib import Path
from datetime import datetime


def count_images(folder: Path) -> int:
    """Đếm số file ảnh trong folder."""
    if not folder.exists():
        return 0
    return len(
        list(folder.glob("*.png"))
        + list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
    )


def main():
    processed = Path("data/processed")

    stats = {
        "created": datetime.now().isoformat(),
        "image_size": "224x224",
        "sources": {
            "real": {
                "cifake": count_images(processed / "real/cifake"),
                "ffhq": count_images(processed / "real/ffhq"),
            },
            "fake_gan": {
                "stylegan": count_images(processed / "fake_gan/stylegan"),
            },
            "fake_diffusion": {
                "cifake": count_images(processed / "fake_diffusion/cifake"),
                "sd15": count_images(processed / "fake_diffusion/sd15"),
            },
            "ood_test": {
                "gemini": count_images(processed / "ood_test/gemini"),
                "flux": count_images(processed / "ood_test/flux"),
                "real_camera": count_images(processed / "ood_test/real_camera"),
            },
        },
    }

    # Tính tổng
    total_real = sum(stats["sources"]["real"].values())
    total_gan = sum(stats["sources"]["fake_gan"].values())
    total_diffusion = sum(stats["sources"]["fake_diffusion"].values())
    total_ood = sum(stats["sources"]["ood_test"].values())
    total_all = total_real + total_gan + total_diffusion + total_ood

    stats["summary"] = {
        "total_real": total_real,
        "total_fake_gan": total_gan,
        "total_fake_diffusion": total_diffusion,
        "total_ood_test": total_ood,
        "total_all": total_all,
    }

    # Kiểm tra acceptance criteria
    stats["acceptance_criteria"] = {
        "real_gte_6k": total_real >= 6000,
        "diffusion_gte_5k": total_diffusion >= 5000,
        "gan_gte_3k": total_gan >= 3000,
        "ood_gemini_gte_100": stats["sources"]["ood_test"]["gemini"] >= 100,
        "ood_flux_gte_100": stats["sources"]["ood_test"]["flux"] >= 100,
        "ood_real_gte_200": stats["sources"]["ood_test"]["real_camera"] >= 200,
    }

    all_pass = all(stats["acceptance_criteria"].values())
    stats["all_criteria_pass"] = all_pass

    # Save
    output_path = Path("data/manifests/dataset_stats.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Print
    print("📊 Dataset Statistics:")
    print(f"  Real:       {total_real:,} ảnh (cần ≥6K)")
    print(f"  GAN Fake:   {total_gan:,} ảnh (cần ≥3K)")
    print(f"  Diff Fake:  {total_diffusion:,} ảnh (cần ≥5K)")
    print(f"  OOD Test:   {total_ood:,} ảnh")
    print(f"  ────────────────────")
    print(f"  TOTAL:      {total_all:,} ảnh")
    print(f"\n{'✅ ALL CRITERIA PASS!' if all_pass else '❌ SOME CRITERIA NOT MET:'}")

    if not all_pass:
        for key, val in stats["acceptance_criteria"].items():
            if not val:
                print(f"  ❌ {key}: FAILED")

    print(f"\n📄 Saved: {output_path}")


if __name__ == "__main__":
    main()
```

```bash
python scripts/dataset_stats.py
```

---

## Bước 8: Validation & Data Integrity

Trước khi tuyên bố "xong", cần kiểm tra data không bị lỗi:

```python
# scripts/validate_dataset.py
"""
Kiểm tra data integrity:
1. Ảnh không bị corrupt (mở được bằng PIL)
2. Tất cả ảnh đều là 224×224
3. Không có file 0 bytes
4. Không có folder rỗng
"""
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def validate_folder(folder: Path) -> dict:
    """Kiểm tra toàn bộ ảnh trong folder."""
    results = {"total": 0, "valid": 0, "corrupt": [], "wrong_size": [], "zero_bytes": []}

    images = sorted(
        list(folder.glob("*.png"))
        + list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
    )
    results["total"] = len(images)

    for img_path in tqdm(images, desc=f"Validating {folder.name}", leave=False):
        # Check 0 bytes
        if img_path.stat().st_size == 0:
            results["zero_bytes"].append(str(img_path))
            continue

        try:
            img = Image.open(img_path)
            img.verify()  # Kiểm tra file không corrupt

            # Re-open (verify closes the file)
            img = Image.open(img_path)

            if img.size != (224, 224):
                results["wrong_size"].append(f"{img_path.name}: {img.size}")
            else:
                results["valid"] += 1

        except Exception as e:
            results["corrupt"].append(f"{img_path.name}: {e}")

    return results


def main():
    processed = Path("data/processed")

    folders = [
        processed / "real/cifake",
        processed / "real/ffhq",
        processed / "fake_gan/stylegan",
        processed / "fake_diffusion/cifake",
        processed / "fake_diffusion/sd15",
        processed / "ood_test/gemini",
        processed / "ood_test/flux",
        processed / "ood_test/real_camera",
    ]

    all_ok = True

    for folder in folders:
        if not folder.exists():
            print(f"⏭️ {folder.relative_to(processed)} — not found")
            continue

        results = validate_folder(folder)
        status = "✅" if results["valid"] == results["total"] else "⚠️"

        print(f"{status} {folder.relative_to(processed)}: "
              f"{results['valid']}/{results['total']} valid")

        if results["corrupt"]:
            all_ok = False
            print(f"  ❌ Corrupt: {results['corrupt'][:3]}")
        if results["wrong_size"]:
            all_ok = False
            print(f"  ❌ Wrong size: {results['wrong_size'][:3]}")
        if results["zero_bytes"]:
            all_ok = False
            print(f"  ❌ Zero bytes: {results['zero_bytes'][:3]}")

    print(f"\n{'='*50}")
    print(f"{'✅ ALL DATA VALID!' if all_ok else '❌ SOME ISSUES FOUND — fix before proceeding'}")


if __name__ == "__main__":
    main()
```

```bash
python scripts/validate_dataset.py
```

---

## Bước 9: Commit & PR

### Commit scripts (KHÔNG commit data)

```bash
# Kiểm tra status
git status

# Thêm scripts đã tạo
git add scripts/subset_cifake.py
git add scripts/subset_ffhq.py
git add scripts/download_stylegan_faces.py
git add scripts/resize_all.py
git add scripts/dataset_stats.py
git add scripts/validate_dataset.py

# Thêm file stats (file nhỏ, nên commit)
git add data/manifests/dataset_stats.json

# KHÔNG add data/raw/ hay data/processed/ (đã có trong .gitignore)

# Commit
git commit -m "feat(data): add data collection scripts + dataset stats

- subset_cifake.py: Random subset CIFAKE dataset
- subset_ffhq.py: Random subset FFHQ faces
- download_stylegan_faces.py: Scrape StyleGAN faces
- resize_all.py: Resize all images to 224x224
- dataset_stats.py: Generate dataset statistics
- validate_dataset.py: Data integrity check

Refs: TASK_1.2"

# Push
git push origin feat/s1/data-collection
```

### Tạo PR trên GitHub

Mở: `https://github.com/EurusDevSec/HolmHz/compare/main...feat/s1/data-collection`

PR description mẫu:

```markdown
## Task 1.2: Data Collection

### Thay đổi

- Thêm 6 scripts cho data collection pipeline
- Dataset stats: [paste output dataset_stats.py]

### Dataset Summary

| Source            | Count       | Type           |
| ----------------- | ----------- | -------------- |
| CIFAKE Real       | 7,000       | Real           |
| FFHQ              | 3,000-5,000 | Real (faces)   |
| StyleGAN          | 3,000-5,000 | GAN Fake       |
| CIFAKE Fake       | 7,000       | Diffusion Fake |
| SD v1.5 self-gen  | 2,000-3,000 | Diffusion Fake |
| Gemini (OOD)      | 100-200     | OOD Test       |
| Flux (OOD)        | 100-200     | OOD Test       |
| Real camera (OOD) | 200         | OOD Test       |

### Acceptance Criteria

- [ ] ≥6k Real ✅
- [ ] ≥5k Diffusion Fake ✅
- [ ] ≥3k GAN Fake ✅
- [ ] OOD test set ready ✅
- [ ] All images 224×224 ✅
- [ ] dataset_stats.json ✅
```

---

## Checklist hoàn thành

Trước khi đánh dấu Task 1.2 ✅ DONE, kiểm tra hết:

### Data đủ số lượng

- [ ] ≥6K ảnh Real (CIFAKE Real subset + FFHQ) trong `data/processed/`
- [ ] ≥5K ảnh Diffusion Fake (CIFAKE Fake + SD v1.5) trong `data/processed/`
- [ ] ≥3K ảnh GAN Fake (StyleGAN) trong `data/processed/`
- [ ] 100-200 ảnh Gemini trong `data/processed/ood_test/gemini/`
- [ ] 100-200 ảnh Flux trong `data/processed/ood_test/flux/`
- [ ] ≥200 ảnh Real camera trong `data/processed/ood_test/real_camera/`
- [ ] Ảnh trong `imgs/` đã copy vào OOD test set

### Data quality

- [ ] Tất cả ảnh processed đã resize về 224×224
- [ ] `python scripts/validate_dataset.py` → ALL VALID
- [ ] Không có file 0 bytes hoặc corrupt

### Organization

- [ ] Folder structure: `data/processed/{real,fake_gan,fake_diffusion,ood_test}/`
- [ ] File `data/manifests/dataset_stats.json` tồn tại và chính xác
- [ ] `python scripts/dataset_stats.py` → ALL CRITERIA PASS

### Git

- [ ] Branch: `feat/s1/data-collection`
- [ ] Scripts committed
- [ ] PR Created trên GitHub
- [ ] `ruff check .` clean (không có lint errors trong scripts)

---

## Troubleshooting

### Q: Kaggle download chậm hoặc bị lỗi 403

**A**: Tạo Kaggle API token → cài `pip install kaggle` → dùng CLI thay web. Hoặc download bằng Colab notebook (tốc độ Google server nhanh hơn).

### Q: CIFAKE 32×32 resize lên 224×224 quá mờ, model có học được không?

**A**: Có. EfficientNet-B0 vẫn capture được texture features ở low resolution. Nhiều paper dùng CIFAKE cho deepfake detection. Nếu AUC quá thấp (< 0.80) → tăng FFHQ (1024×1024) và SD v1.5 (512×512) data.

### Q: thispersondoesnotexist.com bị block (HTTP 429)

**A**: Tăng `time.sleep()` lên 3-5 giây. Hoặc chuyển sang Kaggle dataset "fake faces" (nhanh hơn nhiều).

### Q: Colab disconnect giữa chừng khi generate SD v1.5

**A**: Script đã thiết kế resume — chạy lại sẽ skip ảnh đã generate (nhờ biến `existing`). Ảnh save vào Google Drive nên không mất.

### Q: Không đủ 200 ảnh Gemini/Flux (tạo thủ công mệt quá)

**A**: Tối thiểu 100 ảnh mỗi loại vẫn OK cho research. OOD test set nhỏ = variance cao hơn nhưng vẫn có ý nghĩa thống kê nếu AUC > 0.65.

### Q: Disk space không đủ cho 60K ảnh CIFAKE

**A**: Dùng subset (7K) thay vì toàn bộ. Script `subset_cifake.py` đã hỗ trợ. 7K ảnh × 224×224 × 3 channels × 1 byte ≈ **1GB** — rất nhỏ.

---

## Phân công Luân

Gửi cho Luân hướng dẫn sau (copy & paste):

---

### Hướng dẫn cho Luân — Task 1.2

**Việc cần làm** (ước tính ~2 giờ):

1. **Download CIFAKE** (30 phút):
   - Vào: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
   - Đăng nhập/tạo tài khoản Kaggle (miễn phí)
   - Click **Download** → file `.zip` ~500MB
   - Giải nén → gửi Hoàng (hoặc upload Google Drive chung)

2. **Download FFHQ** (30 phút):
   - Vào Kaggle, tìm "FFHQ faces"
   - Download bản thumbnail 128×128 (nhẹ hơn)
   - Gửi Hoàng

3. **Tạo ảnh Gemini** (1 giờ — tùy chọn, chia với Hoàng):
   - Vào https://gemini.google.com
   - Nhập các prompt: "Generate a realistic portrait photo of a young woman", v.v.
   - Save mỗi ảnh: chuột phải → Save Image As → `gemini_001.png`, ...
   - Mục tiêu: 50-100 ảnh (Hoàng sẽ tạo thêm phần còn lại)

**Deadline**: 28/02/2026 (trước target 02/03 để Hoàng còn xử lý)

**Gửi kết quả**: Upload lên Google Drive chung hoặc USB/hard disk.

---

> 📝 **Next step**: Sau khi Task 1.2 xong → chuyển sang Task 1.3 (Data Pipeline) — viết code đọc ảnh từ `data/processed/` vào PyTorch DataLoader để train.

---

**Last Updated**: 24/02/2026  
**Author**: Generated by GitHub Copilot for Lê Văn Hoàng  
**Version**: 1.0 (aligned with PROJECT_PLAN.md v4.0 — Revised 24/02/2026)
