# 📖 HƯỚNG DẪN CHI TIẾT SPRINT 1: TASK 1.1 → 1.6

> **Dành cho**: Lê Văn Hoàng — người chưa có nền tảng ML/DL, học qua thực hành  
> **Triết lý**: Mỗi bước không chỉ hướng dẫn **làm gì** mà giải thích **tại sao làm vậy**  
> **Thời gian**: ~4-6 tuần (02/2026 → 03/2026)  
> **Tiền đề**: Đã chạy thử 3 dự án SOTA (CNNDetection, UniversalFakeDetect, DeepfakeBench) ✅

---

## 📋 Mục lục

- [Bức tranh tổng thể: Sprint 1 là gì?](#bức-tranh-tổng-thể-sprint-1-là-gì)
- [TASK 1.1: Environment Setup](#task-11-environment-setup)
- [TASK 1.2: Data Collection](#task-12-data-collection)
- [TASK 1.3: Data Pipeline](#task-13-data-pipeline)
- [TASK 1.4: Model Architecture](#task-14-model-architecture)
- [TASK 1.5: Training Pipeline](#task-15-training-pipeline)
- [TASK 1.6: Baseline Training](#task-16-baseline-training)
- [Tổng kết Sprint 1](#tổng-kết-sprint-1)

---

## Bức tranh tổng thể: Sprint 1 là gì?

Trước khi đi vào từng task, hãy hiểu Sprint 1 nằm ở đâu trong dự án:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DỰ ÁN HOLMHZ                                 │
│                                                                      │
│  Phase 0 (Đã xong ✅):  Chạy thử 3 dự án SOTA, hiểu bài toán       │
│                                                                      │
│  ► Phase 1 - Sprint 1 (BẠN ĐANG Ở ĐÂY):                            │
│    Task 1.1  Setup môi trường dự án                                  │
│    Task 1.2  Thu thập dữ liệu (ảnh thật + ảnh giả)                  │
│    Task 1.3  Xây dựng data pipeline (code đọc & xử lý ảnh)          │
│    Task 1.4  Thiết kế kiến trúc model (EfficientNet-B0)              │
│    Task 1.5  Xây dựng training pipeline (code huấn luyện)            │
│    Task 1.6  Train baseline model (chạy huấn luyện thực tế)          │
│                                                                      │
│  Phase 1 - Sprint 2:  Đánh giá, so sánh SOTA, Grad-CAM, Export      │
│  Phase 2:             Web demo, báo cáo, bảo vệ                     │
└──────────────────────────────────────────────────────────────────────┘
```

**Mục tiêu cuối Sprint 1**: Có một model EfficientNet-B0 đã được train, đạt AUC ≥ 0.88 trên tập validation (in-domain). Đây là "baseline" — điểm xuất phát để so sánh cải thiện ở các Sprint sau.

### Sơ đồ phụ thuộc giữa các Task

```
Task 1.1 (Environment)
    │
    ├───► Task 1.2 (Data Collection)  ←── có thể làm song song với 1.1
    │         │
    │         ▼
    ├───► Task 1.3 (Data Pipeline)
    │         │
    ├───► Task 1.4 (Model Architecture)  ←── có thể song song với 1.3
    │         │
    │         ▼
    └───► Task 1.5 (Training Pipeline)  ←── cần 1.3 + 1.4 xong
              │
              ▼
          Task 1.6 (Baseline Training)  ←── cần 1.5 xong
```

> 💡 **Gợi ý**: Bạn có thể làm 1.1 và 1.2 cùng lúc (download data trong khi setup env). Tương tự 1.3 và 1.4 có thể song song.

---

## TASK 1.1: Environment Setup

### 🎯 Mục tiêu

Thiết lập môi trường phát triển: cài đặt thư viện, tổ chức folder, đảm bảo mọi thứ sẵn sàng để code.

### 🧠 Tại sao cần làm bước này?

Khi bạn chạy thử 3 dự án SOTA (CNNDetection, UniversalFakeDetect, DeepfakeBench), bạn đã thấy:

- **DeepfakeBench** bị "dependency hell" — cài thư viện xung đột, phải mock dlib, imgaug crash.
- **CNNDetection** thì đơn giản nhưng code bị gom vào ít file, khó mở rộng.
- **UniversalFakeDetect** không có `requirements.txt`.

> **Bài học rút ra**: Nếu ngay từ đầu không setup cẩn thận, về sau sẽ mất hàng giờ debug chỉ vì xung đột thư viện. Đây là lý do DevOps luôn ưu tiên infrastructure trước.

### 📚 Kiến thức nền cần hiểu

#### Virtual Environment là gì? Tại sao cần?

Hãy tưởng tượng máy tính của bạn giống một căn phòng chung. Mỗi dự án Python cần thư viện khác phiên bản nhau (ví dụ: dự án A cần `numpy 1.24`, dự án B cần `numpy 2.0`). Nếu cài chung vào một chỗ → xung đột.

**Virtual environment** = tạo "phòng riêng" cho mỗi dự án, mỗi phòng có bộ thư viện riêng, không ảnh hưởng nhau.

```
Máy tính của bạn
├── Python System (không đụng vào)
├── .venv/ cho HolmHz      ← PyTorch 2.1, timm 1.0
├── .venv/ cho DeepfakeBench ← PyTorch 2.0, dlib
└── .venv/ cho Web Project   ← Django, không có PyTorch
```

#### `pip install -e .` là gì? Tại sao không chỉ `pip install`?

Khi bạn code dự án có cấu trúc phức tạp (nhiều folder lồng nhau), Python cần biết "package `holmhz` nằm ở đâu?". Có 2 cách:

1. **Hack xấu** (`sys.path.append`): Thêm đường dẫn thủ công. Dễ vỡ, không chuyên nghiệp.
2. **Cách chuẩn** (`pip install -e .`): Đăng ký package của bạn với Python. Flag `-e` (editable) nghĩa là "cài nhưng liên kết tới source code gốc" — sửa code là có hiệu lực ngay, không cần cài lại.

```python
# Sau khi pip install -e ., bạn import như thư viện chuẩn:
from holmhz.backbones.efficientnet import EfficientNetBackbone
from holmhz.data.transforms import get_train_transforms

# Thay vì hack:
# import sys; sys.path.append('../../src')  ← KHÔNG LÀM THẾ NÀY
```

#### YAML Config là gì? Tại sao không hardcode?

Khi train model AI, bạn cần thay đổi rất nhiều tham số: learning rate, batch size, số epochs... Nếu viết thẳng vào code:

```python
# ❌ Hardcode — mỗi lần đổi tham số phải sửa code
lr = 0.001
batch_size = 32
epochs = 30
```

Thay vào đó, dùng file YAML:

```yaml
# ✅ Config file — đổi tham số chỉ cần sửa file YAML
training:
  lr: 0.001
  batch_size: 32
  epochs: 30
```

> **Bài học từ DeepfakeBench**: Họ dùng YAML config cho mỗi detector riêng (`config/detector/efficientnet.yaml`). Khi muốn thử model khác, chỉ đổi file config, không sửa code. Pattern rất hay mà HolmHz nên học.

### 🛠️ Hướng dẫn thực hiện

#### Bước 1: Tạo Virtual Environment

```bash
# Di chuyển vào thư mục dự án
cd R:/_Projects/Eurus_Workspace/HolmHz

# Tạo virtual environment bằng Python
python -m venv .venv

# Kích hoạt (Windows)
.venv\Scripts\activate

# Kiểm tra — nên thấy (.venv) đầu dòng terminal
python --version
pip --version
```

> 💡 **Tại sao dùng `.venv` chứ không phải `venv`?**: Dấu chấm đầu tên khiến folder bị ẩn trên Linux/Mac, và `.gitignore` đã có sẵn để ignore nó. Đây là convention phổ biến.

#### Bước 2: Cài đặt PyTorch (quan trọng nhất)

PyTorch là "xương sống" của dự án. Cài riêng trước vì nó cần chọn đúng phiên bản GPU/CPU.

```bash
# Nếu BẠN CÓ GPU NVIDIA (kiểm tra bằng: nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Nếu KHÔNG CÓ GPU (chỉ CPU — chậm hơn nhiều nhưng vẫn chạy được)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> **Tại sao cần GPU?** Training model AI = hàng triệu phép nhân ma trận. GPU có hàng nghìn nhân tính toán song song, nhanh hơn CPU 10-100 lần. Tuy nhiên, HolmHz sẽ train chính trên **Google Colab** (free GPU T4), máy local chỉ cần CPU để phát triển code.

Kiểm tra PyTorch chạy được:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

#### Bước 3: Cài đặt toàn bộ dependencies

```bash
# Cài package holmhz ở chế độ editable + dev dependencies
pip install -e ".[dev]"

# Nếu lỗi, thử cài từ requirements:
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Giải thích từng nhóm thư viện** (đã khai báo trong `pyproject.toml`):

| Nhóm         | Thư viện                  | Vai trò                     | Ví dụ tương tự         |
| ------------ | ------------------------- | --------------------------- | ---------------------- |
| **Core ML**  | `torch`, `torchvision`    | Framework DL chính          | Như Rails cho web      |
| **Core ML**  | `timm`                    | Thư viện pre-trained models | Kho vũ khí có sẵn      |
| **Core ML**  | `pytorch-grad-cam`        | Tạo heatmap giải thích      | "Tại sao AI nghĩ vậy?" |
| **Data**     | `pillow`, `opencv-python` | Đọc/ghi ảnh                 | Photoshop cho code     |
| **Data**     | `albumentations`          | Tăng cường dữ liệu          | "Photocopy biến dạng"  |
| **Training** | `wandb`                   | Ghi log thí nghiệm          | Dashboard theo dõi     |
| **Training** | `omegaconf`               | Đọc YAML config             | Quản lý cài đặt        |
| **Web**      | `fastapi`, `gradio`       | API + UI demo               | Giao diện cho user     |
| **Quality**  | `ruff`, `pytest`          | Lint code + test            | Kiểm tra chất lượng    |

#### Bước 4: Tạo file `.env.example`

```bash
# Tạo file .env.example (template cho mọi người)
```

Nội dung file `.env.example`:

```env
# === HolmHz Environment Variables ===

# Weights & Biases (đăng ký miễn phí tại wandb.ai)
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=holmhz

# Dataset paths (điều chỉnh theo máy bạn)
DATA_ROOT=./data
CHECKPOINT_DIR=./outputs/checkpoints

# Device
DEVICE=cuda  # hoặc cpu nếu không có GPU
```

> **Tại sao dùng `.env`?** Để không hardcode API key hay đường dẫn vào code. Mỗi người có thể có key khác nhau, path khác nhau. File `.env` KHÔNG được commit lên Git (đã có trong `.gitignore`). Chỉ commit `.env.example` làm template.

#### Bước 5: Setup Weights & Biases (W&B)

W&B là công cụ theo dõi thí nghiệm ML. Thay vì nhìn terminal lướt số, W&B tạo dashboard đẹp hiển thị loss, accuracy theo thời gian thực.

```bash
# 1. Đăng ký tài khoản miễn phí tại: https://wandb.ai/
# 2. Lấy API key từ: https://wandb.ai/authorize
# 3. Login trên terminal:
wandb login
# Paste API key khi được hỏi
```

> **Tại sao cần tracking?** Khi train model AI, bạn sẽ chạy hàng chục thí nghiệm: thay đổi learning rate, thêm augmentation, đổi batch size... Nếu không ghi lại, bạn sẽ quên thí nghiệm nào dùng tham số gì, kết quả ra sao. W&B tự động ghi tất cả.

#### Bước 6: Viết config YAML cơ bản

Tạo nội dung cho `configs/train.yaml`:

```yaml
# ============================================
# HolmHz Training Configuration
# ============================================
# File này chứa TẤT CẢ tham số training.
# Muốn thử nghiệm khác → copy file → đổi giá trị → chạy lại.

model:
  name: efficientnet_b0 # Backbone model
  pretrained: true # Dùng weights đã train trên ImageNet
  num_classes: 1 # Binary: 1 output → Sigmoid → P(Fake)
  dropout: 0.3 # Xác suất "tắt ngẫu nhiên" neuron (chống overfitting)
  freeze_backbone: true # Ban đầu đóng băng backbone, chỉ train head

training:
  epochs: 30 # Số vòng lặp qua toàn bộ dữ liệu
  batch_size: 32 # Số ảnh xử lý cùng lúc (GPU T4: 32 an toàn)
  learning_rate: 0.001 # Tốc độ học — quá cao: model nhảy loạn; quá thấp: học chậm
  optimizer: adamw # Thuật toán tối ưu
  weight_decay: 0.0001 # Regularization — giảm overfitting
  scheduler: cosine # Giảm LR dần theo đường cong cosine
  early_stopping:
    patience: 5 # Dừng nếu 5 epochs liên tiếp không cải thiện
    monitor: val_auc # Theo dõi AUC trên validation set

data:
  train_manifest: data/manifests/train.json
  val_manifest: data/manifests/val.json
  image_size: 224 # EfficientNet-B0 chuẩn: 224x224
  num_workers: 4 # Số thread đọc data song song
  augmentation: true # Bật/tắt data augmentation

wandb:
  project: holmhz
  entity: null # Để null = dùng user mặc định
  log_every_n_steps: 10
```

Tạo nội dung cho `configs/detectors/efficientnet_b0.yaml`:

```yaml
# ============================================
# EfficientNet-B0 Detector Configuration
# ============================================
# Pattern học từ DeepfakeBench: mỗi detector 1 file config riêng.
# Khi muốn thử CLIP, chỉ cần tạo file clip_vit.yaml.

detector:
  name: efficientnet_b0
  backbone:
    name: efficientnet_b0
    pretrained: true
    features_dim: 1280 # EfficientNet-B0 output 1280 features
  head:
    dropout: 0.3
    num_classes: 1 # Binary classification
  loss:
    name: bce_with_logits # Binary Cross-Entropy (nhận logits, tự tính sigmoid)
```

#### Bước 7: Viết file `__init__.py` cho package chính

```python
# src/holmhz/__init__.py
"""
HolmHz - Synthetic Image Detection System
==========================================
Hệ thống phát hiện ảnh tổng hợp bằng CNN với Explainable AI (Grad-CAM).

Modules:
    backbones   - Mạng trích xuất đặc trưng (EfficientNet-B0)
    detectors   - Bộ phát hiện (backbone + classification head)
    data        - Dataset classes và data transforms
    training    - Training loop, early stopping, schedulers
    losses      - Hàm mất mát (BCE)
    metrics     - Đánh giá (AUC, Accuracy)
    evaluation  - Benchmark và so sánh
    xai         - Giải thích mô hình (Grad-CAM)
    exports     - Xuất model (ONNX)
    utils       - Tiện ích chung
"""

__version__ = "0.1.0"
```

#### Bước 8: Kiểm tra linting và test

```bash
# Chạy ruff kiểm tra code style
ruff check src/

# Chạy pytest (hiện chưa có test nào, nhưng framework phải chạy được)
pytest tests/ -v

# Kiểm tra import holmhz (sau khi pip install -e .)
python -c "import holmhz; print(f'HolmHz v{holmhz.__version__}')"
```

### ✅ Checklist hoàn thành Task 1.1

- [ ] `.venv/` tạo xong, `pip install -e .` thành công
- [ ] `python -c "import torch; import timm; import holmhz"` không lỗi
- [ ] `.env.example` có đầy đủ placeholder
- [ ] `configs/train.yaml` có nội dung hợp lệ
- [ ] `wandb login` thành công
- [ ] `ruff check src/` chạy clean (0 errors)

---

## TASK 1.2: Data Collection

### 🎯 Mục tiêu

Thu thập **≥25,000 ảnh** gồm: ảnh thật (Real) + ảnh giả GAN + ảnh giả Diffusion + ảnh OOD test.

### 🧠 Tại sao Data lại quan trọng nhất?

Đây là **bài học số 1** bạn đã rút ra từ Phase 0:

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

### 📚 Kiến thức nền: GAN vs Diffusion — Ảnh giả khác nhau thế nào?

#### GAN (Generative Adversarial Network) — "Thế hệ cũ" (2014-2022)

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

#### Diffusion — "Thế hệ mới" (2022-nay)

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

### 📚 Kiến thức nền: In-Domain vs Out-of-Distribution (OOD)

Đây là khái niệm **cực kỳ quan trọng** trong ML, và là tiêu chí đánh giá chính của hội đồng:

| Thuật ngữ                     | Ý nghĩa                                | Ví dụ                                       |
| ----------------------------- | -------------------------------------- | ------------------------------------------- |
| **In-Domain (ID)**            | Dữ liệu cùng loại với training         | Train trên StyleGAN2 → Test trên StyleGAN2  |
| **Out-of-Distribution (OOD)** | Dữ liệu KHÁC loại, chưa thấy khi train | Train trên StyleGAN2 → Test trên **Gemini** |

**Tại sao OOD quan trọng?** Trong thế giới thực, kẻ xấu sẽ dùng công cụ mới nhất để tạo ảnh giả. Model của bạn không thể biết trước họ dùng công cụ gì. Nếu model chỉ đạt điểm cao trên ID mà thất bại hoàn toàn trên OOD → vô dụng ngoài thực tế.

> **KPI từ plan.md**: AUC ≥ 0.92 (ID) và AUC ≥ 0.85 (OOD)  
> **KPI điều chỉnh thực tế (PROJECT_PLAN.md)**: AUC ≥ 0.90 (ID) và AUC ≥ 0.75 (OOD)

### 🛠️ Hướng dẫn thực hiện

#### Bước 1: Hiểu chiến lược chia data

```
HolmHz Dataset Strategy:
═══════════════════════════════════════════════════════════════

   TRAIN (70%)              VAL (15%)           TEST OOD
   ──────────────           ─────────           ─────────────
   Real:                    Real:               Real:
   • FFHQ (10k)            • FFHQ (2k)         • Camera thật (500)

   Fake GAN:               Fake GAN:           Fake GAN (OOD):
   • StyleGAN2 (8k)        • ProGAN (2k)       • StarGAN
   • DFFD-GAN (7k)

   Fake Diffusion:         Fake Diffusion:     Fake Diffusion (OOD):
   • SD v1.5 (8k)          • SD v2.1 (2k)      • SDXL
   • GenImage (7k)                              • Gemini (200-500)
                                                • Flux (200-500)
```

**Giải thích logic chia**:

- **Train**: Model học từ đây. Cần ĐA DẠNG nhất có thể (nhiều nguồn GAN + Diffusion).
- **Validation (Val)**: Kiểm tra model có đang học tốt không TRONG quá trình train. Dùng nguồn TƯƠNG TỰ train nhưng KHÁC ảnh.
- **Test OOD**: Đánh giá cuối cùng trên nguồn model CHƯA HỀ THẤY. Đây là thước đo thực tế nhất.

> **Tại sao ProGAN nằm ở Val mà không ở Train?** Vì ProGAN và StyleGAN2 cùng "họ" (đều là GAN), nhưng khác nguồn. Nếu model train trên StyleGAN2 mà val tốt trên ProGAN → chứng tỏ nó đang học "phát hiện GAN nói chung" chứ không chỉ "nhớ mặt StyleGAN2".

#### Bước 2: Download ảnh Real (FFHQ)

**FFHQ** (Flickr-Faces-HQ) là bộ dataset chuẩn 70K ảnh khuôn mặt thật từ Flickr. Toàn bộ cộng đồng AI dùng nó, nên kết quả của bạn sẽ tương đương với paper quốc tế.

```bash
# Cách 1: Download từ Kaggle (nhanh hơn, ~10GB cho 70k ảnh)
# 1. Tạo tài khoản Kaggle → Settings → API → Create New Token
# 2. File kaggle.json sẽ download
# 3. Copy vào C:\Users\<bạn>\.kaggle\kaggle.json

pip install kaggle
kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq
# Giải nén vào data/raw/real/ffhq/

# Cách 2: Download trực tiếp từ Google Drive (link trong FFHQ repo)
# https://github.com/NVlabs/ffhq-dataset
# Chỉ cần download thumbs128 hoặc thumbs256 (nhẹ hơn nhiều)
```

> **Tại sao chọn FFHQ?** Vì đây là dataset chuẩn mà mọi paper deepfake detection đều dùng. Dùng cùng dataset = dễ so sánh kết quả.

> **Chỉ cần 10-12K ảnh từ FFHQ** (không cần hết 70K). Chọn ngẫu nhiên bằng script:

```python
# scripts/subset_ffhq.py — Chọn random 12K ảnh từ FFHQ
import shutil
import random
from pathlib import Path

src = Path("data/raw/real/ffhq_full")  # Folder chứa toàn bộ FFHQ
dst = Path("data/raw/real/ffhq")
dst.mkdir(parents=True, exist_ok=True)

all_images = list(src.glob("*.png")) + list(src.glob("*.jpg"))
selected = random.sample(all_images, min(12000, len(all_images)))

for img_path in selected:
    shutil.copy2(img_path, dst / img_path.name)

print(f"Copied {len(selected)} images to {dst}")
```

#### Bước 3: Download ảnh Diffusion (GenImage) ⭐ Quan trọng nhất

**GenImage** là bộ dataset chứa ảnh sinh từ nhiều model Diffusion (Stable Diffusion, Midjourney, DALL-E...). Đây là nguồn **quyết định thành bại** của HolmHz.

```bash
# GenImage dataset: https://github.com/GenImage-Dataset/GenImage
# Tổng ~50GB. Bạn CHỈ CẦN download subset:
#   - imagenet_sdv14 (Stable Diffusion v1.4)
#   - imagenet_sdv15 (Stable Diffusion v1.5)

# Download từ Google Drive links trong repo GenImage README
# Giải nén vào data/raw/fake_diffusion/genimage/
```

> **Nếu GenImage quá lớn hoặc khó download**, có phương án B: Tự generate bằng Stable Diffusion:

```python
# Dùng Hugging Face pipeline để generate ảnh Diffusion
# (Chạy trên Colab vì cần GPU)
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

prompts = [
    "a portrait photo of a young woman, realistic",
    "a headshot of a middle-aged man, studio lighting",
    # ... thêm nhiều prompt
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(f"data/raw/fake_diffusion/sd15/sd15_{i:05d}.png")
```

#### Bước 4: Download ảnh GAN (StyleGAN2)

```bash
# Cách 1: Download ảnh GAN có sẵn
# iFakeFaceDB: https://github.com/socialabubi/iFakeFaceDB (87K ảnh StyleGAN)
# generated.photos: https://generated.photos/datasets (100K ảnh)

# Cách 2: Dùng thispersondoesnotexist.com
# Mỗi lần refresh = 1 ảnh StyleGAN3 mới
# Script tự động download (cần cẩn thận rate limit):
```

```python
# scripts/download_gan_faces.py
import requests
from pathlib import Path
import time

dst = Path("data/raw/fake_gan/stylegan")
dst.mkdir(parents=True, exist_ok=True)

for i in range(5000):
    try:
        resp = requests.get("https://thispersondoesnotexist.com", timeout=10)
        if resp.status_code == 200:
            (dst / f"stylegan_{i:05d}.jpg").write_bytes(resp.content)
            if i % 100 == 0:
                print(f"Downloaded {i}/5000")
        time.sleep(1)  # Tránh bị block
    except Exception as e:
        print(f"Error {i}: {e}")
        time.sleep(5)
```

#### Bước 5: Chuẩn bị ảnh OOD Test

Đây là ảnh từ nguồn model chưa hề xuất hiện trong train/val. Bạn đã có sẵn một số ảnh Gemini trong folder `imgs/`.

```bash
# Copy ảnh Gemini đã có
cp -r imgs/Fake_AI_generated/* data/raw/ood_test/gemini/
cp -r imgs/Real/* data/raw/ood_test/real_camera/

# Generate thêm từ Gemini (https://gemini.google.com)
# Generate từ Flux (https://replicate.com/black-forest-labs/flux-schnell)
# Mỗi nguồn cần 200-500 ảnh
```

#### Bước 6: Resize và tổ chức folder cuối cùng

Tất cả ảnh phải cùng kích thước (224×224 cho EfficientNet-B0):

```python
# preprocessing/resize_all.py
from PIL import Image
from pathlib import Path
from tqdm import tqdm

TARGET_SIZE = (224, 224)

def resize_folder(src_dir: str, dst_dir: str):
    """Resize tất cả ảnh trong folder về TARGET_SIZE."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    images = list(src.glob("*.jpg")) + list(src.glob("*.png")) + list(src.glob("*.jpeg"))

    for img_path in tqdm(images, desc=f"Resizing {src.name}"):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)  # LANCZOS = chất lượng cao nhất
            img.save(dst / f"{img_path.stem}.png")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Resize từng folder
resize_folder("data/raw/real/ffhq", "data/processed/real/ffhq")
resize_folder("data/raw/fake_gan/stylegan", "data/processed/fake_gan/stylegan")
resize_folder("data/raw/fake_diffusion/genimage", "data/processed/fake_diffusion/genimage")
# ... tương tự cho các folder khác
```

> **Tại sao resize về 224×224?**
>
> - EfficientNet-B0 được thiết kế tối ưu cho input 224×224.
> - Ảnh gốc có thể là 1024×1024 (FFHQ) hoặc 512×512 (GenImage). Nếu không resize, mỗi ảnh chiếm gấp 20 lần bộ nhớ → GPU hết RAM.
> - **Quan trọng**: Resize về cùng size = mọi ảnh "ngang hàng", model không bị thiên vị bởi kích thước.

#### Bước 7: Tạo file thống kê

```python
# scripts/dataset_stats.py
import json
from pathlib import Path

stats = {
    "real": {
        "ffhq_train": len(list(Path("data/processed/real/ffhq").glob("*"))),
    },
    "fake_gan": {
        "stylegan": len(list(Path("data/processed/fake_gan/stylegan").glob("*"))),
    },
    "fake_diffusion": {
        "genimage": len(list(Path("data/processed/fake_diffusion/genimage").glob("*"))),
    },
    "ood_test": {
        "gemini": len(list(Path("data/raw/ood_test/gemini").glob("*"))),
        "flux": len(list(Path("data/raw/ood_test/flux").glob("*"))),
    }
}

total = sum(v for group in stats.values() for v in group.values())
stats["total"] = total

with open("data/manifests/dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(json.dumps(stats, indent=2))
print(f"\nTotal images: {total}")
```

### ✅ Checklist hoàn thành Task 1.2

- [ ] ≥10K ảnh Real (FFHQ) trong `data/processed/`
- [ ] ≥10K ảnh Diffusion (GenImage / SD) trong `data/processed/`
- [ ] ≥5K ảnh GAN (StyleGAN) trong `data/processed/`
- [ ] ≥200 ảnh OOD (Gemini) trong `data/raw/ood_test/`
- [ ] ≥200 ảnh OOD (Flux) trong `data/raw/ood_test/`
- [ ] File `data/manifests/dataset_stats.json` tồn tại
- [ ] Tất cả ảnh processed đã resize về 224×224

---

## TASK 1.3: Data Pipeline

### 🎯 Mục tiêu

Viết code Python để: đọc ảnh từ disk → xử lý (augment, normalize) → đóng gói thành batch → đưa vào model.

### 🧠 Tại sao cần Data Pipeline riêng?

Hãy tưởng tượng bạn có 25,000 tấm ảnh trong folder. Model AI không thể "ăn" cả 25K ảnh cùng lúc (GPU sẽ hết RAM). Cũng không thể "ăn" từng ảnh một (quá chậm). Giải pháp:

```
25,000 ảnh trên disk
       │
       ▼
 ┌─────────────┐
 │  Dataset     │ ← Biết cách tìm & đọc từng ảnh
 │  (Thực đơn)  │
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  Transform   │ ← Augment (xoay, lật, thêm nhiễu) + Normalize
 │  (Chế biến)  │
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  DataLoader  │ ← Gom thành batch (32 ảnh/lần), tải song song
 │  (Bồi bàn)   │
 └──────┬──────┘
        │
        ▼
    Model AI
    (Thực khách)
```

**Tương tự nhà hàng:**

- **Dataset** = thực đơn (biết có món gì, ở đâu)
- **Transform** = bếp (chế biến từ nguyên liệu thô thành món ăn chuẩn)
- **DataLoader** = bồi bàn (mang đúng 32 món/lần, nhanh, không để khách chờ)

### 📚 Kiến thức nền: Tensor là gì?

Khi bạn nhìn ảnh, bạn thấy màu sắc. Khi máy tính nhìn ảnh, nó thấy **số**:

```
Ảnh 224×224 pixel, 3 kênh màu (Red, Green, Blue)
= mảng 3 chiều: [3, 224, 224]
= 3 × 224 × 224 = 150,528 con số

Batch 32 ảnh:
= mảng 4 chiều: [32, 3, 224, 224]
= "32 tấm ảnh, mỗi tấm có 3 kênh, mỗi kênh 224×224 pixel"
```

**Tensor** chính là "mảng nhiều chiều" này — cấu trúc dữ liệu nền tảng của mọi framework DL (PyTorch, TensorFlow...).

### 📚 Kiến thức nền: Data Augmentation — Tại sao cần "phá" ảnh?

Giả sử bạn có 10K ảnh thật. Model nhìn đi nhìn lại 10K ảnh này 30 lần (30 epochs) → **Học thuộc** (overfitting). Nó nhớ "ảnh số 1234 là thật" thay vì "ảnh có đặc điểm X nên là thật".

**Giải pháp: Data Augmentation** — mỗi lần model nhìn lại 1 ảnh, ảnh đó bị biến đổi nhẹ:

```
Ảnh gốc → Epoch 1: xoay 5°
         → Epoch 2: lật ngang
         → Epoch 3: thêm JPEG compression
         → Epoch 4: crop ngẫu nhiên
         → ...
```

Model không bao giờ thấy 2 bản giống hệt nhau → buộc phải học **đặc trưng tổng quát** thay vì **thuộc lòng**.

> **Bài học từ CNNDetection**: Paper nổi tiếng của Wang et al. cho thấy **JPEG compression augmentation + Gaussian blur** là 2 augmentation quan trọng nhất cho deepfake detection. Vì ảnh trên mạng xã hội thường bị nén JPEG, blur qua resize.

### 📚 Kiến thức nền: Normalization — Tại sao chia cho mấy số lẻ?

Nhớ lại khi chạy CNNDetection:

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Giá trị pixel gốc: `[0, 255]` (0=đen, 255=trắng).  
Chia cho 255 → `[0, 1]`.  
Trừ mean, chia std → **khoảng [-2.x, +2.x]** (phân phối chuẩn, trung bình = 0).

**Tại sao cần?**

- Network học nhanh hơn khi input có trung bình ~0 và độ lệch chuẩn ~1.
- Các số `[0.485, 0.456, 0.406]` là mean RGB tính trên toàn bộ **ImageNet** (1.2 triệu ảnh). Vì EfficientNet đã được train trên ImageNet, nên khi fine-tune, ảnh mới PHẢI được normalize giống cách nó đã quen.

> **Sai số normalization = model chạy sai kết quả.** Đây là lỗi phổ biến nhất khi copy code giữa các project (nhớ bài học từ UniversalFakeDetect: CLIP dùng số normalize KHÁC ImageNet).

### 🛠️ Hướng dẫn thực hiện

#### Bước 1: Tạo Manifest JSON (build_splits.py)

Manifest JSON = "danh sách" chứa đường dẫn + nhãn mỗi ảnh. Model không biết folder nào trong disk, nó chỉ đọc file JSON này.

```python
# preprocessing/build_splits.py
"""
Tạo manifest JSON cho train/val/test splits.

Manifest format:
[
    {"path": "data/processed/train/real/ffhq_00001.png", "label": 0, "source": "ffhq"},
    {"path": "data/processed/train/fake/genimage_00001.png", "label": 1, "source": "genimage_sd15"},
    ...
]

label: 0 = Real, 1 = Fake
source: tên nguồn gốc (để phân tích sau này)
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def collect_images(folder: Path, label: int, source: str) -> List[Dict]:
    """Thu thập tất cả ảnh trong folder, gắn label và source."""
    entries = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for img_path in folder.glob(ext):
            entries.append({
                "path": str(img_path.as_posix()),  # Dùng / thay \ cho cross-platform
                "label": label,
                "source": source,
            })
    return entries


def split_data(entries: List[Dict], train_ratio=0.7, val_ratio=0.15):
    """Chia data thành train/val/test theo tỷ lệ."""
    random.shuffle(entries)
    n = len(entries)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = entries[:n_train]
    val = entries[n_train:n_train + n_val]
    test = entries[n_train + n_val:]

    return train, val, test


def save_manifest(data: List[Dict], path: str):
    """Lưu manifest ra file JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} entries to {path}")


def main():
    random.seed(42)  # Seed cố định để kết quả reproducible

    # === Thu thập ảnh từ các folder ===
    all_real = []
    all_fake = []

    # Real images
    real_dirs = {
        "data/processed/real/ffhq": "ffhq",
        # Thêm nguồn real khác nếu có
    }
    for folder, source in real_dirs.items():
        p = Path(folder)
        if p.exists():
            entries = collect_images(p, label=0, source=source)
            all_real.extend(entries)
            print(f"Real [{source}]: {len(entries)} images")

    # Fake GAN images
    gan_dirs = {
        "data/processed/fake_gan/stylegan": "stylegan2",
        # Thêm nguồn GAN khác nếu có
    }
    for folder, source in gan_dirs.items():
        p = Path(folder)
        if p.exists():
            entries = collect_images(p, label=1, source=source)
            all_fake.extend(entries)
            print(f"Fake GAN [{source}]: {len(entries)} images")

    # Fake Diffusion images
    diff_dirs = {
        "data/processed/fake_diffusion/genimage": "genimage_sd15",
        "data/processed/fake_diffusion/sd15": "sd15",
        # Thêm nguồn Diffusion khác nếu có
    }
    for folder, source in diff_dirs.items():
        p = Path(folder)
        if p.exists():
            entries = collect_images(p, label=1, source=source)
            all_fake.extend(entries)
            print(f"Fake Diffusion [{source}]: {len(entries)} images")

    # === Chia train/val/test cho Real và Fake riêng (đảm bảo cân bằng) ===
    real_train, real_val, real_test = split_data(all_real)
    fake_train, fake_val, fake_test = split_data(all_fake)

    # Gộp lại
    train_data = real_train + fake_train
    val_data = real_val + fake_val
    test_data = real_test + fake_test

    # Shuffle lần cuối
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # === Lưu manifests ===
    save_manifest(train_data, "data/manifests/train.json")
    save_manifest(val_data, "data/manifests/val.json")
    save_manifest(test_data, "data/manifests/test.json")

    # === Thống kê ===
    print(f"\n{'='*50}")
    print(f"Train: {len(train_data)} ({sum(1 for x in train_data if x['label']==0)} real, {sum(1 for x in train_data if x['label']==1)} fake)")
    print(f"Val:   {len(val_data)} ({sum(1 for x in val_data if x['label']==0)} real, {sum(1 for x in val_data if x['label']==1)} fake)")
    print(f"Test:  {len(test_data)} ({sum(1 for x in test_data if x['label']==0)} real, {sum(1 for x in test_data if x['label']==1)} fake)")


if __name__ == "__main__":
    main()
```

> **Tại sao chia Real và Fake riêng rồi mới gộp?** Để đảm bảo **tỷ lệ 50/50 real:fake** trong cả train, val, và test. Nếu shuffle chung rồi chia, có thể train bị lệch (ví dụ: 80% fake, 20% real) → model "lười", luôn đoán fake cho dễ.

#### Bước 2: Implement transforms.py (Augmentation + Normalization)

```python
# src/holmhz/data/transforms.py
"""
Data transforms cho HolmHz.

Triết lý:
- Train: augment MẠNH (JPEG, blur, flip, color jitter) để chống overfitting
- Val/Test: KHÔNG augment, chỉ resize + normalize (đo đúng sức thật)

Pattern từ:
- CNNDetection: JPEG compression + Gaussian blur là augmentation QUAN TRỌNG NHẤT
- UniversalFakeDetect: Preprocessing PHẢI match backbone (ImageNet vs CLIP)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# === ImageNet Statistics ===
# Đây là "cách nhìn thế giới" của EfficientNet
# (Tính từ 1.2 triệu ảnh ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Kích thước chuẩn cho EfficientNet-B0
DEFAULT_IMAGE_SIZE = 224


def get_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    """
    Transforms cho TRAINING.

    Tại sao augment mạnh?
    → Mô phỏng điều kiện thực tế: ảnh trên mạng bị nén JPEG, resize,
      chụp lại màn hình, thay đổi ánh sáng...
    → Model phải chịu được tất cả biến dạng này.
    """
    return A.Compose([
        # 1. Resize về kích thước chuẩn
        A.Resize(image_size, image_size),

        # 2. Lật ngang ngẫu nhiên (50% chance)
        # Tại sao: khuôn mặt đối xứng, lật không thay đổi Real/Fake
        A.HorizontalFlip(p=0.5),

        # 3. JPEG Compression (30% chance)
        # ⭐ QUAN TRỌNG NHẤT cho deepfake detection
        # Tại sao: ảnh trên mạng luôn bị nén JPEG. Nếu không augment,
        # model học artifact JPEG thay vì artifact AI → fail khi gặp ảnh nén
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),

        # 4. Gaussian Blur (20% chance)
        # Tại sao: ảnh bị resize/share qua mạng xã hội thường bị blur
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),

        # 5. Thay đổi màu sắc nhẹ (30% chance)
        # Tại sao: ảnh thật chụp dưới nhiều điều kiện ánh sáng
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),

        # 6. Thêm nhiễu Gaussian nhẹ (10% chance)
        # Tại sao: camera giá rẻ thường có nhiễu
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.1),

        # 7. Normalize (BẮT BUỘC, luôn áp dụng)
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # 8. Chuyển sang PyTorch tensor
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    """
    Transforms cho VALIDATION và TEST.

    Tại sao KHÔNG augment?
    → Validation/test là "bài kiểm tra cuối kỳ".
    → Muốn đo đúng sức mạnh thật của model, không "giúp đỡ" bằng augmentation.
    → Chỉ resize + normalize (giống điều kiện inference khi deploy).
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
```

#### Bước 3: Implement ImageDataset (Dataset class)

```python
# src/holmhz/data/image_dataset.py
"""
Dataset class cho HolmHz.

Trong PyTorch, Dataset là "hợp đồng" (interface) định nghĩa:
  1. __len__(): Có bao nhiêu mẫu dữ liệu?
  2. __getitem__(index): Lấy mẫu thứ `index` ra.

DataLoader sẽ gọi 2 hàm này tự động:
  - Gọi __len__() để biết khi nào hết data (1 epoch)
  - Gọi __getitem__(0), __getitem__(1), ... để lấy từng mẫu
  - Tự động gom 32 mẫu thành 1 batch
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class ImageDataset(Dataset):
    """
    Dataset đọc ảnh từ manifest JSON file.

    Manifest format:
    [{"path": "...", "label": 0/1, "source": "ffhq"}, ...]

    Parameters:
        manifest_path: Đường dẫn tới file JSON manifest
        transform: Albumentations transform pipeline (augment + normalize)

    Pattern học từ:
        - CNNDetection: ImageFolder đơn giản (folder = label)
        - DeepfakeBench: Abstract Dataset + nhiều subclass
        - HolmHz: JSON manifest (linh hoạt hơn folder, đơn giản hơn Abstract)
    """

    def __init__(
        self,
        manifest_path: str,
        transform: Optional[A.Compose] = None,
    ):
        self.transform = transform

        # Load manifest
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.data: List[Dict] = json.load(f)

        # Thống kê nhanh
        self.num_real = sum(1 for item in self.data if item["label"] == 0)
        self.num_fake = sum(1 for item in self.data if item["label"] == 1)

    def __len__(self) -> int:
        """Trả về tổng số ảnh trong dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Trả về 1 mẫu dữ liệu: ảnh (tensor) + nhãn.

        Flow:
        1. Đọc path và label từ manifest
        2. Load ảnh bằng OpenCV (nhanh hơn PIL cho augmentation)
        3. Chuyển BGR → RGB (OpenCV mặc định đọc BGR)
        4. Áp dụng transforms (augment + normalize + to tensor)
        5. Trả về dict {"image": tensor, "label": tensor, "source": str}
        """
        item = self.data[index]

        # Load ảnh
        image = cv2.imread(item["path"])
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {item['path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR → RGB

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]  # Đã là tensor sau ToTensorV2()

        # Label
        label = torch.tensor(item["label"], dtype=torch.float32)

        return {
            "image": image,                # shape: [3, 224, 224]
            "label": label,                # scalar: 0.0 hoặc 1.0
            "source": item.get("source", "unknown"),
        }

    def __repr__(self) -> str:
        return (
            f"ImageDataset(total={len(self)}, "
            f"real={self.num_real}, fake={self.num_fake})"
        )
```

> **Tại sao dùng OpenCV thay vì PIL?**  
> Albumentations (thư viện augmentation) hoạt động với numpy array (OpenCV format). PIL thì phải convert qua lại. Dùng OpenCV từ đầu = nhanh hơn, ít bug hơn.

#### Bước 4: Tạo DataLoader và kiểm tra

```python
# Kiểm tra nhanh trong Python console hoặc notebook:
from holmhz.data.image_dataset import ImageDataset
from holmhz.data.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader

# Tạo dataset
train_ds = ImageDataset(
    manifest_path="data/manifests/train.json",
    transform=get_train_transforms(224),
)
print(train_ds)  # ImageDataset(total=17500, real=8400, fake=9100)

# Tạo DataLoader — "bồi bàn" tự động gom batch
train_loader = DataLoader(
    train_ds,
    batch_size=32,       # 32 ảnh/batch
    shuffle=True,        # Xáo trộn thứ tự (quan trọng cho training!)
    num_workers=4,       # 4 thread đọc data song song
    pin_memory=True,     # Tăng tốc chuyển data CPU→GPU
)

# Lấy 1 batch kiểm tra
batch = next(iter(train_loader))
print(f"Image batch shape: {batch['image'].shape}")   # [32, 3, 224, 224] ✅
print(f"Label batch shape: {batch['label'].shape}")    # [32] ✅
print(f"Image value range: [{batch['image'].min():.2f}, {batch['image'].max():.2f}]")
# Khoảng [-2.x, +2.x] sau normalize ✅
```

> **Tại sao `shuffle=True` cho train?** Nếu không shuffle, model luôn thấy 1000 ảnh real liên tiếp rồi 1000 ảnh fake liên tiếp → gradient bị lệch → học không ổn định. Shuffle = thứ tự ngẫu nhiên mỗi epoch.

> **Tại sao `num_workers=4`?** DataLoader dùng nhiều thread để đọc + augment ảnh từ disk SONG SONG với việc GPU đang train. GPU tính xong batch hiện tại → batch tiếp theo đã sẵn sàng. Không phải chờ đợi.

### ✅ Checklist hoàn thành Task 1.3

- [ ] `preprocessing/build_splits.py` chạy tạo ra 3 file JSON (train/val/test)
- [ ] `src/holmhz/data/transforms.py` có `get_train_transforms()` và `get_val_transforms()`
- [ ] `src/holmhz/data/image_dataset.py` có class `ImageDataset`
- [ ] DataLoader chạy được, 1 batch shape = `[32, 3, 224, 224]`
- [ ] Image value range sau normalize nằm trong khoảng `[-3, +3]`
- [ ] Viết unit test trong `tests/test_data.py` kiểm tra shape, dtype

---

## TASK 1.4: Model Architecture

### 🎯 Mục tiêu

Implement EfficientNet-B0 binary classifier: nhận ảnh 224×224 → trả về xác suất ảnh đó là Fake (0.0 → 1.0).

### 🧠 Tại sao chọn EfficientNet-B0? (Không phải ResNet, không phải CLIP)

Bạn đã chạy thử 3 models:

| Model                                   | Params   | Ưu điểm               | Nhược điểm                        |
| --------------------------------------- | -------- | --------------------- | --------------------------------- |
| **ResNet-50** (CNNDetection)            | 25M      | Đơn giản, nhanh       | Cũ, 50 layers không đủ sâu        |
| **CLIP ViT-L/14** (UniversalFakeDetect) | 427M     | Generalize tốt nhất   | **Quá nặng** cho web demo ≤2s/CPU |
| **EfficientNet-B4** (DeepfakeBench)     | 19M      | Mạnh, cân bằng        | Vẫn hơi lớn                       |
| **EfficientNet-B0** (HolmHz)            | **5.3M** | **Nhẹ nhất**, đủ mạnh | Cần data tốt để bù                |

**Lý do chọn EfficientNet-B0:**

1. **Nhẹ (5.3M params)**: Chạy được trên CPU laptop ≤ 2 giây — đúng yêu cầu KPI.
2. **Hiệu quả**: EfficientNet "scale" thông minh (đồng đều chiều rộng, chiều sâu, resolution) thay vì chỉ xếp thêm layers như ResNet.
3. **Pre-trained trên ImageNet**: Đã "thấy" 1.2 triệu ảnh tự nhiên → biết phân biệt cấu trúc ảnh thật.
4. **timm library hỗ trợ**: Một dòng code = load model + weights. Không cần code kiến trúc từ đầu.

> **Fallback plan**: Nếu EfficientNet-B0 không đạt OOD target (AUC < 0.70), sẽ thử CLIP ViT (đã kiểm chứng generalize tốt hơn). Nhưng bắt đầu với cái nhẹ nhất trước.

### 📚 Kiến thức nền: Transfer Learning — Tại sao không train từ đầu?

**Transfer Learning** = lấy kiến thức model đã học từ bài toán A (ImageNet classification) để áp dụng cho bài toán B (deepfake detection).

```
BÀI TOÁN A (ImageNet — Đã có lời giải):
   Input: 1.2 triệu ảnh    →  Model: EfficientNet  →  Output: 1000 loại
   (chó, mèo, xe, hoa...)      (đã train xong)          (phân loại)

   → Model đã HỌC ĐƯỢC cách nhìn: cạnh, góc, texture, hình dạng, khuôn mặt...
   → Kiến thức "nhìn" này là TỔNG QUÁT, dùng được cho nhiều bài toán khác.

BÀI TOÁN B (Deepfake Detection — Bài của chúng ta):
   Input: 25K ảnh          →  Model: EfficientNet   →  Output: Real/Fake
   (real/fake faces)            (lấy kiến thức A)         (2 loại)

   → KHÔNG cần train "cách nhìn" từ đầu (đã biết rồi)
   → CHỈ CẦN train "cách phán đoán" Real/Fake (lớp cuối)
   → Tiết kiệm 100x thời gian + data
```

**Tương tự đời thực:** Một bác sĩ chuyên khoa Mắt muốn chuyển sang khoa Da liễu. Họ KHÔNG cần học lại 6 năm đại học y (kiến thức nền tảng đã có). Chỉ cần học thêm 1-2 năm chuyên sâu về da liễu. Transfer Learning = "chuyển ngành" của model AI.

### 📚 Kiến thức nền: Freeze vs Unfreeze Backbone

```
EfficientNet-B0 (Pretrained on ImageNet):
┌──────────────────────────────────────────────────────────────┐
│  BACKBONE (Layers 1-16)                  │  HEAD (Lớp cuối) │
│  ────────────────────                    │  ───────────────  │
│  Conv → Conv → Conv → ... → Global Pool  │  Dropout → Linear│
│                                          │  (1280 → 1)      │
│  "MẮT" của model                         │  "NÃO" phán đoán │
│  Biết nhìn cạnh, góc, texture...         │  Quyết định R/F   │
│                                          │                   │
│  ⭐ Phase 1: ĐÓNG BĂNG (Freeze)          │  ⭐ Phase 1: TRAIN│
│     Không thay đổi kiến thức cũ          │     Học phán đoán │
│                                          │                   │
│  ⭐ Phase 2: MỞ KHÓA (Unfreeze)         │     Continues     │
│     Tinh chỉnh kiến thức cho deepfake    │     training      │
└──────────────────────────────────────────────────────────────┘
```

**Tại sao freeze trước, unfreeze sau?**

1. **Phase 1 (Freeze)**: Backbone đã biết nhìn ảnh tốt rồi (ImageNet). Nếu mở khóa ngay + learning rate cao → phá hỏng kiến thức cũ. Chỉ train HEAD (2 layers) = nhanh (3 phút/epoch), kiểm tra pipeline chạy đúng.

2. **Phase 2 (Unfreeze)**: Sau khi HEAD đã ổn định, mở khóa backbone + learning rate rất nhỏ. Model tinh chỉnh "cách nhìn" cho phù hợp deepfake detection (ví dụ: chú ý vùng mắt, tóc, texture da hơn).

### 🛠️ Hướng dẫn thực hiện

#### Bước 1: Implement BaseBackbone (Abstract class)

```python
# src/holmhz/backbones/base.py
"""
Base class cho Backbone.

Tại sao cần Abstract Base Class?
→ Định nghĩa "hợp đồng": mọi backbone (EfficientNet, ResNet, CLIP...)
  PHẢI có method extract_features() trả về vector features.
→ Khi đổi backbone, phần code khác KHÔNG cần sửa (chỉ đổi config).

Pattern từ DeepfakeBench: AbstractDetector định nghĩa interface chung.
"""

from abc import ABC, abstractmethod
import torch.nn as nn


class BaseBackbone(ABC, nn.Module):
    """Abstract base class cho tất cả backbones."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract_features(self, x):
        """
        Trích xuất features từ ảnh input.

        Args:
            x: Tensor [B, 3, H, W] — batch ảnh đã normalize
        Returns:
            features: Tensor [B, features_dim] — vector đặc trưng
        """
        pass

    @abstractmethod
    def get_features_dim(self) -> int:
        """Trả về kích thước vector features (ví dụ: 1280 cho EfficientNet-B0)."""
        pass

    def freeze(self):
        """Đóng băng tất cả parameters — không cho gradient chạy qua."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Mở khóa tất cả parameters — cho phép training."""
        for param in self.parameters():
            param.requires_grad = True
```

> **Tại sao dùng ABC (Abstract Base Class)?**  
> Giả sử mai sau bạn muốn thử CLIP backbone. Bạn chỉ cần tạo class `CLIPBackbone(BaseBackbone)` implement 2 method bắt buộc. Phần code Detector, Trainer... KHÔNG cần sửa gì — vì chúng chỉ gọi `backbone.extract_features()`, không quan tâm bên trong là EfficientNet hay CLIP. Đây gọi là **Open/Closed Principle** trong OOP.

#### Bước 2: Implement EfficientNetBackbone

```python
# src/holmhz/backbones/efficientnet.py
"""
EfficientNet-B0 Backbone sử dụng thư viện timm.

timm (PyTorch Image Models): thư viện chứa 700+ pre-trained models.
Thay vì tự code kiến trúc EfficientNet (rất phức tạp), ta import từ timm.
"""

import timm
import torch
import torch.nn as nn
from .base import BaseBackbone


class EfficientNetBackbone(BaseBackbone):
    """
    EfficientNet-B0 feature extractor.

    Kiến trúc:
        Input [B, 3, 224, 224]
        → EfficientNet Layers (pretrained)
        → Global Average Pooling
        → Output [B, 1280]  ← vector features

    Params: ~4M (backbone only, không tính head)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load EfficientNet-B0 từ timm
        # num_classes=0: bỏ lớp classification cuối cùng
        # → chỉ lấy phần feature extractor
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,  # True = load weights ImageNet
            num_classes=0,          # Bỏ lớp FC cuối (ta tự thêm sau)
        )

        self._features_dim = 1280  # EfficientNet-B0 output dimension

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất 1280-dim feature vector từ ảnh.

        Args:
            x: [B, 3, 224, 224] — batch ảnh đã normalize
        Returns:
            [B, 1280] — feature vector cho mỗi ảnh
        """
        return self.model(x)  # timm đã bao gồm Global Average Pooling

    def get_features_dim(self) -> int:
        return self._features_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass = extract_features (cho compatibility với nn.Module)."""
        return self.extract_features(x)
```

#### Bước 3: Implement BaseDetector + EfficientNetDetector

```python
# src/holmhz/detectors/base.py
"""
Base class cho Detector.

Detector = Backbone + Head (classification layers).
Backbone "nhìn" ảnh → trích xuất đặc trưng.
Head "phán đoán" → Real hay Fake.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseDetector(ABC, nn.Module):
    """Abstract base class cho tất cả detectors."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: ảnh → xác suất Fake.

        Args:
            x: [B, 3, H, W] — batch ảnh
        Returns:
            [B, 1] — P(Fake) cho mỗi ảnh (0.0 = Real, 1.0 = Fake)
        """
        pass

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Dự đoán nhãn (0 hoặc 1).

        threshold: ngưỡng phân loại (mặc định 0.5)
        P(Fake) > 0.5 → Fake (1)
        P(Fake) ≤ 0.5 → Real (0)
        """
        with torch.no_grad():
            probs = self.forward(x)
            return (probs > threshold).long()
```

```python
# src/holmhz/detectors/efficientnet_detector.py
"""
EfficientNet-B0 Deepfake Detector.

Kiến trúc tổng thể:
    Input [B, 3, 224, 224]
    → EfficientNet-B0 Backbone [B, 1280]   (Pretrained ImageNet)
    → Dropout(0.3)                          (Chống overfitting)
    → Linear(1280, 1)                       (1 output = P(Fake))
    → Sigmoid                               (Ép về [0, 1])
    → Output [B, 1]

Tại sao 1 output + Sigmoid thay vì 2 output + Softmax?
→ Binary classification (Real/Fake) chỉ cần 1 con số:
  - P(Fake) = 0.95 → 95% là Fake → P(Real) = 0.05
  - Đơn giản hơn, nhanh hơn, dùng BCELoss
→ CNNDetection và UniversalFakeDetect đều dùng pattern này
"""

import torch
import torch.nn as nn

from ..backbones.efficientnet import EfficientNetBackbone
from .base import BaseDetector


class EfficientNetDetector(BaseDetector):
    """
    Detector sử dụng EfficientNet-B0 backbone.

    Args:
        pretrained: Load pretrained ImageNet weights
        dropout: Tỷ lệ dropout (0.3 = tắt 30% neuron ngẫu nhiên khi train)
        freeze_backbone: Đóng băng backbone (Phase 1 transfer learning)
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Backbone: EfficientNet-B0
        self.backbone = EfficientNetBackbone(pretrained=pretrained)

        # Head: Dropout + Linear
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.get_features_dim(), 1),
            # Không có Sigmoid ở đây!
            # Lý do: BCEWithLogitsLoss tự tính Sigmoid bên trong
            # → Ổn định hơn về mặt số học (numerical stability)
        )

        # Freeze backbone nếu được yêu cầu
        if freeze_backbone:
            self.backbone.freeze()
            print("Backbone frozen: only training head layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, 3, 224, 224]
        Returns:
            logits: [B, 1] — raw scores (chưa qua Sigmoid)

        Lưu ý: Output là LOGITS, không phải probabilities.
        Để lấy P(Fake), cần: probs = torch.sigmoid(logits)
        Trong training, BCEWithLogitsLoss tự xử lý.
        """
        features = self.backbone.extract_features(x)  # [B, 1280]
        logits = self.head(features)                    # [B, 1]
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về probability (đã qua Sigmoid). Dùng khi inference."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def get_feature_layer(self) -> nn.Module:
        """
        Trả về layer cuối cùng của backbone — dùng cho Grad-CAM.
        Grad-CAM cần biết "nhìn vào layer nào" để tạo heatmap.
        """
        # EfficientNet-B0: layer cuối trước pooling
        return self.backbone.model.conv_head
```

> **Tại sao Dropout?**  
> Khi train, 30% neuron trong lớp head bị "tắt" ngẫu nhiên mỗi batch. Model không thể "phụ thuộc" vào bất kỳ neuron đơn lẻ nào → buộc phải phân tán kiến thức → chống overfitting. Khi inference, Dropout tự tắt (mọi neuron hoạt động).

#### Bước 4: Implement Registry Pattern

```python
# src/holmhz/utils/registry.py
"""
Registry Pattern — Factory cho Detectors.

Tại sao cần Registry?
→ Khi có nhiều detector (EfficientNet, CLIP, ResNet...), bạn KHÔNG muốn:
    if name == "efficientnet_b0":
        model = EfficientNetDetector(...)
    elif name == "clip_vit":
        model = CLIPDetector(...)
    elif ...

→ Registry cho phép:
    model = DETECTOR_REGISTRY.get("efficientnet_b0")(config)
    # Chỉ đổi string → đổi model. Code khác không sửa.

Pattern từ DeepfakeBench: @DETECTOR.register_module()
"""

from typing import Dict, Type, Any


class Registry:
    """
    Registry quản lý mapping: tên (str) → class.

    Usage:
        DETECTOR_REGISTRY = Registry("detector")

        @DETECTOR_REGISTRY.register("efficientnet_b0")
        class EfficientNetDetector(BaseDetector):
            ...

        # Sau đó:
        model = DETECTOR_REGISTRY.build("efficientnet_b0", pretrained=True)
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: str):
        """Decorator để đăng ký class vào registry."""
        def decorator(cls):
            if name in self._registry:
                raise ValueError(f"{name} already registered in {self.name}")
            self._registry[name] = cls
            return cls
        return decorator

    def build(self, name: str, **kwargs) -> Any:
        """Tạo instance từ tên đã đăng ký."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[name](**kwargs)

    def list(self):
        """Liệt kê tất cả tên đã đăng ký."""
        return list(self._registry.keys())


# === Global Registries ===
BACKBONE_REGISTRY = Registry("backbone")
DETECTOR_REGISTRY = Registry("detector")
```

#### Bước 5: Đăng ký và kiểm tra

```python
# Cập nhật src/holmhz/detectors/__init__.py
from ..utils.registry import DETECTOR_REGISTRY
from .efficientnet_detector import EfficientNetDetector

# Đăng ký detector
DETECTOR_REGISTRY.register("efficientnet_b0")(EfficientNetDetector)
```

Kiểm tra:

```python
# Test nhanh
import torch
from holmhz.detectors.efficientnet_detector import EfficientNetDetector

# Tạo model
model = EfficientNetDetector(pretrained=True, freeze_backbone=True)

# Input giả (batch 4 ảnh, 3 kênh, 224x224)
dummy_input = torch.randn(4, 3, 224, 224)

# Forward pass
logits = model(dummy_input)
print(f"Output shape: {logits.shape}")  # [4, 1] ✅

probs = torch.sigmoid(logits)
print(f"Probabilities: {probs.squeeze().tolist()}")  # [0.xx, 0.xx, 0.xx, 0.xx]

# Đếm parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")       # ~5.3M
print(f"Trainable params: {trainable_params:,}")  # ~1,281 (chỉ head khi freeze)
```

### ✅ Checklist hoàn thành Task 1.4

- [ ] `EfficientNetBackbone` extract features shape `[B, 1280]`
- [ ] `EfficientNetDetector` output shape `[B, 1]`
- [ ] Freeze backbone → trainable params ~1,281 (chỉ head)
- [ ] Unfreeze → trainable params ~5.3M (toàn bộ)
- [ ] Registry pattern hoạt động: `DETECTOR_REGISTRY.build("efficientnet_b0")`
- [ ] Unit tests trong `tests/test_backbones.py` và `tests/test_detectors.py` pass

---

## TASK 1.5: Training Pipeline

### 🎯 Mục tiêu

Viết code training hoàn chỉnh: train loop → validate → log metrics → early stopping → save checkpoint.

### 🧠 Tại sao Training Pipeline phức tạp hơn bạn nghĩ?

Nhiều người nghĩ training = gọi `model.fit(data)` (như scikit-learn). Nhưng trong Deep Learning, bạn phải code **từng bước** của quá trình học:

```
MỘT EPOCH (1 vòng lặp qua toàn bộ dữ liệu):
═══════════════════════════════════════════════

  Bắt đầu epoch 1
       │
       ▼
  ┌─── TRAIN PHASE ─────────────────────────────────────────────┐
  │  for each batch (32 ảnh):                                   │
  │    1. Forward: ảnh → model → logits (dự đoán)               │
  │    2. Loss: so sánh dự đoán vs nhãn thật → tính "sai bao nhiêu"│
  │    3. Backward: tính gradient (hướng cần điều chỉnh)         │
  │    4. Update: optimizer cập nhật weights (sửa sai)           │
  │    5. Log: ghi lại loss                                      │
  └──────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─── VALIDATION PHASE ────────────────────────────────────────┐
  │  for each batch (32 ảnh từ val set):                        │
  │    1. Forward: ảnh → model → logits                         │
  │    2. Loss: tính loss (KHÔNG backward, KHÔNG update)         │
  │    3. Metrics: tính AUC, accuracy                           │
  │  → Dùng để kiểm tra model có đang overfitting không          │
  └──────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─── QUYẾT ĐỊNH ──────────────────────────────────────────────┐
  │  Val AUC cải thiện? → Lưu checkpoint (best_model.pt)        │
  │  Val AUC không đổi 5 epochs? → DỪNG SỚM (Early Stopping)   │
  │  Chưa → Tiếp tục epoch 2, 3, ...                           │
  └──────────────────────────────────────────────────────────────┘
```

### 📚 Kiến thức nền: Loss Function — "Sai bao nhiêu"

**Loss function** (hàm mất mát) = thước đo "model sai bao nhiêu". Training = cố gắng **giảm loss xuống nhỏ nhất**.

Cho bài toán binary (Real/Fake), ta dùng **Binary Cross-Entropy (BCE)**:

```
Ví dụ: Ảnh thật (label = 0 = Real)
  - Model đoán: P(Fake) = 0.05 → Loss = -log(1 - 0.05) = 0.05  (thấp = TỐT)
  - Model đoán: P(Fake) = 0.95 → Loss = -log(1 - 0.95) = 3.00  (cao = TỆ)

Ví dụ: Ảnh fake (label = 1 = Fake)
  - Model đoán: P(Fake) = 0.95 → Loss = -log(0.95) = 0.05  (thấp = TỐT)
  - Model đoán: P(Fake) = 0.05 → Loss = -log(0.05) = 3.00  (cao = TỆ)
```

> **BCEWithLogitsLoss** = BCE + Sigmoid gộp lại. Thay vì model → Sigmoid → BCE Loss, ta chỉ cần model → BCEWithLogitsLoss. Ổn định hơn về mặt số học (tránh log(0) = -∞).

### 📚 Kiến thức nền: Optimizer — "Sửa sai thế nào"

**Optimizer** quyết định **cách model cập nhật weights** dựa trên gradient (hướng sai).

| Optimizer | Mô tả                                     | Dùng khi                |
| --------- | ----------------------------------------- | ----------------------- |
| **SGD**   | Đơn giản, đi theo hướng gradient          | Cần kiểm soát chặt      |
| **Adam**  | Tự điều chỉnh learning rate cho mỗi param | Default phổ biến        |
| **AdamW** | Adam + weight decay (regularization)      | **HolmHz dùng cái này** |

**AdamW** = "Adam thông minh" — tự tăng/giảm tốc độ học cho từng parameter, kèm regularization (weight decay) giúp model không "quá khớp" data.

### 📚 Kiến thức nền: Learning Rate Scheduler

**Learning Rate (LR)** = tốc độ model cập nhật weights. Tưởng tượng:

- LR quá cao: model nhảy lung tung, không hội tụ (như chạy quá nhanh, trượt mất đích)
- LR quá thấp: model học chậm, tốn thời gian (như đi bộ tới đích xa)
- **Mẹo**: Bắt đầu LR cao → giảm dần

**CosineAnnealingLR**: LR giảm theo đường cong cosine (nhanh đầu, chậm cuối):

```
LR  ▲
    │  ╲
    │    ╲
    │      ╲
    │        ╲___
    │            ╲_______________
    └───────────────────────────► Epoch
     1                         30
```

### 📚 Kiến thức nền: Early Stopping — Biết lúc nào nên dừng

Overfitting = model "thuộc bài" trên training data nhưng không generalize được:

```
Loss ▲
     │   Train loss ──────── xuống liên tục
     │            ╲
     │  Val loss ──╲────── đi xuống rồi ĐI LÊN ← Overfitting bắt đầu!
     │              ╲     ╱
     │               ╲   ╱
     │                ╲_╱  ← Điểm tốt nhất (nên dừng ở đây)
     │
     └─────────────────────────────────────► Epoch
                   ^
                   Early stopping trigger
```

**Early Stopping**: Nếu validation metric (AUC) không cải thiện sau `patience=5` epochs liên tiếp → DỪNG. Lưu lại checkpoint tại điểm tốt nhất.

### 🛠️ Hướng dẫn thực hiện

#### Bước 1: Implement Early Stopping

```python
# src/holmhz/training/early_stopping.py
"""
Early Stopping: dừng training khi model bắt đầu overfitting.

Tại sao cần?
→ Không có Early Stopping, model sẽ train cho đến hết 30 epochs.
→ Nếu epoch 15 đã là tốt nhất, 15 epochs còn lại chỉ làm model TỆ HƠN.
→ Early Stopping = "biết dừng đúng lúc".
"""

from pathlib import Path
import torch


class EarlyStopping:
    """
    Dừng training khi monitored metric không cải thiện.

    Args:
        patience: Số epochs chờ trước khi dừng
        mode: 'max' (AUC, accuracy) hoặc 'min' (loss)
        min_delta: Thay đổi tối thiểu để coi là "cải thiện"
        checkpoint_path: Đường dẫn lưu best model
    """

    def __init__(
        self,
        patience: int = 5,
        mode: str = "max",
        min_delta: float = 0.001,
        checkpoint_path: str = "outputs/checkpoints/best_model.pt",
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_score = None
        self.counter = 0        # Đếm số epochs không cải thiện
        self.should_stop = False

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Kiểm tra xem có nên dừng training không.

        Args:
            score: Metric giá trị hiện tại (ví dụ: val_auc = 0.87)
            model: Model hiện tại (để lưu checkpoint)

        Returns:
            True nếu nên dừng, False nếu tiếp tục.
        """
        if self.best_score is None:
            # Epoch đầu tiên → luôn là "tốt nhất"
            self.best_score = score
            self._save_checkpoint(model)
            return False

        # Kiểm tra cải thiện
        improved = False
        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        elif self.mode == "min":
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
            print(f"  ✅ New best {self.mode}: {score:.4f} — saved checkpoint")
        else:
            self.counter += 1
            print(f"  ⏳ No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  🛑 Early stopping triggered! Best: {self.best_score:.4f}")

        return self.should_stop

    def _save_checkpoint(self, model: torch.nn.Module):
        """Lưu model state dict."""
        torch.save(model.state_dict(), self.checkpoint_path)
```

#### Bước 2: Implement BCE Loss wrapper

```python
# src/holmhz/losses/bce.py
"""
Binary Cross-Entropy Loss cho deepfake detection.

Tại sao wrap thay vì dùng trực tiếp?
→ Để sau này dễ đổi sang Focal Loss (xử lý dữ liệu mất cân bằng)
   mà không sửa Trainer code.
"""

import torch
import torch.nn as nn


class BCEWithLogitsLossWrapper(nn.Module):
    """
    Wrapper cho BCEWithLogitsLoss.

    BCEWithLogitsLoss = Sigmoid + BCE gộp lại:
    - Nhận logits (raw output từ model)
    - Tự tính Sigmoid bên trong
    - Tính BCE loss

    → Ổn định hơn tính Sigmoid riêng rồi BCE riêng
      (tránh log(0) = -infinity khi sigmoid output = 0 hoặc 1)
    """

    def __init__(self, pos_weight: float = None):
        """
        Args:
            pos_weight: Trọng số cho class positive (Fake).
                       Nếu data bị lệch (nhiều Real hơn Fake),
                       set pos_weight > 1 để model chú ý Fake hơn.
                       None = cân bằng (50/50).
        """
        super().__init__()
        weight = torch.tensor([pos_weight]) if pos_weight else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Tính loss.

        Args:
            logits: [B, 1] — raw model output
            targets: [B] — ground truth labels (0.0 hoặc 1.0)
        Returns:
            scalar — loss value
        """
        # Đảm bảo targets cùng shape với logits
        targets = targets.view_as(logits)
        return self.criterion(logits, targets)
```

#### Bước 3: Implement Trainer (trung tâm toàn bộ pipeline)

```python
# src/holmhz/training/trainer.py
"""
Trainer: điều phối toàn bộ quá trình training.

Đây là file QUAN TRỌNG NHẤT trong training pipeline.
Mọi thứ nối lại ở đây: model + data + loss + optimizer + scheduler + logging.

Flow:
    trainer.train()
    → for epoch in epochs:
        → train_one_epoch()   (forward + backward + update)
        → validate()          (forward only, đo metrics)
        → early_stopping()    (save best, dừng nếu cần)
        → log_to_wandb()      (ghi metrics lên dashboard)
"""

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from sklearn.metrics import roc_auc_score, accuracy_score

from ..losses.bce import BCEWithLogitsLossWrapper
from .early_stopping import EarlyStopping


class Trainer:
    """
    Training orchestrator cho HolmHz.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # --- Loss Function ---
        self.criterion = BCEWithLogitsLossWrapper()

        # --- Optimizer ---
        # Chỉ optimize parameters có requires_grad=True
        # (khi freeze backbone, chỉ head params được optimize)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = AdamW(
            trainable_params,
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )

        # --- Scheduler ---
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["epochs"],  # Giảm LR trong toàn bộ epochs
        )

        # --- Early Stopping ---
        self.early_stopping = EarlyStopping(
            patience=config["training"]["early_stopping"]["patience"],
            mode="max",          # AUC càng cao càng tốt
            checkpoint_path=config.get(
                "checkpoint_path", "outputs/checkpoints/best_model.pt"
            ),
        )

        # --- W&B Logging ---
        if HAS_WANDB and config.get("wandb", {}).get("project"):
            wandb.init(
                project=config["wandb"]["project"],
                config=config,
            )
            self.use_wandb = True
        else:
            self.use_wandb = False

    def train(self):
        """
        Main training loop.

        Đây là hàm "điều phối" chính. Mỗi epoch gồm:
        1. Train trên toàn bộ training data
        2. Validate trên validation data
        3. Kiểm tra early stopping
        4. Log metrics
        """
        epochs = self.config["training"]["epochs"]

        print(f"\n{'='*60}")
        print(f"Starting training: {epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # === TRAIN PHASE ===
            train_metrics = self._train_one_epoch(epoch)

            # === VALIDATION PHASE ===
            val_metrics = self._validate(epoch)

            # === SCHEDULER STEP ===
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # === LOGGING ===
            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch}/{epochs} ({elapsed:.0f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "val/loss": val_metrics["loss"],
                    "val/auc": val_metrics["auc"],
                    "val/accuracy": val_metrics["accuracy"],
                    "learning_rate": current_lr,
                })

            # === EARLY STOPPING ===
            if self.early_stopping(val_metrics["auc"], self.model):
                print(f"\n🛑 Training stopped at epoch {epoch}")
                break

        # Thông báo kết thúc
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best Val AUC: {self.early_stopping.best_score:.4f}")
        print(f"Best checkpoint: {self.early_stopping.checkpoint_path}")
        print(f"{'='*60}")

        if self.use_wandb:
            wandb.finish()

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train model qua 1 epoch.

        Mỗi batch:
        1. Forward pass: ảnh → model → logits
        2. Tính loss: so sánh logits với label thật
        3. Backward pass: tính gradient (hướng sửa)
        4. Optimizer step: cập nhật weights
        5. Zero grad: reset gradient (không tích lũy)
        """
        self.model.train()  # Bật training mode (Dropout hoạt động, BatchNorm cập nhật)

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"  Train Epoch {epoch}", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)     # [B, 3, 224, 224]
            labels = batch["label"].to(self.device)     # [B]

            # 1. Forward
            logits = self.model(images)                  # [B, 1]

            # 2. Loss
            loss = self.criterion(logits, labels)

            # 3. Backward
            self.optimizer.zero_grad()  # Reset gradient từ batch trước
            loss.backward()             # Tính gradient cho mỗi parameter

            # 4. Update weights
            self.optimizer.step()       # weights = weights - lr * gradient

            # 5. Logging
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"loss": total_loss / num_batches}

    @torch.no_grad()  # Tắt gradient computation → nhanh hơn, tiết kiệm memory
    def _validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model trên validation set.

        Khác với train:
        - KHÔNG tính gradient (torch.no_grad)
        - KHÔNG cập nhật weights
        - CHỈ đo metrics (loss, AUC, accuracy)

        Tại sao NO_GRAD?
        → Tiết kiệm ~50% GPU memory (không lưu computation graph)
        → Nhanh hơn ~30%
        → Validation chỉ cần forward pass
        """
        self.model.eval()  # Tắt training mode (Dropout off, BatchNorm dùng running stats)

        total_loss = 0.0
        all_probs = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc=f"  Val Epoch {epoch}", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Thu thập predictions
            probs = torch.sigmoid(logits).cpu().squeeze()
            total_loss += loss.item()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().tolist())

        # Tính metrics
        avg_loss = total_loss / len(self.val_loader)

        # AUC: Area Under ROC Curve
        # Giá trị 0.5 = đoán mò, 1.0 = hoàn hảo
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5  # Nếu chỉ có 1 class trong batch

        # Accuracy
        preds = [1 if p > 0.5 else 0 for p in all_probs]
        accuracy = accuracy_score(all_labels, preds)

        return {
            "loss": avg_loss,
            "auc": auc,
            "accuracy": accuracy,
        }
```

#### Bước 4: Implement scripts/train.py (Entry point)

```python
# scripts/train.py
"""
CLI entry point cho training.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --device cpu
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

# Import từ holmhz package (nhờ pip install -e .)
from holmhz.detectors.efficientnet_detector import EfficientNetDetector
from holmhz.data.image_dataset import ImageDataset
from holmhz.data.transforms import get_train_transforms, get_val_transforms
from holmhz.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="HolmHz Training")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda or cpu (auto-detect if not specified)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # === DATA ===
    image_size = config["data"]["image_size"]

    train_dataset = ImageDataset(
        manifest_path=config["data"]["train_manifest"],
        transform=get_train_transforms(image_size),
    )
    val_dataset = ImageDataset(
        manifest_path=config["data"]["val_manifest"],
        transform=get_val_transforms(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,  # Val không cần shuffle
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=(device == "cuda"),
    )

    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} images, {len(val_loader)} batches")

    # === MODEL ===
    model = EfficientNetDetector(
        pretrained=config["model"]["pretrained"],
        dropout=config["model"].get("dropout", 0.3),
        freeze_backbone=config["model"].get("freeze_backbone", True),
    )

    # === TRAIN ===
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
```

### ✅ Checklist hoàn thành Task 1.5

- [ ] `EarlyStopping` hoạt động (test với score tăng/giảm giả)
- [ ] `BCEWithLogitsLossWrapper` tính loss đúng shape
- [ ] `Trainer.train()` chạy 1 epoch trên mini dataset (100 ảnh) không lỗi
- [ ] `scripts/train.py --config configs/train.yaml` chạy được
- [ ] W&B dashboard hiển thị metrics (loss, auc, accuracy, lr)
- [ ] Checkpoint được save vào `outputs/checkpoints/`

---

## TASK 1.6: Baseline Training

### 🎯 Mục tiêu

Train EfficientNet-B0 trên dataset đầy đủ (≥25K ảnh). Đạt **AUC ≥ 0.88 trên validation** (in-domain).

### 🧠 Tại sao có 2 Phase training?

Từ kiến thức Transfer Learning đã học ở Task 1.4:

```
PHASE 1: FREEZE BACKBONE + TRAIN HEAD
════════════════════════════════════════
Mục đích: Kiểm tra pipeline, stabilize head weights
Setting:
  - Backbone: FROZEN (không học)
  - Head: TRAINING (Dropout + Linear)
  - LR: 1e-3 (cao, vì chỉ train head)
  - Epochs: 10
  - Thời gian: ~30 phút (Colab T4)

Kỳ vọng: Val AUC ≥ 0.80
(Nếu < 0.70 → có bug trong data/pipeline, KHÔNG phải vấn đề model)


PHASE 2: UNFREEZE + FINE-TUNE TOÀN BỘ
════════════════════════════════════════
Mục đích: Model tinh chỉnh backbone cho deepfake detection
Setting:
  - Backbone: UNFROZEN (tất cả layers học)
  - Head: Tiếp tục training
  - LR: 1e-4 (thấp hơn 10x, tránh phá backbone)
  - Epochs: 20
  - Thời gian: ~5 giờ (Colab T4)

Kỳ vọng: Val AUC ≥ 0.88
(Nếu < 0.85 → cần thêm data hoặc điều chỉnh augmentation)
```

**Tại sao Phase 1 trước?**

1. **Debug nhanh**: Mất 30 phút thay vì 5 giờ. Nếu có bug (wrong labels, bad augmentation), phát hiện sớm.
2. **Stabilize head**: Head vừa khởi tạo random. Nếu unfreeze ngay + gradient từ head random lan về backbone → phá hỏng pretrained weights.
3. **Baseline**: AUC ~0.80 chỉ từ frozen backbone = chứng minh features ImageNet có ích cho deepfake detection.

### 📚 Kiến thức nền: Hyperparameter Tuning

Hyperparameters = "nút vặn" mà BẠN phải chọn (model không tự học):

| Hyperparameter    | Giá trị thử      | Ảnh hưởng                                        |
| ----------------- | ---------------- | ------------------------------------------------ |
| **Learning Rate** | 5e-4, 1e-4, 5e-5 | Quá cao → không hội tụ; quá thấp → học chậm      |
| **Batch Size**    | 16, 32           | Lớn → stable gradient nhưng tốn RAM; nhỏ → noisy |
| **Dropout**       | 0.2, 0.3, 0.5    | Thấp → overfitting; cao → underfitting           |
| **Weight Decay**  | 1e-4, 1e-5       | Regularization: cao → model đơn giản hơn         |

**Chiến lược**: Chạy grid search nhỏ (3 LR × 2 batch = 6 experiments). Dùng W&B so sánh.

### 🛠️ Hướng dẫn thực hiện

#### Bước 1: Phase 1 — Freeze Backbone

Tạo file config riêng cho Phase 1:

```yaml
# configs/train_phase1_freeze.yaml
model:
  name: efficientnet_b0
  pretrained: true
  num_classes: 1
  dropout: 0.3
  freeze_backbone: true # ← FREEZE

training:
  epochs: 10 # Ít epochs (kiểm tra nhanh)
  batch_size: 32
  learning_rate: 0.001 # LR cao (chỉ train 1281 params)
  optimizer: adamw
  weight_decay: 0.0001
  scheduler: cosine
  early_stopping:
    patience: 5
    monitor: val_auc

data:
  train_manifest: data/manifests/train.json
  val_manifest: data/manifests/val.json
  image_size: 224
  num_workers: 4

wandb:
  project: holmhz
```

Chạy:

```bash
python scripts/train.py --config configs/train_phase1_freeze.yaml
```

**Kết quả kỳ vọng:**

```
Epoch 10/10 | Train Loss: 0.35 | Val Loss: 0.40 | Val AUC: 0.82 | Val Acc: 0.78
Best Val AUC: 0.83
```

**Phân tích kết quả:**

- AUC ≥ 0.80 → ✅ Pipeline hoạt động đúng. Proceed to Phase 2.
- AUC 0.70-0.80 → ⚠️ Kiểm tra data balance (50/50 real:fake?).
- AUC < 0.70 → ❌ Có bug. Kiểm tra: labels có đúng không? Augmentation có quá mạnh? Normalize đúng mean/std?

#### Bước 2: Phase 2 — Unfreeze + Fine-tune

```yaml
# configs/train_phase2_finetune.yaml
model:
  name: efficientnet_b0
  pretrained: true
  num_classes: 1
  dropout: 0.3
  freeze_backbone: false # ← UNFREEZE

training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.0001 # LR thấp hơn 10x (bảo vệ backbone)
  optimizer: adamw
  weight_decay: 0.0001
  scheduler: cosine
  early_stopping:
    patience: 5
    monitor: val_auc

data:
  train_manifest: data/manifests/train.json
  val_manifest: data/manifests/val.json
  image_size: 224
  num_workers: 4

# Load checkpoint từ Phase 1
checkpoint: outputs/checkpoints/best_model.pt

wandb:
  project: holmhz
```

> **Lưu ý**: Cần thêm logic load checkpoint vào `scripts/train.py`:

```python
# Thêm vào scripts/train.py, sau khi tạo model:
if config.get("checkpoint"):
    state_dict = torch.load(config["checkpoint"], map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {config['checkpoint']}")
```

Chạy:

```bash
python scripts/train.py --config configs/train_phase2_finetune.yaml
```

**Kết quả kỳ vọng sau Phase 2:**

```
Epoch 15/20 | Train Loss: 0.12 | Val Loss: 0.18 | Val AUC: 0.91 | Val Acc: 0.87
🛑 Early stopping at epoch 15! Best Val AUC: 0.91
```

#### Bước 3: Hyperparameter Search (Nếu Phase 2 chưa đạt target)

Tạo nhiều config files, chạy lần lượt:

```bash
# Thử LR khác nhau
python scripts/train.py --config configs/experiments/lr_5e-4.yaml
python scripts/train.py --config configs/experiments/lr_1e-4.yaml
python scripts/train.py --config configs/experiments/lr_5e-5.yaml

# So sánh trên W&B dashboard
```

#### Bước 4: Smoke Test trên ảnh thực tế

Sau khi train xong, kiểm tra nhanh trên ảnh bạn đã có:

```python
# scripts/quick_test.py
"""Kiểm tra nhanh model trên vài ảnh thật."""
import torch
from PIL import Image
from holmhz.detectors.efficientnet_detector import EfficientNetDetector
from holmhz.data.transforms import get_val_transforms

# Load model
model = EfficientNetDetector(pretrained=False, freeze_backbone=False)
model.load_state_dict(torch.load("outputs/checkpoints/best_model.pt", map_location="cpu"))
model.eval()

# Transform
transform = get_val_transforms(224)

# Test ảnh
test_images = [
    ("imgs/Real/photo1.jpg", "Real Photo"),
    ("imgs/Fake_AI_generated/gemini1.png", "Gemini Generated"),
]

for path, name in test_images:
    img = Image.open(path).convert("RGB")
    import numpy as np
    img_np = np.array(img)
    transformed = transform(image=img_np)
    input_tensor = transformed["image"].unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()

    label = "FAKE" if prob > 0.5 else "REAL"
    print(f"{name}: {label} (P(Fake)={prob:.4f})")
```

### 📊 Cách đọc kết quả Training

Khi train xong, bạn sẽ thấy các metric trên W&B. Đây là cách đọc:

| Metric            | Ý nghĩa                             | Giá trị tốt              | Giá trị tệ                               |
| ----------------- | ----------------------------------- | ------------------------ | ---------------------------------------- |
| **Train Loss**    | Model sai bao nhiêu trên train data | Giảm dần → 0.x           | Tăng hoặc dao động mạnh                  |
| **Val Loss**      | Model sai bao nhiêu trên val data   | Giảm dần, gần train loss | Tăng khi train loss giảm (= OVERFITTING) |
| **Val AUC**       | Khả năng phân biệt Real/Fake        | ≥ 0.88                   | < 0.70                                   |
| **Val Accuracy**  | % dự đoán đúng                      | ≥ 85%                    | < 70%                                    |
| **Learning Rate** | Tốc độ học hiện tại                 | Giảm dần (cosine)        | Nhảy lung tung                           |

**Dấu hiệu Overfitting:**

- Train Loss giảm nhưng Val Loss TĂNG
- Train Accuracy 99% nhưng Val Accuracy 75%
- → Cần: tăng augmentation, tăng dropout, thêm data

**Dấu hiệu Underfitting:**

- Cả Train Loss và Val Loss đều cao
- AUC < 0.75 sau nhiều epochs
- → Cần: tăng LR, unfreeze sớm hơn, thêm epochs

### ✅ Checklist hoàn thành Task 1.6

- [ ] Phase 1 (Freeze): Val AUC ≥ 0.80
- [ ] Phase 2 (Fine-tune): Val AUC ≥ 0.88
- [ ] Best checkpoint saved (`.pt` file) có kích thước ~20MB
- [ ] W&B dashboard có training curves (loss, AUC)
- [ ] Smoke test trên 5 ảnh thật (Gemini, Flux, Real camera)
- [ ] Training time documented (ví dụ: "6 hours on Colab T4")
- [ ] Không overfitting (val loss ≤ 1.2× train loss)

---

## Tổng kết Sprint 1

### Sprint 1 xong rồi, bạn có gì?

```
✅ Task 1.1: Môi trường dev sẵn sàng
✅ Task 1.2: ≥25K ảnh (Real + GAN + Diffusion + OOD)
✅ Task 1.3: Data pipeline (Dataset + Augmentation + DataLoader)
✅ Task 1.4: EfficientNet-B0 Detector (5.3M params)
✅ Task 1.5: Training pipeline (Trainer + Loss + Scheduler + Early Stopping)
✅ Task 1.6: Baseline model đạt AUC ≥ 0.88 (in-domain)

📦 Deliverables:
├── outputs/checkpoints/best_model.pt    ← Model đã train
├── data/manifests/{train,val,test}.json  ← Dataset manifests
├── W&B dashboard                         ← Training metrics
└── Smoke test results                    ← Quick sanity check
```

### Milestone M1: ✅ hoặc ❌?

| Tiêu chí       | Target         | Kết quả | Pass? |
| -------------- | -------------- | ------- | ----- |
| Dataset        | ≥25K ảnh       | ?       |       |
| Val AUC (ID)   | ≥ 0.88         | ?       |       |
| Checkpoint     | Saved          | ?       |       |
| Training curve | No overfitting | ?       |       |

### Sprint 2 Preview: Những gì tiếp theo

```
Task 2.1: Evaluation Pipeline — Đánh giá model trên test set + OOD
Task 2.2: Benchmark SOTA — So sánh với CNNDetection, UniversalFakeDetect, DeepfakeBench
Task 2.3: Grad-CAM XAI — Tạo heatmap giải thích "model nhìn vào đâu"
Task 2.4: Model Export — Xuất ONNX cho web demo
```

> 🎓 **Nhớ**: Mục tiêu của nghiên cứu này KHÔNG phải tạo model tốt nhất thế giới. Mục tiêu là **hiểu, triển khai, đánh giá, và giải thích** — đúng tinh thần nghiên cứu ứng dụng.

---

_Tài liệu này thuộc dự án HolmHz — Synthetic Image Detection System_  
_Trường Đại học Thủ Dầu Một | Viện Công nghệ số | 2025-2026_
