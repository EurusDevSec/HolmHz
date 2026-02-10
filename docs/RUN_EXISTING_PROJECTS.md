# 🚀 Chạy thử các dự án Deepfake Detection có sẵn

> **Dành cho**: DevOps Engineer muốn hiểu AI qua thực hành
> **Mục tiêu**: Chạy được các dự án tương tự, thấy kết quả trước khi tự build
> **Thời gian**: 2-3 buổi (mỗi buổi 2-3 tiếng)

---

## 📋 Mục lục

1. [Tại sao cần làm bước này?](#1-tại-sao-cần-làm-bước-này)
2. [Chuẩn bị chung](#2-chuẩn-bị-chung)
3. [Dự án 1: CNNDetection (Dễ nhất)](#3-dự-án-1-cnndetection-dễ-nhất)
4. [Dự án 2: UniversalFakeDetect](#4-dự-án-2-universalfakedetect)
5. [Dự án 3: DeepfakeBench](#5-dự-án-3-deepfakebench)
6. [Demo Online (Không cần cài đặt)](#6-demo-online-không-cần-cài-đặt)
7. [Rút ra bài học cho HolmHz](#7-rút-ra-bài-học-cho-holmhz)

---

## 1. Tại sao cần làm bước này?

### Góc nhìn DevOps

```
┌─────────────────────────────────────────────────────────────────┐
│  CÁCH HỌC AI CỦA DEVOPS vs AI ENGINEER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  AI Engineer:                                                   │
│  Lý thuyết → Toán → Code từ đầu → Train → Deploy               │
│  (Mất 6 tháng - 1 năm)                                         │
│                                                                 │
│  DevOps Engineer:                                               │
│  Chạy thử → Thấy kết quả → Hiểu flow → Tùy chỉnh → Deploy      │
│  (Mất 2-4 tuần)                                                 │
│                                                                 │
│  → Bạn đang đi theo hướng DevOps, và đó là ĐÚNG!               │
└─────────────────────────────────────────────────────────────────┘
```

### Lợi ích của việc chạy thử trước

| Lợi ích                    | Mô tả                                         |
| -------------------------- | --------------------------------------------- |
| **Có động lực**            | Thấy kết quả "Ảnh này là FAKE 95%" rất thú vị |
| **Hiểu input/output**      | Biết model cần gì, trả về gì                  |
| **Học cấu trúc dự án**     | Xem cách người ta tổ chức code                |
| **Có baseline để so sánh** | Biết dự án mình cần đạt mức nào               |
| **Tự tin hơn**             | "À, thì ra nó chỉ như vậy thôi"               |

---

## 2. Chuẩn bị chung

### 2.1. Tạo thư mục làm việc

```bash
# Tạo folder chứa các dự án thử nghiệm
mkdir -p ~/ai-experiments/deepfake-detection
cd ~/ai-experiments/deepfake-detection
```

### 2.2. Chuẩn bị ảnh test

Tải về 5-10 ảnh để test:

**Ảnh thật (Real)**:

- Ảnh selfie của bạn
- Ảnh từ Google tìm "portrait photography"
- Ảnh từ Unsplash: https://unsplash.com/s/photos/portrait

**Ảnh giả (Fake)**:

- https://thispersondoesnotexist.com (Mỗi lần refresh = 1 ảnh mới)
- https://generated.photos/faces (Ảnh AI-generated miễn phí)

```bash
# Tạo folder chứa ảnh test
mkdir -p test_images/real test_images/fake

# Download ảnh fake từ thispersondoesnotexist
# (Mở link trong browser, chuột phải > Save Image)
```

### 2.3. Cài đặt môi trường cơ bản

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt các thư viện cơ bản
pip install torch torchvision
pip install pillow numpy
pip install gradio  # Để chạy demo UI
```

---

## 3. Dự án 1: CNNDetection (Dễ nhất)

> **Tác giả**: Wang et al. (2020) - Bài báo gốc về deepfake detection
> **Độ khó**: ⭐ Dễ
> **Thời gian**: 30 phút

### 3.1. Clone và cài đặt

```bash
# Clone repo
git clone https://github.com/PeterWang512/CNNDetection.git
cd CNNDetection

# Cài đặt dependencies
pip install -r requirements.txt
```

### 3.2. Tải model đã train sẵn

```bash
# Tải pretrained weights
# Link trong README của repo, hoặc:
mkdir weights
# Tải file từ Google Drive link trong repo
```

### 3.3. Chạy demo

**Cách 1: Chạy trên 1 ảnh**

```bash
# Lưu ý:
# - Dùng flag -f thay vì --img_path
# - Thêm --use_cpu nếu không có GPU CUDA
python demo.py -f examples/fake.png -m weights/blur_jpg_prob0.5.pth --use_cpu
```

**Cách 2: Chạy Gradio demo (có giao diện)**

```bash
python demo_gradio.py
# Mở browser: http://localhost:7860
```

### 3.4. Kết quả mong đợi

```
Input: fake1.jpg
Output:
  - Prediction: FAKE
  - Confidence: 0.94 (94%)
```

### 3.5. Ghi chú cho HolmHz

````markdown
## Học được gì từ CNNDetection (Đã chạy thành công):

- [x] **Cấu trúc folder**: Đơn giản, gồm `networks/` (chứa ResNet50), `weights/` (chứa .pth), và `demo.py` để chạy chính.
- [x] **Cách load pretrained model**:
  ```python
  model = resnet50(num_classes=1)
  state_dict = torch.load(path, map_location='cpu') # Quan trọng: map_location='cpu'
  model.load_state_dict(state_dict['model'])
  ```
- [x] **Cách preprocess ảnh**: Sử dụng ImageNet normalization:
  ```python
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ```
- [x] **Inference**: Output của model đi qua hàm Sigmoid để ra xác suất (0-1).

Kết quả chạy với các ảnh GAN, STYLE GAN cho kết quả chính xác khoảng 94.62% và dự đoán ảnh real cũng chính xác 0.05 % nhưng thất bại hoàn toàn trước ảnh diffusion với ảnh do gemini (nano banana tạo ra) lại cho kết quả 0.06%
````

---

## 4. Dự án 2: UniversalFakeDetect

> **Tác giả**: Yuheng Li et al. (2023) - SOTA cho cross-dataset
> **Độ khó**: ⭐⭐ Trung bình
> **Thời gian**: 1 tiếng

### 4.1. Clone và cài đặt

```bash
cd ~/ai-experiments/deepfake-detection

# Clone repo
git clone https://github.com/Yuheng-Li/UniversalFakeDetect.git
cd UniversalFakeDetect

# Cài đặt dependencies (Lưu ý: repo không có requirements.txt)
# Cần cài CLIP từ source và các thư viện hỗ trợ
pip install setuptools wheel
pip install torch torchvision ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### 4.2. Tải model

Repo này đã **có sẵn** weight cho classifier trong folder `pretrained_weights/`, bạn không cần tải thêm gì cả.

- Path: `pretrained_weights/fc_weights.pth` (Linear layer weights)
- CLIP Backbone: Tự động tải từ OpenAI khi chạy code lần đầu (khoảng 900MB).

### 4.3. Chạy inference

Tạo file script để test trên 1 ảnh đơn lẻ (vì code gốc `validate.py` thiết kế để chạy cả folder dataset).

```python
# Tạo file test_universal.py
import torch
from PIL import Image
from models import get_model
import os

# 1. Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Load Model (CLIP Backbone + Custom FC Layer)
print("Loading model...")
# Model name phải đúng format "CLIP:..." để kích hoạt CLIP backbone
model = get_model("CLIP:ViT-L/14")
ckpt_path = 'pretrained_weights/fc_weights.pth'

# Load trained weights cho lớp FC
state_dict = torch.load(ckpt_path, map_location='cpu')
model.fc.load_state_dict(state_dict)
model.to(device)
model.eval()

# 3. Load & Preprocess Image
# CLIP đi kèm hàm preprocess riêng (resize, crop, normalize chuẩn CLIP)
transform = model.preprocess

# Dùng ảnh ví dụ có sẵn từ CNNDetection folder bên cạnh (nếu chưa có ảnh riêng)
img_path = '../CNNDetection/examples/fake.png'
# Hoặc: img_path = 'test_images/fake/fake1.jpg'

if not os.path.exists(img_path):
    print(f"Không tìm thấy ảnh tại {img_path}")
    exit()

img = Image.open(img_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)

# 4. Predict
with torch.no_grad():
    output = model(input_tensor)
    prob = output.sigmoid().item()

print(f"Testing on: {img_path}")
print(f"Prediction: {'FAKE' if prob > 0.5 else 'REAL'}")
print(f"Fake Probability: {prob*100:.2f}%")
```

**Kết quả chạy thực tế (Log)**:

```text
Using device: cuda
Loading model...
Testing on: ../CNNDetection/examples/fake.png
Prediction: FAKE
Fake Probability: 100.00%
```

### 4.4. Điểm đặc biệt của dự án này

```
┌─────────────────────────────────────────────────────────────────┐
│  TẠI SAO UNIVERSALFAKEDETECT ĐÁNG HỌC?                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Dùng CLIP (model của OpenAI) làm backbone                  │
│     → Generalize tốt hơn EfficientNet thông thường             │
│                                                                 │
│  2. Không cần train lại nhiều                                  │
│     → Chỉ fine-tune phần classifier                            │
│                                                                 │
│  3. AUC cao trên cross-dataset (OOD)                           │
│     → Đây là điểm HolmHz cần học hỏi                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Dự án 3: DeepfakeBench

> **Tác giả**: Nhiều tác giả - Benchmark 15+ methods
> **Độ khó**: ⭐⭐⭐ Khó hơn (nhưng đầy đủ nhất)
> **Thời gian**: 2-3 tiếng

### 5.1. Clone và cài đặt

```bash
cd ~/ai-experiments/deepfake-detection

# Clone repo
git clone https://github.com/SCLBD/DeepfakeBench.git
cd DeepfakeBench

# Cài đặt dependencies
pip install -r requirements.txt
```

### 5.2. Chạy demo nhanh

```bash
# Chạy với config có sẵn
python demo.py \
    --detector efficientnet \
    --image_path test_images/fake/fake1.jpg
```

### 5.3. So sánh nhiều methods

```bash
# Chạy benchmark với nhiều detector
python benchmark.py \
    --detectors efficientnet,resnet50,xception \
    --test_dir test_images/
```

### 5.4. Kết quả mong đợi

```
┌────────────────┬──────────┬──────────┬──────────┐
│ Detector       │ Accuracy │ AUC      │ Time(ms) │
├────────────────┼──────────┼──────────┼──────────┤
│ EfficientNet   │ 92.3%    │ 0.95     │ 45       │
│ ResNet50       │ 89.1%    │ 0.92     │ 38       │
│ Xception       │ 91.5%    │ 0.94     │ 52       │
└────────────────┴──────────┴──────────┴──────────┘
```

### 5.5. Ghi chú cho HolmHz

```markdown
## Học được gì từ DeepfakeBench:

- [x] Cách tổ chức benchmark nhiều models
- [x] Cách tính metrics (AUC, Accuracy, F1)
- [x] Config system (YAML files)
- [x] Cấu trúc code chuyên nghiệp
```

---

## 6. Demo Online (Không cần cài đặt)

Nếu lười cài đặt, hãy thử các demo online:

### 6.1. Hugging Face Spaces

| Demo                   | Link                                                      | Mô tả                          |
| ---------------------- | --------------------------------------------------------- | ------------------------------ |
| **Deepfake Detection** | https://huggingface.co/spaces/keras-io/deepfake-detection | Upload ảnh, xem kết quả        |
| **AI Image Detector**  | https://huggingface.co/spaces/umm-maybe/AI-image-detector | Detect ảnh AI-generated        |
| **SDXL Detector**      | Tìm trên HuggingFace                                      | Chuyên detect Stable Diffusion |

### 6.2. Cách sử dụng

1. Mở link trong browser
2. Upload ảnh từ `thispersondoesnotexist.com`
3. Xem kết quả: Real/Fake + Confidence
4. Thử với ảnh thật của bạn

### 6.3. Ghi lại kết quả

```markdown
## Test log - Hugging Face Demo

| Ảnh       | Nguồn                  | Kết quả | Confidence |
| --------- | ---------------------- | ------- | ---------- |
| fake1.jpg | thispersondoesnotexist | FAKE    | 97%        |
| fake2.jpg | generated.photos       | FAKE    | 89%        |
| real1.jpg | Ảnh selfie             | REAL    | 82%        |
| real2.jpg | Unsplash               | REAL    | 91%        |

## Nhận xét:

- Model detect tốt ảnh từ thispersondoesnotexist
- Ảnh thật có confidence thấp hơn (có thể vì chất lượng)
```

---

## 7. Rút ra bài học cho HolmHz

### 7.1. Checklist sau khi chạy thử

```markdown
## Sau khi chạy các dự án trên, tôi đã hiểu:

### Input/Output

- [ ] Model nhận ảnh kích thước bao nhiêu? (224x224? 256x256?)
- [ ] Model trả về gì? (Probability? Class?)
- [ ] Cần preprocess ảnh như thế nào?

### Cấu trúc code

- [ ] Folder models/ chứa gì?
- [ ] File config dùng format gì? (YAML? JSON?)
- [ ] Cách load pretrained weights?

### Performance

- [ ] Inference mất bao lâu? (ms/ảnh)
- [ ] AUC đạt được bao nhiêu?
- [ ] Có bị fail case nào không?

### Deployment

- [ ] Họ dùng Gradio hay Streamlit?
- [ ] API endpoint như thế nào?
- [ ] Docker có sẵn không?
```

### 7.2. So sánh với kế hoạch HolmHz

| Aspect     | CNNDetection | UniversalFakeDetect | HolmHz (Plan)    |
| ---------- | ------------ | ------------------- | ---------------- |
| Backbone   | ResNet50     | CLIP                | EfficientNet-B0  |
| XAI        | ❌           | ❌                  | ✅ Grad-CAM      |
| Web Demo   | Gradio basic | ❌                  | Gradio + FastAPI |
| Target AUC | 0.95         | 0.82 (OOD)          | 0.90             |

### 7.3. Action items cho HolmHz

```markdown
## Việc cần làm sau khi chạy thử:

1. [ ] Copy cấu trúc folder từ CNNDetection
2. [ ] Học cách preprocess ảnh từ code của họ
3. [ ] Tham khảo Gradio demo của họ
4. [ ] Benchmark model của họ trên dataset của mình
5. [ ] Bắt đầu Sprint 1.1 (Environment Setup)
```

---

## 🎯 Thứ tự thực hiện

```
Ngày 1: Demo Online (30 phút)
        → Hiểu AI deepfake detection làm được gì

Ngày 2: CNNDetection (1 tiếng)
        → Chạy local, hiểu cấu trúc cơ bản

Ngày 3: UniversalFakeDetect (1-2 tiếng)
        → Hiểu SOTA, so sánh với baseline

Ngày 4: DeepfakeBench (2-3 tiếng) - Optional
        → Nếu muốn hiểu sâu hơn về benchmark

Ngày 5: Bắt đầu HolmHz Sprint 1.1
        → Áp dụng những gì đã học
```

---

👉 **Lời khuyên**: Đừng cố hiểu 100% code của họ. Chỉ cần:

1. Chạy được
2. Thấy kết quả
3. Hiểu input/output
4. Biết cấu trúc folder

Đó là đủ để bắt đầu HolmHz! 🚀
