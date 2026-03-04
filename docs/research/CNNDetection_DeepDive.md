# 🔍 Deep Dive: CNNDetection - Mổ xẻ dự án Deepfake Detection

> **Dành cho**: DevOps Engineer / AI Researcher (HolmHz Team)
> **Mục đích**: Giải thích **"Tại sao nó chạy được?"** và **"Ta học được gì từ nó?"** thay vì chỉ chạy mù quáng.
> **Dự án gốc**: [CNNDetection (Wang et al., CVPR 2020)](https://github.com/PeterWang512/CNNDetection)

---

## 1. Tổng quan: Tại sao cái này quan trọng?

Bạn vừa chạy demo có kết quả **>90%** độ chính xác. Nhưng con số này có ý nghĩa gì?

- **Bí mật**: Model này **KHÔNG** nhìn vào khuôn mặt để xem mắt mũi có lệch không (như con người).
- **Sự thật**: Nó nhìn vào **vết tích của thuật toán (Artifacts)**. Các công cụ sinh ảnh AI (GAN, Diffusion) luôn để lại các "dấu vân tay" vô hình (do cách tính toán Convolution) mà mắt người không thấy, nhưng máy tính thì thấy rõ.
- **Tầm quan trọng**: Đây là baseline (chuẩn mực). Nếu HolmHz muốn xịn, HolmHz phải tốt hơn hoặc bằng cái này.

---

## 2. Giải phẫu cấu trúc dự án (Anatomy)

Một dự án AI chuẩn mực thường chia như sau, và CNNDetection làm rất tốt:

```text
CNNDetection/
├── demo.py            <-- Cửa chính (Entry Point). Bạn chạy cái này.
├── networks/          <-- "Bộ não". Chứa code định nghĩa kiến trúc Model.
│   └── resnet.py      <-- File quan trọng nhất về Model.
├── weights/           <-- "Ký ức". File .pth chứa những gì model đã học.
├── utils/             <-- "Đồ nghề". Các hàm phụ trợ (load ảnh, tính toán).
└── requirements.txt   <-- Danh sách nguyên liệu cần thiết.
```

### 💡 Bài học cho HolmHz:
> Đừng dồn hết code vào 1 file `main.py`. Hãy tách `networks/`, `utils/`, và `configs/` riêng biệt.

---

## 3. Quy trình "chế biến" 1 tấm ảnh (Pipeline)

Khi bạn chạy lệnh:
`python demo.py -f fake.png`

Điều gì thực sự xảy ra bên dưới? Hãy tưởng tượng dây chuyền sản xuất:

### Bước 1: Sơ chế nguyên liệu (Preprocessing)
Ảnh `fake.png` (màu mè, kích thước lung tung) được đưa vào khuôn khổ.

```python
# Trong file demo.py
trans = transforms.Compose([
    transforms.ToTensor(),       # 1. Đổi ảnh từ 0-255 sang 0.0-1.0
    transforms.Normalize(        # 2. "Chuẩn hóa" màu sắc theo chuẩn ImageNet
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])
```
*   **Tại sao lại có mấy số lẻ lẻ kia?** (`0.485...`): Đây là **tiêu chuẩn vàng** của ngành AI. Mọi model AI (Google, Facebook...) đều dùng bộ số này để model nhìn thế giới "cùng một màu".
*   **Học được gì**: HolmHz **BẮT BUỘC** phải có bước Preprocessing giống hệt như này nếu dùng base model chuẩn.

### Bước 2: Bộ não suy luận (Inference)
Ảnh đã sơ chế được đưa vào `resnet50` (mạng nơ-ron 50 tầng).

```python
# Model tính toán ra 1 con số thô (logit), ví dụ: 5.2 hoặc -3.1
raw_output = model(in_tens)

# Hàm Sigmoid ép con số đó về khoảng 0% - 100%
prob = raw_output.sigmoid().item()
```
*   **Logic**: 
    *   Số dương lớn (ví dụ 5.0) -> Sigmoid ra gần 1.0 (99% Fake).
    *   Số âm lớn (ví dụ -5.0) -> Sigmoid ra gần 0.0 (0% Fake - tức là Real).

### Bước 3: Cấu hình linh hoạt (Arguments)
Họ dùng `argparse` để bạn chỉnh sửa mà không cần sửa code:
```python
parser.add_argument('--use_cpu', action='store_true') # Không có GPU thì dùng CPU
parser.add_argument('-m', '--model_path')             # Thích đổi model khác cũng được
```

---

## 4. Những Best Practices cần "chôm" ngay

Dưới đây là checklist những thứ bạn cần copy từ dự án này sang HolmHz:

### ✅ 1. Tách biệt Model Definition
Trong file `networks/resnet.py`, họ chỉ định nghĩa Class `ResNet`. Họ không viết code chạy thử hay load ảnh trong đó.
*   **Lợi ích**: Bạn có thể tái sử dụng file này cho dự án khác, hoặc import nó vào Web App dễ dàng.

### ✅ 2. Xử lý thiết bị (Device Agnostic)
Code của họ chạy được trên cả máy server khủng (GPU) và laptop cùi (CPU).
```python
# Đoạn code vàng ngọc:
if not opt.use_cpu:
    model.cuda()  # Có GPU thì đẩy lên GPU
else:
    # Không có thì thôi, chạy chậm chút trên CPU
    pass 
```

### ✅ 3. Load Model an toàn
```python
# map_location='cpu' giúp tránh lỗi khi bạn train trên GPU server 
# nhưng mang về laptop (không có GPU) để chạy demo.
state_dict = torch.load(opt.model_path, map_location='cpu')
```

### ✅ 4. Binary Classification (0 hoặc 1)
Họ set `num_classes=1`.
*   Nhiều hướng dẫn trên mạng dạy set là 2 (Class 0: Real, Class 1: Fake).
*   Nhưng set là 1 (chỉ cần biết xác suất Fake) + hàm `Sigmoid` ở cuối là cách pro hơn, gọn nhẹ hơn.

---

## 5. Kết luận: HolmHz sẽ khác gì?

Dự án CNNDetection này rất tốt, nhưng nó thiếu những thứ HolmHz sẽ làm:

1.  **AI Explainability (XAI)**:
    *   CNNDetection chỉ phán "99% Fake".
    *   HolmHz sẽ trả lời: "Fake ở vùng mắt, vùng miệng" (Sử dụng Grad-CAM).
2.  **Web Interface chuẩn chỉnh**:
    *   Họ chỉ có script `demo.py` đen ngòm.
    *   HolmHz sẽ có UI kéo thả đẹp mắt.
3.  **Hỗ trợ nhiều Model**:
    *   Họ chỉ có ResNet.
    *   HolmHz sẽ cho phép chọn model (EfficientNet, ViT...).

**👉 Hành động tiếp theo**:
Hãy thử đọc file `demo.py` của họ một lần nữa với tâm thế của người đã hiểu "nội tạng" của nó. Bạn sẽ thấy nó cực kỳ logic và dễ hiểu.
