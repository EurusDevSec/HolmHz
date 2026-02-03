# 🔢 NumPy Best Practices cho Computer Vision

NumPy là nền tảng của xử lý ảnh. Ảnh trong máy tính thực chất chỉ là các ma trận số 3 chiều (Height, Width, Channels).

---

## 1. Anatomy of an Image (Cấu trúc ảnh)

Ghi nhớ quy ước shape (kích thước) cực kỳ quan trọng:

- **NumPy / OpenCV / PIL**: `(H, W, C)` -> Height, Width, Channel.
- **PyTorch**: `(C, H, W)` -> Channel, Height, Width.
- **PyTorch Batch**: `(B, C, H, W)` -> Batch, Channel, Height, Width.

```python
import numpy as np

# Giả lập một ảnh màu 224x224 (RGB)
# Shape: (Height=224, Width=224, Channels=3)
img = np.zeros((224, 224, 3), dtype=np.uint8)

print(f"Chiều cao: {img.shape[0]}")
print(f"Chiều rộng: {img.shape[1]}")
print(f"Số kênh màu: {img.shape[2]}")
```

---

## 2. Array Operations (Thao tác mảng)

### 2.1. Vectorization (Tránh vòng lặp for!)

Nhanh gấp trăm lần so với vòng lặp. Luôn thao tác trên toàn bộ array.

**❌ Bad (Chậm): Làm sáng ảnh bằng Loop**

```python
def brighten_loop(img, value=10):
    h, w, c = img.shape
    new_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                new_img[i,j,k] = min(img[i,j,k] + value, 255)
    return new_img
```

**✅ Good (Nhanh): Vectorization**

```python
def brighten_vector(img, value=10):
    # Cộng value vào TẤT CẢ pixel cùng lúc
    # np.clip để đảm bảo giá trị không vượt quá 255
    return np.clip(img + value, 0, 255).astype(np.uint8)
```

### 2.2. Broadcasting (Cơ chế lan truyền)

NumPy tự động "kéo dãn" các mảng nhỏ để khớp với mảng lớn khi tính toán.

**Ví dụ: Normalization (Chuẩn hóa ảnh)**
Muốn trừ giá trị trung bình (Mean) cho từng kênh màu R, G, B.

- Ảnh: `(224, 224, 3)`
- Mean: `(3,)` -> `[0.485, 0.456, 0.406]`

```python
# Giả sử ảnh đã được đưa về dạng float 0-1
img_float = np.random.rand(224, 224, 3).astype(np.float32)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)

# NumPy tự hiểu: trừ kênh 0 cho 0.485, kênh 1 cho 0.456...
# Nó "broadcast" vector (3,) thành (224, 224, 3) để tính toán
normalized_img = img_float - mean
```

---

## 3. Reshaping & Axes Manipulation

Đây là phần hay gây lỗi nhất khi chuyển từ NumPy sang PyTorch.

### 3.1. Transpose / Permute (Đổi trục)

Chuyển từ `(H, W, C)` sang `(C, H, W)`.

```python
img_hwc = np.random.rand(224, 224, 3) # Chuẩn ảnh thường

# Cách 1: np.moveaxis
img_chw = np.moveaxis(img_hwc, -1, 0) # Đưa axis cuối (-1) lên đầu (0)

# Cách 2: transpose (thường dùng hơn)
# Trục cũ: 0(H), 1(W), 2(C)
# Trật tự mới mong muốn: 2(C), 0(H), 1(W)
img_chw_2 = img_hwc.transpose(2, 0, 1)

print(img_chw_2.shape) # (3, 224, 224)
```

### 3.2. Add/Remove Dimension (Batch dimension)

Deep Learning models luôn cần batch dimension: `(1, C, H, W)`.

```python
# Thêm dimension ở vị trí đầu tiên
input_tensor = np.expand_dims(img_chw_2, axis=0)
print(input_tensor.shape) # (1, 3, 224, 224) -> Sẵn sàng đưa vào model!
```

---

## 📝 Hands-on Challenge cho bạn

Mở `prac/numpy/numpyLearn.ipynb` và thực hiện bài tập sau:

1.  Tạo một mảng random kích thước `(100, 100, 3)` giá trị int từ 0-255 (giả lập ảnh nhiễu).
2.  Tính giá trị trung bình độ sáng của toàn bộ ảnh (mean của tất cả pixels).
3.  Tạo mask: Tìm tất cả các pixel có giá trị > 128 và gán chúng về 255 (trắng), còn lại về 0 (đen). -> Đây là thao tác **Thresholding**.
4.  Chuyển đổi mảng đó sang format PyTorch `(3, 100, 100)`.

```python
# Gợi ý
mask = arr > 128
arr[mask] = 255
```
