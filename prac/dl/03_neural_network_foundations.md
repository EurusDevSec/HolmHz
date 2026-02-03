# 🧠 Neural Networks Foundations "In a Nutshell"

Tài liệu này giải thích cơ chế hoạt động của Neural Network theo cách "Data Flow" thay vì toán học phức tạp, giúp bạn hiểu code PyTorch đang làm gì.

---

## 1. The "Black Box" Model (Forward Pass)

Tưởng tượng Neural Network (NN) là một hàm số phức tạp $f(x)$ nhận đầu vào là Ảnh và trả ra đầu ra là Xác suất (Real/Fake).

### Code thuần túy (không dùng thư viện)

Một nơ-ron đơn giản (Perceptron) thực hiện phép tính:
$$y = \text{activation}(x \cdot w + b)$$

```python
import numpy as np

# Giả sử đầu vào là vector đặc trưng của ảnh (đã flatten)
x = np.array([0.5, 0.2, 0.9])  # 3 features

# Trọng số (Weights) - Nơi lưu trữ "trí thông minh"
w = np.array([0.8, -0.5, 1.0]) # 3 weights tương ứng
b = 0.1                       # Bias (độ lệch)

# 1. Linear combination (Phép nhân và cộng)
z = np.dot(x, w) + b
# z = (0.5*0.8) + (0.2*-0.5) + (0.9*1.0) + 0.1

# 2. Activation Function (Phi tuyến tính hóa)
# Sigmoid: Ép giá trị về khoảng (0, 1) -> Xác suất
def sigmoid(val):
    return 1 / (1 + np.exp(-val))

y_pred = sigmoid(z)
print(f"Dự đoán: {y_pred:.4f}")
```

---

## 2. Loss Function (Đo lường sai số)

Làm sao biết model ngu hay khôn? Chúng ta cần một thước đo sai số (Loss/Cost).
Trong bài toán phân loại nhị phân (Deepfake), ta dùng **Binary Cross Entropy (BCE)**.

- Nếu Label thật = 1 (Fake), Dự đoán = 0.9 (Fake) -> Loss thấp (Tốt).
- Nếu Label thật = 1 (Fake), Dự đoán = 0.1 (Real) -> Loss cao (Tệ).

```python
def binary_loss(y_true, y_pred):
    # Công thức đơn giản hóa để hiểu
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Ví dụ
label = 1 # Fake
pred_good = 0.9
pred_bad = 0.1

print(f"Loss khi đoán đúng: {binary_loss(label, pred_good):.4f}") # Gần 0
print(f"Loss khi đoán sai:  {binary_loss(label, pred_bad):.4f}")  # Rất lớn
```

---

## 3. Backward & Gradient (Học hỏi)

Đây là phép màu của AI. Sau khi biết mình sai bao nhiêu (Loss), model cần biết: **"Cần chỉnh sửa w và b tăng hay giảm bao nhiêu để Loss nhỏ lại?"**.

Đây chính là **Gradient (Đạo hàm)**.

- Gradient dương (+): Giảm weight đi thì Loss sẽ giảm.
- Gradient âm (-): Tăng weight lên thì Loss sẽ giảm.

### PyTorch Autograd (Tự động hóa việc này)

PyTorch sinh ra để bạn không phải tính đạo hàm bằng tay.

```python
import torch

# require_grad=True báo cho PyTorch biết cần theo dõi biến này để học
w = torch.tensor([0.8, -0.5, 1.0], requires_grad=True)
x = torch.tensor([0.5, 0.2, 0.9])
b = torch.tensor(0.1, requires_grad=True)
label = torch.tensor(1.0) # Nhãn thật

# 1. Forward
z = torch.dot(x, w) + b
y_pred = torch.sigmoid(z)

# 2. Compute Loss
loss = - (label * torch.log(y_pred) + (1 - label) * torch.log(1 - y_pred))
print(f"Loss: {loss.item()}")

# 3. Backward (Magic happens here)
# PyTorch tự động tính đạo hàm ngược từ Loss về w và b
loss.backward()

# Xem kết quả: Cần điều chỉnh w như thế nào?
print(f"Gradient của w: {w.grad}")
# Nếu w.grad[0] là số âm -> Cần tăng w[0] lên để Loss giảm
```

---

## 4. Training Loop Architecture

Mọi training loop trong HolmHz sẽ tuân theo quy trình 5 bước bất di bất dịch này:

1.  **Forward**: Đưa ảnh vào, lấy dự đoán.
2.  **Loss**: Tính sai số so với nhãn thật.
3.  **Zero Grad**: Xóa sạch đạo hàm cũ (bước dọn dẹp quan trọng).
4.  **Backward**: Tính đạo hàm mới (Gradient).
5.  **Step (Optimizer)**: Cập nhật trọng số `w = w - learning_rate * gradient`.

---

## 📝 Hands-on Challenge cho bạn

Trong file notebook `prac/dl/nn_practice.ipynb` (hãy tạo nó), thử dùng **PyTorch** để:

1.  Khởi tạo `w`, `b` ngẫu nhiên.
2.  Chạy 1 vòng lặp 100 lần (epochs).
3.  Trong mỗi vòng lặp:
    - Tính forward.
    - Tính loss (giả sử `label=1`).
    - Gọi `loss.backward()`.
    - Cập nhật thủ công: `with torch.no_grad(): w -= 0.1 * w.grad`.
    - `w.grad.zero_()`.
4.  In ra Loss xem nó có giảm dần về 0 không?

```python
# Khung code luyện tập
import torch

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
x = torch.tensor([1.5])
y = torch.tensor([1.0])

lr = 0.01

for i in range(100):
   # Code here...
```
