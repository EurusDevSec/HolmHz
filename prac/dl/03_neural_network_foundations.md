# 🧠 Machine Learning: Dạy Máy Học (Bản Vỡ Lòng)

> **Tư duy cốt lõi**: Đừng nghĩ về não người. Hãy nghĩ về việc **Dạy Trẻ Con**.

---

## 1. Lý thuyết: Máy học là gì?

Tưởng tượng bạn dạy một đứa trẻ (Model) phân biệt **Tiền thật** và **Tiền giả**.

1.  **Bước 1 (Đoán bừa):** Bạn đưa tờ tiền cho bé. Bé chưa biết gì cả, đoán đại: "Tiền giả ạ!".
2.  **Bước 2 (So sánh):** Thực tế đó là tờ **Tiền thật**. => Bé đoán **SAI**.
3.  **Bước 3 (Rút kinh nghiệm):** Bạn cốc đầu bé một cái (nhẹ thôi!). Bé nhận ra: "À, tờ này màu xanh lá là tiền thật". Bé điều chỉnh suy nghĩ.
4.  **Lặp lại:** Làm đi làm lại 1000 lần với các tờ tiền khác nhau. Dần dần bé sẽ đoán chuẩn 99%.

### Trong ngôn ngữ máy tính:

- **Đứa trẻ** = **Model** (Mạng nơ-ron).
- **Cú cốc đầu (Độ sai sót)** = **Loss** (Hàm mất mát).
  - Đoán sai nhiều -> Loss cao -> Cốc đầu đau.
  - Đoán đúng -> Loss thấp -> Được khen.
- **Rút kinh nghiệm** = **Optimizer** (Bộ tối ưu hóa).
  - Là cơ chế giúp model tự sửa lại các dây thần kinh (trọng số) để lần sau đoán đúng hơn.

---

## 2. Code ngây ngô: Dạy máy làm phép nhân đôi

Chúng ta sẽ dạy máy tính học quy luật: `Đầu Ra = Đầu Vào x 2`.
(Ví dụ: Vào 3 -> Ra 6. Vào 5 -> Ra 10).

Chúng ta **không lập trình** công thức `y = x * 2`. Chúng ta bắt máy **tự tìm ra** số 2 đó.

### Bước 1: Chuẩn bị dữ liệu học (Data)

```python
import torch

# Dữ liệu đầu vào (X)
X = torch.tensor([1.0, 2.0, 3.0, 4.0])

# Kết quả đúng tương ứng (Y) - Đây là đáp án để máy so sánh
Y = torch.tensor([2.0, 4.0, 6.0, 8.0])

# Quy luật ngầm là nhân 2, nhưng máy chưa biết.
```

### Bước 2: Tạo đứa trẻ ngây thơ (Model)

Ta giả sử quy luật là `y = x * w`. `w` (Weight - Trọng số) là cái máy cần tìm.
Ban đầu ta cho `w` một số ngẫu nhiên.

```python
# Khởi tạo w là 0.5 (Đoán bừa)
# requires_grad=True nghĩa là: "Cho phép sửa số này trong quá trình học"
w = torch.tensor(0.5, requires_grad=True)

print(f"Khởi đầu, máy đoán số cần tìm là: {w.item()}")
```

### Bước 3: Quá trình Training (Dạy học)

Ta sẽ dạy máy 100 lần (epochs).

```python
# Lặp lại bài học 20 lần
for buoi_hoc in range(20):

    # 1. Forward (Máy đoán)
    # Máy lấy X nhân với w hiện tại (0.5)
    du_doan = X * w

    # 2. Tính Loss (Máy xem sai bao nhiêu)
    # Lấy (Dự đoán - Đáp án thật) bình phương lên cho mất dấu âm
    sai_so = (du_doan - Y).pow(2).mean()

    # 3. Backward (Máy tự suy ngẫm)
    # Lệnh này tính toán xem cần tăng hay giảm w để sai số bé đi
    sai_so.backward()

    # 4. Update (Sửa sai)
    # Tắt chế độ tính toán để sửa w
    with torch.no_grad():
        # Học với tốc độ 0.1 (Learning Rate)
        # Nếu w đang thấp, nó sẽ cộng thêm. Nếu w đang cao, nó sẽ trừ đi.
        w -= 0.1 * w.grad

        # Reset suy nghĩ cũ để chuẩn bị cho buổi học mới
        w.grad.zero_()

    print(f"Buổi {buoi_hoc+1}: Máy đoán w = {w.item():.4f}, Sai số = {sai_so.item():.4f}")

# Kết quả cuối cùng
print(f"\nSau khi học, máy chốt số cần tìm là: {w.item()}")
# Bạn sẽ thấy nó cực kỳ gần số 2.0 (ví dụ 1.999...)
```

---

## 🎯 Bài tập Hands-on 3

Hãy copy đoạn code trên vào file `bai_tap_3.py` và chạy thử. Sau đó hãy thử sửa:

1.  Sửa bộ dữ liệu `Y` sao cho quy luật là **Nhân 3** (ví dụ: X=1 -> Y=3, X=2 -> Y=6).
2.  Chạy lại xem cuối cùng máy có tìm ra số `w` gần bằng 3.0 không?
3.  Thử sửa `Learning Rate` (chỗ `0.1`) thành `0.001` (Học quá chậm). Quan sát xem sau 20 buổi học máy đã tìm ra kết quả chưa hay vẫn còn sai nhiều?

👉 **Mục tiêu**: Hiểu rằng "Learning" thực chất chỉ là quá trình điều chỉnh một con số `w` dần dần cho đến khi kết quả tính toán khớp với đáp án thật.
