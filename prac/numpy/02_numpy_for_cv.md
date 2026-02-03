# 🔢 Hiểu Ảnh Là Các Con Số (NumPy Cơ Bản)

> **Tư duy cốt lõi**: Máy tính không có "Mắt", nó không nhìn thấy hình ảnh cô gái hay con mèo. Nó chỉ nhìn thấy một bảng tính Excel khổng lồ chứa đầy các con số.

---

## 1. Lý thuyết: Pixel là gì?

- Một bức ảnh kỹ thuật số là một lưới các ô vuông (Pixel).
- Mỗi ô vuông chứa một con số thể hiện độ sáng.
  - **0**: Đen thui (Tắt đèn).
  - **255**: Trắng tinh (Bật đèn hết cỡ).
  - Các số ở giữa (ví dụ 100): Xám xám.

### Ảnh màu (RGB) thì sao?

Nó là 3 tấm lưới chồng lên nhau:

- Tấm 1: Chỉ chứa màu **Red** (Đỏ).
- Tấm 2: Chỉ chứa màu **Green** (Xanh lá).
- Tấm 3: Chỉ chứa màu **Blue** (Xanh dương).

--> Vì thế trong code bạn sẽ thấy kích thước ảnh là `(Cao, Rộng, 3)`. Số 3 chính là 3 tấm lưới màu này.

---

## 2. Thực hành: Phẫu thuật một bức ảnh

Chúng ta dùng thư viện `numpy` để làm việc với các bảng số này.

### Bước 1: Tạo một bức ảnh giả (Vuông Đen)

```python
import numpy as np
import matplotlib.pyplot as plt # Thư viện để vẽ hình

# Tạo một "bức ảnh" đen thui kích thước 10x10
# zeros nghĩa là toàn số 0 -> Đen
anh_den = np.zeros((10, 10))

print("Dữ liệu ảnh đen:")
print(anh_den)
```

### Bước 2: Vẽ hình lên ảnh (Thay đổi số)

Hãy nhớ: **Sửa ảnh = Sửa số**.

```python
# Copy ra ảnh mới để vẽ
anh_chu_thap = anh_den.copy()

# Vẽ một đường dọc màu trắng (số 255) ở giữa
# [:, 4] nghĩa là: Lấy tất cả các hàng tại cột số 4
anh_chu_thap[:, 4] = 255

# Vẽ một đường ngang màu trắng ở giữa
# [4, :] nghĩa là: Lấy hàng số 4, tất cả các cột
anh_chu_thap[4, :] = 255

print("\nDữ liệu sau khi vẽ:")
print(anh_chu_thap)

# Code để hiển thị ra màn hình (bạn chạy trong Jupyter Notebook mới thấy hình)
# plt.imshow(anh_chu_thap, cmap='gray')
# plt.show()
```

---

## 3. Các thao tác "Sửa ảnh" cơ bản

Trong dự án Deepfake, ta sẽ phải xử lý ảnh rất nhiều.

### 3.1. Cắt ảnh (Cropping)

Cắt ảnh thực chất là chọn một vùng trong bảng số.

```python
# Cắt lấy vùng trung tâm từ dòng 3 đến 7, cột 3 đến 7
vung_trung_tam = anh_chu_thap[3:7, 3:7]
```

### 3.2. Làm sáng ảnh (Brightness)

Làm sáng nghĩa là cộng thêm giá trị vào tất cả các pixel.

```python
# Cộng thêm 50 đơn vị độ sáng vào toàn bộ ảnh
anh_sang_hon = anh_chu_thap + 50
print("Ảnh đã sáng hơn!")

# Lưu ý: Nếu cộng quá 255 nó sẽ bị lỗi hoặc reset về 0.
# Nhưng ở mức cơ bản ta khoan hãy lo việc đó.
```

### 3.3. Lọc ảnh (Thresholding) - Quan trọng!

Tìm tất cả điểm ảnh sáng hơn một mức nào đó.

```python
# Tìm xem những chỗ nào đang là màu trắng (255)
mask = (anh_chu_thap > 200)

# mask sẽ là một bảng chứa True/False (Đúng/Sai)
# Nơi nào trắng -> True, nơi nào đen -> False
print(mask)
```

---

## 🎯 Bài tập Hands-on 2

Bạn tạo file `bai_tap_2.py` hoặc dùng Jupyter Notebook:

1.  Tạo một mảng numpy toàn số 0 kích thước `(20, 20)` (Ảnh đen 20x20).
2.  Tô một hình vuông màu trắng kích thước 5x5 ở góc trên cùng bên trái.
    - _Gợi ý_: Dùng slice `[0:5, 0:5] = 255`.
3.  Tô một hình vuông màu xám (giá trị 100) ở góc dưới cùng bên phải.
    - _Gợi ý_: Góc dưới cùng là từ 15 đến 20.
4.  Tính giá trị trung bình (`mean`) của toàn bộ bức ảnh xem độ sáng trung bình là bao nhiêu.
    - _Gợi ý_: Search Google "numpy calculate mean of array".

👉 **Mục tiêu**: Hiểu rằng thao tác với ảnh thực chất chỉ là gán giá trị và cắt gọt các mảng số.
