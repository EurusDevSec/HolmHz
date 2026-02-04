# 🔢 NumPy: 5 Bài tập từ Dễ đến Khó (Ngữ cảnh đời thường)

> **Lưu ý**: Các bài tập này dùng ví dụ thực tế dễ hiểu. Bạn chỉ cần biết NumPy là thư viện làm việc với mảng số.

---

## Bài 1: Điểm thi của lớp (Tạo mảng & Thống kê cơ bản)

**Kiến thức**: `np.array()`, `mean()`, `max()`, `min()`.

**Yêu cầu**:

1. Tạo mảng `diem_thi` chứa điểm của 10 học sinh: `[7.5, 8.0, 6.5, 9.0, 5.5, 8.5, 7.0, 6.0, 9.5, 8.0]`
2. Tính điểm trung bình của cả lớp.
3. Tìm điểm cao nhất và thấp nhất.
4. Đếm xem có bao nhiêu học sinh đạt từ 8.0 trở lên.

**Code khung**:

```python
import numpy as np

# Tạo mảng điểm
diem_thi = np.array([7.5, 8.0, 6.5, 9.0, 5.5, 8.5, 7.0, 6.0, 9.5, 8.0])

# Tính trung bình
trung_binh = # dùng np.mean() hoặc diem_thi.mean()
print(f"Điểm trung bình: {trung_binh}")

# Tìm max, min
diem_cao_nhat = # dùng np.max() hoặc diem_thi.max()
diem_thap_nhat = # dùng np.min() hoặc diem_thi.min()
print(f"Cao nhất: {diem_cao_nhat}, Thấp nhất: {diem_thap_nhat}")

# Đếm học sinh giỏi (điểm >= 8.0)
# Gợi ý: diem_thi >= 8.0 sẽ tạo ra mảng True/False
# np.sum() sẽ đếm số True (vì True = 1, False = 0)
so_hs_gioi = np.sum(diem_thi >= 8.0)
print(f"Số học sinh giỏi: {so_hs_gioi}")
```

---

## Bài 2: Bảng lương nhân viên (Mảng 2 chiều)

**Kiến thức**: Mảng 2D, truy cập hàng/cột, `sum()` theo trục.

**Tình huống**: Công ty có 4 nhân viên, mỗi người có lương 3 tháng (tháng 1, 2, 3).

**Yêu cầu**:

1. Tạo mảng 2D `bang_luong` kích thước (4, 3) - 4 nhân viên, 3 tháng.
2. Tính tổng lương của từng nhân viên (tổng theo hàng).
3. Tính tổng lương công ty phải trả mỗi tháng (tổng theo cột).
4. Tăng lương tháng 3 thêm 500 cho tất cả nhân viên.

**Code khung**:

```python
import numpy as np

# Mảng 2D: Hàng = Nhân viên, Cột = Tháng
bang_luong = np.array([
    [5000, 5200, 5100],  # Nhân viên 1
    [6000, 6000, 6500],  # Nhân viên 2
    [4500, 4800, 4700],  # Nhân viên 3
    [7000, 7200, 7100],  # Nhân viên 4
])

print("Bảng lương ban đầu:")
print(bang_luong)

# Tổng lương từng nhân viên (tổng theo hàng, axis=1)
tong_moi_nv = np.sum(bang_luong, axis=1)
print(f"Tổng lương từng NV: {tong_moi_nv}")

# Tổng lương mỗi tháng (tổng theo cột, axis=0)
tong_moi_thang = # Điền code ở đây
print(f"Tổng lương mỗi tháng: {tong_moi_thang}")

# Tăng lương tháng 3 (cột thứ 2, index=2) thêm 500
bang_luong[:, 2] = bang_luong[:, 2] + 500
print("Bảng lương sau khi tăng tháng 3:")
print(bang_luong)
```

---

## Bài 3: Lọc sản phẩm theo giá (Boolean Indexing)

**Kiến thức**: So sánh mảng, lọc bằng điều kiện.

**Tình huống**: Cửa hàng có danh sách giá sản phẩm, cần lọc ra các sản phẩm trong tầm giá.

**Yêu cầu**:

1. Tạo mảng `gia_san_pham` gồm 8 giá khác nhau.
2. Lọc ra các sản phẩm có giá DƯỚI 100.
3. Lọc ra các sản phẩm có giá TỪ 50 ĐẾN 150.
4. Giảm giá 10% cho tất cả sản phẩm trên 200.

**Code khung**:

```python
import numpy as np

gia_san_pham = np.array([50, 120, 30, 200, 85, 300, 150, 75])
print(f"Giá ban đầu: {gia_san_pham}")

# Lọc giá dưới 100
# Bước 1: Tạo mask (mảng True/False)
mask_duoi_100 = gia_san_pham < 100
print(f"Mask: {mask_duoi_100}")  # [True, False, True, ...]

# Bước 2: Dùng mask để lọc
san_pham_re = gia_san_pham[mask_duoi_100]
print(f"Sản phẩm dưới 100: {san_pham_re}")

# Lọc giá từ 50 đến 150 (dùng & để kết hợp 2 điều kiện)
mask_tam_gia = (gia_san_pham >= 50) & (gia_san_pham <= 150)
san_pham_tam_gia = # Điền code ở đây
print(f"Sản phẩm 50-150: {san_pham_tam_gia}")

# Giảm giá 10% cho sản phẩm trên 200
# Gợi ý: gia_san_pham[gia_san_pham > 200] = ...
gia_san_pham[gia_san_pham > 200] = gia_san_pham[gia_san_pham > 200] * 0.9
print(f"Giá sau giảm: {gia_san_pham}")
```

---

## Bài 4: Bảng điểm nhiều môn (Reshape & Transpose)

**Kiến thức**: `reshape()`, `T` (transpose), thay đổi hình dạng mảng.

**Tình huống**: Bạn có điểm 12 bài kiểm tra, cần sắp xếp thành bảng.

**Yêu cầu**:

1. Tạo mảng 1 chiều gồm 12 điểm.
2. Reshape thành bảng 3 hàng x 4 cột (3 học sinh, 4 môn).
3. Transpose để đổi thành 4 hàng x 3 cột (4 môn, 3 học sinh).
4. Tính điểm trung bình của từng học sinh.

**Code khung**:

```python
import numpy as np

# 12 điểm liền nhau
diem = np.array([8, 7, 9, 6, 7, 8, 8, 9, 5, 6, 7, 8])
print(f"Mảng gốc (12 phần tử): {diem}")
print(f"Shape: {diem.shape}")  # (12,)

# Reshape thành 3x4 (3 học sinh, 4 môn)
bang_diem = diem.reshape(3, 4)
print("Bảng điểm 3x4:")
print(bang_diem)
print(f"Shape: {bang_diem.shape}")  # (3, 4)

# Transpose thành 4x3 (4 môn, 3 học sinh)
bang_diem_T = bang_diem.T  # Hoặc np.transpose(bang_diem)
print("Sau transpose 4x3:")
print(bang_diem_T)

# Điểm trung bình từng học sinh (trung bình theo hàng của bảng gốc 3x4)
tb_hoc_sinh = np.mean(bang_diem, axis=1)
print(f"Điểm TB từng HS: {tb_hoc_sinh}")
```

---

## Bài 5: So sánh 2 bảng điểm (Phép toán giữa mảng)

**Kiến thức**: Cộng/trừ mảng, so sánh mảng, `np.where()`.

**Tình huống**: So sánh điểm kỳ 1 và kỳ 2 của học sinh.

**Yêu cầu**:

1. Tạo 2 mảng `diem_ky1` và `diem_ky2` (mỗi mảng 5 phần tử).
2. Tính mảng `chenh_lech = diem_ky2 - diem_ky1`.
3. Đếm số học sinh tiến bộ (điểm kỳ 2 > kỳ 1).
4. Dùng `np.where()` để tạo mảng đánh giá: "Tiến bộ" hoặc "Giảm sút".

**Code khung**:

```python
import numpy as np

diem_ky1 = np.array([7.0, 6.5, 8.0, 5.5, 7.5])
diem_ky2 = np.array([7.5, 6.0, 8.5, 6.0, 7.0])

# Chênh lệch điểm
chenh_lech = diem_ky2 - diem_ky1
print(f"Chênh lệch: {chenh_lech}")  # [0.5, -0.5, 0.5, 0.5, -0.5]

# Đếm học sinh tiến bộ
so_tien_bo = np.sum(chenh_lech > 0)
print(f"Số HS tiến bộ: {so_tien_bo}")

# Đánh giá bằng np.where()
# np.where(điều_kiện, giá_trị_nếu_đúng, giá_trị_nếu_sai)
danh_gia = np.where(chenh_lech > 0, "Tiến bộ", "Giảm sút")
print(f"Đánh giá: {danh_gia}")
# ['Tiến bộ', 'Giảm sút', 'Tiến bộ', 'Tiến bộ', 'Giảm sút']
```

---

👉 **Thứ tự làm**: Bài 1 → 2 → 3 → 4 → 5 (Độ khó tăng dần)

**Mẹo debug**:

- Luôn `print(ten_mang.shape)` để kiểm tra kích thước mảng.
- Nếu báo lỗi index, kiểm tra lại số hàng/cột.
