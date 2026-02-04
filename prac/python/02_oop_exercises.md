# 🐍 Python OOP: 5 Bài tập từ Dễ đến Khó (Ngữ cảnh đời thường)

> **Lưu ý**: Các bài tập này dùng ví dụ đời thường để bạn tập trung vào cú pháp OOP, không cần kiến thức AI/ML.

---

## Bài 1: Quản lý Sinh viên (Cơ bản nhất)

**Kiến thức**: `class`, `__init__`, `self`, tạo object.

**Yêu cầu**:

1. Tạo class `SinhVien` với 3 thuộc tính: `ho_ten`, `tuoi`, `diem_tb`.
2. Viết hàm `xep_loai()`:
   - Nếu `diem_tb >= 8`: return "Giỏi"
   - Nếu `diem_tb >= 6.5`: return "Khá"
   - Nếu `diem_tb >= 5`: return "Trung bình"
   - Còn lại: return "Yếu"
3. Viết hàm `gioi_thieu()`: return chuỗi "Tôi là [tên], [tuổi] tuổi, xếp loại [loại]"
4. Tạo 2 sinh viên và in `gioi_thieu()` của họ.

**Code khung**:

```python
class SinhVien:
    def __init__(self, ho_ten, tuoi, diem_tb):
        # Lưu 3 thuộc tính vào self
        pass

    def xep_loai(self):
        # Dùng if-elif-else để return xếp loại
        pass

    def gioi_thieu(self):
        # Gọi self.xep_loai() để lấy xếp loại
        pass

# Test
sv1 = SinhVien("Hoàng", 20, 8.5)
sv2 = SinhVien("Luân", 19, 6.0)
print(sv1.gioi_thieu())  # Tôi là Hoàng, 20 tuổi, xếp loại Giỏi
print(sv2.gioi_thieu())  # Tôi là Luân, 19 tuổi, xếp loại Khá
```

---

## Bài 2: Tài khoản Ngân hàng (Bảo vệ dữ liệu)

**Kiến thức**: Thuộc tính private (`__`), kiểm tra điều kiện.

**Yêu cầu**:

1. Tạo class `TaiKhoan` với:
   - Thuộc tính `ten_chu_tai_khoan` (công khai)
   - Thuộc tính `__so_du` (private, bắt đầu = 0)
2. Viết hàm `nap_tien(so_tien)`:
   - Nếu `so_tien > 0`: cộng vào `__so_du`, in "Nạp thành công"
   - Nếu không: in "Số tiền không hợp lệ"
3. Viết hàm `rut_tien(so_tien)`:
   - Nếu `so_tien > 0` VÀ `so_tien <= __so_du`: trừ đi, in "Rút thành công"
   - Nếu không đủ tiền: in "Không đủ số dư"
   - Nếu số âm: in "Số tiền không hợp lệ"
4. Viết hàm `xem_so_du()`: return `__so_du`

**Code khung**:

```python
class TaiKhoan:
    def __init__(self, ten):
        self.ten_chu_tai_khoan = ten
        self.__so_du = 0  # Private - không ai sửa trực tiếp được

    def nap_tien(self, so_tien):
        # Kiểm tra rồi cộng vào __so_du
        pass

    def rut_tien(self, so_tien):
        # Kiểm tra rồi trừ khỏi __so_du
        pass

    def xem_so_du(self):
        return self.__so_du

# Test
tk = TaiKhoan("Hoàng")
tk.nap_tien(1000000)      # Nạp thành công
tk.nap_tien(-500)         # Số tiền không hợp lệ
tk.rut_tien(300000)       # Rút thành công
tk.rut_tien(9000000)      # Không đủ số dư
print(tk.xem_so_du())     # 700000
```

---

## Bài 3: Hệ thống Thú cưng (Kế thừa)

**Kiến thức**: Class cha, class con, `super()`, override.

**Yêu cầu**:

1. Tạo class cha `ThuCung`:
   - `__init__` nhận `ten`, `tuoi`
   - Hàm `keu()`: return "..."
   - Hàm `gioi_thieu()`: return "[tên], [tuổi] tuổi, kêu: [keu()]"
2. Tạo class con `Cho` kế thừa `ThuCung`:
   - Ghi đè `keu()`: return "Gâu gâu!"
3. Tạo class con `Meo` kế thừa `ThuCung`:
   - Ghi đè `keu()`: return "Meo meo!"
4. Tạo 1 chó, 1 mèo, gọi `gioi_thieu()` của cả hai.

**Code khung**:

```python
class ThuCung:
    def __init__(self, ten, tuoi):
        self.ten = ten
        self.tuoi = tuoi

    def keu(self):
        return "..."

    def gioi_thieu(self):
        return f"{self.ten}, {self.tuoi} tuổi, kêu: {self.keu()}"

class Cho(ThuCung):
    def keu(self):
        # Ghi đè hàm keu của cha
        pass

class Meo(ThuCung):
    def keu(self):
        # Ghi đè hàm keu của cha
        pass

# Test
cho = Cho("Milu", 3)
meo = Meo("Kitty", 2)
print(cho.gioi_thieu())  # Milu, 3 tuổi, kêu: Gâu gâu!
print(meo.gioi_thieu())  # Kitty, 2 tuổi, kêu: Meo meo!
```

---

## Bài 4: Giỏ hàng mua sắm (List trong Class)

**Kiến thức**: Quản lý list bên trong class, tính tổng.

**Yêu cầu**:

1. Tạo class `GioHang`:
   - `__init__`: tạo list rỗng `self.san_pham = []`
2. Viết hàm `them_san_pham(ten, gia)`:
   - Thêm dictionary `{"ten": ten, "gia": gia}` vào list
3. Viết hàm `xoa_san_pham(ten)`:
   - Tìm và xóa sản phẩm có tên tương ứng khỏi list
4. Viết hàm `tinh_tong()`:
   - Dùng vòng lặp cộng tất cả `gia` lại
5. Viết hàm `in_hoa_don()`:
   - In từng sản phẩm và giá, cuối cùng in tổng

**Code khung**:

```python
class GioHang:
    def __init__(self):
        self.san_pham = []

    def them_san_pham(self, ten, gia):
        # Thêm dict vào list
        pass

    def xoa_san_pham(self, ten):
        # Duyệt list, tìm và xóa
        pass

    def tinh_tong(self):
        tong = 0
        # Duyệt list, cộng dồn gia
        return tong

    def in_hoa_don(self):
        print("=== HÓA ĐƠN ===")
        # Duyệt và in từng sản phẩm
        print(f"TỔNG: {self.tinh_tong()} VND")

# Test
gio = GioHang()
gio.them_san_pham("Áo thun", 150000)
gio.them_san_pham("Quần jean", 350000)
gio.them_san_pham("Giày", 500000)
gio.xoa_san_pham("Quần jean")
gio.in_hoa_don()
# Output:
# === HÓA ĐƠN ===
# Áo thun: 150000 VND
# Giày: 500000 VND
# TỔNG: 650000 VND
```

---

## Bài 5: Máy tính bỏ túi (Magic method `__call__`)

**Kiến thức**: Biến object thành hàm gọi được.

**Yêu cầu**:

1. Tạo class `PhepCong`:
   - Hàm `__call__(self, a, b)`: return `a + b`
2. Tạo class `PhepNhan`:
   - Hàm `__call__(self, a, b)`: return `a * b`
3. Tạo object từ mỗi class, rồi gọi nó như hàm.

**Code khung**:

```python
class PhepCong:
    def __call__(self, a, b):
        # Return tổng
        pass

class PhepNhan:
    def __call__(self, a, b):
        # Return tích
        pass

# Test
cong = PhepCong()
nhan = PhepNhan()

# Gọi object như gọi hàm!
print(cong(5, 3))   # 8
print(nhan(5, 3))   # 15

# Ứng dụng: Tạo list các phép tính
phep_tinh = [PhepCong(), PhepNhan()]
for phep in phep_tinh:
    print(phep(10, 2))  # 12 rồi 20
```

---

👉 **Thứ tự làm**: Bài 1 → 2 → 3 → 4 → 5 (Độ khó tăng dần)
