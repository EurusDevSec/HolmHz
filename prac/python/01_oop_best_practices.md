# 🐍 Python OOP: Từ Con Số 0 (Dễ hiểu nhất)

> **Dành cho người mới**: Nếu bạn biết biến (`x = 5`) và hàm (`def func():`), bạn đã sẵn sàng để học cái này.

---

## 1. Lý thuyết: Class và Object là cái quái gì?

Hãy tưởng tượng bạn là một **Kiến trúc sư** thiết kế nhà.

1.  **Class (Lớp)**: Chính là **Bản vẽ thiết kế** (Blueprint).
    - Trên bản vẽ, bạn quy định: Nhà có mấy tầng? Màu sơn là gì? Có bấm chuông được không?
    - _Lưu ý_: Bản vẽ chỉ là giấy, chưa ở được.

2.  **Object (Đối tượng)**: Là **Ngôi nhà thật** được xây từ bản vẽ đó.
    - Từ 1 bản vẽ (Class), bạn có thể xây 100 ngôi nhà (Objects) khác nhau (nhà A màu xanh, nhà B màu đỏ).

### Tại sao AI cần cái này?

Trong AI, chúng ta thiết kế một "bộ não" (Model).

- **Class**: Là thiết kế của bộ não (có bao nhiêu nơ-ron, nối với nhau thế nào).
- **Object**: Là bộ não cụ thể đang chạy trong máy tính của bạn.

---

## 2. Cú pháp cơ bản (Vừa đọc vừa gõ)

### Bước 1: Tạo bản vẽ (Class)

Chúng ta dùng từ khóa `class` để bắt đầu vẽ. Hàm `__init__` là hàm quan trọng nhất - nó chạy ngay khi "động thổ" xây nhà.

```python
class Robot:
    # __init__ giống như phiếu điền thông tin khi xuất xưởng
    # self chính lá "cái robot này" (để phân biệt với robot khác)
    def __init__(self, ten, mau_sac):
        self.ten = ten          # Lưu tên vào bộ nhớ robot
        self.mau_sac = mau_sac  # Lưu màu vào bộ nhớ robot
        self.pin = 100          # Mặc định pin đầy

    # Hành động robot có thể làm (Method)
    def chao(self):
        print(f"Xin chào, ta là {self.ten}, màu {self.mau_sac}")

    def chay(self):
        self.pin = self.pin - 10 # Chạy thì tốn pin
        print(f"{self.ten} đang chạy... Pin còn {self.pin}%")
```

### Bước 2: Xây robot (Tạo Object)

```python
# Tạo ra 2 robot từ 1 bản vẽ
robot_1 = Robot("Wall-E", "Vàng")
robot_2 = Robot("Baymax", "Trắng")

# Bắt chúng hoạt động
robot_1.chao()  # In: Xin chào, ta là Wall-E...
robot_2.chao()  # In: Xin chào, ta là Baymax...

robot_1.chay()  # Wall-E chạy, pin giảm còn 90
print(robot_2.pin) # Baymax chưa chạy, pin vẫn 100
```

---

## 3. Ứng dụng vào AI (Đơn giản hóa)

Trong dự án HolmHz, chúng ta sẽ quản lý dữ liệu ảnh. Thay vì lưu lung tung, ta tạo một Class để quản lý nó.

### Bước 3: Class quản lý Dataset

```python
class KhoAnh:
    def __init__(self, duong_dan_thu_muc):
        self.thu_muc = duong_dan_thu_muc
        self.danh_sach_anh = ["anh1.jpg", "anh2.jpg", "anh3.jpg"] # Giả vờ có 3 ảnh

    # Hàm đặc biệt __len__: Để khi hỏi len(kho_anh) nó trả lời được
    def __len__(self):
        return len(self.danh_sach_anh)

    # Hàm đặc biệt __getitem__: Để lấy ảnh theo số thứ tự
    def __getitem__(self, vi_tri):
        ten_anh = self.danh_sach_anh[vi_tri]
        return f"Đang lấy ảnh ở {self.thu_muc}/{ten_anh}"
```

### Chạy thử đoạn code này:

```python
# Tạo kho ảnh
kho_cua_toi = KhoAnh("R:/Data/HolmHz")

# Kiểm tra số lượng
print(f"Tổng số ảnh: {len(kho_cua_toi)}")
# Máy hiểu lệnh len() nhờ hàm __len__ ta viết ở trên

# Lấy ảnh thứ 0
print(kho_cua_toi[0])
# Máy hiểu ngoặc vuông [] nhờ hàm __getitem__ ta viết ở trên
```

---

## 🎯 Bài tập Hands-on 1 (Làm ngay)

Bạn hãy tạo một file `bai_tap_1.py` và viết code sau:

1.  Tạo một class tên là `ChoNghiepVu` (Chó nghiệp vụ).
2.  Hàm `__init__` nhận vào `ten` và `khu_vuc_truc` (ví dụ: Cổng A, Nhà kho).
3.  Thêm thuộc tính `so_lan_sua` mặc định bằng 0.
4.  Viết hàm `phat_hien_trom()`. Khi gọi hàm này:
    - In ra "Gâu gâu! Có trộm ở [khu vực trực]!"
    - Tăng `so_lan_sua` lên 1.
5.  Tạo ra 1 chú chó tên "Micky", trực ở "Cổng chính". Gọi hàm `phat_hien_trom()` 2 lần rồi in `so_lan_sua` ra xem đúng bằng 2 không.

👉 **Mục tiêu**: Hiểu được `self` dùng để lưu trữ thông tin riêng của từng đối tượng.
