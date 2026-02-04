# 🐍 Python OOP: Luyện tập chuyên sâu (5 Bài tập thực tế)

Sau khi đã nắm được khái niệm "Bản vẽ (Class)" và "Ngôi nhà (Object)", hãy thực hành với các bài toán mô phỏng hệ thống thực tế trong dự án AI.

---

## Bài 1: Hệ thống Quản lý Dataset (Đa hình & Kế thừa)

**Mục tiêu**: Hiểu về kế thừa (`Inheritance`) - Viết 1 lần dùng nhiều nơi.

**Yêu cầu**:

1.  Tạo class cha `Dataset` có thuộc tính `ten_dataset` và hàm `bao_cao()` in ra "Đây là dataset...".
2.  Tạo class con `ImageDataset` kế thừa từ `Dataset`.
    - Thêm thuộc tính `so_luong_anh`.
    - Ghi đè (override) hàm `bao_cao()` để in ra: "Dataset hình ảnh [tên] có [số] ảnh."
3.  Tạo class con `AudioDataset` kế thừa từ `Dataset`.
    - Thêm thuộc tính `so_gio_thu_am`.
    - Ghi đè `bao_cao()` để in ra theo format phù hợp.
4.  Tạo một list chứa lộn xộn cả `ImageDataset` và `AudioDataset`. Dùng vòng lặp for để gọi hàm `bao_cao()` của từng cái.

**Gợi ý**:

```python
class Dataset:
    def __init__(self, ten):
        self.ten = ten
    def bao_cao(self):
        print("Dataset chung chung")

class ImageDataset(Dataset):
    def __init__(self, ten, so_luong):
        super().__init__(ten) # Gọi hàm init của cha
        self.so_luong = so_luong
    # Viết tiếp...
```

---

## Bài 2: Mô phỏng Training Log (Encapsulation)

**Mục tiêu**: Hiểu về tính đóng gói - Bảo vệ dữ liệu quan trọng.

**Yêu cầu**:

1.  Tạo class `TrainingLogger`.
2.  Bên trong có một list ẩn `__logs` (dấu 2 gạch dưới để private, không cho bên ngoài sửa trực tiếp).
3.  Viết hàm `add_log(epoch, loss)`: Chỉ cho phép thêm log nếu `loss` là số dương. Nếu loss âm thì in ra cảnh báo "Lỗi dữ liệu" và không thêm.
4.  Viết hàm `get_average_loss()`: Tính trung bình cộng của tất cả loss đang có trong `__logs`.
5.  Thử tạo object, thêm vài log (cả đúng và sai), sau đó tính trung bình.

---

## Bài 3: Quản lý thư mục ảnh (Static Method)

**Mục tiêu**: Hiểu `staticmethod` - Hàm tiện ích không cần tạo object cũng dùng được.

**Yêu cầu**:

1.  Tạo class `PathUtils`.
2.  Viết một hàm `is_image_file(filename)` nhận vào tên file (string).
    - Trả về `True` nếu đuôi file là .jpg, .png, .jpeg.
    - Trả về `False` nếu là đuôi khác (.txt, .pdf).
    - Đánh dấu hàm này là `@staticmethod`.
3.  Viết chương trình chính: Có một list tên file `["anh1.jpg", "bao_cao.txt", "meo.png"]`. Dùng `PathUtils.is_image_file` để lọc ra chỉ lấy tên ảnh.
    - _Lưu ý_: Không cần `p = PathUtils()` mà gọi trực tiếp `Class.ham()`.

---

## Bài 4: Hệ thống biến hình ảnh (Magic method `__call__` nâng cao)

**Mục tiêu**: Hiểu sâu hơn về `__call__` - Biến object thành hàm (Rất hay dùng trong PyTorch Transforms).

**Yêu cầu**:

1.  Tạo class `ResizeImage`.
    - `__init__` nhận vào `width`, `height`.
2.  Viết hàm `__call__(self, image_name)`:
    - In ra: "Đang resize ảnh [image_name] về kích thước [width]x[height]".
3.  Tạo class `GrayScale`:
    - `__init__` không nhận gì cả.
    - `__call__(self, image_name)`: In ra "Đang chuyển ảnh [image_name] sang trắng đen".
4.  Tạo một list các bước xử lý: `pipeline = [ResizeImage(224, 224), GrayScale()]`.
5.  Dùng vòng lặp chạy qua `pipeline` để xử lý file "avatar.jpg".

---

## Bài 5: Custom Exception (Tự tạo lỗi)

**Mục tiêu**: Học cách thông báo lỗi chuyên nghiệp.

**Yêu cầu**:

1.  Tạo class `LowQualityImageError` kế thừa từ `Exception`.
2.  Tạo class `ImageChecker`.
    - Hàm `check(resolution)`: Nhận vào độ phân giải (ví dụ 720).
    - Nếu `resolution < 1080`: Dùng lệnh `raise LowQualityImageError("Ảnh quá mờ để train AI!")`.
    - Nếu tốt: In ra "Ảnh đạt chuẩn".
3.  Viết khối `try...except` để bắt lỗi này.
    - Thử check với 720 (phải bắt được lỗi và in ra "Bỏ qua ảnh này").
    - Thử check với 4000 (thành công).

---

👉 **Lời khuyên**: Hãy tạo file `bai_tap_oop_nang_cao.py` và giải từng bài một. Kẹt chỗ nào hãy hỏi ngay!
