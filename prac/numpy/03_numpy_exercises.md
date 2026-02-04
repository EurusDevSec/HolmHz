# 🔢 NumPy: Luyện tập chuyên sâu (5 Bài tập thực tế Computer Vision)

Các bài tập này mô phỏng chính xác những gì bạn sẽ làm khi xử lý dữ liệu cho HolmHz.

---

## Bài 1: Normalization (Chuẩn hóa dữ liệu)

**Tình huống**: Trong Data Science, dữ liệu đầu vào thường chênh lệch nhau (vd: tuổi 20-80, lương 5tr-100tr). Neural Network học rất tệ nếu số liệu không cùng thang đo. Ta cần đưa tất cả về khoảng 0-1.

**Yêu cầu**:

1.  Tạo một mảng numpy `data` gồm 10 số ngẫu nhiên từ 0 đến 1000.
2.  Tìm giá trị `min` và `max` của mảng.
3.  Áp dụng công thức Min-Max Scaling cho cả mảng:
    $$X_{new} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
4.  In kết quả ra (Tất cả các số phải nằm trong khoảng 0 đến 1).

---

## Bài 2: Color Channel Splitting (Tách kênh màu)

**Tình huống**: Bạn muốn phân tích riêng kênh màu Đỏ của bức ảnh xem nó có gì lạ không (Deepfake thường để lại dấu vết lạ ở các kênh màu riêng biệt).

**Yêu cầu**:

1.  Tạo một "ảnh giả" kích thước `(100, 100, 3)` bằng số ngẫu nhiên `np.random.randint(0, 256, ...)` (Kiểu dữ liệu `uint8`).
2.  Tách riêng 3 kênh màu ra 3 biến: `red_channel`, `green_channel`, `blue_channel`.
    - Gợi ý: Dùng slicing `[:, :, 0]`, `[:, :, 1]`, ...
3.  Làm kênh màu Đỏ sáng lên 50 đơn vị (nhớ không cho vượt quá 255 bằng `np.clip` hoặc thủ thuật khác).
4.  Gộp lại thành bức ảnh mới `anh_moi`.
    - Gợi ý: Dùng `np.stack((r, g, b), axis=2)`.

---

## Bài 3: Image Masking (Cắt nền xanh)

**Tình huống**: Kỹ thuật "Phông xanh" (Green Screen). Ta muốn tách vật thể ra khỏi nền.

**Yêu cầu**:

1.  Tạo một ảnh `(50, 50)` toàn màu xanh lá cây `[0, 255, 0]`.
2.  Vẽ một hình vuông màu đỏ `[255, 0, 0]` ở giữa ảnh (đóng vai người mẫu).
3.  Tạo một `mask` (mặt nạ) logic: Tìm tất cả các điểm ảnh MÀU XANH LÁ.
    - _Khó_: Màu xanh lá là `[0, 255, 0]`. Bạn cần so sánh cả 3 kênh. `(img[:,:,0] == 0) & (img[:,:,1] == 255) & (img[:,:,2] == 0)`.
4.  Dùng mask này để đổi hết nền xanh thành màu đen `[0, 0, 0]`.

---

## Bài 4: Data Augmentation (Lật ảnh thủ công)

**Tình huống**: Để AI học tốt hơn, ta cần tạo ra nhiều biến thể của ảnh (nghiêng, lật, xoay). Hãy tự viết code lật ảnh mà không dùng thư viện có sẵn.

**Yêu cầu**:

1.  Tạo một mảng 2 chiều `(4, 4)` với các số từ 1 đến 16 (để dễ nhìn vị trí).
    ```
    [[ 1,  2,  3,  4],
     [ 5,  6,  7,  8],
     ...]
    ```
2.  Thực hiện **Lật Ngang** (Horizontal Flip): Cột trái sang phải, phải sang trái.
    - Gợi ý: Dùng slicing với bước nhảy âm `[:, ::-1]`.
3.  Thực hiện **Lật Dọc** (Vertical Flip): Hàng trên xuống dưới.
    - Gợi ý: Dùng slicing `[::-1, :]`.

---

## Bài 5: Tính Mean Square Error (MSE) thủ công

**Tình huống**: Tự tay viết hàm tính sai số (Loss Function) giữa 2 bức ảnh. Đây là nền tảng của Deep Learning.

**Yêu cầu**:

1.  Tạo `anh_that`: mảng `(10, 10)` toàn số 100.
2.  Tạo `anh_du_doan`: mảng `(10, 10)` toàn số 110 (sai mười đơn vị).
3.  Tính MSE theo công thức:
    - Lấy `anh_that` TRỪ `anh_du_doan`.
    - Bình phương lên (để mất dấu âm).
    - Tính trung bình cộng của tất cả các ô.
4.  Kết quả phải ra 100. (Vì lệch 10, bình phương là 100, trung bình vẫn là 100).

---

👉 **Lời khuyên**: Hãy tạo file `bai_tap_numpy_nang_cao.py` hoặc notebook để làm. Bài 3 khá khó về logic mask, hãy kiên nhẫn!
