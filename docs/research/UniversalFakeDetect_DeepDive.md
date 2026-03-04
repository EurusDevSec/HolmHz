# 🔍 Deep Dive: UniversalFakeDetect - Sức mạnh của Foundation Models

> **Dành cho**: DevOps Engineer / AI Researcher (HolmHz Team)
> **Mục đích**: Giải mã tại sao một file weight siêu nhẹ (chỉ vài KB) lại có thể đánh bại các model khổng lồ về khả năng phát hiện ảnh Fake.
> **Dự án gốc**: [UniversalFakeDetect (Ojhal et al., CVPR 2023)](https://github.com/Yuheng-Li/UniversalFakeDetect)

---

## 1. Sự thật ngỡ ngàng: "Gã khổng lồ đứng trên vai người khổng lồ"

Khi bạn chạy thử dự án này, bạn có thấy điều lạ lùng không?

1.  File weight `fc_weights.pth` bạn dùng để chạy **cực kỳ nhẹ** (chỉ vài MB so với hàng trăm MB của ResNet).
2.  Nhưng mỗi lần chạy, nó lại download một cái gì đó gần **1GB** từ OpenAI.

**✨ Bí mật nằm ở đây:**
Dự án này **KHÔNG** train một model deepfake từ đầu.
Nó sử dụng **CLIP** (Contrastive Language-Image Pre-Training) của OpenAI - một model đã được train trên 400 triệu cặp ảnh-văn bản từ internet. Nó đã "nhìn" thấy cả thế giới rồi.

- **Chiến thuật**: Lấy bộ não thiên tài của CLIP để "nhìn" ảnh, sau đó chỉ dạy thêm 1 lớp neuron đơn giản ở cuối để phán quyết: "Cái này là Fake hay Real?".
- **Kết quả**: Nó phát hiện được cả những loại Deepfake mà nó **chưa từng được học** (Zero-shot/Few-shot generalization).

---

## 2. Giải phẫu cấu trúc (Architecture Anatomy)

Khác với `ResNet` truyền thống (train tất cả các lớp), kiến trúc ở đây chia làm 2 phần rõ rệt:

```python
# File: models/clip_models.py (Đã đơn giản hóa)

class CLIPModel(nn.Module):
    def __init__(self, name):
        # Phần 1: BACKBONE (Của OpenAI - Freeze, không học nữa)
        # Đây là "Gã khổng lồ". Nó chuyển ảnh thành features (vector số).
        self.model, self.preprocess = clip.load(name, device="cpu")

        # Phần 2: CLASSIFIER (Của chúng ta - Train cái này)
        # Chỉ là 1 lớp Linear đơn giản nối từ Backbone ra output.
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        # B1: Nhờ CLIP nhìn ảnh -> Ra đặc trưng (features)
        features = self.model.encode_image(x)

        # B2: Lớp Linear phán quyết dựa trên đặc trưng đó
        return self.fc(features)
```

### 💡 Bài học cho HolmHz:

> **Transfer Learning** là chìa khóa. Không nhất thiết phải train lại cả model EfficientNet. Có thể thử phương án: Load EfficientNet đã train trên ImageNet -> Freeze lại -> Chỉ train 1-2 lớp cuối. Tiết kiệm tài nguyên và thời gian huấn luyện cực lớn.

---

## 3. Quy trình xử lý ảnh (Pipeline khác biệt)

Nếu bạn copy code xử lý ảnh từ CNNDetection sang đây, model sẽ chạy sai ngay lập tức! Tại sao?

### Preprocessing: "Nhập gia tùy tục"

Mỗi model "lớn" (Backbone) đều có một cách nhìn thế giới riêng.

- **ResNet/EfficientNet (ImageNet)**: Thích chuẩn hóa kiểu `{mean: [0.485...], std: [0.229...]}`.
- **CLIP (OpenAI)**: Thích chuẩn riêng của nó.

Trong code `validate.py`:

```python
MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip":     [0.48145466, 0.4578275, 0.40821073] # <-- Số lạ hoắc!
}
```

**⚠️ Cảnh báo:** Nếu HolmHz sau này định switch giữa các backbone (lúc dùng EfficientNet, lúc dùng CLIP), hãy chắc chắn hệ thống Preprocessing pipeline tự động đổi thông số `mean/std` tương ứng. Sai cái này là Accuracy tụt không phanh.

---

## 4. Tại sao nó lại tốt hơn ResNet ở OOD (Out-of-Distribution)?

- **CNNDetection (ResNet)**: Học vẹt các "dấu vết" cụ thể của ProGAN (tập train). Gặp loại Deepfake mới (như Stable Diffusion), nó ngơ ngác vì "dấu vết" này không giống cái nó đã học.
- **UniversalFakeDetect (CLIP)**: CLIP được học để hiểu **nội dung** và **cấu trúc** ngữ nghĩa của ảnh. Nó nhận ra sự bất thường ở mức độ cao hơn (high-level features) chứ không chỉ chăm chăm soi từng pixel.

> **Ví dụ**:
>
> - ResNet thấy "có vân lạ ở góc ảnh" -> FAKE.
> - CLIP thấy "cấu trúc khuôn mặt này có gì đó vô lý so với hàng triệu mặt người thật tao từng thấy" -> FAKE.

---

## 5. Reality Check: Khi "Universal" gặp đối thủ 2025 (Flux, Gemini)

**Thực tế phũ phàng**:
Khi test thực tế (10/02/2026) với các ảnh sinh bởi **Flux.1** (SOTA 2024-2025) và **Gemini** (Google), model này cho kết quả **thất bại thảm hại** (Fake Probability < 10% - tức là phán thành Real).

**(Xem log chi tiết trong file RUN_EXISTING_PROJECTS.md)**

### Tại sao lại thất bại?

1.  **Training Data cũ (Pre-2020)**: File weights `fc_weights.pth` được train chủ yếu trên dataset của Wang2020 (chủ yếu là GANs như ProGAN, StyleGAN).
2.  **Khoảng cách công nghệ (Domain Gap)**:
    - **GANs**: Để lại các mẫu bàn cờ (checkerboard artifacts) do lớp Deconvolution.
    - **Diffusion (Flux/SDXL)**: Sinh ảnh bằng cách khử nhiễu (denoising). Nhiễu của nó giống "hạt ảnh film" (grain) tự nhiên hơn nhiều nên model CLIP+Linear không bắt được nếu chỉ được học trên GAN.
3.  **Linear Probe quá đơn giản**: Một lớp Linear duy nhất có thể phân tách tốt GAN vs Real trong không gian CLIP, nhưng Flux vs Real có thể nằm trộn lẫn vào nhau phức tạp hơn mà 1 đường thẳng (Linear) không cắt ngăn được.

---

## 6. Tổng kết & Áp dụng cho HolmHz

### Những điểm "chôm" được:

1.  **Fine-tuning Strategy**: Chiến lược khóa (freeze) backbone và chỉ train lớp cuối (`model.fc`). Đây là cách nhanh nhất để HolmHz có MVP (Minimum Viable Product).
2.  **CLIP as a Service**: Nếu model EfficientNet của chúng ta thất bại với ảnh Midjourney/DALL-E 3 mới ra, hãy nhớ đến CLIP như một "cứu cánh" (Fallback solution).
3.  **Code Structure của `validate.py`**: Họ viết code validation rất kỹ, hỗ trợ `find_best_threshold` (tìm ngưỡng confidence tốt nhất thay vì mặc định 0.5). HolmHz nên học hàm này để tối ưu F1-score.

### Bài học xương máu:

**Không có "Universal" mãi mãi.** Model 2023 không thể bắt được ảnh Fake 2025 (Flux, Gemini) nếu không được update.
-> **HolmHz Action Plan**: Chúng ta bắt buộc phải đưa **GenImage** (tập dữ liệu Diffusion) vào train và tự tạo thêm dữ liệu **Flux/SDXL** mới nhất, đừng tin tưởng hoàn toàn vào Pretrained weights.

### Kế hoạch tiếp theo:

Bạn đã nắm được 2 thái cực:

- **Cổ điển & Chuyên sâu**: CNNDetection (Soi pixel).
- **Hiện đại & Tổng quát**: UniversalFakeDetect (Dùng kiến thức có sẵn).

Dự án HolmHz của chúng ta sẽ đi con đường trung đạo: **EfficientNet-B0**. Nó nhẹ hơn CLIP (chạy nhanh hơn trên web/mobile) nhưng kiến trúc hiện đại hơn ResNet. Hãy chuẩn bị tinh thần để bắt đầu xây dựng HolmHz từ con số 0 trong Sprint tới!
