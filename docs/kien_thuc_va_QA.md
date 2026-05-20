# KIẾN THỨC DỰ ÁN HolmHz — Chuẩn bị nghiệm thu

## PHẦN 1: TÓM TẮT KIẾN THỨC CỐT LÕI

### 1. Bài toán
- Phát hiện ảnh tổng hợp (Synthetic Image Detection): phân loại ảnh Real/Fake
- Bao phủ cả GAN (StyleGAN, ProGAN) và Diffusion (DALL-E, Midjourney, Stable Diffusion)
- Tích hợp XAI (Grad-CAM) để giải thích kết quả

### 2. Dữ liệu — Dataset v2
- **35.454 ảnh** từ 5 nguồn Kaggle, 8+ loại generator
- Train: 28.220 | Val: 3.526 | Test ID: 3.526 | Test OOD: 182
- OOD dùng nguồn Camera vs AI — hoàn toàn không có trong training
- Tỷ lệ Real/Fake ≈ 1:1 ở mọi split

### 3. Kiến trúc mô hình
- Pipeline: Input 224×224 → Backbone (pretrained ImageNet) → Dropout(0,3) → Linear(1) → Sigmoid → P(Fake)
- 4 backbone: EfficientNet-B0 (4M), ResNet-18 (11M), ViT-Small/16 (22M), Swin-Tiny (28M)
- Loss: BCEWithLogitsLoss, Optimizer: AdamW (lr=3×10⁻⁴), Scheduler: Cosine Annealing
- Early Stopping patience=7, seed=42

### 4. Kết quả chính
| Mô hình | ID AUC | OOD AUC | Tham số |
|---------|--------|---------|---------|
| EfficientNet-B0 v9 | **0,998** | **0,896** | 4M |
| ResNet-18 | 0,995 | 0,865 | 11M |
| ViT-Small/16 | 0,974 | 0,833 | 22M |
| Swin-Tiny | 0,620 | 0,811* | 28M |

*Swin-Tiny: training thất bại (best epoch=0), OOD AUC phản ánh pretrained weights

### 5. So sánh baseline (zero-shot, không retrain)
| Baseline | ID AUC | OOD AUC |
|----------|--------|---------|
| CNNDetection (ResNet-50, 23M) | 0,662 | 0,325 |
| UniversalFakeDetect (CLIP, 304M) | 0,722 | 0,486 |
| DeepfakeBench (EfficientNet-B4, 19M) | 0,439 | 0,536 |

### 6. Phát hiện quan trọng: JPEG Augmentation
- v7 (không JPEG aug): OOD AUC = 0,440
- v9 (JPEG quality 50-95, p=0,7): OOD AUC = 0,896 (+0,456)
- Lưu ý: single seed, chưa ablation tách biệt

### 7. Web Demo
- Framework: Gradio + EfficientNet-B0 ONNX + EXIF Analyzer
- Latency: ~1,5s/ảnh trên CPU
- Tính năng: Upload → Real/Fake (%) + Grad-CAM heatmap + EXIF metadata

### 8. KPI: Đạt 5/5
| KPI | Mục tiêu | Đạt |
|-----|----------|-----|
| Dataset ≥ 20.000 | 20.000 | 28.220 (141%) |
| ID AUC ≥ 0,92 | 0,92 | 0,998 |
| ID Acc ≥ 90% | 90% | 98,4% |
| OOD AUC ≥ 0,85 | 0,85 | 0,896 |
| Web demo ≤ 2s | 2s | ~1,5s |

---

## PHẦN 2: CÁC KHÁI NIỆM CẦN NẮM VỮNG

### GAN vs Diffusion
- **GAN**: Generator + Discriminator đối kháng. Up-sampling tạo spectral artifacts → dễ phát hiện bằng frequency analysis
- **Diffusion**: Thêm nhiễu T bước → học khử nhiễu ngược. Không có up-sampling → không có spectral artifacts → khó phát hiện hơn

### Transfer Learning
- Tái sử dụng weights từ ImageNet (1,2M ảnh, 1000 lớp)
- Full fine-tuning: unfreeze toàn bộ backbone vì dataset đủ lớn (28.220)
- Lý do: bài toán phát hiện ảnh AI cần điều chỉnh cả đặc trưng cấp thấp

### EfficientNet-B0
- Tìm bởi NAS (Neural Architecture Search), không thiết kế thủ công
- Compound Scaling: scale đồng thời depth/width/resolution
- MBConv + Squeeze-and-Excitation → 4M params nhưng hiệu quả bằng ResNet-50 (26M)

### Grad-CAM
- Gradient-weighted Class Activation Mapping
- Lấy gradient tại lớp conv cuối → tạo heatmap chỉ vùng ảnh hưởng quyết định
- Ảnh Fake: tập trung vào mặt/tóc/mắt. Ảnh Real: phân tán đều

### AUC vs Accuracy
- AUC: đánh giá ở mọi threshold, không phụ thuộc ngưỡng cố định
- AUC=1,0: hoàn hảo. AUC=0,5: random. AUC<0,5: phản-tương quan
- Phù hợp hơn Accuracy khi dữ liệu không cân bằng

### ONNX
- Open Neural Network Exchange: format xuất model cho inference nhanh
- Loại bỏ overhead Python/PyTorch, hỗ trợ quantization INT8
- Giảm latency từ ~3s xuống ~1,5s trên CPU

---

## PHẦN 3: CÂU HỎI HỘI ĐỒNG DỰ KIẾN VÀ CÂU TRẢ LỜI

### Q1: Tại sao mô hình nhỏ nhất (4M) lại tốt nhất?
**Trả lời:** Ba lý do:
1. **Inductive bias CNN** phù hợp dataset nhỏ-trung (28K ảnh). Transformer (ViT, Swin) cần hàng triệu ảnh mới vượt CNN.
2. **EfficientNet được NAS tối ưu** — mỗi tham số được tận dụng tối đa nhờ Compound Scaling và MBConv.
3. **JPEG Augmentation** buộc model học đặc trưng bền vững thay vì dựa vào artifacts — yếu tố có ảnh hưởng lớn nhất đến OOD.

### Q2: Tập OOD 182 ảnh có đủ để kết luận không?
**Trả lời:** Chúng tôi thừa nhận đây là hạn chế. Với n=182, khoảng tin cậy 95% (Wilson score) là ±6%. Vì vậy báo cáo ghi rõ: "kết luận OOD mang tính chỉ báo sơ bộ, cần tập OOD ≥ 1.000 ảnh để xác nhận". Tuy nhiên, xu hướng rõ ràng: cả 3 baseline đều AUC < 0,54, trong khi HolmHz đạt 0,896 — khoảng cách quá lớn để giải thích bằng sai số thống kê.

### Q3: Swin-Tiny thất bại — có phải kiến trúc Swin kém?
**Trả lời:** Không. Đây là **hạn chế thiết kế thí nghiệm**, không phải hạn chế kiến trúc. LR=3×10⁻⁴ quá cao cho Transformer — Swin cần lr=5×10⁻⁵, warmup 3 epochs, layer-wise LR decay 0,65. Chúng tôi dùng cùng hyperparameters cho cả 4 model để đảm bảo công bằng, nhưng điều này bất lợi cho Transformer. Best epoch=0 nghĩa là pretrained weights ban đầu tốt hơn mọi epoch fine-tune — gradient quá lớn phá hủy attention patterns đã học.

### Q4: JPEG augmentation lấy cảm hứng từ đâu? Có phải tự nghĩ ra?
**Trả lời:** Lấy cảm hứng từ CNNDetection (Wang et al., CVPR 2020) — họ dùng blur + JPEG augmentation để phát hiện GAN. Chúng tôi áp dụng vào bối cảnh Diffusion với dải quality 50-95 (bao phủ Facebook 80-85, Instagram 85-90, WhatsApp 60-75). Đóng góp của đề tài là **xác nhận kỹ thuật này hiệu quả trên Diffusion** — OOD AUC tăng từ 0,440 lên 0,896. Tuy nhiên cần lưu ý: chưa ablation tách biệt JPEG khỏi các augmentation khác.

### Q5: So sánh baseline có công bằng không? Họ zero-shot, các bạn retrain.
**Trả lời:** Đúng, đây là so sánh không hoàn toàn đối xứng và báo cáo đã ghi rõ điều này. Baselines dùng zero-shot (pretrained weights gốc), không retrain trên Dataset v2. Mục đích: kiểm tra xem các phương pháp hiện có có **hoạt động được ngay** trên dữ liệu Diffusion không — câu trả lời là không (AUC < 0,73). Đây chính là khoảng trống nghiên cứu mà đề tài giải quyết. Nếu retrain baselines trên cùng dataset, kết quả có thể khác — nhưng nằm ngoài phạm vi đề tài.

### Q6: Grad-CAM có thực sự giải thích được mô hình?
**Trả lời:** Grad-CAM là công cụ minh họa **định tính** (qualitative), không phải bằng chứng định lượng. Nó cho thấy vùng ảnh ảnh hưởng nhiều nhất đến quyết định, nhưng không giải thích chính xác mô hình dùng đặc trưng nào. Báo cáo đã ghi nhận hạn chế này và đề xuất dùng LIME, SHAP trong hướng phát triển. Tuy nhiên, Grad-CAM vẫn có giá trị thực tiễn: giúp người dùng cuối hiểu "tại sao" — tốt hơn nhiều so với hộp đen.

### Q7: Tại sao đề tài dùng CNN mà không dùng Transformer?
**Trả lời:** Đề tài **có dùng Transformer** — ViT-Small/16 và Swin-Tiny là Transformer. Tuy nhiên, kết quả cho thấy CNN (EfficientNet-B0, ResNet-18) tốt hơn trên dataset này vì: (1) dataset 28K ảnh quá nhỏ cho Transformer, (2) CNN có inductive bias phù hợp cho ảnh (locality, translation invariance). Transformer mạnh khi có hàng triệu mẫu — với dataset nhỏ, CNN là lựa chọn thực tiễn hơn.

### Q8: Hệ thống có phát hiện được video deepfake không?
**Trả lời:** Hiện tại chỉ xử lý ảnh tĩnh. Để phát hiện video, cần thêm phân tích temporal consistency (sự nhất quán giữa các frame liên tiếp) — đây là hướng phát triển #1 trong báo cáo. Tuy nhiên, hệ thống hiện tại có thể trích xuất từng frame từ video và phân tích riêng lẻ — chỉ là chưa tối ưu.

### Q9: Chi phí triển khai thực tế là bao nhiêu?
**Trả lời:** Kiến trúc AWS đề xuất (Serverless): < $5/tháng cho ~1.000 request/ngày. Lambda tính tiền theo request (không tốn tiền khi idle), S3 Lifecycle tự xóa heatmap sau 24h, CloudFront cache giảm số lần gọi Lambda. Với Gradio local: hoàn toàn miễn phí, chỉ cần laptop có CPU.

### Q10: Dữ liệu 5 nguồn Kaggle có trùng lặp không?
**Trả lời:** Báo cáo thừa nhận đây là hạn chế: chưa thực hiện deduplication bằng perceptual hashing hoặc feature-level matching. Các nguồn Kaggle độc lập nên nguy cơ trùng lặp tồn tại nhưng không cao (khác chủ đề: RVF10K là khuôn mặt CelebA, DeepDetect-2025 là ảnh đa dạng, CIPLab là face manipulation). Hướng phát triển: cần chạy perceptual hash để xác nhận.

### Q11: Tại sao chọn AUC làm chỉ số chính thay vì Accuracy?
**Trả lời:** AUC đánh giá khả năng phân biệt ở **mọi ngưỡng** (threshold-free), không phụ thuộc vào việc chọn threshold 0,5 hay 0,7. Accuracy chỉ đo tại 1 ngưỡng cố định, dễ bị ảnh hưởng bởi class imbalance. Ví dụ: nếu model predict tất cả là Real, accuracy vẫn đạt ~50% nhưng AUC sẽ thấp. Đây là tiêu chuẩn trong lĩnh vực deepfake detection (Wang et al., Ojha et al. đều dùng AUC).

### Q12: WeightedRandomSampler hoạt động thế nào?
**Trả lời:** Mỗi source (rvf10k, dd2025, ciplab...) được gán weight = max_count / source_count. Source ít ảnh (camera: 218) được gán weight cao hơn source nhiều ảnh (rvf10k: 8.000). Kết quả: mỗi epoch, tất cả sources được sample cân bằng mà không cần duplicate dữ liệu vật lý — hiệu quả bộ nhớ tốt hơn oversampling.

### Q13: Có thể bypass hệ thống bằng adversarial attack không?
**Trả lời:** Có khả năng. Hệ thống chưa được kiểm tra adversarial robustness. Kẻ gian có thể thêm nhiễu nhỏ (imperceptible perturbation) vào ảnh AI để lừa detector. Đây là hạn chế cần giải quyết trước khi triển khai sản xuất — đặc biệt trong ứng dụng bảo mật như eKYC. Hướng giải quyết: adversarial training, randomized smoothing.

### Q14: EXIF Analyzer trong web demo hoạt động thế nào?
**Trả lời:** EXIF (Exchangeable Image File Format) chứa metadata ảnh: loại camera, cài đặt (ISO, aperture, shutter speed), GPS, timestamp, phần mềm chỉnh sửa. Ảnh chụp bằng camera thật thường có EXIF đầy đủ, trong khi ảnh AI-generated thường thiếu EXIF hoặc có EXIF bất thường (không có thông tin camera thật). EXIF Analyzer là lớp phân tích bổ sung — không thay thế mô hình CNN mà kết hợp để tăng độ tin cậy.

### Q15: Tên đề tài ghi CNN nhưng có dùng cả Transformer, có mâu thuẫn?
**Trả lời:** ViT và Swin được đưa vào với mục đích **đối chứng kiến trúc** (architectural comparison), không phải đóng góp chính. Đóng góp chính tập trung vào nhóm CNN: EfficientNet-B0 và ResNet-18. Tên đề tài phản ánh đúng phương pháp chính (CNN), còn Transformer là thực nghiệm bổ sung để so sánh. Nếu muốn tên bao quát hơn: "Xây dựng hệ thống phát hiện ảnh tổng hợp sử dụng học sâu và giải thích mô hình".
