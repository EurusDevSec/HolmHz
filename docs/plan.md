TRƯỜNG ĐẠI HỌC THỦ DẦU MỘT
VIỆN CÔNG NGHỆ SỐ

THUYẾT MINH ĐỀ CƯƠNG

ĐỀ TÀI NGHIÊN CỨU KHOA HỌC CỦA SINH VIÊN
NĂM HỌC 2025-2026

XÂY DỰNG HỆ THỐNG PHÁT HIỆN ẢNH TỔNG HỢP BẰNG MẠNG NƠ-RON TÍCH CHẬP (CNN)

Nhóm SV: 1. Lê Văn Hoàng (Nhóm trưởng) 2. Ngô Huỳnh Bảo Luân

GVHD: ThS. Nguyễn Trung Kiệt

Thành phố Hồ Chí Minh, ngày 11 tháng 11 năm 2025

 
UBND THÀNH PHỐ HỒ CHÍ MINH
TRƯỜNG ĐẠI HỌC THỦ DẦU MỘT

THUYẾT MINH ĐỀ TÀI
NGHIÊN CỨU KHOA HỌC SINH VIÊN
Năm học 2025-2026

1. Tên đề tài: Xây dựng hệ thống phát hiện ảnh tổng hợp bằng Mạng nơ-ron tích chập (CNN)

2. Loại hình nghiên cứu:  Cơ bản  Ứng dụng  Triển khai

3. Lĩnh vực nghiên cứu:
    Khoa học Xã hội và Nhân văn  Khoa học Kỹ thuật và Công nghệ
    Kinh tế  Khoa học Tự nhiên
    Khoa học Giáo dục

4. Thời gian thực hiện: 7 tháng
   Từ tháng 11 năm 2025 đến tháng 05 năm 2026

5. Đơn vị quản lý về chuyên môn:
   Đơn vị: Viện công nghệ số Khoa/Chương trình: Công nghệ thông tin

6. Giáng viên hướng dẫn:
   Họ và tên: Nguyễn Trung Kiệt Học vị: ThS
   Đơn vị công tác: Viện Công nghệ số - Trường Đại học Thủ Dầu Một  
   Địa chỉ nhà riêng: 42/2 Huỳnh Thị Tươi, Tân Thắng, Tân Đông Hiệp, thành phố Hồ Chí Minh
   Di động: 0847754828 E-mail: kietnt@tdmu.edu.vn

7. Sinh viên/Nhóm sinh viên thực hiện đề tài:  
   Các thành viên tham gia đề tài (không quá 05 sinh viên):

TT Họ và tên MSSV Lớp Ghi chú
1 Lê Văn Hoàng 2224802010279 D22CNTT02 Nhóm trưởng
2 Ngô Huỳnh Bảo Luân 2524802010327 D25CNTT10 Thành viên
Thông tin SV chịu trách nhiệm chính:
Họ và tên: Lê Văn Hoàng
SĐT: 0399354603
Email: 2224802010279@student.tdmu.edu.vn

8. Tính cấp thiết của đề tài:

Sự bùng nổ của các mô hình Trí tuệ nhân tạo tạo sinh (Generative AI) đã tạo ra những hình ảnh tổng hợp (synthetic images) với độ chân thực cực cao, gần như không thể phân biệt bằng mắt thường. Tình trạng này đang bị lạm dụng nghiêm trọng cho các mục đích xấu như tạo tin giả (fake news), lừa đảo trực tuyến, bôi nhọ danh dự và thao túng dư luận, gây ra các lo ngại lớn về an ninh và thông tin sai lệch.

Trong bối cảnh đó, việc xây dựng các công cụ phát hiện tự động là vô cùng cấp thiết. Mặc dù đã có nhiều nghiên cứu sử dụng Mạng nơ-ron tích chập (CNN) đạt kết quả cao, một thách thức lớn và cấp thiết hiện nay là khả năng tổng quát hóa (generalization). Rất nhiều mô hình được huấn luyện tốt trên các bộ dữ liệu cũ (ảnh GAN) nhưng lại thất bại khi phát hiện ảnh từ các mô hình Diffusion thế hệ mới.

Đề tài này có tính cấp thiết cao vì nó giải quyết trực tiếp "khoảng trống" đó: nghiên cứu và xây dựng một hệ thống có khả năng cập nhật, phát hiện hiệu quả các ảnh tổng hợp từ cả công nghệ cũ và mới, góp phần cung cấp một công cụ kiểm chứng đáng tin cậy.

Hơn nữa, các hệ thống hiện tại thường hoạt động như một 'hộp đen' (black-box), chỉ trả về kết quả đúng/sai mà không chỉ ra được vùng bất thường trên ảnh. Do đó, việc nghiên cứu tích hợp khả năng Giải thích mô hình (Explainable AI) để trực quan hóa các vùng bị can thiệp (ví dụ: vùng mắt, răng, tóc) là bước tiến cần thiết để thuyết phục người dùng và nâng cao độ tin cậy của hệ thống.

Tại Việt Nam, vấn nạn lừa đảo trực tuyến sử dụng Deepfake đang gia tăng đáng báo động (Theo thông tin từ BỘ KHOA HỌC VÀ CÔNG NGHỆ). Các đối tượng xấu sử dụng hình ảnh/video giả mạo người thân để lừa chuyển tiền hoặc bôi nhọ danh dự trên mạng xã hội. Do đó, đề tài này không chỉ mang tính học thuật mà còn hướng tới ứng dụng thực tiễn, cung cấp một công cụ hỗ trợ kiểm chứng bước đầu (không cam kết phát hiện mọi biến thể mới) cho cộng đồng và các đơn vị truyền thông trong nước

9. Mục tiêu đề tài:

Mục tiêu tổng quát: Xây dựng mô hình CNN nhẹ (transfer learning) phát hiện ảnh chân dung tổng hợp (GAN & Diffusion phổ biến) đạt AUC ≥ 0,92 trên tập kiểm tra nội bộ và AUC ≥ 0,85 trên tập ngoài miền; tích hợp Grad-CAM và web demo suy luận ≤ 2 giây/ảnh trên máy tính cá nhân.

• Mục tiêu cụ thể (KPI):

1. Dataset: tối thiểu 20.000 ảnh (50% thật, 50% giả; gồm ít nhất 3 nguồn GAN và 2 nguồn Diffusion), chia train/val/test ngoài miền rõ ràng.
2. Hiệu năng: Accuracy ≥ 90%, F1 ≥ 0,90 nội miền; AUC ≥ 0,85 ngoài miền (valid trên 2 bộ chưa thấy khi huấn luyện).
3. Tính bền vững: kiểm tra ảnh nén JPEG (q=60) và scale/crop nhẹ; giảm suy hao AUC ≤ 5%.
4. XAI: Grad-CAM hiển thị vùng nghi ngờ (mắt/da/tóc) với ví dụ minh họa trong báo cáo.
5. Web demo: upload ảnh 512×512, trả kết quả + heatmap; thời gian phản hồi ≤ 2s trên CPU laptop phổ thông.

(Các con số có thể điều chỉnh theo năng lực tính toán; quan trọng là có KPI định lượng.)

10. Tổng quan tình hình nghiên cứu liên quan đến đề tài:

Lĩnh vực phát hiện ảnh tổng hợp (Synthetic Image Detection) đã thu hút sự quan tâm lớn từ cộng đồng nghiên cứu quốc tế. Dựa trên các công bố khoa học gần đây, các phương pháp tiếp cận có thể được phân loại thành 04 nhóm chính:

10.1. Nhóm phương pháp dựa trên Mạng nơ-ron tích chập (CNN)
Đây là hướng tiếp cận nền tảng và phổ biến nhất. Các nghiên cứu tập trung khai thác khả năng trích xuất đặc trưng cục bộ của CNN để tìm ra các lỗi cấu trúc trong ảnh giả.
• Rössler et al. (2019) [1]: Giới thiệu bộ dữ liệu chuẩn FaceForensics++ và chứng minh mạng XceptionNet đạt độ chính xác rất cao (>99%) trên dữ liệu nén thấp. Tuy nhiên, hiệu suất giảm đáng kể khi gặp ảnh nén chất lượng thấp hoặc ảnh từ nguồn lạ.
• Wang et al. (2020) [2]: Trong công trình "CNN-generated images are surprisingly easy to spot", nhóm tác giả sử dụng ResNet-50 huấn luyện trên bộ dữ liệu ProGAN. Kết quả cho thấy khả năng tổng quát hóa tốt (Average Precision 100%) trên các mô hình GAN cũ, nhưng vẫn gặp khó khăn với các kiến trúc Diffusion mới gần đây.

10.2. Nhóm phương pháp dựa trên Transformer và Cơ chế Attention
Để khắc phục hạn chế về tầm nhìn cục bộ của CNN, các nghiên cứu mới áp dụng Vision Transformer (ViT).
• Wodajo et al. (2021) [3]: Đề xuất kiến trúc Convolutional Vision Transformer, kết hợp CNN và ViT. Mô hình đạt độ chính xác 91.5% trên bộ dữ liệu DeepfakeDetection (DFDC), cho thấy khả năng nắm bắt sự bất thường trong bối cảnh toàn cục (global context) tốt hơn CNN thuần túy.
• Cao et al. (2022) [4]: Sử dụng cơ chế Attention để tập trung vào các vùng ranh giới (blending boundary) trong ảnh ghép mặt, cải thiện đáng kể khả năng phát hiện các dạng deepfake thay đổi khuôn mặt.

10.3. Nhóm phương pháp dựa trên Phân tích miền tần số (Frequency Analysis)
Nhiều mô hình AI tạo sinh để lại các "dấu vết số" (artifacts) trong miền tần số và các đặc trưng nhiễu (noise residuals) mà mắt thường hoặc miền không gian (RGB) khó phát hiện.
• Frank et al. (2020) [5]: Nghiên cứu "Leveraging Frequency Analysis for Deep Fake Image Recognition" sử dụng biến đổi DCT (Discrete Cosine Transform). Kết quả thực nghiệm trên nhiều kiến trúc GAN khác nhau cho thấy các lỗi phổ tần số lặp lại bất thường, đạt Accuracy trên 90% với chi phí tính toán rất thấp. Tuy nhiên, đối với các mô hình Diffusion hiện đại, các phương pháp dựa trên Bộ lọc SRM (Spatial Rich Model) để trích xuất vân tay nhiễu (noise fingerprints) đang cho thấy hiệu quả vượt trội hơn so với phân tích phổ đơn thuần
• Durall et al. (2020) [6]: Đề xuất phương pháp phân tích phổ dựa trên DFT, chứng minh rằng các bước up-sampling (phóng to ảnh) trong mô hình tạo sinh làm mất đi các đặc trưng tần số cao của ảnh thật.

10.4. Nhóm phương pháp Hỗn hợp và Giải thích mô hình (Hybrid & XAI)
Đây là xu hướng mới nhằm tăng độ tin cậy. Thay vì chỉ đưa ra kết quả nhị phân, các nghiên cứu tích hợp XAI (Explainable AI).
• Tuy nhiên, đa số các nghiên cứu hiện tại (như các nhóm trên) thường hoạt động như một "hộp đen" (black-box). Các nghiên cứu tích hợp Grad-CAM để trực quan hóa vùng giả mạo vẫn còn hạn chế và chưa được tối ưu hóa cho các mô hình Diffusion thế hệ mới.

10.5. Bảng so sánh tổng hợp các nghiên cứu tiêu biểu
Nhóm phương pháp Nghiên cứu tiêu biểu Dữ liệu & Kết quả chính Ưu điểm Nhược điểm/Hạn chế
CNN (Spatial) Rössler et al. (2019)

Wang et al. (2020) Data: FaceForensics++, ProGAN

Acc: ~99% (nội miền) Tốc độ xử lý nhanh, dễ triển khai, độ chính xác nội miền cực cao. Dễ bị Overfitting, hiệu suất giảm mạnh khi gặp dữ liệu từ nguồn mới (Cross-dataset kém).
Transformer Wodajo et al. (2021) Data: DFDC

Acc: ~91.5% Nắm bắt tốt ngữ cảnh toàn cục của ảnh. Yêu cầu tài nguyên tính toán lớn, tốc độ suy luận chậm hơn CNN.
Frequency (Tần số) Frank et al. (2020) Data: StyleGAN, BigGAN

Acc: >90% Phát hiện được các lỗi cấu trúc ẩn, bền vững với nén ảnh. Đôi khi bỏ qua các lỗi ngữ nghĩa (ví dụ: mắt lệch) mà miền không gian thấy rõ.
Đề tài đề xuất (Đề tài này) Data: GAN + Diffusion (Midjourney/SD)

Target: AUC > 0.9 Kết hợp CNN + XAI + Dữ liệu cập nhật. Tập trung vào bài toán ứng dụng, không đề xuất kiến trúc mạng mới hoàn toàn.
10.6. Khoảng trống nghiên cứu (Research Gap)
Mặc dù đã có nhiều thành tựu, tổng quan tài liệu cho thấy tồn tại những khoảng trống rõ rệt mà đề tài này tập trung giải quyết:

1. Thiếu khả năng tổng quát hóa trên Diffusion Models: Đa số các nghiên cứu kinh điển (2019-2021) tập trung vào GAN (như DeepFakes, Face2Face). Hiện tại thiếu các đánh giá chuyên sâu trên các mô hình Diffusion mới (Midjourney v6, Flux) đang bùng nổ.
2. Thiếu tính minh bạch (XAI): Người dùng cuối cần hiểu "tại sao ảnh này là giả". Các mô hình hiện tại thiếu tích hợp sẵn cơ chế giải thích trực quan (như Heatmap) trên giao diện người dùng.
3. Nhu cầu về mô hình nhẹ: Các mô hình Transformer quá nặng để triển khai trên thiết bị cá nhân phổ thông.

=> Đề tài tập trung vào việc: Xây dựng một quy trình (pipeline) dữ liệu cập nhật (GAN & Diffusion), sử dụng mô hình CNN nhẹ kết hợp phân tích tần số để tối ưu hóa khả năng cross-dataset, đồng thời tích hợp XAI để minh bạch hóa kết quả. 11. Đối tượng, phạm vi nghiên cứu, cách tiếp cận và phương pháp nghiên cứu:

11.1. Đối tượng nghiên cứu
• Các kiến trúc Mạng nơ-ron tích chập (CNN): Tập trung vào các kiến trúc hiện đại (State-of-the-Art) phù hợp cho bài toán phân loại ảnh.
• Các đặc trưng nhận dạng:
o Đặc trưng miền không gian (Spatial domain): Các bất thường về cấu trúc khuôn mặt, ánh sáng, màu sắc.
o Đặc trưng miền tần số (Frequency domain): Các dấu vết phổ (spectrum artifacts) để lại do quá trình up-sampling của các mô hình AI tạo sinh.

11.2. Phạm vi nghiên cứu
• Dữ liệu: Tập trung vào ảnh tĩnh (static images), giới hạn ở ảnh chân dung người (human faces) để đảm bảo tính nhất quán.
• Nguồn sinh ảnh (Generative Models):
o Mô hình GANs: StyleGAN, ProGAN (đại diện cho công nghệ cũ).
o Mô hình Diffusion: Stable Diffusion, Midjourney (đại diện cho công nghệ mới).
• Hệ thống: Xây dựng ứng dụng Web Demo (Proof-of-Concept) chạy trên môi trường máy tính cá nhân/Google Colab, không yêu cầu xử lý thời gian thực (real-time) độ trễ thấp < 30ms.

11.3. Cách tiếp cận
• Tiếp cận thực nghiệm (Experimental Research): Xây dựng giả thuyết, thiết kế mô hình, chạy thí nghiệm nhiều lần với các tham số khác nhau để tìm ra cấu hình tối ưu.
• Tiếp cận Học có giám sát (Supervised Learning): Mô hình được huấn luyện trên tập dữ liệu đã gán nhãn chính xác (0: Real, 1: Fake).

11.4. Phương pháp nghiên cứu
Đề tài áp dụng quy trình nghiên cứu chặt chẽ gồm 4 trụ cột chính:
a. Chiến lược Dữ liệu và Chia tập (Data Strategy) Để giải quyết vấn đề tổng quát hóa (Generalization), dữ liệu được tổ chức như sau:
• Nguồn dữ liệu: Tổng hợp từ các bộ dataset công khai (FFHQ, CelebA-HQ) và tự sinh thêm từ các model mới.
• Tập Train/Val (Nội miền - In-domain): Sử dụng tối thiểu 20.000 ảnh.
o Bao gồm ít nhất 3 nguồn GAN (ví dụ: StyleGAN2, ProGAN, StarGAN) và 1-2 nguồn Diffusion phổ biến (Stable Diffusion v1.5).
o Mục tiêu: Giúp mô hình học được các đặc trưng cơ bản.
• Tập Test (Ngoài miền - Out-of-Distribution/OOD):
o Giữ lại hoàn toàn 2 nguồn chưa từng xuất hiện trong quá trình huấn luyện (ví dụ: Midjourney v6, Flux.1) để kiểm thử khả năng phát hiện các dạng deepfake chưa biết (unseen attacks).
o Bổ sung các biến thể nhiễu: Nén JPEG, Resize, Crop để đánh giá độ bền của mô hình.
b. Kiến trúc Mô hình và Tiền xử lý
• Mô hình song song (Parallel Pipeline): Thiết kế kiến trúc 2 nhánh:
o Nhánh 1 (Spatial): Đầu vào là ảnh RGB, sử dụng backbone ConvNeXt-Tiny hoặc EfficientNet-B0 để trích xuất đặc trưng hình ảnh.
o Nhánh 2 (Noise & Frequency): Sử dụng các bộ lọc SRM (Spatial Rich Model) để trích xuất bản đồ nhiễu (Noise Map), kết hợp với biến đổi DCT/FFT để phát hiện các bất thường trong cả phân phối nhiễu và Phổ biên độ (Magnitude Spectrum) (đặc biệt hiệu quả với ảnh Diffusion).
o Cơ chế kết hợp (Fusion): Sử dụng cơ chế Attention-based Fusion (Hợp nhất dựa trên sự chú ý) để mô hình tự học cách trọng số hóa đặc trưng quan trọng từ hai nhánh trước khi phân loại.

c. Phương pháp Huấn luyện và Chống Overfitting
• Học chuyển giao (Transfer Learning): Sử dụng trọng số pre-trained từ ImageNet, thực hiện chiến lược "Unfreeze từng phần" (đóng băng các lớp đầu, chỉ huấn luyện các lớp sâu).
• Hàm mất mát (Loss Function): Sử dụng Focal Loss thay cho Cross-Entropy thông thường để giải quyết vấn đề mất cân bằng dữ liệu (nếu số lượng ảnh thật/giả chênh lệch).
• Kỹ thuật chống Overfitting: Áp dụng Early Stopping (dừng sớm), kỹ thuật tăng cường dữ liệu mạnh (JPEG Compression, Gaussian Blur, Mixup, Cutout) để mô phỏng môi trường thực tế và kiểm chứng chéo theo nguồn (Source-wise Cross Validation).
d. Phương pháp Đánh giá và Minh bạch hóa (XAI)
• Bộ chỉ số đánh giá toàn diện: Không chỉ dùng độ chính xác (Accuracy), đề tài sẽ báo cáo:
o AUC (Area Under Curve) & EER (Equal Error Rate): Đánh giá ngưỡng quyết định tối ưu.
o Biểu đồ ROC: Trực quan hóa hiệu suất.
o Ma trận nhầm lẫn (Confusion Matrix): Phân tích tỷ lệ báo động giả (False Positive).
• Giải thích mô hình (Explainable AI): Tích hợp thuật toán Grad-CAM để sinh ra bản đồ nhiệt (heatmap), làm sáng tỏ các vùng điểm ảnh (pixels) đóng góp nhiều nhất vào quyết định của AI.
11.5. Yêu cầu Tài nguyên tính toán
• Môi trường thực nghiệm: Google Colab Pro hoặc Máy chủ cục bộ.
• Phần cứng dự kiến:
o GPU: NVIDIA Tesla T4 hoặc tương đương (yêu cầu VRAM tối thiểu 12GB để chạy các model EfficientNet/ResNet với batch-size phù hợp).
o Lưu trữ: ~50GB cho bộ dữ liệu và checkpoints mô hình.

12. Nội dung nghiên cứu và tiến độ thực hiện:
    12.1. Nội dung nghiên cứu (trình bày dưới dạng đề cương nghiên cứu chi tiết)

Chương 1: Mở đầu:

Trình bày tính cấp thiết, mục tiêu, đối tượng và phạm vi của đề tài.

Chương 2: Cơ sở lý thuyết và Tổng quan:

- Tổng quan về Trí tuệ nhân tạo tạo sinh (GAN, Diffusion Models).
- Cơ sở lý thuyết về Mạng nơ-ron tích chập (CNN) và Học chuyển giao.
- Tổng quan các công trình nghiên cứu liên quan đã được thực hiện (dựa trên Mục 10).
- Lý thuyết về phân tích ảnh trong miền tần số (Frequency Domain Analysis).
- Tổng quan về Explainable AI và thuật toán Grad-CAM trong thị giác máy tính

Chương 3: Phương pháp và Xây dựng hệ thống:

- Quy trình thu thập, xử lý và xây dựng bộ dữ liệu.
- Thiết kế kiến trúc mô hình CNN kết hợp tiền xử lý miền tần số.
- Phương pháp huấn luyện và tinh chỉnh mô hình.
- Thiết kế kiến trúc hệ thống web (Frontend, Backend API, và Mô hình AI).
- Cài đặt module trực quan hóa kết quả (Heatmap Visualization) bằng Grad-CAM
  Chương 4: Kết quả thực nghiệm và Đánh giá:

- Trình bày môi trường, tham số thực nghiệm.
- Trình bày kết quả đánh giá mô hình (Accuracy, Precision, Recall, F1, Ma trận nhầm lẫn).
- Phân tích, so sánh và bàn luận về kết quả đạt được.

Chương 5: Kết luận và Hướng phát triển:

- Tổng kết các kết quả đạt được so với mục tiêu đề ra.
- Nêu lên các hạn chế của đề tài.
- Đề xuất các hướng phát triển trong tương lai (ví dụ: mở rộng sang video, cải thiện tốc độ).

  12.2. Tiến độ thực hiện
  Thời gian Nội dung công việc Sản phẩm và KPI (Mốc nghiệm thu) Người thực hiện
  Giai đoạn 1

(T11/2025) Nghiên cứu tổng quan & Xây dựng dữ liệu

- Chốt thiết kế hệ thống và bộ chỉ số đánh giá.

- Thu thập và gán nhãn dữ liệu từ các nguồn GAN/Diffusion.

- Chia tập dữ liệu chuẩn: Train/Val/Test-OOD. Dataset v1 + Tài liệu mô tả

- KPI Dữ liệu: ≥ 10.000 ảnh (tỷ lệ cân bằng 50/50).

- KPI Chia tập: Có tập Test-OOD tách biệt (nguồn chưa thấy). Lê Văn Hoàng

Ngô Huỳnh Bảo Luân
Giai đoạn 2

(T12/2025) Xây dựng mô hình cơ sở (Baseline)

- Thiết lập môi trường huấn luyện (GPU/Colab).

- Huấn luyện mô hình CNN cơ bản (ResNet/EfficientNet) trên ảnh RGB.

- Đánh giá sơ bộ hiệu năng nội miền. Mô hình Model-RGB v1

- KPI Hiệu năng: AUC ≥ 0.88 (trên tập nội miền).

- Báo cáo kết quả chạy thử lần 1. Lê Văn Hoàng
  Giai đoạn 3

(T01/2026) Phát triển mô hình Fusion (Tần số)

- Cài đặt module tiền xử lý miền tần số (DCT/FFT).

- Tích hợp nhánh mạng tần số vào kiến trúc CNN.

- Huấn luyện và tinh chỉnh mô hình đa nhánh. Mô hình Model-Fusion v1

- KPI Hiệu năng: AUC ≥ 0.90 (nội miền), AUC ≥ 0.83 (ngoài miền).

- Source code module xử lý tần số. Lê Văn Hoàng
  Giai đoạn 4

(T02 - T03/2026) Tối ưu hóa và Tích hợp XAI

- Tinh chỉnh siêu tham số (Hyper-parameter tuning) để đạt hiệu năng cao nhất.

- Cài đặt thuật toán Grad-CAM để xuất bản đồ nhiệt.

- Kiểm thử độ bền (Robustness test) với ảnh nén/resize. Mô hình hoàn thiện (Final Model)

- KPI Final: AUC ≥ 0.92 (nội miền); AUC ≥ 0.85 (ngoài miền).

- KPI XAI: Heatmap hiển thị đúng vùng giả mạo trên 50 ảnh mẫu. Lê Văn Hoàng

Ngô Huỳnh Bảo Luân
Giai đoạn 5

(T03 - T04/2026) Xây dựng Web Demo

- Xây dựng Backend API (Python/Flask) tích hợp model.

- Xây dựng Frontend cho người dùng upload ảnh.

- Tối ưu hóa tốc độ phản hồi. Ứng dụng Web Demo v1

- KPI Hệ thống: Chạy ổn định trên máy cá nhân.

- KPI Tốc độ: Latency ≤ 2 giây/ảnh (trên CPU). Lê Văn Hoàng

Ngô Huỳnh Bảo Luân
Giai đoạn 6

(T04 - T05/2026) Tổng kết và Báo cáo

- Viết báo cáo tổng kết đề tài.

- Quay video demo sản phẩm.

- Chuẩn bị slide và hồ sơ bảo vệ. Hồ sơ nghiệm thu đầy đủ

- Báo cáo tổng kết (Docx + PDF).

- Slide thuyết trình.

- Video demo + Mã nguồn đóng gói. Lê Văn Hoàng

Ngô Huỳnh Bảo Luân

13. Sản phẩm và khả năng ứng dụng:
    13.1. Danh mục sản phẩm bàn giao (Hồ sơ nghiệm thu)
    Nhóm nghiên cứu cam kết bàn giao đầy đủ các hạng mục sau, được đóng gói và chuẩn hóa kỹ thuật:
    13.1.1. Bộ Mã nguồn (Source Code Package):
    • Toàn bộ mã nguồn huấn luyện (Training scripts), suy luận (Inference scripts) và ứng dụng Web (Backend/Frontend).
    • File README.md chi tiết: Hướng dẫn cài đặt môi trường, danh sách thư viện phụ thuộc (requirements.txt), và yêu cầu phần cứng tối thiểu (CPU/GPU).
    • Giấy phép mã nguồn: MIT License (mã nguồn mở).
    13.1.2. Mô hình đã huấn luyện (Pre-trained Models):
    • Định dạng: File trọng số .pt (PyTorch) và bản tối ưu hóa .onnx (áp dụng Quantization INT8/FP16 để tăng tốc suy luận) (để chạy trên web).
    • Bảng phiên bản (Model Versioning): Kèm theo file log ghi rõ mã commit, siêu tham số (hyper-parameters), và độ chính xác tương ứng của từng phiên bản model.
    13.1.3. Bộ Dữ liệu (Dataset Manifest):
    • Cung cấp cấu trúc thư mục dữ liệu đã gán nhãn (Train/Val/Test-OOD).
    • Lưu ý về bản quyền: Đối với các ảnh thuộc bộ dữ liệu gốc (như FFHQ) có bản quyền hạn chế, nhóm sẽ cung cấp file manifest (danh sách đường dẫn) và script tải xuống tự động (download scripts) thay vì phân phối trực tiếp file ảnh, nhằm tuân thủ quy định sở hữu trí tuệ.
    13.1.4. Báo cáo đánh giá chi tiết:
    • Báo cáo hiệu năng mô hình trên các tập dữ liệu nội miền (In-domain) và ngoài miền (Out-of-Distribution).
    • Đánh giá độ bền (Robustness) khi ảnh bị nén, thay đổi kích thước.
    • Minh họa kết quả XAI: Ảnh kết quả kèm bản đồ nhiệt Grad-CAM giải thích quyết định của AI.
    13.2. Khả năng ứng dụng
    • Hỗ trợ truyền thông: Cung cấp công cụ rà soát nhanh (Quick-check tool) cho các biên tập viên báo chí và quản trị viên nội dung để sàng lọc hình ảnh trước khi đăng tải.
    • Tích hợp nền tảng: Có thể phát triển thành API hoặc Plugin trình duyệt để cảnh báo người dùng mạng xã hội về các nội dung nghi ngờ là giả mạo.
    • Giáo dục cộng đồng: Nâng cao nhận thức của người dùng cá nhân về sự tồn tại và mức độ tinh vi của Deepfake.
    13.3. Giới hạn và Tuyên bố miễn trừ trách nhiệm (Disclaimer)
    • Phạm vi: Đây là sản phẩm nghiên cứu khoa học ở quy mô thử nghiệm (Proof-of-Concept).
    • Giá trị pháp lý: Kết quả dự đoán của mô hình chỉ mang tính chất tham khảo, không có giá trị thay thế cho các kết luận giám định kỹ thuật số hoặc phán quyết pháp lý.
    • Sai số: Người dùng cần đối chiếu kết quả với các công cụ khác hoặc đánh giá của con người, đặc biệt trong các trường hợp ảnh hưởng đến danh dự/nhân phẩm.

14. Kinh phí thực hiện đề tài:.
    Tổng kinh phí dự kiến: 4.900.000 VNĐ (Bằng chữ: Bốn triệu chín trăm nghìn đồng chẵn). Chi tiết các khoản mục chi tiêu phục vụ trực tiếp cho việc nghiên cứu và xây dựng hệ thống:
    TT Nội dung chi Đơn vị Số lượng Đơn giá (VNĐ) Thành tiền (VNĐ) Ghi chú
    1 Thuê hạ tầng tính toán (Cloud GPU) Tháng 3 800.000 2.400.000 Thuê dịch vụ Google Colab Pro+ hoặc VM GPU để huấn luyện Deep Learning trong giai đoạn cao điểm (3 tháng).
    2 Thiết bị lưu trữ dữ liệu Chiếc 1 1.500.000 1.500.000 Ổ cứng SSD di động (1TB) để lưu trữ Dataset và Checkpoint Model (do dữ liệu ảnh rất lớn).
    3 Vật tư, in ấn, văn phòng phẩm Gói 1 500.000 500.000 In báo cáo thuyết minh, báo cáo tổng kết, poster bảo vệ.
    4 Chi phí dự phòng (10%) Gói 1 500.000 500.000 Chi phí phát sinh (tên miền demo, hosting, tài liệu tham khảo...).
    TỔNG CỘNG 4.900.000

15. Tài Liệu Tham khảo
    [1] Rössler, A., et al. (2019). "FaceForensics++: Learning to Detect Manipulated Facial Images". Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
    [2] Wang, S. Y., et al. (2020). "CNN-generated images are surprisingly easy to spot... for now". Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
    [3] Wodajo, D., & Atnafu, S. (2021). "Deepfake Video Detection Using Convolutional Vision Transformer". arXiv preprint arXiv:2102.11126.
    [4] Cao, J., et al. (2022). "End-to-End Reconstruction-Classification Learning for Face Forgery Detection". Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
    [5] Frank, J., et al. (2020). "Leveraging Frequency Analysis for Deep Fake Image Recognition". International Conference on Machine Learning (ICML).
    [6] Durall, R., et al. (2020). "Watch your Up-Convolution: CNN Based Generative Colorization Exposes Artificial Artifacts". Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
    [7] GitHub Repository. "Awesome-Deepfakes-Detection". Available at: https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection.

Giảng viên hướng dẫn
đề tài
(Ký, ghi rõ họ tên)
Ngày 11tháng 11 năm 2025
Sinh viên
chịu trách nhiệm chính
(Ký, ghi rõ họ tên)

Thành phố Hồ Chí Minh, ngày …… tháng …… năm 20…
TRƯỞNG ĐƠN VỊ
(Ký, ghi rõ họ tên)
