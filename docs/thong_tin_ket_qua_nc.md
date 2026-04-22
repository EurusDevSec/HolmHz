THÔNG TIN KẾT QUẢ NGHIÊN CỨU CỦA ĐỀ TÀI

1. Thông tin chung:
- Tên đề tài: Xây dựng hệ thống phát hiện ảnh tổng hợp bằng Mạng nơ-ron tích chập (CNN) 
- Sinh viên/ nhóm sinh viên thực hiện: Lê Văn Hoàng
STT	Họ và tên	MSSV	Lớp	Khoa	Năm thứ/ Số năm đào tạo
1	Lê Văn Hoàng	2224802010279	D22CNTT02	Viện Công
Nghệ Số	Năm thứ tư
2	Ngô Huỳnh Bảo Luân	2524802010327	D25CNTT10	Viện Công
Nghệ Số	Năm thứ nhất

- Người hướng dẫn: ThS. Nguyễn Trung Kiệt

2. Mục tiêu đề tài:

Mục tiêu tổng quát: Xây dựng hệ thống phát hiện ảnh tổng hợp (Synthetic Image Detection) dựa trên Mạng nơ-ron tích chập (CNN) với kỹ thuật Học chuyển giao (Transfer Learning), có khả năng phân biệt ảnh thật và ảnh được tạo ra bởi các mô hình AI thế hệ mới (GAN và Diffusion Models), tích hợp giải thích trực quan bằng Grad-CAM và triển khai dưới dạng web demo thực tế.

Mục tiêu cụ thể:
- Xây dựng bộ dữ liệu chuẩn hóa ≥ 20.000 ảnh từ ≥ 5 nguồn, bao phủ GAN (StyleGAN, ProGAN) và Diffusion (Stable Diffusion, DALL-E, Midjourney).
- Huấn luyện và so sánh 4 kiến trúc: EfficientNet-B0, ResNet-18, ViT-Small/16, Swin-Tiny bằng Transfer Learning từ ImageNet.
- Đạt ID AUC ≥ 0,92 và OOD AUC ≥ 0,85 trên tập kiểm thử độc lập.
- Tích hợp Grad-CAM để trực quan hóa vùng ảnh bị nghi ngờ là giả mạo.
- Xây dựng Web Demo với thời gian phản hồi ≤ 2 giây/ảnh trên CPU phổ thông.
- Benchmark công bằng với 3 nghiên cứu SOTA quốc tế trên cùng bộ dữ liệu.

3. Tính mới và sáng tạo:

a) Bộ dữ liệu đa nguồn, đa thế hệ generator: Đề tài tổng hợp 35.454 ảnh từ 5 nguồn Kaggle, bao phủ đồng thời cả công nghệ GAN (StyleGAN, ProGAN, DeepFaceLab, Face2Face, FaceShifter, NeuralTextures) và Diffusion thế hệ mới (Stable Diffusion, DALL-E, Midjourney) — vượt trội so với các nghiên cứu trước chỉ tập trung vào GAN. Tập Test OOD (182 ảnh từ nguồn camera thực tế vs AI) hoàn toàn tách biệt với tập huấn luyện, đánh giá đúng khả năng tổng quát hóa.

b) Phát hiện vai trò then chốt của JPEG Augmentation: Đề tài chứng minh thực nghiệm rằng kỹ thuật JPEG compression ngẫu nhiên (quality 50–95) trong pipeline augmentation cải thiện OOD AUC từ 0,440 lên 0,896 (+103,6%) — đóng góp thực tiễn có giá trị cho cộng đồng nghiên cứu phát hiện ảnh tổng hợp.

c) Benchmark công bằng, toàn diện: Lần đầu tiên tại TDMU, 4 mô hình tự huấn luyện được đặt lên bàn cân với 3 phương pháp SOTA quốc tế (CNNDetection, UniversalFakeDetect, DeepfakeBench) trên cùng một bộ dữ liệu và cùng điều kiện đánh giá — thay vì so sánh gián tiếp qua số liệu trong bài báo.

d) Tích hợp XAI (Explainable AI) vào web demo: Hệ thống không chỉ trả kết quả Real/Fake mà còn hiển thị bản đồ nhiệt Grad-CAM chỉ ra vùng ảnh khiến mô hình nghi ngờ — tăng tính minh bạch và tin cậy cho người dùng cuối, điều mà các nghiên cứu kinh điển trong lĩnh vực chưa chú trọng.

e) Phân tích thất bại có hệ thống: Sự thất bại của Swin-Tiny (best epoch = 0, do learning rate không phù hợp với Transformer) được phân tích chi tiết và trở thành bài học về fine-tuning recipe cho Transformer trên dataset vừa và nhỏ.

4. Kết quả nghiên cứu:

a) Bộ dữ liệu HolmHz-v2:
- Tổng cộng 35.454 ảnh (28.220 Train | 3.526 Validation | 3.526 Test ID | 182 Test OOD)
- 5 nguồn dữ liệu: RVF10K, DeepDetect-2025, Diffusion Fakes, CIPLab Faces, Camera vs AI
- 8+ loại AI generator được bao phủ

b) Kết quả huấn luyện 4 mô hình:

STT | Mô hình       | Tham số | Val AUC | Best Epoch
1   | EfficientNet-B0 (v9) | 4M  | 0,9993  | 25/30
2   | ResNet-18     | 11M     | 0,9956  | 28/30
3   | ViT-Small/16  | 22M     | 0,9735  | 29/30
4   | Swin-Tiny†    | 28M     | 0,6198  | 0/30 (thất bại)

c) Benchmark 7 mô hình (ID test: 3.526 ảnh | OOD test: 182 ảnh):

Nhóm        | Phương pháp          | Tham số | ID AUC | OOD AUC
Baseline    | CNNDetection         | ~23M    | 0,662  | 0,325
Baseline    | UniversalFakeDetect  | ~304M   | 0,722  | 0,486
Baseline    | DeepfakeBench        | ~19M    | 0,439  | 0,536
Ours (tốt nhất) | EfficientNet-B0 v9 | 4M  | 0,998  | 0,896 ✅
Ours        | ResNet-18            | 11M     | 0,995  | 0,865 ✅
Ours        | ViT-Small/16         | 22M     | 0,974  | 0,833
Ours        | Swin-Tiny†           | 28M     | 0,620  | 0,811

d) Kết quả đánh giá KPI:

KPI                          | Mục tiêu | Đạt được     | Trạng thái
Dataset ≥ 20.000 ảnh         | 20.000   | 28.220       | ✅ Vượt 41%
ID AUC ≥ 0,92                | 0,92     | 0,998        | ✅ Vượt 8%
ID Accuracy ≥ 90%            | 90%      | 98,4%        | ✅ Vượt 9%
OOD AUC ≥ 0,85               | 0,85     | 0,896        | ✅ Vượt 5%
Web demo ≤ 2 giây/ảnh (CPU)  | 2s       | ~1,5s        | ✅ Đạt

Đề tài đạt 5/5 KPI đề ra. EfficientNet-B0 v9 (4M tham số) vượt trội 3 phương pháp SOTA quốc tế có số tham số lớn hơn 5–75 lần.

e) Phát hiện khoa học: JPEG Augmentation (quality 50–95, p=0.7) là yếu tố then chốt cải thiện khả năng tổng quát hóa: OOD AUC tăng từ 0,440 (EfficientNet-B0 v7, không có JPEG aug) lên 0,896 (EfficientNet-B0 v9, có JPEG aug v3), tương đương tăng 103,6%.

f) Web Demo: Ứng dụng Gradio (Python) với ResNet-18 ONNX, latency ~1,5 giây/ảnh trên CPU, hiển thị kết quả Real/Fake (%) kèm bản đồ nhiệt Grad-CAM.

5. Đóng góp về mặt kinh tế - xã hội, giáo dục và đào tạo, an ninh, quốc phòng và khả năng áp dụng của đề tài: 

- An ninh: Hỗ trợ kiểm chứng nguồn gốc ảnh số, giảm thiểu nguy cơ lan truyền thông tin sai lệch từ deepfake trong các vụ việc an ninh mạng.
- Giáo dục: Cung cấp giải pháp kỹ thuật thực tiễn cho sinh viên CNTT tiếp cận công nghệ phát hiện AI-generated content (AIGC).
- Khả năng áp dụng: Web demo có tính di động cao, dễ tích hợp vào các hệ thống kiểm duyệt tin tức hoặc các diễn đàn trực tuyến để sàng lọc nội dung hình ảnh tự động.

6. Công bố khoa học của sinh viên từ kết quả nghiên cứu của đề tài (ghi rõ họ tên tác giả, nhan đề và các yếu tố về xuất bản nếu có) hoặc nhận xét, đánh giá của cơ sở đã áp dụng các kết quả nghiên cứu (nếu có): 



                                                      Thành phố Hồ Chí Minh, ngày      tháng     năm 2026  
	SINH VIÊN
CHỊU TRÁCH NHIỆM CHÍNH
(ký, họ và tên)



Nhận xét của người hướng dẫn về những đóng góp khoa học của sinh viên thực hiện đề tài (phần này do người hướng dẫn ghi):

                                                                                    Bình Dương, ngày      tháng    năm  
TRƯỞNG ĐƠN VỊ
(ký, họ và tên)	GIẢNG VIÊN HƯỚNG DẪN
(ký, họ và tên)

