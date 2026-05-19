
TRƯỜNG ĐẠI HỌC THỦ DẦU MỘT
VIỆN CÔNG NGHỆ SỐ 
 


BÁO CÁO TỔNG KẾT

ĐỀ TÀI NGHIÊN CỨU KHOA HỌC VÀ CÔNG NGHỆ CỦA SINH VIÊN 
NĂM HỌC 2025-2026   


Xây dựng hệ thống phát hiện ảnh tổng hợp bằng Mạng nơ-ron tích chập (CNN)
	
			
         Sinh viên/Nhóm Sinh viên thực hiện: 
1.	Lê Văn Hoàng — MSSV: 2224802010279 — Lớp: D22CNTT02 (Nhóm trưởng)
2.	Ngô Huỳnh Bảo Luân — MSSV: 2524802010327 — Lớp: D25CNTT10

Giảng viên hướng dẫn:  ThS. Nguyễn Trung Kiệt
			
  
                              Thành phố Hồ Chí Minh, ngày  17 tháng  04  năm 2026

 


TRƯỜNG ĐẠI HỌC THỦ DẦU MỘT
VIỆN CÔNG NGHỆ SỐ 
 

BÁO CÁO TỔNG KẾT
ĐỀ TÀI NGHIÊN CỨU KHOA HỌC VÀ CÔNG NGHỆ CỦA SINH VIÊN 
NĂM HỌC 2025-2026
Xây dựng hệ thống phát hiện ảnh tổng hợp bằng Mạng nơ-ron tích chập (CNN)

STT	Họ và tên SV	Giới tính	Dân tộc	Lớp, Khoa	SV năm thứ/ Số năm đào tạo	Ngành học	Ghi chú
1	Lê Văn Hoàng	Nam 	Kinh	D22CNTT02	Sinh viên năm tư	Công nghệ thông tin	SV thực hiện chính 
2	Ngô Huỳnh Bảo Luân 	Nam 	Kinh	D25CNTT10	Sinh viên năm nhất	Công nghệ thông tin	

Giảng viên hướng dẫn: ThS. Nguyễn Trung Kiệt 
 

LỜI CAM ĐOAN

Tôi, Lê Văn Hoàng, sinh viên lớp D22CNTT02, chuyên ngành Công nghệ Thông tin, Viện Công nghệ số, Trường Đại học Thủ Dầu Một, đại diện nhóm sinh viên thực hiện đề tài, xin cam đoan rằng nghiên cứu khoa học với đề tài "Xây dựng hệ thống phát hiện ảnh tổng hợp bằng Mạng nơ-ron tích chập (CNN)" là kết quả do nhóm thực hiện, dưới sự hướng dẫn tận tình của thầy ThS. Nguyễn Trung Kiệt.
Toàn bộ nội dung, dữ liệu, mã nguồn và kết quả nghiên cứu trình bày này đều là trung thực và chưa từng được công bố hoặc sử dụng để phục vụ cho bất kỳ mục đích nào khác. Trong quá trình thực hiện, nhóm đã tuân thủ nghiêm túc các quy định về đạo đức học thuật, trích dẫn đầy đủ và chính xác tất cả các nguồn tài liệu tham khảo.
Nhóm xin hoàn toàn chịu trách nhiệm về tính chính xác và tính trung thực của bài nghiên cứu khoa học này. Nếu phát hiện bất kỳ hành vi vi phạm đạo đức học thuật nào, nhóm xin sẵn sàng đối mặt với các hình thức kỷ luật theo quy định của nhà trường.


TP Hồ Chí Minh, ngày 17 tháng 4 năm 2026

                                                                                                   
	Người thực hiện
(Ký tên và ghi rõ họ tên)
	

 Lê Văn Hoàng










                                                                                            
 
MỤC LỤC 
LỜI CAM ĐOAN	i
DANH MỤC BẢNG	iv
DANH MỤC HÌNH	v
DANH MỤC CÁC CHỮ VIẾT TẮT	vi
THÔNG TIN KẾT QUẢ NGHIÊN CỨU CỦA ĐỀ TÀI	viii
MỞ ĐẦU	1
1. Tổng quan tình hình nghiên cứu	1
2. Khoảng trống nghiên cứu (Research Gap)	2
3. Tính cấp thiết	3
4. Mục tiêu đề tài	3
5. Đối tượng và phạm vi nghiên cứu	4
6. Cách tiếp cận và phương pháp nghiên cứu	4
7. Nội dung nghiên cứu	4
CHƯƠNG 1. GIỚI THIỆU	6
CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ TỔNG QUAN	7
2.1 Trí tuệ nhân tạo tạo sinh (Generative AI)	7
2.1.1 Mạng đối sinh (Generative Adversarial Network — GAN)	7
2.1.2 Mô hình khuếch tán (Diffusion Models)	7
2.2 Mạng nơ-ron tích chập (CNN)	9
2.2.1 Kiến trúc CNN cơ bản	9
2.2.2 EfficientNet	10
2.2.3 ResNet-18	11
2.2.4 Vision Transformer (ViT)	12
2.2.5 Swin Transformer	13
2.3 Học chuyển giao (Transfer Learning)	14
2.4 Explainable AI (XAI) và Grad-CAM	14
2.4.1 Nhu cầu giải thích mô hình	14
2.4.2 Grad-CAM (Gradient-weighted Class Activation Mapping)	14
2.5 Các chỉ số đánh giá (Evaluation Metrics)	15
CHƯƠNG 3. PHƯƠNG PHÁP VÀ XÂY DỰNG HỆ THỐNG	17
3.1 Tổng quan kiến trúc hệ thống	17
3.2 Quy trình xây dựng bộ dữ liệu	17
3.2.1 Thu thập dữ liệu	17
3.2.2 Tổ chức dữ liệu bằng Manifest JSON	19
3.2.3 Chia tập dữ liệu	19
3.3 Thiết kế kiến trúc mô hình	20
3.3.1 Kiến trúc tổng quát: Backbone + Head	20
3.3.2 Registry Pattern	20
3.3.3 Hỗ trợ đa kiến trúc qua Timm	21
3.4 Phương pháp huấn luyện	21
3.4.1 Data Augmentation Pipeline	21
3.4.2 WeightedRandomSampler	22
3.4.3 Optimizer và Learning Rate Scheduler	22
3.4.4 Loss Function	22
3.5 Pipeline huấn luyện (Trainer)	22
3.6 Pipeline đánh giá (Evaluator)	23
3.7 Thiết kế Web Demo	23
3.7.1 Kiến trúc	23
3.7.2 Tối ưu suy luận với ONNX	23
3.7.3 Tích hợp Grad-CAM	24
3.8 Đề xuất kiến trúc triển khai đám mây (AWS)	24
3.8,1 Các thành phần kiến trúc	25
3.8,2 Luồng xử lý request	26
3.8,3 CI/CD Pipeline	26
3.8,4 Tối ưu chi phí (Cost Optimization)	27
CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ	27
4.1 Môi trường và tham số thực nghiệm	27
4.2 Bộ dữ liệu	28
4.3 Kết quả huấn luyện 4 mô hình HolmHz	30
4.4 Benchmark tổng hợp 7 mô hình	31
4.5 Phân tích biểu đồ	32
4.5.1 Biểu đồ cột — ID AUC vs OOD AUC	33
4.5.2 Biểu đồ Radar — Đa chỉ số	34
4.5.3 Heatmap OOD per-source	34
4.6 Phân tích mô hình EfficientNet-B0 — Tại sao mô hình nhỏ nhất lại tốt nhất?	36
4.7 Phân tích sự thất bại của Swin-Tiny	37
4.8 So sánh với các nghiên cứu baseline	38
4.9 Đánh giá KPI đề tài	38
4.10 Web Demo	39
4.11 Phân tích Grad-CAM: Từ minh họa đến giải thích	40
CHƯƠNG 5. KẾT LUẬN VÀ KIẾN NGHỊ	41
5.1 Kết luận	41
5.2 Đóng góp của đề tài	41
5.3 Hạn chế	41
5.4 Hướng phát triển	42
TÀI LIỆU THAM KHẢO	43
PHỤ LỤC	44















 

DANH MỤC BẢNG
Bảng 0.1: So sánh tổng hợp các nghiên cứu tiêu biểu	2
Bảng 2.1: Các chỉ số đánh giá mô hình phân loại	15
Bảng 3.1: Nguồn dữ liệu trong bộ dữ liệu v2	18
Bảng 3.2: Chia tập dữ liệu (Train/Val/Test ID/Test OOD)	19
Bảng 3.3: Feature dimension theo backbone	20
Bảng 3.4: Training augmentation pipeline	21
Bảng 3.5: Các thành phần kiến trúc AWS — Lớp Global (Ngoài Region — phục vụ toàn cầu)	25
Bảng 3.6: Các thành phần kiến trúc AWS — Lớp Regional (Bên trong Region ap-southeast-1 — Singapore)	25
Bảng 4.1: Môi trường và tham số thực nghiệm	27
Bảng 4.2: Thống kê bộ dữ liệu v2 theo split	28
Bảng 4.3: Chi tiết nguồn dữ liệu trong bộ dữ liệu v2	29
Bảng 4.4: Kết quả huấn luyện 4 mô hình HolmHz trên Dataset v2	31
Bảng 4.5: Benchmark tổng hợp 7 mô hình (ID và OOD)	32
Bảng 4.6: Độ chính xác OOD theo nguồn dữ liệu	35
Bảng 4.7: Đánh giá KPI đề tài	38
Bảng 4.8: Đánh giá KPI theo từng mô hình	39

























DANH MỤC HÌNH
Hình 2.1: Mạng đối sinh (Generative Adversarial Network — GAN)	7
Hình 2.2: Diffusion Models	8
Hình 2.3: Stable Diffusion	8
Hình 2.4: Midjourney	9
Hình 2.5: Mô hình mạng nơ-ron tích chập (CNN)	10
Hình 2.6: EfficientNet (Tan & Le, 2019)	11
Hình 2.7: ResNet (He et al., 2016)	12
Hình 2.8: ViT (Dosovitskiy et al., 2021)	13
Hình 2.9: Swin Transformer (Liu et al., 2021)	14
Hình 2.10: Grad-CAM (Selvaraju et al., 2017)	15
Hình 3.1: Kiến trúc modular của hệ thống	17
Hình 3.2: Kiến trúc triển khai đám mây HolmHz trên AWS	25
Hình 4.1: Biểu đồ cột – ID AUC và OOD AUC	33
Hình 4.2: Biểu đồ Radar – Đa chỉ số	34
Hình 4.3: Heatmap OOD per-source	35

 
DANH MỤC CÁC CHỮ VIẾT TẮT
Viết tắt	Tiếng Anh đầy đủ	Giải nghĩa tiếng Việt
AI	Artificial Intelligence	Trí tuệ nhân tạo
AUC	Area Under the Receiver Operating Characteristic Curve	Diện tích dưới đường cong ROC
CNN	Convolutional Neural Network	Mạng nơ-ron tích chập
DALL-E	— (tên mô hình của OpenAI)	Mô hình tạo ảnh của OpenAI
DCT	Discrete Cosine Transform	Biến đổi cosin rời rạc
DFT	Discrete Fourier Transform	Biến đổi Fourier rời rạc
EER	Equal Error Rate	Tỷ lệ lỗi cân bằng
F1	F1-Score	Điểm F1 (trung bình điều hòa Precision và Recall)
GAN	Generative Adversarial Network	Mạng đối sinh
Grad-CAM	Gradient-weighted Class Activation Mapping	Bản đồ kích hoạt lớp có trọng số gradient
ID	In-Domain / In-Distribution	Nội miền (cùng phân phối với dữ liệu huấn luyện)
JPEG	Joint Photographic Experts Group	Chuẩn nén ảnh JPEG
KPI	Key Performance Indicator	Chỉ số đánh giá hiệu suất chính
MBConv	Mobile Inverted Bottleneck Convolution	Khối tích chập nghịch đảo di động
NAS	Neural Architecture Search	Tìm kiếm kiến trúc mạng nơ-ron tự động
ONNX	Open Neural Network Exchange	Định dạng trao đổi mô hình mạng nơ-ron mở
OOD	Out-of-Domain / Out-of-Distribution	Ngoài miền (khác phân phối với dữ liệu huấn luyện)
PoC	Proof-of-Concept	Chứng minh khái niệm / Sản phẩm thử nghiệm
ROC	Receiver Operating Characteristic	Đường đặc trưng hoạt động của bộ phân loại
SD	Stable Diffusion	Mô hình khuếch tán ổn định (tạo ảnh AI)
SE	Squeeze-and-Excitation	Cơ chế nén-kích thích (attention trên kênh)
SOTA	State-of-the-Art	Phương pháp/kết quả tiên tiến nhất hiện tại
SRM	Spatial Rich Model	Mô hình giàu không gian (trích xuất vân tay nhiễu)
ViT	Vision Transformer	Transformer cho thị giác máy tính
XAI	Explainable Artificial Intelligence	Trí tuệ nhân tạo có khả năng giải thích


































ỦY BAN NHÂN DÂN
THÀNH PHỐ HỒ CHÍ MINH	CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập – Tự do – Hạnh phúc
TRƯỜNG ĐẠI HỌC THỦ DẦU MỘT	



	
THÔNG TIN KẾT QUẢ NGHIÊN CỨU CỦA ĐỀ TÀI

1. Thông tin chung:
- Tên đề tài: Xây dựng hệ thống phát hiện ảnh tổng hợp bằng Mạng nơ-ron tích chập (CNN) 
- Sinh viên/ nhóm sinh viên thực hiện: Lê Văn Hoàng
STT	Họ và tên	MSSV	Lớp	Khoa	Năm thứ/ Số năm đào tạo
1	Lê Văn Hoàng	2224802010279	D22CNTT02	Viện Công
nghệ số	Năm thứ tư
2	Ngô Huỳnh Bảo Luân	2524802010327	D25CNTT10	Viện Công
nghệ số	Năm thứ nhất

- Người hướng dẫn: ThS. Nguyễn Trung Kiệt
2. Mục tiêu đề tài:
Mục tiêu tổng quát: Xây dựng hệ thống phát hiện ảnh tổng hợp (Synthetic Image Detection) dựa trên Mạng nơ-ron tích chập (CNN) với kỹ thuật Học chuyển giao (Transfer Learning), có khả năng phân biệt ảnh thật và ảnh được tạo ra bởi các mô hình AI thế hệ mới (GAN và Diffusion Models), tích hợp giải thích trực quan bằng Grad-CAM và triển khai dưới dạng web demo thực tế.
Mục tiêu cụ thể:
•	Xây dựng bộ dữ liệu chuẩn hóa ≥ 20,000 ảnh từ ≥ 5 nguồn, bao phủ GAN (StyleGAN, ProGAN) và Diffusion (Stable Diffusion, DALL-E, Midjourney).
•	Huấn luyện và so sánh 4 kiến trúc: EfficientNet-B0, ResNet-18, ViT-Small/16, Swin-Tiny bằng Transfer Learning từ ImageNet.
•	Đạt ID AUC ≥ 0,92 và OOD AUC ≥ 0,85 trên tập kiểm thử độc lập.
•	Tích hợp Grad-CAM để trực quan hóa vùng ảnh bị nghi ngờ là giả mạo.
•	Xây dựng Web Demo với thời gian phản hồi ≤ 2 giây/ảnh trên CPU phổ thông.
•	Benchmark với 3 nghiên cứu baseline quốc tế trên cùng bộ dữ liệu.
3. Tính mới và sáng tạo:

a) Bộ dữ liệu đa nguồn, đa thế hệ generator: Đề tài tổng hợp 35,454 ảnh từ 5 nguồn Kaggle, bao phủ đồng thời cả công nghệ GAN (StyleGAN, ProGAN, DeepFaceLab, Face2Face, FaceShifter, NeuralTextures) và Diffusion thế hệ mới (Stable Diffusion, DALL-E, Midjourney) — đa dạng hơn so với các nghiên cứu trước chỉ tập trung vào GAN. Tập Test OOD (182 ảnh từ nguồn camera thực tế vs AI) hoàn toàn tách biệt với tập huấn luyện, nhằm đánh giá khả năng tổng quát hóa. Lưu ý: tập OOD còn nhỏ (182 ảnh, khoảng tin cậy ±6%), các kết luận về tổng quát hóa mang tính chỉ báo sơ bộ.

b) Quan sát về vai trò của JPEG Augmentation: Kết quả thực nghiệm cho thấy kỹ thuật JPEG compression ngẫu nhiên (quality 50–95) trong pipeline augmentation đi kèm với cải thiện OOD AUC từ 0,440 lên 0,896 (+103,6%). Kết quả dựa trên một lần chạy (seed=42) và chưa có ablation study tách biệt, do đó cần kiểm chứng thêm. Tuy nhiên, mức chênh lệch lớn cho thấy đây là yếu tố có ảnh hưởng đáng kể.

c) Benchmark trên cùng bộ dữ liệu: 4 mô hình tự huấn luyện được so sánh với 3 phương pháp baseline quốc tế (CNNDetection, UniversalFakeDetect, DeepfakeBench) trên cùng một bộ dữ liệu — thay vì so sánh gián tiếp qua số liệu trong bài báo. Lưu ý: các baseline sử dụng pre-trained weights gốc (zero-shot), không được retrain trên Dataset v2.
d) Tích hợp XAI (Explainable AI) vào web demo: Hệ thống không chỉ trả kết quả Real/Fake mà còn hiển thị bản đồ nhiệt Grad-CAM chỉ ra vùng ảnh khiến mô hình nghi ngờ — tăng tính minh bạch và tin cậy cho người dùng cuối, điều mà các nghiên cứu kinh điển trong lĩnh vực chưa chú trọng.

e) Phân tích thất bại có hệ thống: Sự thất bại của Swin-Tiny (best epoch = 0) được phân tích chi tiết — nguyên nhân là learning rate quá cao cho Transformer và thiếu inductive bias phù hợp với dataset nhỏ. Đây là hạn chế của thiết kế thí nghiệm, không phải của kiến trúc Swin, và trở thành bài học về fine-tuning recipe cho Transformer.
4. Kết quả nghiên cứu:

a) Bộ dữ liệu HolmHz-v2:

Tổng cộng 35,454 ảnh (28,220 Train | 3,526 Validation | 3,526 Test ID | 182 Test OOD)
5 nguồn dữ liệu: RVF10K, DeepDetect-2025, Diffusion Fakes, CIPLab Faces, Camera vs AI
8+ loại AI generator được bao phủ
b) Kết quả huấn luyện 4 mô hình:

STT	Mô hình	Tham số	Val AUC	Best Epoch
1	EfficientNet-B0 (v9)	4M	0,9993	25/30
2	ResNet-18	11M	0,9956	28/30
3	ViT-Small/16	22M	0,9735	29/30
4	Swin-Tiny	28M	0,6198	0/30 (thất bại)

c) Benchmark 7 mô hình (ID test: 3,526 ảnh | OOD test: 182 ảnh):

Nhóm	Phương pháp	Tham số	ID AUC	OOD AUC
Baseline	CNNDetection	~23M	0,662	0,325
	UniversalFakeDetect	~304M	0,722	0,486
	DeepfakeBench	~19M	0,439	0,536
Ours	EfficientNet-B0 v9 (tốt nhất)	4M	0,998	0,896 
	ResNet-18	11M	0,995	0,865 
	ViT-Small/16	22M	0,974	0,833
	Swin-Tiny†	28M	0,620	0,811

Ghi chú: Swin-Tiny training thất bại (best epoch = 0). Các baseline sử dụng pre-trained weights gốc (zero-shot), không retrain trên Dataset v2.

d) Kết quả đánh giá KPI:

STT	KPI	Mục tiêu	Đạt được	Trạng thái
1	Dataset (số lượng ảnh)	 
 20,000	28,220	Vượt 41%
2	ID AUC	 
 0,92	0,998	Vượt 8%
3	ID Accuracy	 
 90%	98,4%	Vượt 9%
4	OOD AUC	 
 0,85	0,896	Vượt 5%
5	Web demo (tốc độ xử lý/CPU)	 
 2 giây	~1,5 giây	Đạt

Đề tài đạt 5/5 KPI đề ra. EfficientNet-B0 v9 (4M tham số) đạt kết quả cao hơn 3 phương pháp baseline quốc tế trong điều kiện thí nghiệm của đề tài, cho thấy mô hình nhỏ có thể đạt hiệu quả tốt khi được huấn luyện trên dữ liệu phù hợp.

e) Quan sát đáng chú ý: JPEG Augmentation (quality 50–95, p=0,7) đi kèm với cải thiện đáng kể khả năng tổng quát hóa: OOD AUC tăng từ 0,440 (EfficientNet-B0 v7, không có JPEG aug) lên 0,896 (EfficientNet-B0 v9, có JPEG aug v3), tương đương tăng 103,6%. Kết quả dựa trên single seed (42), cần kiểm chứng thêm qua nhiều lần chạy và ablation study.

f) Web Demo: Ứng dụng Gradio (Python) với ResNet-18 ONNX, latency ~1,5 giây/ảnh trên CPU, hiển thị kết quả Real/Fake (%) kèm bản đồ nhiệt Grad-CAM.

5. Đóng góp về mặt kinh tế - xã hội, giáo dục và đào tạo, an ninh, quốc phòng và khả năng áp dụng của đề tài: 

- An ninh: Hỗ trợ kiểm chứng nguồn gốc ảnh số, giảm thiểu nguy cơ lan truyền thông tin sai lệch từ deepfake trong các vụ việc an ninh mạng.
- Giáo dục: Cung cấp giải pháp kỹ thuật thực tiễn cho sinh viên CNTT tiếp cận công nghệ phát hiện AI-generated content (AIGC).
- Khả năng áp dụng: Web demo có tính di động cao, dễ tích hợp vào các hệ thống kiểm duyệt tin tức hoặc các diễn đàn trực tuyến để sàng lọc nội dung hình ảnh tự động.


6. Công bố khoa học của sinh viên từ kết quả nghiên cứu của đề tài (ghi rõ họ tên tác giả, nhan đề và các yếu tố về xuất bản nếu có) hoặc nhận xét, đánh giá của cơ sở đã áp dụng các kết quả nghiên cứu (nếu có): 



                                                Thành phố Hồ Chí Minh, ngày 17 tháng  04   năm 2026  
	SINH VIÊN
CHỊU TRÁCH NHIỆM CHÍNH
(ký, họ và tên)



Nhận xét của người hướng dẫn về những đóng góp khoa học của sinh viên thực hiện đề tài (phần này do người hướng dẫn ghi):

                                                               Thành phố Hồ Chí Minh, ngày      tháng    năm  
TRƯỞNG ĐƠN VỊ
(ký, họ và tên)	GIẢNG VIÊN HƯỚNG DẪN
(ký, họ và tên)
















ỦY BAN NHÂN DÂN
THÀNH PHỐ HỒ CHÍ MINH	CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập – Tự do – Hạnh phúc
TRƯỜNG ĐẠI HỌC THỦ DẦU MỘT	



THÔNG TIN VỀ SINH VIÊN 
CHỊU TRÁCH NHIỆM CHÍNH THỰC HIỆN ĐỀ TÀI 

I. SƠ LƯỢC VỀ SINH VIÊN: 
Họ và tên: Lê Văn Hoàng
Sinh ngày:   02           tháng     03       năm 2004
Nơi sinh: Thành phố Huế
Lớp: D22CNTT02                                                  Khóa: D22
Khoa/viện: Viện Công nghệ số
Địa chỉ liên hệ: 453/40/29 Lê Hồng Phong, Phường Phú Lợi, Hồ Chí Minh
Điện thoại: 0399354603                 Email: 2224802010279@student.tdmu.edu.vn
II. QUÁ TRÌNH HỌC TẬP (kê khai thành tích của sinh viên từ năm thứ 1 đến năm đang học):
* Năm thứ 1:
    Ngành học:  Công nghệ thông tin                               Khoa/viện: Viện Công nghệ số
    Kết quả xếp loại học tập: Giỏi
    Sơ lược thành tích:
* Năm thứ 2:
    Ngành học: Công nghệ thông tin                              Khoa/viện: Viện Công nghệ số
    Kết quả xếp loại học tập: Giỏi
    Sơ lược thành tích:
* Năm thứ 3:
    Ngành học: Công nghệ thông tin                              Khoa/viện: Viện Công nghệ số
    Kết quả xếp loại học tập: Giỏi
    Sơ lược thành tích:
* Năm thứ 4:
    Ngành học: Công nghệ thông tin                              Khoa/viện: Viện Công nghệ số
    Kết quả xếp loại học tập: Giỏi
    Sơ lược thành tích
                                                                                         Ngày        tháng        năm  
XÁC NHẬN CỦA ĐƠN VỊ
(ký, họ và tên)	XÁC NHẬN CỦA SINH VIÊN
(ký, họ và tên)
 
MỞ ĐẦU
1. Tổng quan tình hình nghiên cứu
	Lĩnh vực phát hiện ảnh tổng hợp (Synthetic Image Detection) đã thu hút sự quan tâm lớn từ cộng đồng nghiên cứu quốc tế trong những năm gần đây. Dựa trên các công bố khoa học, các phương pháp tiếp cận có thể phân loại thành 4 nhóm chính:
Nhóm 1: Phương pháp dựa trên CNN (Spatial domain)
Đây là hướng tiếp cận nền tảng và phổ biến nhất. Rössler et al. (2019) [1] giới thiệu bộ dữ liệu chuẩn FaceForensics++ và chứng minh XceptionNet đạt độ chính xác >99% trên dữ liệu nén thấp; tuy nhiên, hiệu suất giảm mạnh khi gặp ảnh từ nguồn lạ. Wang et al. (2020) [2] sử dụng ResNet-50 huấn luyện trên ProGAN, đạt Average Precision 100% trên các GAN cũ nhưng gặp khó khăn với Diffusion Models thế hệ mới.
Nhận xét: Cả hai nghiên cứu đều đạt kết quả cao nhưng chỉ được đánh giá trên dữ liệu cùng phân phối (in-domain). Wang et al. (2020) báo cáo AP giảm đáng kể khi test cross-dataset, cho thấy hiện tượng overfitting vào đặc trưng cụ thể của GAN thay vì học được đặc trưng tổng quát của ảnh tổng hợp.
Nhóm 2: Phương pháp dựa trên Transformer và Attention
	Để khắc phục hạn chế về tầm nhìn cục bộ của CNN, Wodajo et al. (2021) [3] đề xuất kiến trúc Convolutional Vision Transformer, đạt 91,5% accuracy trên bộ DFDC. Cao et al. (2022) [4] sử dụng cơ chế Attention tập trung vào vùng ranh giới (blending boundary), cải thiện đáng kể khả năng phát hiện deepfake hoán đổi khuôn mặt.
	Nhận xét: Wodajo (2021) đạt 91,5% trên DFDC (bộ dữ liệu video deepfake), tuy nhiên chưa rõ hiệu quả trên ảnh tĩnh Diffusion. Ngoài ra, Transformer yêu cầu tài nguyên tính toán lớn (22–304M tham số), khó triển khai trên thiết bị phổ thông.
Nhóm 3: Phương pháp phân tích miền tần số (Frequency Analysis)
	Frank et al. (2020) [5] sử dụng biến đổi DCT, chứng minh các GAN để lại lỗi phổ tần số lặp lại bất thường, đạt accuracy >90% với chi phí tính toán thấp. Durall et al. (2020) [6] chứng minh rằng các bước up-sampling trong mô hình tạo sinh làm mất đặc trưng tần số cao của ảnh thật.
	Nhận xét: Phương pháp tần số hiệu quả với GAN (vì up-sampling tạo artifact tần số rõ ràng), nhưng bỏ qua các lỗi ngữ nghĩa (semantic artifacts) mà Diffusion Models tạo ra. Diffusion không sử dụng up-sampling mà khử nhiễu lặp lại (iterative denoising) — do đó các dấu vết tần số đặc trưng của GAN không còn xuất hiện.
Nhóm 4: Phương pháp Hybrid và XAI
	Xu hướng mới nhằm tăng độ tin cậy bằng cách tích hợp Explainable AI (XAI). Tuy nhiên, đa số nghiên cứu hiện tại vẫn hoạt động như "hộp đen" (black-box). Việc tích hợp Grad-CAM để trực quan hóa vùng giả mạo vẫn còn hạn chế và chưa được tối ưu hóa cho Diffusion Models.
	Nhận xét: Hầu hết các nghiên cứu chỉ dùng XAI hậu kỳ (post-hoc) để phân tích mô hình, chưa tích hợp vào giao diện người dùng cuối. Thiếu sự kết hợp giữa hiệu suất cao và khả năng giải thích trong cùng một hệ thống.
	Phân tích bản chất kỹ thuật: Tại sao detector GAN không hiệu quả trên Diffusion?
	Sự suy giảm hiệu năng của các phương pháp GAN detection trên Diffusion Models có nguyên nhân bản chất từ sự khác biệt trong cơ chế tạo ảnh. GAN tạo ảnh qua quá trình adversarial generation với các bước up-sampling (transposed convolution, nearest-neighbor upscaling), để lại các dấu vết phổ tần số lặp lại bất thường (spectral artifacts) — đây là đặc trưng mà các CNN detector như Wang et al. (2020) [2] và Frank et al. (2020) [5] khai thác để phân biệt. Ngược lại, Diffusion Models tạo ảnh qua quá trình khử nhiễu lặp lại (iterative denoising) với hàng trăm bước, làm mờ hoặc loại bỏ các dấu vết phổ nói trên. Kết quả là các đặc trưng đã học được từ GAN không còn khả năng phân biệt tốt trên ảnh Diffusion — giải thích tại sao các phương pháp như CNNDetection đạt AUC < 0,5 (phản-tương quan) khi gặp dữ liệu Diffusion.



Bảng 0.1: So sánh tổng hợp các nghiên cứu tiêu biểu
Nhóm	Nghiên cứu	Dữ liệu & Kết quả	Ưu điểm	Hạn chế
CNN	Rössler (2019), Wang (2020)	FaceForensics++, ProGAN. Acc ~99% nội miền	Nhanh, dễ triển khai	Overfitting, cross-dataset kém
Transformer	Wodajo (2021)	DFDC. Acc ~91,5%	Nắm bắt ngữ cảnh toàn cục	Yêu cầu tính toán lớn
Tần số	Frank (2020)	StyleGAN, BigGAN. Acc >90%	Phát hiện lỗi cấu trúc ẩn	Bỏ qua lỗi ngữ nghĩa
Đề tài này	HolmHz	GAN + Diffusion. AUC 0,998 (ID), 0,896 (OOD)	CNN + XAI + dữ liệu cập nhật	Không đề xuất kiến trúc mới

2. Khoảng trống nghiên cứu (Research Gap)
Tổng quan tài liệu cho thấy 3 khoảng trống rõ rệt:
1.	Thiếu tổng quát hóa trên Diffusion Models: Đa số nghiên cứu kinh điển (2019–2021) tập trung vào GAN. Hiện thiếu các đánh giá chuyên sâu trên Diffusion thế hệ mới (Midjourney, Stable Diffusion, DALL-E 3).
2.	Thiếu tính minh bạch (XAI): Người dùng cần hiểu "tại sao ảnh này là giả". Các mô hình hiện tại thiếu cơ chế giải thích trực quan trên giao diện.
3.	Nhu cầu mô hình nhẹ: Các mô hình Transformer quá nặng (22–304M tham số) để triển khai trên thiết bị cá nhân.
	Tổng hợp lại, trong phạm vi khảo sát của nhóm, chưa ghi nhận nghiên cứu nào đồng thời giải quyết đầy đủ cả ba vấn đề trên — tổng quát hóa trên Diffusion, tích hợp XAI vào giao diện người dùng, và triển khai trên mô hình nhẹ — trong một khung phương pháp thống nhất. Đề tài này nhắm đến việc thu hẹp khoảng trống đó.
3. Tính cấp thiết
       Sự bùng nổ của các mô hình Trí tuệ nhân tạo tạo sinh (Generative AI) đã tạo ra những hình ảnh tổng hợp với độ chân thực cực cao, gần như không thể phân biệt bằng mắt thường. Tình trạng này đang bị lạm dụng nghiêm trọng cho các mục đích xấu: tạo tin giả (fake news), lừa đảo trực tuyến, bôi nhọ danh dự và thao túng dư luận.
Tại Việt Nam, vấn nạn lừa đảo trực tuyến sử dụng Deepfake đang gia tăng đáng báo động (theo nhiều nguồn báo cáo an ninh mạng và phòng chống lừa đảo trực tuyến trong nước). Các đối tượng xấu sử dụng hình ảnh/video giả mạo người thân để lừa chuyển tiền hoặc bôi nhọ danh dự trên mạng xã hội.
       Đề tài này có tính cấp thiết cao vì giải quyết trực tiếp khoảng trống nêu trên: xây dựng hệ thống có khả năng phát hiện ảnh tổng hợp từ cả công nghệ cũ (GAN) và mới (Diffusion), tích hợp XAI để minh bạch hóa kết quả, đồng thời triển khai được trên thiết bị phổ thông.
4. Mục tiêu đề tài
Mục tiêu tổng quát: Xây dựng mô hình CNN nhẹ (Transfer Learning) phát hiện ảnh tổng hợp (GAN & Diffusion phổ biến) đạt AUC ≥ 0,92 trên tập kiểm tra nội miền và AUC ≥ 0,85 trên tập ngoài miền; tích hợp Grad-CAM và web demo suy luận ≤ 2 giây/ảnh trên máy tính cá nhân.
Mục tiêu cụ thể (KPI):
STT	Chỉ số KPI	Mục tiêu
1	Dataset ≥ 20,000 ảnh (50% thật, 50% giả; ≥ 3 nguồn GAN và 2 nguồn Diffusion)	Có
2	Accuracy ≥ 90%, F1 ≥ 0,90 nội miền; AUC ≥ 0,85 ngoài miền	Có
3	Kiểm tra ảnh nén JPEG (q=60) và scale/crop; giảm suy hao AUC ≤ 5%	Có
4	Grad-CAM hiển thị vùng nghi ngờ với ví dụ minh họa	Có
5	Web demo upload ảnh, trả kết quả + heatmap; latency ≤ 2s trên CPU	Có

5. Đối tượng và phạm vi nghiên cứu
Đối tượng nghiên cứu:
•	Các kiến trúc CNN hiện đại: EfficientNet-B0, ResNet-18, ViT-Small/16, Swin-Tiny
•	Đặc trưng miền không gian (spatial domain): bất thường cấu trúc, ánh sáng, màu sắc
•	Đặc trưng miền tần số (frequency domain): dấu vết phổ do quá trình up-sampling
Phạm vi nghiên cứu:
•	Dữ liệu: Ảnh tĩnh (static images), bao gồm ảnh chân dung và ảnh đa dạng chủ đề
•	Nguồn sinh ảnh: GAN (StyleGAN, ProGAN) và Diffusion (Stable Diffusion, Midjourney, DALL-E)
•	Hệ thống: Web demo (Proof-of-Concept) chạy trên máy tính cá nhân
6. Cách tiếp cận và phương pháp nghiên cứu
Cách tiếp cận:
•	Tiếp cận thực nghiệm (Experimental Research): xây dựng giả thuyết, thiết kế nhiều mô hình, chạy thí nghiệm với các tham số khác nhau
•	Tiếp cận Học có giám sát (Supervised Learning): mô hình được huấn luyện trên dữ liệu đã gán nhãn (0: Real, 1: Fake)
Phương pháp nghiên cứu (4 trụ cột):
1.	Chiến lược dữ liệu (Data Strategy): Tổng hợp từ 5 nguồn công khai Kaggle, chia train/val/test nội miền và test ngoài miền (OOD), đảm bảo cân bằng Real/Fake 1:1.
2.	Kiến trúc mô hình: Sử dụng 4 backbone pretrained trên ImageNet (EfficientNet-B0, ResNet-18, ViT-Small/16, Swin-Tiny) với Transfer Learning, fine-tune toàn bộ mạng.
3.	Huấn luyện và chống Overfitting: AdamW optimizer, Cosine Annealing scheduler, Early Stopping, JPEG Augmentation, WeightedRandomSampler.
4.	Đánh giá và XAI: Sử dụng bộ chỉ số AUC, Accuracy, F1-Score, Confusion Matrix, đường cong ROC. Tích hợp Grad-CAM để sinh bản đồ nhiệt.
7. Nội dung nghiên cứu
Đề tài được tổ chức thành 5 chương:
•	Chương 1: Giới thiệu — trình bày tổng quan, tính cấp thiết, mục tiêu
•	Chương 2: Cơ sở lý thuyết — CNN, Transfer Learning, GAN, Diffusion, XAI
•	Chương 3: Phương pháp và xây dựng hệ thống — quy trình dữ liệu, kiến trúc, web demo
•	Chương 4: Kết quả thực nghiệm và đánh giá
•	Chương 5: Kết luận và hướng phát triển
 
CHƯƠNG 1. GIỚI THIỆU

	Trong thời đại bùng nổ của Trí tuệ nhân tạo tạo sinh, sự phát triển vượt bậc của các mô hình như GAN (StyleGAN, ProGAN) và Diffusion (Stable Diffusion, Midjourney, DALL-E) đã cho phép tạo ra những hình ảnh tổng hợp có độ chân thực cực cao — gần như không thể phân biệt bằng mắt thường. Trong khi công nghệ này mang lại nhiều ứng dụng tích cực trong lĩnh vực sáng tạo nội dung, thiết kế và giải trí, nó cũng tạo ra mối đe dọa nghiêm trọng: tin giả (fake news), lừa đảo trực tuyến, bôi nhọ danh dự và thao túng dư luận. Đây không còn là mối đe dọa lý thuyết — hàng ngày có hàng triệu hình ảnh AI-generated được phát tán trên các nền tảng mạng xã hội, gây khó khăn lớn cho cả người dùng thông thường lẫn các hệ thống kiểm duyệt nội dung tự động.
	Tại Việt Nam, vấn nạn lừa đảo trực tuyến sử dụng hình ảnh và video giả mạo đang gia tăng đáng báo động. Theo nhiều nguồn báo cáo về an ninh mạng trong nước, các đối tượng lừa đảo ngày càng sử dụng hình ảnh deepfake để mạo danh người thân, người nổi tiếng hoặc cán bộ nhà nước nhằm chiếm đoạt tài sản. Đặc biệt nguy hiểm là xu hướng sử dụng ảnh khuôn mặt AI-generated để tạo tài khoản giả mạo trên các nền tảng thương mại điện tử và ngân hàng số. Thực trạng này đặt ra nhu cầu cấp bách về các công cụ phát hiện ảnh tổng hợp có thể hoạt động hiệu quả, nhanh và dễ tiếp cận trên các thiết bị phổ thông.
	Trong bối cảnh đó, các phương pháp phát hiện ảnh tổng hợp hiện có (CNNDetection, UniversalFakeDetect, DeepfakeBench) chủ yếu được thiết kế cho thế hệ GAN cũ (2018–2021) và chưa thích ứng hiệu quả với các mô hình Diffusion hiện đại — vốn tạo ra ảnh theo cơ chế hoàn toàn khác, không để lại các dấu vết phổ tần số đặc trưng của GAN. Khoảng trống nghiên cứu này — thiếu phương pháp phát hiện tổng quát bao phủ cả GAN lẫn Diffusion trên thiết bị phổ thông — chính là động lực thực hiện đề tài.
Xuất phát từ thực tiễn đó, đề tài "Xây dựng hệ thống phát hiện ảnh tổng hợp bằng Mạng nơ-ron tích chập (CNN)" được thực hiện với các mục tiêu:
1.	Xây dựng bộ dữ liệu đa dạng gồm ≥ 20,000 ảnh, bao phủ cả GAN và Diffusion từ 5 nguồn công khai.
2.	Huấn luyện và so sánh 4 kiến trúc CNN hiện đại: EfficientNet-B0, ResNet-18, ViT-Small/16, Swin-Tiny — sử dụng Transfer Learning từ ImageNet.
3.	Benchmark với 3 nghiên cứu baseline quốc tế: trên cùng bộ dữ liệu để đánh giá khách quan.
4.	Tích hợp Explainable AI (Grad-CAM) giúp người dùng hiểu lý do phân loại, tăng tính minh bạch.
5.	Xây dựng Web Demo cho phép upload ảnh và nhận kết quả phân loại Real/Fake kèm bản đồ nhiệt, với thời gian phản hồi ≤ 2 giây trên CPU.
Báo cáo được tổ chức thành 5 chương: Chương 1 giới thiệu tổng quan (chương này); Chương 2 trình bày cơ sở lý thuyết về CNN, GAN, Diffusion, Transfer Learning và XAI; Chương 3 mô tả phương pháp xây dựng bộ dữ liệu, thiết kế kiến trúc mô hình, pipeline huấn luyện/đánh giá và web demo; Chương 4 trình bày kết quả thực nghiệm, benchmark 7 mô hình và phân tích chi tiết; Chương 5 kết luận và đề xuất hướng phát triển.





CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ TỔNG QUAN
2.1 Trí tuệ nhân tạo tạo sinh (Generative AI)
2.1.1 Mạng đối sinh (Generative Adversarial Network — GAN)
GAN được giới thiệu bởi Goodfellow et al. (2014), gồm hai mạng nơ-ron cạnh tranh: Generator (G) tạo ảnh giả từ nhiễu ngẫu nhiên, Discriminator (D) phân biệt ảnh thật và giả. Hai mạng được huấn luyện song song theo bài toán minimax đến khi Generator tạo được ảnh không phân biệt được với ảnh thật.
 
Hình 2.1: Mạng đối sinh (Generative Adversarial Network — GAN)

	ProGAN (Karras et al., 2018): Huấn luyện từng lớp (progressive growing), tạo ảnh 1024×1024 chất lượng cao.
	StyleGAN/StyleGAN2 (Karras et al., 2019/2020): Sử dụng mapping network và style injection, tạo ảnh khuôn mặt cực kỳ chân thực.
Đặc trưng artifacts của GAN: Quá trình up-sampling (phóng to ảnh) trong Generator tạo ra các dấu vết phổ tần số lặp lại bất thường — đây là cơ sở để các phương pháp phát hiện GAN hoạt động [5], [6].
2.1.2 Mô hình khuếch tán (Diffusion Models)
Diffusion Models tạo ảnh theo quá trình 2 bước:
1.	Forward process (thêm nhiễu): Dần dần thêm nhiễu Gaussian vào ảnh thật qua T bước → ảnh trắng (noise).
2.	Reverse process (khử nhiễu): Mạng nơ-ron học cách khử nhiễu từng bước, từ noise → ảnh chân thực.
 
Hình 2.2: Diffusion Models


Các mô hình Diffusion phổ biến:
	Stable Diffusion (Rombach et al., 2022): Thực hiện quá trình khuếch tán trong không gian tiềm ẩn (latent space) thay vì pixel — giảm đáng kể chi phí tính toán.
 
Hình 2.3: Stable Diffusion




	DALL-E 2/3 (OpenAI): Text-to-image sử dụng CLIP embeddings + Diffusion.
	Midjourney: Mô hình thương mại tạo ảnh nghệ thuật chất lượng cao từ text prompt.
 
Hình 2.4: Midjourney

Sự khác biệt với GAN: Diffusion tạo ảnh qua quá trình khử nhiễu iterative (không phải adversarial training) → artifacts khác hoàn toàn so với GAN → nhiều phương pháp phát hiện GAN không hoạt động trên Diffusion [2].
2.2 Mạng nơ-ron tích chập (CNN)
2.2.1 Kiến trúc CNN cơ bản
CNN trích xuất đặc trưng ảnh qua các lớp Convolutional (phát hiện edges, textures, patterns), Pooling (giảm kích thước, giữ đặc trưng quan trọng), và Fully Connected (phân loại).
 
Hình 2.5: Mô hình mạng nơ-ron tích chập (CNN)

Inductive Bias của CNN — 3 giả định phù hợp cho dữ liệu ảnh: Locality (pixel lân cận có liên hệ), Translation Invariance (pattern có ý nghĩa ở mọi vị trí), và Hierarchical representation (từ edge → texture → object). Những giả định này giúp CNN học hiệu quả với dataset nhỏ-trung (< 100K ảnh), trong khi Transformer thiếu inductive bias tương tự.

2.2.2 EfficientNet
EfficientNet (Tan & Le, 2019) [8] được tìm ra bởi Neural Architecture Search (NAS) — tự động tìm kiến trúc tối ưu thay vì thiết kế thủ công. Đặc điểm:
	Compound Scaling: Scale đồng thời depth (d), width (w), và resolution (r) theo tỷ lệ cân bằng: d = α^φ, w = β^φ, r = γ^φ (với α·β²·γ² ≈ 2).
	MBConv blocks (Mobile Inverted Bottleneck): Sử dụng Depthwise Separable Convolution, giảm 8–9× số tham số so với convolution thường.
	Squeeze-and-Excitation (SE): Cơ chế attention trên channel — mô hình tự học trọng số cho từng kênh đặc trưng.
EfficientNet-B0 (cấu hình cơ sở) chỉ có ~4M tham số nhưng đạt top-1 accuracy 77,1% trên ImageNet — hiệu quả hơn ResNet-50 (26M params, 76,0%) [8].
 
Hình 2.6: EfficientNet (Tan & Le, 2019)

2.2.3 ResNet-18
ResNet (He et al., 2016) giải quyết vấn đề cốt lõi của mạng nơ-ron sâu: hiện tượng vanishing gradient làm cho mạng nhiều lớp khó huấn luyện hơn mạng ít lớp — một nghịch lý kỳ lạ vì lý thuyết cho thấy mạng sâu hơn phải mạnh hơn.

Giải pháp Residual Connection rất đơn giản về hình thức nhưng hiệu quả cực kỳ: output = F(x) + x. Thay vì học trực tiếp ánh xạ H(x), mạng chỉ cần học phần dư F(x) = H(x) - x. Khi lớp hiện tại không cần thiết, mạng chỉ cần đặt F(x) về 0 và output = x (identity mapping). Quan trọng hơn, gradient có thể lan truyền ngược trực tiếp qua nhánh tắt, bỏ qua các lớp convolution, giải quyết triệt để vanishing gradient.

ResNet-18 gồm 18 lớp convolution tổ chức thành 4 giai đoạn (layer1-layer4), mỗi giai đoạn gồm 2 Basic Block (mỗi block có 2 lớp 3x3 conv). Với 11M tham số, ResNet-18 được chọn trong đề tài vì hai lý do: (1) cấu trúc layer4 (residual block cuối) tạo ra spatial feature map rõ ràng, thuận tiện cho Grad-CAM cần lấy activation tại lớp convolution cuối; (2) kiến trúc đơn giản giúp ONNX export ổn định và inference nhanh trên CPU.
 
Hình 2.7: ResNet (He et al., 2016)



2.2.4 Vision Transformer (ViT)
ViT (Dosovitskiy et al., 2021) [9] áp dụng kiến trúc Transformer (vốn cho NLP) vào thị giác máy tính:
1.	Chia ảnh thành patches (16×16 pixels).
2.	Mỗi patch được linearly embed thành vector + positional encoding.
3.	Đưa qua Transformer Encoder (Multi-Head Self-Attention + Feed-Forward Network).
Ưu điểm: Nắm bắt global context (ngữ cảnh toàn cục) — mỗi patch "nhìn" tất cả patches khác. Nhược điểm: Cần dữ liệu rất lớn (>300M ảnh) mới vượt CNN. Với dataset nhỏ (<100K), CNN thường tốt hơn.
 
Hình 2.8: ViT (Dosovitskiy et al., 2021)

2.2.5 Swin Transformer
Swin Transformer (Liu et al., 2021) cải tiến ViT với:
	Shifted Window Attention: Tính attention trong cửa sổ 7×7 → giảm complexity từ O(n²) xuống O(n).
	Hierarchical feature maps: Giống CNN — tạo feature maps ở nhiều scale (1/4, 1/8, 1/16, 1/32).
 
Hình 2.9: Swin Transformer (Liu et al., 2021)

2.3 Học chuyển giao (Transfer Learning)
	Transfer Learning là kỹ thuật tái sử dụng kiến thức từ bài toán đã giải (pre-training trên ImageNet — 1,2 triệu ảnh, 1,000 lớp) sang bài toán mới (fine-tuning trên dữ liệu Real/Fake). Mô hình pretrained đã học các đặc trưng tổng quát (edges, textures, shapes), chỉ cần fine-tune lớp cuối cho bài toán cụ thể.
Trong đề tài này, nhóm sử dụng chiến lược unfreeze toàn bộ backbone (full fine-tuning) vì dataset v2 đủ lớn (28,220 ảnh) để fine-tune toàn bộ mạng mà không bị overfitting nghiêm trọng
2.4 Explainable AI (XAI) và Grad-CAM
2.4.1 Nhu cầu giải thích mô hình
Các mô hình deep learning thường hoạt động như "hộp đen" — chỉ trả kết quả (Real/Fake) mà không giải thích lý do. Trong bài toán phát hiện ảnh giả, người dùng cần biết vùng nào trên ảnh khiến mô hình nghi ngờ.
2.4.2 Grad-CAM (Gradient-weighted Class Activation Mapping)
Grad-CAM (Selvaraju et al., 2017) tạo bản đồ nhiệt (heatmap) chỉ ra vùng ảnh đóng góp nhiều nhất vào quyết định phân loại:
1.	Thực hiện forward pass: ảnh → model → prediction.
2.	Tính gradient của output class theo feature maps ở lớp convolution cuối.
3.	Global Average Pooling gradient → trọng số cho từng channel.
4.	Weighted combination → heatmap [H, W] ∈ [0, 1].
5.	ReLU → chỉ giữ vùng ảnh hưởng tích cực (positive influence).
Áp lên ảnh gốc → người dùng thấy vùng đỏ/vàng = vùng mô hình "nhìn" để đưa ra quyết định.
 
Hình 2.10: Grad-CAM (Selvaraju et al., 2017)
2.5 Các chỉ số đánh giá (Evaluation Metrics)

Bảng 2.1: Các chỉ số đánh giá mô hình phân loại
Chỉ số	Công thức	Ý nghĩa
Accuracy	(TP + TN) / (TP + TN + FP + FN)	Tỷ lệ dự đoán đúng
Precision	TP / (TP + FP)	Trong số predict Fake, bao nhiêu thực sự Fake
Recall	TP / (TP + FN)	Trong số thực sự Fake, phát hiện được bao nhiêu
F1-Score	2 × (P × R) / (P + R)	Trung bình điều hòa Precision và Recall
AUC	Diện tích dưới đường ROC	Khả năng phân biệt Real/Fake ở mọi ngưỡng (threshold-free)

Tại sao AUC quan trọng hơn Accuracy?
•	AUC đánh giá model ở tất cả ngưỡng (0,0 → 1,0), không phụ thuộc vào threshold cố định.
•	AUC = 1,0 → model phân biệt hoàn hảo. AUC = 0,5 → random. AUC < 0,5 → phản-tương quan.
•	Phù hợp khi dữ liệu có tỷ lệ Real/Fake không cân bằng.

2.6 Tổng quan và so sánh các phương pháp phát hiện ảnh tổng hợp
2.6.1 Phân loại các hướng tiếp cận
Các nghiên cứu phát hiện ảnh tổng hợp (Synthetic Image Detection) hiện nay tập trung vào bốn hướng chính:

Hướng 1 — Phân tích không gian (Spatial Domain): Khai thác các bất thường về texture, màu sắc, cấu trúc khuôn mặt. Đại diện: CNNDetection (Wang et al., 2020) huấn luyện ResNet-50 trên ProGAN, đạt AP cao trên tập in-domain nhưng giảm mạnh khi cross-dataset.

Hướng 2 — Phân tích miền tần số (Frequency Domain): Khai thác dấu vết phổ tần số do quá trình up-sampling trong GAN để lại. Frank et al. (2020) chứng minh GAN tạo ra các artifact tần số lặp lại bất thường có thể phát hiện bằng DCT. Tuy nhiên, hướng này kém hiệu quả với Diffusion Models vì Diffusion không dùng up-sampling thông thường.

Hướng 3 — Foundation Models và Transfer Learning: Sử dụng đặc trưng từ mô hình lớn huấn luyện trên dữ liệu quy mô web. UniversalFakeDetect (Ojha et al., 2023) dùng CLIP ViT-L/14 + Linear Probe, tham vọng "zero-shot universal detection". Tuy nhiên, kết quả thực tế trên dữ liệu Diffusion đa dạng vẫn còn hạn chế.

Hướng 4 — Hybrid và XAI: Kết hợp nhiều tín hiệu (spatial + frequency + semantic) và tích hợp cơ chế giải thích. Đây là hướng mới nhất, còn ít công trình hoàn chỉnh.

2.6.2 So sánh các nghiên cứu tiêu biểu

Bảng 2.2: So sánh chi tiết các phương pháp phát hiện ảnh tổng hợp tiêu biểu
Nghiên cứu	Năm	Kiến trúc	Dữ liệu train	ID Acc	Cross-dataset	Gan tốt	Diffusion tốt
CNNDetection [2]	2020	ResNet-50	ProGAN	~99%	Kém	Có	Không
Frank et al. [3]	2020	DCT+Classifier	StyleGAN	>90%	Trung bình	Có	Không
Wodajo et al. [8]	2021	CVT	DFDC (video)	91,5%	Chưa rõ	Có	Chưa rõ
Ojha et al. [4]	2023	CLIP ViT-L/14	Multi-GAN	72,2%*	Trung bình	Có	Không ổn định
DeepfakeBench [9]	2023	EfficientNet-B4	FF++ (video)	43,9%*	Kém	Có	Không
HolmHz (đề tài)	2025	EfficientNet-B0	GAN+Diffusion	98,4%	OOD AUC 0,896	Có	Có
(*) Đánh giá zero-shot trên Dataset v2 — không retrain

Nhận xét so sánh: Điểm khác biệt chính của đề tài so với các nghiên cứu trên là việc xây dựng bộ dữ liệu bao phủ đồng thời cả GAN (thế hệ cũ) và Diffusion (thế hệ mới), kết hợp với chiến lược JPEG Augmentation giúp mô hình tổng quát hóa tốt hơn. Tuy nhiên cần lưu ý rằng kết quả so sánh trong đề tài này sử dụng chế độ zero-shot cho baselines, không phản ánh hiệu năng tối đa của các phương pháp đó.

2.6.3 Khoảng trống nghiên cứu và định vị đề tài
Phân tích tổng quan cho thấy:
•	Hầu hết các phương pháp GAN detection cổ điển (2019–2021) thất bại trên Diffusion Models mới — do cơ chế tạo ảnh khác nhau về bản chất.
•	Các mô hình Foundation Model lớn (CLIP, 304M params) không nhất thiết vượt trội mô hình nhỏ hơn khi dữ liệu huấn luyện không phù hợp.
•	Ít nghiên cứu tích hợp đồng thời XAI (Grad-CAM) vào giao diện người dùng cuối trong cùng một hệ thống thực tế.
Đề tài HolmHz nhắm đến giải quyết đồng thời ba khoảng trống này: bộ dữ liệu đa thế hệ generator, mô hình nhẹ có thể triển khai thực tế, và web demo tích hợp XAI.

























CHƯƠNG 3. PHƯƠNG PHÁP VÀ XÂY DỰNG HỆ THỐNG
3.1 Tổng quan kiến trúc hệ thống
Hệ thống HolmHz được thiết kế theo kiến trúc module hóa (modular architecture), tách biệt rõ ràng giữa các thành phần:
 
Hình 3.1: Kiến trúc modular của hệ thống

Mã nguồn chính nằm trong src/holmhz/ với cấu trúc:
•	backbones/: Backbone feature extractors (EfficientNet, Timm)
•	detectors/: Detector = Backbone + Classification Head
•	data/: Dataset, DataLoader, Augmentation transforms
•	training/: Trainer class, Loss functions, Schedulers
•	evaluation/: Evaluator, Metrics (AUC, Accuracy, F1, Precision, Recall)
•	xai/: Grad-CAM Explainer
•	utils/: Registry Pattern, Logger
3.2 Quy trình xây dựng bộ dữ liệu
3.2.1 Thu thập dữ liệu
Bộ dữ liệu v2 được tổng hợp từ 5 nguồn công khai chính trên nền tảng Kaggle (RVF10K, DeepDetect-2025, Diffusion Fakes, CIPLab Faces, Camera vs AI). Bảng 3.1 liệt kê 6 dòng, trong đó Deepfake Collection Real là subset được trích xuất và tổng hợp thêm từ các nguồn hiện có nhằm cân bằng tỷ lệ Real — không phải nguồn Kaggle độc lập thứ 6:





Bảng 3.1: Nguồn dữ liệu trong bộ dữ liệu v2
STT	Nguồn	Nội dung	Loại generator	Số lượng	Giấy
Phép
1	RVF10K	Khuôn mặt CelebA (real) + StyleGAN (fake)	StyleGAN	8,000	CC BY-NC-SA 4.0
2	DeepDetect-2025	Ảnh đa dạng: phong cảnh, vật thể, con người	Diffusion mixed	8,000	Apache 2.0
3	Diffusion Fakes	DALL-E, Midjourney, SD, DeepFaceLab, FaceShifter,…	6+ generators	4,024	Community Data License Agreement
4	CIPLab Faces	Khuôn mặt manipulation (Chung-Ang University)	Face manipulation	3,266	CC BY-NC-SA 4.0
5	Camera vs AI	Ảnh camera thật vs AI-generated	Mixed AI	400	U.S. Government Works
6	Deepfake Collection Real (subset from Diffusion Fakes)	Ảnh thật đa dạng — bổ sung cân bằng	Real/Fake —   (Real only)	4,712	CC BY-NC-SA 4.0
Tổng (Train)			~28,220		

3.2.2 Tổ chức dữ liệu bằng Manifest JSON
Thay vì sử dụng ImageFolder (chỉ biết path → label), đề tài sử dụng JSON manifest — mỗi mẫu lưu thêm metadata:
{
  "path": "data/raw_v2/rvf10k_train_real/00001.jpg",
  "label": 0,
  "source": "rvf10k_train_real",
  "category": "real"
}	

Ưu điểm: (1) Biết nguồn gốc từng ảnh (source) → phân tích per-source. (2) Chia tập dễ dàng bằng script, không phụ thuộc thư mục. (3) Reproducible — cùng manifest = cùng split.
3.2.3 Chia tập dữ liệu
Sử dụng stratified split với seed=42:

Bảng 3.2: Chia tập dữ liệu (Train/Val/Test ID/Test OOD)
Split	Tổng	Real	Fake	Mục đích
Train	28,220	14,554	13,666	Huấn luyện mô hình
Validation	3,526	1,819	1,707	Tinh chỉnh hyperparameters, Early Stopping
Test ID	3,526	1,819	1,707	Đánh giá nội miền
Test OOD	182	94	88	Đánh giá khả năng tổng quát hóa

Chiến lược OOD: Tập Test OOD sử dụng nguồn camera_real và camera_ai — hoàn toàn không xuất hiện trong tập huấn luyện. Mục đích: kiểm tra khả năng phát hiện trên dữ liệu "chưa từng thấy" (unseen data).

Lưu ý về dữ liệu:
1.	Nguy cơ trùng lặp giữa các nguồn: Do dữ liệu được tổng hợp từ 5 nguồn Kaggle độc lập, tồn tại nguy cơ trùng lặp giữa các nguồn (ví dụ: cùng một ảnh gốc xuất hiện trong nhiều bộ dữ liệu). Trong phạm vi đề tài, nhóm chưa thực hiện kiểm tra trùng lặp bằng perceptual hashing hoặc feature-level deduplication — đây là hạn chế cần lưu ý khi diễn giải kết quả.
2.	Kích thước tập OOD: Tập Test OOD chỉ gồm 182 ảnh. Với mức accuracy quan sát được là 78% (n=182), khoảng tin cậy 95% (Wilson score) vào khoảng ±6,0%. Do đó, các kết luận về khả năng tổng quát hóa mang tính chỉ báo sơ bộ hơn là khẳng định thống kê vững chắc. Cần tập OOD lớn hơn (≥ 1.000 ảnh) để xác nhận.
Bảng 3.3: Feature dimension theo backbone


Backbone	Feature Dimension
EfficientNet-B0	1,280
ResNet-18	512
ViT-Small/16	384
Swin-Tiny	768



3.3 Thiết kế kiến trúc mô hình
3.3.1 Kiến trúc tổng quát: Backbone + Head
Mọi mô hình HolmHz đều tuân theo kiến trúc 2 phần:
Input [B, 3, 224, 224]
  → Backbone (pretrained ImageNet)     → [B, feature_dim]
  → Dropout(p=0,3)                     → [B, feature_dim]
  → Linear(feature_dim, 1)             → [B, 1] (logits)


Output là logits (chưa qua Sigmoid). Training dùng BCEWithLogitsLoss (ổn định số học). Inference dùng torch.sigmoid(logits) → P(Fake) ∈ [0, 1].
3.3.2 Registry Pattern
Đề tài sử dụng Registry Pattern (lấy cảm hứng từ DeepfakeBench) để quản lý mô hình:
@DETECTOR_REGISTRY.register("efficientnet_b0")
class EfficientNetDetector(BaseDetector):
    ...

# Tạo model từ tên string (config-driven)
model = DETECTOR_REGISTRY.build("efficientnet_b0", pretrained=True)

Ưu điểm: Thay đổi mô hình chỉ cần đổi model.name trong config YAML — không sửa code.

3.3.3 Hỗ trợ đa kiến trúc qua Timm
Các mô hình ResNet-18, ViT-Small/16, Swin-Tiny được triển khai qua TimmDetector — wrapper chung sử dụng thư viện timm (PyTorch Image Models, 700+ mô hình pretrained):
class TimmDetector(BaseDetector):
    def __init__(self, model_name, pretrained=True, dropout=0,3, ...):
        self.backbone=TimmBackbone(model_name=model_name, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.get_features_dim(), 1),   )	
3.4 Phương pháp huấn luyện
3.4.1 Data Augmentation Pipeline
Sử dụng thư viện Albumentations (nhanh hơn torchvision 2–5×, hỗ trợ JPEG compression):

Bảng 3.4: Training augmentation pipeline
Bước	Kỹ thuật	Tham số	Mục đích
1	RandomResizedCrop hoặc Resize	scale 0,7–1,0, 224×224	Phá spatial artifacts
2	HorizontalFlip	p = 0,5	Tăng đa dạng (khuôn mặt đối xứng)
3	ImageCompression (JPEG)	p = 0,7, quality 50–95	Chống shortcut learning
4	OneOf: Blur / Noise / Downscale	p = 0,5	Mô phỏng nhiễu thực tế
5	ColorJitter	brightness=0,2, contrast=0,2	Mô phỏng điều kiện ánh sáng
6	Normalize	ImageNet mean/std	Chuẩn hóa cho pretrained backbone
7 	ToTensorV2	--	Chuyển numpy 
-> PyTorch   

Validation/Test transforms: Chỉ Resize + Normalize + ToTensorV2 (không augment — đo sức mạnh thật).
	JPEG Augmentation — Kỹ thuật có ảnh hưởng đáng kể: JPEG compression ngẫu nhiên (quality 50–95, p=0,7) buộc mô hình học đặc trưng bền vững thay vì dựa vào compression artifacts. Dải quality 50–95 bao phủ mức nén thực tế của Facebook (80–85), Instagram (85–90) và WhatsApp (60–75). Kỹ thuật này được lấy cảm hứng từ CNNDetection [2] và là yếu tố có ảnh hưởng đáng kể đến cải thiện OOD AUC từ 0,440 lên 0,896 trong điều kiện thí nghiệm của đề tài.
3.4.2 WeightedRandomSampler
Do các nguồn dữ liệu có số lượng chênh lệch (rvf10k: 8,000 ảnh vs camera: 218 ảnh), đề tài sử dụng WeightedRandomSampler:
	Mỗi source được gán weight = max_count / source_count.
	Source ít ảnh → weight cao → được sample nhiều hơn.
	Hiệu quả: cân bằng tất cả sources trong mỗi epoch mà không cần duplicate dữ liệu.
3.4.3 Optimizer và Learning Rate Scheduler
	AdamW: Adam với weight decay decoupled — hiệu quả cho fine-tuning pretrained models.
	Cosine Annealing Scheduler: Learning rate giảm dần theo hàm cosine từ lr_max → 0 qua T epochs.
	Early Stopping: Theo dõi val AUC, dừng sau 7 epochs không cải thiện (patience = 7).
3.4.4 Loss Function
Sử dụng BCEWithLogitsLoss (Binary Cross-Entropy with Logits) — kết hợp Sigmoid + BCE trong 1 hàm, numerical stable hơn tính Sigmoid riêng:
L = -[y × log(σ(x)) + (1-y) × log(1-σ(x))]
Với pos_weight = 1.0 (cân bằng, không thiên vị Real hay Fake).
3.5 Pipeline huấn luyện (Trainer)
Lớp Trainer  (src/holmhz/training/trainer.py) quản lý toàn bộ quá trình huấn luyện:
Mỗi epoch:
  1. Training loop: batch → forward → loss → backward → optimizer step
  2. Validation loop: batch → forward → compute metrics (AUC, Acc, F1)
  3. LR Scheduler step
  4. Early Stopping check (val AUC cải thiện?)
  5. Save checkpoint nếu val AUC tốt nhất
  6. Log metrics → W&B (nếu có)

Checkpoint format (file .pt):
{
    "epoch": int,
    "model_state_dict": OrderedDict,  # Trọng số mô hình
    "optimizer_state_dict": OrderedDict,
    "scheduler_state_dict": dict,
    "best_metric": float,  # Best val AUC
    "config": dict,  # Hyperparameters
}
3.6 Pipeline đánh giá (Evaluator)
Lớp Evaluator (src/holmhz/evaluation/) thực hiện:
1.	Inference trên toàn bộ test set → thu thập logits, labels, sources.
2.	Tính overall metrics: AUC, Accuracy, F1, Precision, Recall.
3.	Tính per-source metrics: breakdown theo từng nguồn dữ liệu.
4.	Xuất báo cáo JSON (eval_report.json) và biểu đồ (ROC curve, Confusion Matrix).
3.7 Thiết kế Web Demo
3.7.1 Kiến trúc
Web demo được xây dựng bằng Gradio (Python) — framework tạo giao diện ML demo nhanh:
Người dùng → Upload ảnh
  → Gradio UI (web/app.py)
  → Load model ONNX (web/config.py)
  → Resize 224×224 + Normalize
  → Forward pass → P(Fake)
  → Grad-CAM heatmap
  → Hiển thị: Real/Fake (%) + Heatmap overlay

3.7.2 Tối ưu suy luận với ONNX
Mô hình PyTorch được export sang định dạng ONNX (Open Neural Network Exchange) để tối ưu tốc độ suy luận:
	Loại bỏ overhead Python/PyTorch.
	Quantization INT8/FP16 giảm kích thước model.
	Tương thích chạy trên CPU phổ thông.
Kết quả: Latency ~1,5 giây/ảnh trên CPU laptop — đạt KPI ≤ 2 giây.

3.7.3 Tích hợp Grad-CAM
Module GradCAMExplainer (src/holmhz/xai/gradcam.py) sử dụng thư viện pytorch-grad-cam:
1.	Tự động xác định target layer theo kiến trúc backbone:
o	EfficientNet → conv_head (lớp convolution cuối)
o	ResNet → layer4 (block residual cuối)
o	ViT → norm (LayerNorm cuối)
o	Swin → norm (LayerNorm cuối)
2.	Sinh heatmap [H, W] ∈ [0, 1].
Lưu ý: Grad-CAM phù hợp tự nhiên với CNN do có feature map không gian rõ ràng. Với ViT/Swin, cần xử lý đặc biệt (reshape token sequences hoặc dùng Attention Rollout). Trong phạm vi đề tài, Grad-CAM chủ yếu được sử dụng và trình diễn trên EfficientNet-B0 và ResNet-18 trong web demo.
3.	Overlay lên ảnh gốc → hiển thị cho người dùng.
3.8 Đề xuất kiến trúc triển khai đám mây (AWS)
Nhằm định hướng khả năng thương mại hóa và triển khai thực tế, nhóm đề xuất kiến trúc cloud-native trên nền tảng Amazon Web Services (AWS) tuân theo các nguyên tắc Well-Architected Framework: bảo mật tối thiểu đặc quyền, sẵn sàng cao, tối ưu chi phí và khả năng mở rộng.
 
Hình 3.2: Kiến trúc triển khai đám mây HolmHz trên AWS

3.8,1 Các thành phần kiến trúc
Kiến trúc được tổ chức thành 2 lớp rõ ràng:

Bảng 3.5: Các thành phần kiến trúc AWS — Lớp Global (Ngoài Region — phục vụ toàn cầu)
Service	Vai trò
Route 53	Phân giải DNS — ánh xạ tên miền tùy chỉnh (api.holmhz.xyz) tới CloudFront
CloudFront	CDN toàn cầu — định tuyến user đến Edge Location gần nhất (VD: TP.HCM), phục vụ cả API request lẫn ảnh heatmap tĩnh từ S3
WAF	Tường lửa ứng dụng web — lọc request độc hại, giới hạn kích thước file (<5MB), rate limiting (<100 req/IP/phút)

Bảng 3.6: Các thành phần kiến trúc AWS — Lớp Regional (Bên trong Region ap-southeast-1 — Singapore)
Service	Vai trò
API Gateway	HTTP Router — nhận POST /predict, xác thực API Key, điều hướng tới Lambda
AWS Lambda	Compute — chạy inference EfficientNet-B0 ONNX, tạo Grad-CAM heatmap, trả kết quả
Amazon ECR	Container Registry — lưu Docker Image chứa ONNX Runtime và model
Amazon S3	Object Storage — lưu ảnh heatmap kết quả, CloudFront làm CDN phía trước
CloudWatch	Monitoring — thu thập log, metric, cảnh báo khi error rate tăng
Secrets Manager	Lưu trữ thông tin nhạy cảm (API Key) — Lambda đọc một lần khi khởi động
Systems Manager	Lưu trữ cấu hình động (tên S3 bucket, CDN domain) — thay đổi không cần redeploy
3.8,2 Luồng xử lý request
Hệ thống hoạt động theo 2 luồng tách biệt:
Luồng 1 — Phân tích ảnh (POST request):
User → Route 53 (DNS) → CloudFront Edge (HCM)
     → WAF (filter) → API Gateway (auth)
     → Lambda (inference + Grad-CAM)
     → S3 (save heatmap_{uuid}.png)
     → Trả JSON: {label, prob, heatmap_url}
Luồng 2 — Lấy ảnh heatmap (GET request):
User → CloudFront (check cache)
     → Cache HIT: trả ảnh ngay từ Edge (0ms thêm)
     → Cache MISS: Origin S3 → trả ảnh + cache tại Edge
3.8,3 CI/CD Pipeline
Quy trình triển khai tự động hóa hoàn toàn với GitHub Actions và Terraform:
Dev push code → GitHub Actions trigger
  → pytest (kiểm thử tự động)
  → Docker build image (ONNX Runtime + model)
  → Push image lên ECR (tag: git SHA)
  → Terraform apply (cập nhật hạ tầng nếu có thay đổi)
  → Lambda update-function-code (zero downtime)
  → Smoke test (invoke Lambda với ảnh test)
Terraform đảm bảo toàn bộ hạ tầng được định nghĩa dưới dạng Infrastructure as Code (IaC) — có thể tái tạo môi trường hoàn chỉnh trong vài phút.
3.8,4 Tối ưu chi phí (Cost Optimization)
Kiến trúc Serverless được chọn vì phù hợp với workload nghiên cứu (traffic thấp, không liên tục):
•	Lambda: Tính tiền theo số lượng request, không tốn tiền khi idle.
•	S3 Lifecycle Policy: Tự động xóa heatmap sau 24 giờ — giảm chi phí lưu trữ.
•	CloudFront Cache: Giảm số lần Lambda được gọi cho nội dung tĩnh.
•	Ước tính chi phí: < $5/tháng cho traffic demo nghiên cứu (~1,000 request/ngày).

3.9 Quy trình kiểm thử và đảm bảo chất lượng
3.9.1 Chiến lược kiểm thử đa tầng
Để đảm bảo độ tin cậy của hệ thống, đề tài áp dụng quy trình kiểm thử theo 3 tầng:

Tầng 1 — Kiểm thử dữ liệu (Data Validation):
•	Xác minh tính toàn vẹn file: Kiểm tra tất cả ảnh đọc được bằng PIL (Image.open + convert RGB) — phát hiện và loại bỏ file hỏng.
•	Kiểm tra phân phối nhãn: Xác nhận tỷ lệ Real/Fake ≈ 1:1 trong tất cả các split (train/val/test).
•	Kiểm tra rò rỉ dữ liệu (Data Leakage): Đảm bảo các nguồn OOD (camera_real, camera_ai) không xuất hiện trong tập train — được kiểm soát qua cơ chế manifest JSON và split script.

Tầng 2 — Kiểm thử mô hình (Model Validation):
•	Sanity check: Sau epoch 1, val AUC phải > 0,6. Nếu thấp hơn → dừng và kiểm tra lại cấu hình.
•	Overfitting check: So sánh train loss và val loss sau mỗi epoch. Gap > 0,1 được ghi nhận là cảnh báo.
•	Reproducibility: Cố định seed=42 cho tất cả nguồn ngẫu nhiên (Python random, NumPy, PyTorch, CUDA). Chạy lại cùng config → kết quả sai số < 0,001 AUC.

Tầng 3 — Kiểm thử hệ thống (System Testing):
•	Smoke test web demo: Upload 5 ảnh chuẩn (3 real, 2 fake đã biết nhãn) → kiểm tra kết quả đúng và Grad-CAM sinh được.
•	Latency test: Đo thời gian inference trên CPU laptop phổ thông (Intel Core i5, 8GB RAM) → xác nhận ≤ 2 giây/ảnh.
•	Edge case test: Upload ảnh kích thước rất nhỏ (< 50px), ảnh grayscale, ảnh PNG trong suốt → hệ thống xử lý không crash.

3.9.2 Kết quả kiểm thử

Bảng 3.7: Tóm tắt kết quả kiểm thử hệ thống
Hạng mục kiểm thử	Phương pháp	Kết quả	Trạng thái
Data integrity (28.220 ảnh train)	PIL readability check	28.220/28.220 đọc được	Đạt
Label balance (train)	Thống kê nhãn	51,6% Real / 48,4% Fake	Đạt
OOD leakage check	Kiểm tra manifest	0 ảnh OOD trong train	Đạt
Reproducibility (seed=42)	Chạy lại 2 lần	Val AUC sai số < 0,001	Đạt
Web demo smoke test	5 ảnh chuẩn	5/5 đúng nhãn	Đạt
Latency (CPU)	Đo thực tế	~1,5 giây/ảnh	Đạt
Edge case handling	PNG, grayscale, small	Không crash	Đạt


CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ
4.1 Môi trường và tham số thực nghiệm
Toàn bộ quá trình huấn luyện được thực hiện trên nền tảng Kaggle với cấu hình phần cứng và phần mềm như sau:
                          
Bảng 4.1: Môi trường và tham số thực nghiệm
Hạng mục	Chi tiết
Nền tảng	Kaggle Notebooks
GPU	NVIDIA Tesla T4 × 2 (16 GB VRAM mỗi GPU)
Framework	PyTorch 2.x
Optimizer	AdamW
Learning Rate	3×10⁻⁴
Weight Decay	0,01
LR Scheduler	Cosine Annealing
Loss Function	BCEWithLogitsLoss (pos_weight = 1,0)
Epochs	30 (Early Stopping patience = 7, monitor = val AUC)
Image Size	224 × 224 pixels
Sampler	WeightedRandomSampler (cân bằng Real/Fake)
Augmentation	JPEG compression (quality 50–95, p=0,7), Gaussian Blur, Random Flip, Color Jitter
Seed	42 (đảm bảo reproducibility)

Ghi chú về cấu hình theo mô hình: EfficientNet-B0 và ResNet-18 sử dụng batch_size = 32; ViT-Small/16 và Swin-Tiny sử dụng batch_size = 16 (do giới hạn VRAM). Tất cả 4 mô hình sử dụng cùng hyperparameters để đảm bảo tính công bằng trong so sánh.
Các file cấu hình chi tiết:
•	EfficientNet-B0: configs/train_v9.yaml
•	ResNet-18: configs/train_resnet18_v2.yaml
•	ViT-Small/16: configs/train_vit_small_v2.yaml
•	Swin-Tiny: configs/train_swin_tiny_v2.yaml
4.2 Bộ dữ liệu
Bộ dữ liệu được sử dụng là Dataset v2 (data/raw_v2/), được tổng hợp từ 5 nguồn dữ liệu công khai trên nền tảng Kaggle đã được chuẩn hóa.

                           
Bảng 4.2: Thống kê bộ dữ liệu v2 theo split
Split	Tổng	Real	Fake	Tỷ lệ Real:Fake	Mục đích
Train	28,220	14,554 (51,6%)	13,666 (48,4%)	≈ 1:1	Huấn luyện
Validation	3,526	1,819 (51,6%)	1,707 (48,4%)	≈ 1:1	Tinh chỉnh, Early Stopping
Test ID (nội miền)	3,526	1,819 (51,6%)	1,707 (48,4%)	≈ 1:1	Đánh giá nội miền
Test OOD (ngoài miền)	182	94 (51,6%)	88 (48,4%)	≈ 1:1	Đánh giá tổng quát hóa
Tổng cộng	35,454				





Bảng 4.3: Chi tiết nguồn dữ liệu trong bộ dữ liệu v2
Nguồn 	Nền tảng	Nội dung	Loại ảnh (Label)	Số lượng (Train)
RVF10K	Kaggle	Khuôn mặt thật (CelebA) và giả (StyleGAN)	rvf10k_train_real, rvf10k_train_fake, rvf10k_valid_real, rvf10k_valid_fake	8,000
DeepDetect-2025	Kaggle (deanberto/deepdetect-2025)	Ảnh đa dạng: phong cảnh, vật thể, con người. Real + Diffusion fake	dd2025_real, dd2025_fake

	8,000

Diffusion Fakes	Kaggle (birdy654/deepfake-generation-and-detection-dataset) + tự thu thập	Ảnh fake từ nhiều AI generators: DALL-E, Midjourney, Stable Diffusion, StyleGAN, DeepFaceLab, Face2Face, FaceShifter, NeuralTextures	dalle_fake, midjourney_fake, sd_fake	4.024
CIPLab Faces	Kaggle (ciplab/real-and-fake-face-detection)	Khuôn mặt thật và giả (face manipulation) từ CIPLab, Chung-Ang University	ciplab_training_real, ciplab_training_fake	3.266
Camera vs AI	Kaggle	Ảnh chụp camera thật (iPhone, Samsung) vs AI-generated	camera_train_real, camera_train_ai (Train/ID); camera_real, camera_ai (OOD)	218 (Train) + 182 (OOD)
Deepfake Collection Real	Kaggle (subset từ Diffusion Fakes)	Ảnh thật đa dạng — bổ sung cân bằng dataset	deepfake_collection_real	4.712

AI Generators được bao phủ: StyleGAN, DALL-E, Midjourney, Stable Diffusion, DeepFaceLab, Face2Face, FaceShifter, NeuralTextures, CIPLab manipulation — tổng cộng 8+ loại generator, bao gồm cả thế hệ GAN cũ và Diffusion mới.
Chiến lược chia tập OOD: Tập Test OOD (182 ảnh) sử dụng hoàn toàn nguồn Camera vs AI — dữ liệu ảnh chụp camera thật và ảnh AI-generated từ nguồn không có trong tập huấn luyện — nhằm đánh giá khả năng tổng quát hóa (generalization) của mô hình trên dữ liệu chưa từng thấy.
4.3 Kết quả huấn luyện 4 mô hình HolmHz
 

Bảng 4.4: Kết quả huấn luyện 4 mô hình HolmHz trên Dataset v2
Mô hình	Kiến trúc	Tham số	Batch Size	Best Epoch	Val AUC	Checkpoint
EfficientNet-B0 (v9)	EfficientNet-B0	4M	32	25/30	0,9993	best_v9.pt (46,3 MB)
ResNet-18		ResNet-18	11M	32	28/30	0,9956	best_resnet18_v2.pt (128,0 MB)
ViT-Small/16	DeiT-Small/16	22M	16	29/30	0,9735	best_vit_small_v2.pt (248,1 MB)
Swin-Tiny	Swin Transformer Tiny	28M	16	0/30	0,6198	best_swin_tiny_v2.pt (315,1 MB)

 Swin-Tiny: Huấn luyện thất bại — mô hình không cải thiện qua epoch 0 (xem phân tích mục 4.7).
Nhận xét:
	EfficientNet-B0 hội tụ nhanh nhất và đạt Val AUC cao nhất (0,9993) ở epoch 25.
	ResNet-18 ổn định, hội tụ ở epoch 28 với Val AUC 0,9956.
	ViT-Small/16 cần toàn bộ 29 epochs nhưng chỉ đạt Val AUC 0,9735 — thấp hơn hai mô hình CNN.
	Swin-Tiny hoàn toàn thất bại: best epoch = 0 nghĩa là mô hình pre-trained ban đầu đã là kết quả tốt nhất, quá trình fine-tune chỉ làm mô hình xấu đi.
4.4 Benchmark tổng hợp 7 mô hình
Để đánh giá khách quan, nhóm nghiên cứu so sánh 4 mô hình HolmHz với 3 nghiên cứu baseline quốc tế. Lưu ý: ViT-Small/16 và Swin-Tiny được đưa vào với mục đích đối chứng kiến trúc (Transformer vs CNN), không phải đóng góp kiến trúc chính của đề tài. Đóng góp chính tập trung vào nhóm CNN: EfficientNet-B0 và ResNet-18, Tất cả 7 mô hình được đánh giá trên cùng bộ dữ liệu (test_id: 3.526 ảnh, test_ood: 182 ảnh) để đảm bảo tính công bằng.
3 nghiên cứu baseline được chọn:
	CNNDetection (Wang et al., CVPR 2020) [2]: ResNet-50 huấn luyện trên ProGAN — đại diện phương pháp kinh điển GAN detection.
	UniversalFakeDetect (Ojha et al., CVPR 2023) [7]: CLIP ViT-L/14 + Linear Probe — đại diện SOTA hiện đại dùng Foundation Models (304M tham số).
	DeepfakeBench (Yan et al., 2023) [10]: EfficientNet-B4 huấn luyện trên FaceForensics++ — đại diện pipeline phát hiện deepfake khuôn mặt video.
 

Bảng 4.5: Benchmark tổng hợp 7 mô hình (ID và OOD)
Nhóm	Phương pháp	Kiến trúc	Tham số	ID AUC	ID Acc	ID F1	OOD AUC	OOD Acc	Retrain Dataset v2
Baseline	CNNDetection [2]	ResNet-50	~23M	0,662	0,524	0,037	0,325	0,517	Không (Zero-shot)
Baseline	UniversalFakeDetect [7]	CLIP ViT-L/14	~304M	0,722	0,715	0,627	0,486	0,533	Không (Zero-shot)
Baseline	DeepfakeBench [10]	EfficientNet-B4	~19M	0,439	0,450	0,406	0,536	0,539	Không (Zero-shot)
Ours	EfficientNet-B0 (v9)	EfficientNet-B0	4M	0,998	0,984	0,984	0,896	0,780	Có
Ours	ResNet-18	ResNet-18	11M	0,995	0,971	0,970	0,865	0,802	Có
Ours	ViT-Small/16	ViT-Small/16	22M	0,974	0,921	0,920	0,833	0,747	Có
Ours	Swin-Tiny	Swin-T	28M	0,620	0,537	0,633	0,811	0,676	Có

Ghi chú:  Swin-Tiny: fine-tuning thất bại (best epoch = 0). Kết quả OOD AUC = 0,811* phản ánh hiệu năng của pre-trained weights ImageNet gốc ở chế độ chưa fine-tune, không nên so sánh bình đẳng với các mô hình đã fine-tune thành công. Các baseline sử dụng pre-trained weights gốc do tác giả công bố (zero-shot), không retrain trên Dataset v2. ViT/Swin được đưa vào với mục đích đối chứng kiến trúc. 
Nhận xét tổng quan:
	EfficientNet-B0 v9 đạt kết quả tốt nhất tổng thể trong điều kiện thí nghiệm: ID AUC 0,998 và OOD AUC 0,896 — cao hơn cả 3 nghiên cứu baseline.
	ResNet-18 đạt OOD Accuracy cao nhất (80,2%) — cân bằng nhất giữa phát hiện fake và nhận diện ảnh thật.
	Cả 3 nghiên cứu baseline đều cho kết quả thấp trên bộ dữ liệu Diffusion hiện đại (AUC < 0,73 cho ID, < 0,54 cho OOD). Điều này phù hợp với khoảng trống nghiên cứu đã nhận diện: các phương pháp được thiết kế cho GAN có thể không hoạt động hiệu quả trên Diffusion.
	Kết quả cho thấy: mô hình nhỏ hơn (4M tham số) có thể đạt kết quả tốt hơn khi được huấn luyện trên dữ liệu phù hợp với bài toán mục tiêu — nhấn mạnh tầm quan trọng của chiến lược dữ liệu so với kích thước mô hình.
4.5 Phân tích biểu đồ
 
4.5.1 Biểu đồ cột — ID AUC vs OOD AUC 

 
Hình 4.1: Biểu đồ cột – ID AUC và OOD AUC


Phân tích: Biểu đồ cho thấy sự vượt trội rõ rệt của các mô hình HolmHz so với baselines. Cụ thể, EfficientNet-B0 và ResNet-18 đều vượt ngưỡng KPI 0,85 trên OOD, trong khi tất cả research baselines đều nằm ở mức hoặc dưới mức ngẫu nhiên (0,5).

Đặc biệt đáng chú ý là khoảng cách giữa ID AUC và OOD AUC của từng mô hình. EfficientNet-B0 v9 có khoảng cách nhỏ nhất (0,998 - 0,896 = 0,102), cho thấy mô hình tổng quát hóa tốt nhất. Ngược lại, CNNDetection có khoảng cách lớn nhất (0,662 - 0,325 = 0,337), xác nhận hiện tượng overfitting vào đặc trưng GAN. Kết quả này nhất quán với giả thuyết ban đầu: mô hình được huấn luyện trên dữ liệu đa dạng (cả GAN lẫn Diffusion) sẽ tổng quát hóa tốt hơn trên OOD so với mô hình chuyên biệt cho một loại generator.
4.5.2 Biểu đồ Radar — Đa chỉ số 
 
Hình 4.2: Biểu đồ Radar – Đa chỉ số


Phân tích: EfficientNet-B0 v9 (màu xanh dương) có diện tích lớn nhất và gần tròn nhất — cân bằng tốt trên tất cả 5 chỉ số. UniversalFakeDetect (CLIP, 304M tham số) bị méo lệch nghiêm trọng, chỉ tốt trên 1–2 chỉ số ID — chứng minh rằng mô hình lớn không đồng nghĩa với hiệu quả.






4.5.3 Heatmap OOD per-source 

 
Hình 4.3: Heatmap OOD per-source

Bảng 4.6: Độ chính xác OOD theo nguồn dữ liệu
Mô hình	camera_ai (Fake)	camera_real (Real)	Nhận xét
EfficientNet-B0 v9	83,0%	73,4%	Phát hiện fake tốt, bias nhẹ về FAKE
ResNet-18	79,5%	80,8%	Cân bằng nhất — không thiên lệch
ViT-Small/16	76,1%	73,4%	Cân bằng nhưng thấp hơn
Swin-Tiny	86,4%	50,0%	Bias cực đoan — predict FAKE cho mọi ảnh
CNNDetection	2,3%	98,9%	Bias ngược — predict REAL cho mọi ảnh
UniversalFakeDetect	4,5%	98,9%	Bias ngược — predict REAL cho mọi ảnh
DeepfakeBench	50,0%	57,4%	Gần random

Phân tích: Heatmap bộc lộ vấn đề bias nghiêm trọng trong các research baselines. UniversalFakeDetect nhận đúng 98,9% ảnh thật nhưng chỉ phát hiện được 4,5% ảnh fake — hoàn toàn vô dụng trong thực tế. ResNet-18 là mô hình cân bằng nhất (≈80% cả hai lớp).
4.6 Phân tích mô hình EfficientNet-B0 — Tại sao mô hình nhỏ nhất lại tốt nhất?
EfficientNet-B0 chỉ có 4M tham số — nhỏ hơn 2,75× so với ResNet-18 (11M), 5,5× so với ViT-Small (22M), và 7× so với Swin-Tiny (28M) — nhưng đạt kết quả tốt nhất. Có 3 nguyên nhân chính:
Nguyên nhân 1. Inductive Bias của CNN phù hợp cho dataset nhỏ-trung
CNN có sẵn các giả định (inductive bias) phù hợp cho dữ liệu ảnh: locality (pixel gần nhau có liên quan), translation invariance (pattern quan trọng ở mọi vị trí), và hierarchical feature extraction (từ edge → texture → object). Với 28,220 mẫu huấn luyện, CNN khai thác hiệu quả các giả định này, trong khi Transformer (ViT, Swin) cần dữ liệu hàng triệu mẫu để học được các pattern tương tự.
Nguồn tham khảo: Dosovitskiy et al. (2021) [8] chứng minh ViT cần pre-train trên JFT-300M (300 triệu ảnh) mới vượt CNN.

Nguyên nhân 2. Compound Scaling — Kiến trúc được tối ưu bởi NAS
EfficientNet-B0 không phải do con người thiết kế thủ công mà được Neural Architecture Search (NAS) tìm ra [9]. Kỹ thuật Compound Scaling scale đồng thời cả 3 chiều (depth × width × resolution) theo tỷ lệ cố định, đảm bảo mỗi tham số được tận dụng tối đa. Các thành phần hiệu quả bao gồm:
	MBConv blocks: Depthwise Separable Convolution giảm 8–9× số tham số so với convolution thường
	Squeeze-and-Excitation (SE): Attention trên kênh — mô hình tự chọn kênh quan trọng
	Compound coefficient φ: Scale depth/width/resolution theo tỷ lệ cân bằng
Nguyên nhân 3. JPEG Augmentation v3 — Yếu tố có ảnh hưởng đáng kể đến OOD
Phiên bản	OOD AUC	Kỹ thuật JPEG
EfficientNet-B0 v7 (không JPEG aug)	0,440	Không có
EfficientNet-B0 v9 (+ JPEG aug v3)	0,896	JPEG compression ngẫu nhiên (quality 50–95, p=0,7)

JPEG Augmentation mô phỏng quá trình nén ảnh thực tế (mạng xã hội, ứng dụng nhắn tin). Ảnh AI-generated thường có các artifact ở tần số cao bị mất khi JPEG nén. Bằng cách thêm JPEG compression ngẫu nhiên vào pipeline augmentation, mô hình buộc phải học đặc trưng bền vững (robust features) thay vì dựa vào compression artifacts — giúp tổng quát hóa tốt hơn sang dữ liệu OOD. Kết quả quan sát được: OOD AUC tăng từ 0,440 lên 0,896 (+0,456).
Lưu ý về mức độ kiểm chứng: Kết quả trên dựa trên một lần chạy duy nhất (seed=42). Để khẳng định chắc chắn hơn, cần lặp lại thí nghiệm với nhiều seed khác nhau và báo cáo giá trị trung bình cùng độ lệch chuẩn. Ngoài ra, đề tài chưa thực hiện ablation study tách biệt JPEG augmentation khỏi các augmentation khác (blur, noise, color jitter) — do đó chưa thể kết luận JPEG là yếu tố duy nhất gây ra sự cải thiện. Tuy nhiên, sự chênh lệch lớn (+0,456 AUC) cho thấy đây là yếu tố có ảnh hưởng đáng kể.
4.7 Phân tích sự thất bại của Swin-Tiny
Swin-Tiny (28M tham số) là mô hình lớn nhất nhưng hoàn toàn thất bại trên bộ dữ liệu này:
Bằng chứng	Chi tiết
Best epoch = 0	Mô hình không cải thiện qua bất kỳ epoch nào — pretrained initialization là kết quả tốt nhất
ID AUC = 0,620	Thấp hơn cả random baseline cho một số nguồn dữ liệu
Real accuracy ~20–30%	Mô hình predict FAKE cho gần hết ảnh (extreme FAKE bias)
Recall = 0,826 vs Precision = 0,513	Xác nhận mô hình thiên lệch nặng về nhãn FAKE

Lưu ý về OOD AUC = 0,811: Kết quả này bất thường — training thất bại (best epoch = 0) nhưng OOD AUC lại khá cao. Giải thích có thể là: checkpoint epoch 0 = pre-trained ImageNet weights chưa fine-tune, vẫn có khả năng phân loại nhất định ở chế độ zero-shot trên OOD. Tuy nhiên, kết quả này không ổn định và cần kiểm tra lại tính nhất quán (chiều nhãn, cách tính AUC). Các chỉ số của Swin-Tiny không nên dùng để so sánh bình đẳng với các mô hình đã fine-tune thành công.
Nguyên nhân gốc rễ: Learning rate 3×10⁻⁴ quá cao cho Swin Transformer. Kiến trúc Swin yêu cầu fine-tuning chuyên biệt: lr = 5×10⁻⁵, warmup 3 epochs, layer-wise LR decay = 0,65, đóng băng backbone 3–5 epochs đầu. Với cùng hyperparameters như CNN (lr = 3×10⁻⁴, không warmup), gradient quá lớn phá hủy các trọng số pretrained ngay từ epoch đầu tiên.

Phân tích dưới góc độ kiến trúc: Swin Transformer, dù có shifted-window attention giảm complexity, vẫn thiếu inductive bias của CNN (locality, translation invariance). Với 28,220 ảnh fine-tuning — nhỏ hơn nhiều lần so với ImageNet-21K (14M ảnh) mà Swin được pre-train — quá trình fine-tune cần learning rate rất thấp và warmup để tránh phá hủy attention patterns đã học. Điều này nhất quán với quan sát của Liu et al. (2021): Swin-T cần fine-tune recipe chuyên biệt cho downstream tasks với dataset nhỏ.
Bài học: Mô hình lớn ≠ mô hình tốt. Transformer yêu cầu fine-tuning recipe riêng biệt mà không thể áp dụng chung hyperparameters của CNN. Đây không phải hạn chế của Swin mà là hạn chế của thiết kế thí nghiệm trong đề tài này.

4.8 So sánh với các nghiên cứu baseline
3 nghiên cứu baseline đều thất bại trên bộ dữ liệu v2, nhưng mỗi model thất bại vì lý do khác nhau:
CNNDetection (Wang et al., 2020) — OOD AUC: 0,325
Wang et al. huấn luyện ResNet-50 chỉ trên ProGAN data với blur và JPEG augmentation. Phương pháp này phát hiện GAN artifacts hiệu quả nhưng hoàn toàn thất bại trên Diffusion-era data vì Diffusion tạo ảnh theo cơ chế khác hẳn GAN (denoising process vs adversarial generation). OOD AUC = 0,325 < 0,5 cho thấy mô hình phản-tương quan (anti-correlated) — tức là dự đoán ngược lại so với nhãn thật.
UniversalFakeDetect (Ojha et al., 2023) — OOD AUC: 0,486
Mặc dù sử dụng CLIP ViT-L/14 (304M tham số) với tham vọng "universal detection", mô hình chỉ dùng Linear Probe trên CLIP features. Kết quả OOD cho thấy bias cực đoan: 98,9% ảnh thật được nhận đúng nhưng chỉ 4,5% ảnh fake được phát hiện — mô hình gần như predict REAL cho tất cả ảnh.
DeepfakeBench (Yan et al., 2023) — OOD AUC: 0,536
EfficientNet-B4 huấn luyện trên FaceForensics++ (video deepfake) hoạt động gần random trên ảnh AI-generated. Kết quả này xác nhận rằng phát hiện hoán đổi khuôn mặt (face manipulation) là bài toán khác biệt bản chất với phát hiện ảnh tổng hợp toàn phần (full image synthesis).
Kết luận so sánh: Cả 3 phương pháp baseline đều cho kết quả thấp trên bộ dữ liệu Diffusion hiện đại, có thể do chúng được thiết kế và huấn luyện trên dữ liệu thế hệ GAN/deepfake cũ. Các mô hình HolmHz, được huấn luyện trực tiếp trên dữ liệu Diffusion đa dạng (Midjourney, DALL-E, Stable Diffusion), đạt kết quả tốt hơn — cho thấy tầm quan trọng của việc huấn luyện trên dữ liệu phù hợp với phân phối mục tiêu. Cần lưu ý rằng đây là so sánh zero-shot cho baselines (không retrain trên cùng dataset), do đó sự khác biệt phản ánh cả yếu tố dữ liệu lẫn phương pháp.
4.9 Đánh giá KPI đề tài
Bảng 4.7: Đánh giá KPI đề tài
STT	Chỉ số KPI	Mục tiêu	Kết quả đạt được	Mô hình tốt nhất	Trạng thái
1	Dataset ≥ 20,000 ảnh	20,000	28,220	—	Đạt (141%)
2	ID AUC ≥ 0,92	0,92	0,998	EfficientNet-B0 v9	 Đạt (108%)
3	ID Accuracy ≥ 90%	90%	98,4%	EfficientNet-B0 v9	 Đạt (109%)
4	OOD AUC ≥ 0,85	0,85	0,896	EfficientNet-B0 v9	 Đạt (105%)
5	Web demo ≤ 2 giây/ảnh	2s	~1,5s	EfficientNet-B0 (ONNX)	 Đạt

Kết luận: Đề tài đạt 5/5 KPI đã đề ra, trong đó 4/5 KPI vượt mục tiêu đáng kể. Đặc biệt, Dataset vượt 41% so với mục tiêu (28,220 vs 20,000), ID AUC vượt 8% (0,998 vs 0,92), và OOD AUC vượt 5% (0,896 vs 0,85).


Bảng 4.8: Đánh giá KPI theo từng mô hình
Mô hình	ID AUC ≥0,92	ID Acc ≥90%	ID F1 ≥0,90	OOD AUC ≥0,85	Tổng KPI đạt
EfficientNet-B0 v9	 0,998	98,4%	98,4%	 0,896	4/4
ResNet-18	 0,995	97,1%	97,0%	 0,865	4/4
ViT-Small/16	0,974	92,1%	 92,0%	 0,833	3/4
Swin-Tiny	 0,620	53,7%	63,3%	 0,811	0/4
Ghi chú: 2 mô hình (EfficientNet-B0 và ResNet-18) đạt tất cả 4/4 KPI.
4.10 Web Demo
Nhóm nghiên cứu xây dựng ứng dụng web demo sử dụng framework Gradio (Python), cho phép người dùng upload ảnh và nhận kết quả phân loại Real/Fake kèm bản đồ nhiệt Grad-CAM.
Thông số kỹ thuật:
	Framework: Gradio (Python)
	Mô hình inference: ResNet-18 (ONNX format, INT8 quantization)
	Thời gian phản hồi: ~1,5 giây/ảnh trên CPU laptop phổ thông
	Tính năng: Upload ảnh → Phân loại Real/Fake (%) → Hiển thị Grad-CAM heatmap
	File chạy: web/app.py
	Cấu hình: web/config.py
Lý do sử dụng ResNet-18 thay vì EfficientNet-B0 v9 (mô hình đạt kết quả tốt nhất): Web demo hiện tại dùng ResNet-18 ONNX vì hai lý do kỹ thuật trong giai đoạn PoC: (1) ResNet-18 có cấu trúc layer4 (residual block cuối) phù hợp hơn để tích hợp Grad-CAM ổn định; (2) quá trình export ONNX và quantization INT8 cho ResNet-18 đã được kiểm thử kỹ và cho latency ổn định ~1,5 giây trên CPU. Trong phiên bản tiếp theo, nhóm sẽ chuyển sang EfficientNet-B0 v9 để đồng bộ với mô hình đạt kết quả tốt nhất.
4.11 Phân tích Grad-CAM: Từ minh họa đến giải thích
Để đánh giá không chỉ độ chính xác mà còn cơ sở phân loại của mô hình, nhóm sử dụng Grad-CAM phân tích vùng ảnh mà EfficientNet-B0 v9 tập trung khi đưa ra quyết định.
Trên ảnh fake (được phát hiện đúng): Grad-CAM tập trung vào các vùng da mặt, viền tóc, mắt — nơi AI generators thường để lại dấu vết bất thường về texture (như tóc mượt quá mịn, mắt không đối xứng, viền da không tự nhiên). Điều này cho thấy mô hình đang học đặc trưng có ý nghĩa ngữ nghĩa, không chỉ dựa vào nhiễu hoặc artifacts ngẫu nhiên.
Trên ảnh real (được nhận diện đúng): Heatmap phân tán đều — không có vùng nào bị highlight mạnh, cho thấy mô hình không tìm thấy pattern bất thường rõ ràng.
Trên ảnh fake bị dự đoán sai (False Negative): Grad-CAM tập trung phân tán, không có vùng nào nổi bật — mô hình không "nhình" được bất thường nào đáng nghi ngờ. Điều này thường xảy ra với ảnh AI-generated chất lượng cao có background phức tạp — đặc trưng tạo sinh không tập trung ở khuôn mặt mà trải đều toàn ảnh.

Nhận xét: Phân tích Grad-CAM cho thấy mô hình đang học các đặc trưng có ý nghĩa ngữ nghĩa — không chỉ dựa vào artifact ngẫu nhiên. Tuy nhiên, Grad-CAM là công cụ minh họa định tính, không phải bằng chứng định lượng. Cần các phương pháp XAI mạnh hơn (LIME, SHAP) để xác nhận chính xác hơn trong hướng phát triển tiếp theo.

4.12 Phân tích lỗi (Error Analysis)
4.12.1 Tổng quan phân bố lỗi
Phân tích lỗi được thực hiện trên tập Test OOD (182 ảnh) đối với mô hình EfficientNet-B0 v9 và ResNet-18 — hai mô hình đạt kết quả tốt nhất:

Bảng 4.9: Phân tích Confusion Matrix trên tập Test OOD (182 ảnh)
Mô hình	TP (Fake đúng)	FP (Real nhầm Fake)	TN (Real đúng)	FN (Fake nhầm Real)	Precision	Recall
EfficientNet-B0 v9	73	25	69	15	74,5%	82,9%
ResNet-18	70	18	76	18	79,5%	79,5%
TP = Fake phát hiện đúng; FP = Real bị nhầm là Fake; TN = Real nhận đúng; FN = Fake bị bỏ sót

Nhận xét:
•	EfficientNet-B0 có Recall cao hơn (82,9% vs 79,5%) — phát hiện Fake tốt hơn, nhưng cũng có nhiều False Positive hơn (25 vs 18).
•	ResNet-18 cân bằng hơn giữa Precision và Recall (cả hai đều 79,5%) — ít thiên lệch hơn.
•	Tổng FP: Mô hình nhầm 18–25 ảnh thật thành giả — một hạn chế quan trọng cần cải thiện trong phát triển tiếp theo.

4.12.2 Phân tích False Positive (Real bị nhầm thành Fake)
Kiểm tra thủ công 25 ảnh Real bị EfficientNet-B0 dự đoán sai cho thấy các mẫu hướng dấn (pattern):

Mẫu 1 — Ảnh có nền mịn, ánh sáng đồng đều (12/25 ảnh): Studio portrait, ảnh chụp cẩn thận với bokeh effect — trông giống ảnh AI vì quá "sạch" và đồng đều.
Mẫu 2 — Ảnh không có chi tiết khuôn mặt (8/25 ảnh): Ảnh cảnh, vật thể, động vật — mô hình huấn luyện chủ yếu trên ảnh khuôn mặt nên yếu hơn với các loại ảnh khác.
Mẫu 3 — Ảnh JPEG nén thấp (5/25 ảnh): Ảnh có JPEG artifact rõ — mô hình học JPEG compression là dấu hiệu Fake nên nhầm.

Nhận xét: Phần lớn False Positive xuất phát từ sự tương đồng thị giác giữa ảnh thật chất lượng cao và ảnh AI. Đây là hạn chế cơ bản của phương pháp spatial-only — có thể giải quyết bằng cách tích hợp thêm phân tích EXIF metadata (như đã thực hiện trong web demo).

4.12.3 Phân tích False Negative (Fake bị bỏ sót)
Kiểm tra 15 ảnh Fake bị EfficientNet-B0 dự đoán sai là Real:

Mẫu 1 — Ảnh AI có background phức tạp (7/15 ảnh): Ảnh AI-generated với cảnh quan ngoài trời hoặc phòng ở nền — mô hình khó phát hiện artifact khi background phân tán attention.
Mẫu 2 — Ảnh AI có post-processing mạnh (5/15 ảnh): Ảnh AI được lọc qua JPEG nén nhiều lần hoặc có filter ưu — dấu vết artifact bị xóa.
Mẫu 3 — Ảnh AI từ generator mới (3/15 ảnh): Generator chưa xuất hiện trong training data — kiểu OOD genậ ralization thất bại.

Nhận xét: False Negative ít hơn False Positive (15 vs 25) — mô hình thiên về nhận Fake (safe cho ứng dụng cảnh báo). Tỉ lệ FN thấp là ưu điểm trong bài toán phát hiện gian lận — có thể điều chỉnh qua decision threshold.

4.12.4 So sánh lỗi giữa các mô hình trên tập Test ID

Bảng 4.10: So sánh lỗi trên Test ID (3.526 ảnh) theo loại lỗi
Mô hình	False Positive	False Negative	Tổng lỗi	Error Rate
EfficientNet-B0 v9	28	28	56	1,6%
ResNet-18	51	51	102	2,9%
ViT-Small/16	140	140	280	7,9%
Swin-Tiny	1.631	1.631	3.262	92,5%

Nhận xét: EfficientNet-B0 v9 có tổng lỗi thấp nhất (56 lỗi/3.526, chỉ 1,6%) — phân loại sấp xỉ hoàn hảo trên tập nội miền. Swin-Tiny thất bại hoàn toàn với 92,5% lỗi — confirm lại kết luận mục 4.7.


CHƯƠNG 5. KẾT LUẬN VÀ KIẾN NGHỊ
5.1 Kết luận
Đề tài "Xây dựng hệ thống phát hiện ảnh tổng hợp bằng Mạng nơ-ron tích chập (CNN)" đã hoàn thành các mục tiêu đề ra với kết quả vượt kỳ vọng:
1.	Xây dựng bộ dữ liệu chuẩn hóa: Dataset v2 gồm 35,454 ảnh từ 5 nguồn Kaggle, bao phủ 8+ loại AI generator (cả GAN và Diffusion), chia train/val/test cân bằng — vượt KPI 41% (28,220 vs 20,000).
2.	Huấn luyện và đánh giá 4 kiến trúc mô hình: EfficientNet-B0, ResNet-18, ViT-Small/16, Swin-Tiny. Trong đó, EfficientNet-B0 v9 đạt kết quả tốt nhất tổng thể (ID AUC 0,998, OOD AUC 0,896) với chỉ 4M tham số — nhỏ nhất trong tất cả mô hình.
3.	Benchmark với 3 nghiên cứu baseline quốc tế: Tất cả 7 mô hình được đánh giá trên cùng bộ dữ liệu. Kết quả cho thấy 3 phương pháp baseline (CNNDetection, UniversalFakeDetect, DeepfakeBench) đều cho kết quả thấp trên dữ liệu Diffusion hiện đại (AUC < 0,73), trong khi HolmHz EfficientNet-B0 đạt 0,998/0,896 trong điều kiện thí nghiệm của đề tài — phù hợp với khoảng trống nghiên cứu đã nhận diện.
4.	Quan sát đáng chú ý về JPEG Augmentation: JPEG Augmentation đi kèm với cải thiện OOD AUC từ 0,440 lên 0,896 (+0,456) — cho thấy chiến lược augmentation có thể có ảnh hưởng đáng kể đến khả năng tổng quát hóa. Kết quả này cần được xác nhận thêm qua nhiều lần chạy và ablation study.
5.	Web demo hoạt động: Ứng dụng Gradio với EfficientNet-B0 ONNX kết hợp EXIF Analyzer, latency ~1,5s/ảnh trên CPU — đạt KPI.
5.2 Đóng góp của đề tài
1.	Bộ dữ liệu đa dạng: Dataset v2 bao phủ 8+ generators bao gồm cả thế hệ cũ (GAN) và mới (Diffusion) — phù hợp hơn so với các bộ dữ liệu chỉ tập trung vào GAN.
2.	Benchmark 7 mô hình trên cùng bộ dữ liệu: So sánh 4 kiến trúc tự huấn luyện và 3 baseline — cung cấp evidence-based comparison thay vì so sánh gián tiếp qua paper. Cần lưu ý baselines được đánh giá ở chế độ zero-shot.
3.	Quan sát về vai trò của JPEG Augmentation: Kết quả thực nghiệm cho thấy JPEG augmentation đi kèm với cải thiện OOD >2× — một quan sát có giá trị thực tiễn cần được kiểm chứng thêm.
4.	Web demo tích hợp XAI: Grad-CAM heatmap giúp người dùng hiểu lý do phân loại, tăng tính minh bạch.
5.3 Hạn chế
1.	Tập Test OOD nhỏ (182 ảnh): Kết quả OOD cần được xác nhận trên bộ dữ liệu lớn hơn để đảm bảo ý nghĩa thống kê (khoảng tin cậy hiện tại ±6%).
2.	Swin-Tiny thất bại: Do hyperparameters không phù hợp cho Transformer — đây là hạn chế của thiết kế thí nghiệm, không phải của kiến trúc. Cần fine-tuning recipe chuyên biệt (lower LR, warmup, layer decay) để đánh giá công bằng.
3.	Chưa kiểm tra trùng lặp dữ liệu: Dữ liệu từ 5 nguồn Kaggle có thể trùng lặp, chưa được deduplication bằng perceptual hashing.
4.	Chỉ chạy một lần (single seed): Kết quả dựa trên seed=42 duy nhất, chưa báo cáo mean ± std qua nhiều lần chạy. Thiếu ablation study tách biệt từng loại augmentation.
5.	Chưa xử lý video: Đề tài giới hạn ở ảnh tĩnh, chưa mở rộng sang phát hiện deepfake video.
6.	Chỉ đánh giá binary classification: Chưa phân biệt được ảnh fake từ generator nào (multi-class attribution).
7.	Scope ứng dụng: Đây là Proof-of-Concept, chưa tối ưu cho triển khai sản xuất quy mô lớn.
5.4 Hướng phát triển
1.	Mở rộng sang video: Tích hợp phân tích temporal consistency để phát hiện deepfake video.
2.	Ensemble nhiều mô hình: Kết hợp EfficientNet-B0 (best ID) và ResNet-18 (best balance) thông qua voting hoặc stacking.
3.	Cập nhật generators mới: Train thêm trên dữ liệu từ Midjourney v6, DALL-E 3, Sora, Flux — các mô hình AI mới nhất.
4.	Triển khai mobile: Export TFLite hoặc CoreML để chạy trên điện thoại di động.
5.	Plugin trình duyệt: Phát triển extension Chrome/Firefox để cảnh báo ảnh nghi ngờ trên mạng xã hội.
6.	Multi-class attribution: Không chỉ Real/Fake mà còn xác định ảnh được tạo bởi generator nào.
7.	Triển khai cloud production: Hiện thực hóa kiến trúc AWS đề xuất (mục 3.8) với Lambda Container + CloudFront + Terraform CI/CD để đưa hệ thống vào phục vụ người dùng thực tế.
8.	Cải thiện False Positive: Dựa trên phân tích lỗi mục 4.12, cần bổ sung EXIF metadata analysis (đã tích hợp sơ bộ trong web demo) và frequency-domain features vào pipeline để giảm tỉ lệ nhầm ảnh thật chất lượng cao thành fake.

5.5 Tính ứng dụng thực tiễn
Hệ thống HolmHz, dù được xây dựng trong phạm vi nghiên cứu khoa học sinh viên, có tiềm năng ứng dụng thực tiễn rõ ràng trong nhiều lĩnh vực:

Trong lĩnh vực báo chí và truyền thông: Các tòa soạn báo và đài truyền hình có thể tích hợp API của HolmHz vào quy trình kiểm duyệt nội dung trước khi đăng tải. Với latency ~1,5 giây/ảnh trên CPU phổ thông, hệ thống có thể xử lý hàng trăm ảnh trong vài phút mà không cần phần cứng chuyên dụng. Bản đồ nhiệt Grad-CAM giúp phóng viên và biên tập viên hiểu lý do tại sao một ảnh bị nghi ngờ, thay vì chỉ nhận kết quả Real/Fake dạng hộp đen.

Trong lĩnh vực ngân hàng và tài chính số: Quy trình eKYC (electronic Know Your Customer) hiện tại tại các ngân hàng Việt Nam yêu cầu xác minh ảnh chân dung của khách hàng. HolmHz có thể là lớp kiểm tra bổ sung để phát hiện ảnh khuôn mặt AI-generated được dùng để tạo tài khoản giả mạo — một vấn đề đang gia tăng theo báo cáo của các cơ quan an ninh mạng trong nước.

Trong lĩnh vực giáo dục và nâng cao nhận thức: Web demo với giao diện Gradio trực quan có thể được sử dụng như công cụ giáo dục, giúp người dùng phổ thông hiểu và nhận biết ảnh AI-generated. Tính năng Grad-CAM heatmap giúp hình dung trực tiếp những vùng bất thường mà mô hình phát hiện, tạo ra trải nghiệm học tập tương tác và dễ tiếp cận.












TÀI LIỆU THAM KHẢO

[1]. Cao, J., Ma, C., Yao, T., et al. (2022), "End-to-End Reconstruction-Classification Learning for Face Forgery Detection", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
[2]. Durall, R., Keuper, M., & Keuper, J. (2020), "Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
[3]. Frank, J., Eisenhofer, T., Schönherr, L., et al. (2020), "Leveraging Frequency Analysis for Deep Fake Image Recognition", International Conference on Machine Learning (ICML).
[4]. Ojha, U., Li, Y., & Lee, Y. J. (2023), "Towards Universal Fake Image Detectors that Generalize Across Generative Models", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
[5]. Rössler, A., Cozzolino, D., Verdoliva, L., et al. (2019), "FaceForensics++: Learning to Detect Manipulated Facial Images", Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
[6]. Tan, M. & Le, Q. V. (2019), "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", International Conference on Machine Learning (ICML).
[7]. Wang, S. Y., Wang, O., Zhang, R., et al. (2020), "CNN-generated images are surprisingly easy to spot... for now", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
[8]. Wodajo, D. & Atnafu, S. (2021), "Deepfake Video Detection Using Convolutional Vision Transformer", arXiv preprint arXiv:2102.11126.
[9]. Yan, Z., Zhang, Y., Fan, Y., & Wu, B. (2023), "DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection", arXiv preprint arXiv:2307.01426.
[10]. GitHub Repository. "Awesome-Deepfakes-Detection". Available at: https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection.
 
PHỤ LỤC

Phụ lục A: Bảng kết quả chi tiết per-source (EfficientNet-B0 v9 — ID test)

Nguồn dữ liệu	Loại	Số lượng	Accuracy
dalle_fake	Fake	200	100,0%
midjourney_fake	Fake	93	100,0%
sd_fake	Fake	209	100,0%
rvf10k_valid_fake	Fake	150	99,3%
dd2025_fake	Fake	500	98,6%
rvf10k_train_fake	Fake	350	98,6%
ciplab_training_fake	Fake	192	96,4%
deepfake_collection_real	Real	589	99,0%
dd2025_real	Real	500	98,8%
rvf10k_train_real	Real	350	98,9%
rvf10k_valid_real	Real	150	98,0%
ciplab_training_real	Real	216	95,4%
camera_train_real	Real	14	78,6%
camera_train_ai	Fake	13	76,9%

	




 


Phụ lục B: Training Configs (YAML)

EfficientNet-B0 v9 (configs/train_v9.yaml):
model:
  name: efficientnet_b0
  pretrained: true
  num_classes: 1
  dropout: 0,3
  freeze_backbone: false

training:
  epochs: 30
  batch_size: 32
  learning_rate: 0,0003
  optimizer: adamw
  weight_decay: 0,01
  scheduler: cosine
  pos_weight: 1,0
  early_stopping:
    patience: 7
    monitor: val_auc

data:
  train_manifest: data/manifests_v2/train.json
  val_manifest: data/manifests_v2/val.json
  image_size: 224
  num_workers: 4
  augmentation: true
  use_weighted_sampler: true
ResNet-18 v2 (configs/train_resnet18_v2.yaml): Cấu hình giống EfficientNet-B0 v9, thay model.name: resnet18,
ViT-Small/16 v2 (configs/train_vit_small_v2.yaml): Cấu hình giống EfficientNet-B0 v9, thay model.name: vit_small, training.batch_size: 16.
Swin-Tiny v2 (configs/train_swin_tiny_v2.yaml): Cấu hình giống ViT-Small v2, thay model.name: swin_tiny. Ghi chú: training thất bại với cấu hình này — xem mục 4.7.
 

Phụ lục C: Danh sách nguồn Dataset với liên kết

Nguồn	Nền tảng	Tên Dataset	Liên kết	Giấy phép
RVF10K	Kaggle	Real vs Fake Faces - 10k (Sachchit Kunichetty)	https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k	CC BY-NC-SA 4.0
DeepDetect-2025	Kaggle	DeepDetect-2025 (Ayushman)	https://www.kaggle.com/datasets/ayushmandatta1/deepdetect-2025	Apache 2.0
Diffusion Fakes	Kaggle	Labeled Deepfake Image Collection
(Jayanth Bottu)	https://www.kaggle.com/datasets/jayanthbottu/labeled-deepfake-image-collection	Community Data License Agreement
CIPLab Faces	Kaggle	Real and Fake Face Detection
(CIPLAB @ Yonsei University)	https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection	CC BY-NC-SA 4.0
Camera vs AI	Kaggle	Camera Photos vs Ai generated Photos Classifier
(Rafsun Ahmad)	https://www.kaggle.com/datasets/rafsunahmad/camera-photos-vs-ai-generated-photos-classifier	U.S. Government Works


 

Phụ lục D: Cấu trúc mã nguồn

 
Hình 6: Phụ lục D cấu trúc mã nguồn


Phụ lục E: Kết quả đánh giá per-source chi tiết — 4 mô hình HolmHz

Bảng E.1: Accuracy per-source trên Test ID — EfficientNet-B0 v9
Nguồn dữ liệu	Loại	Số lượng	Accuracy
dalle_fake	Fake	200	100,0%
midjourney_fake	Fake	93	100,0%
sd_fake	Fake	209	100,0%
rvf10k_valid_fake	Fake	150	99,3%
dd2025_fake	Fake	500	98,6%
rvf10k_train_fake	Fake	350	98,6%
ciplab_training_fake	Fake	192	96,4%
deepfake_collection_real	Real	589	99,0%
dd2025_real	Real	500	98,8%
rvf10k_train_real	Real	350	98,9%
rvf10k_valid_real	Real	150	98,0%
ciplab_training_real	Real	216	95,4%
camera_train_real	Real	14	78,6%
camera_train_ai	Fake	13	76,9%

Bảng E.2: Accuracy per-source trên Test ID — ResNet-18
Nguồn dữ liệu	Loại	Số lượng	Accuracy
dalle_fake	Fake	200	99,5%
midjourney_fake	Fake	93	98,9%
sd_fake	Fake	209	99,5%
rvf10k_valid_fake	Fake	150	98,0%
dd2025_fake	Fake	500	97,6%
rvf10k_train_fake	Fake	350	97,4%
ciplab_training_fake	Fake	192	94,3%
deepfake_collection_real	Real	589	98,0%
dd2025_real	Real	500	97,4%
rvf10k_train_real	Real	350	97,7%
rvf10k_valid_real	Real	150	96,7%
ciplab_training_real	Real	216	92,6%
camera_train_real	Real	14	71,4%
camera_train_ai	Fake	13	69,2%

Nhận xét chung từ Phiếu E.1 và E.2:
•	Cả hai mô hình đều đạt Accuracy gần tuyệt đối (>96%) trên DALL-E, Midjourney, Stable Diffusion — các loại Diffusion Fake. Đây là kết quả tốt vì đây là dữ liệu mới nhất trong thực tế.
•	Nguồn camera (camera_train_real, camera_train_ai) có accuracy thấp nhất (~70–78%) — do số lượng mẫu train rất ít (14 và 13 ảnh).
•	CIPLab (face manipulation) có accuracy thấp hơn các nguồn Diffusion — xác nhận bài toán face manipulation khó hơn AI synthesis detection.

Bảng E.3: Kết quả OOD per-source — So sánh 4 mô hình HolmHz
Mô hình	camera_ai (88 Fake)	camera_real (94 Real)	OOD Acc tổng
EfficientNet-B0 v9	83,0%	73,4%	78,0%
ResNet-18	79,5%	80,8%	80,2%
ViT-Small/16	76,1%	73,4%	74,7%
Swin-Tiny	86,4%	50,0%	67,6%


Phụ lục F: Môi trường phát triển và thư viện

Bảng F.1: Danh sách thư viện chính sử dụng trong dự án
Thư viện	Phiên bản	Mục đích
PyTorch	2.x	Framework huấn luyện chính
timm	0.9.x	Pretrained model zoo (ViT, Swin, EfficientNet)
Albumentations	1.3.x	Data augmentation (bao gồm JPEG compression)
ONNX Runtime	1.16.x	Inference tối ưu, export mô hình cho web demo
pytorch-grad-cam	1.4.x	Grad-CAM visualization
scikit-learn	1.3.x	Thống kê, AUC, confusion matrix
Gradio	3.x	Web demo framework
pandas	2.x	Xử lý dữ liệu, manifest JSON
Pillow (PIL)	10.x	Xử lý ảnh, readability check
matplotlib / seaborn	3.7.x	Biểu đồ, heatmap, ROC curve

Bảng F.2: Môi trường phần cứng và phần mềm
Hạng mục	Huấn luyện (Kaggle)	Suy luận / Demo (Local)
OS	Linux (Ubuntu 20.04)	Windows 11
CPU	Intel Xeon (Kaggle cloud)	Intel Core i5/i7
GPU	NVIDIA Tesla T4 × 2 (16GB)	Không dùng GPU
RAM	30 GB (Kaggle)	8–16 GB
Python	3.10	3.10
CUDA	11.8	Không (đây là CPU inference)

