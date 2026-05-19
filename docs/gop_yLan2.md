1. Nhận xét tổng quan
Đây là một đề tài có tính thời sự rất cao, phù hợp với bối cảnh AI tạo sinh phát triển mạnh và ảnh tổng hợp ngày càng khó phân biệt bằng mắt thường. Việc sinh viên chọn hướng Synthetic Image Detection là hợp lý, có ý nghĩa khoa học và thực tiễn, đặc biệt trong các vấn đề như tin giả, lừa đảo trực tuyến, deepfake, kiểm duyệt nội dung số và an toàn thông tin.
Báo cáo có cấu trúc khá tốt: phần tổng quan nghiên cứu được chia theo các nhóm phương pháp CNN, Transformer, phân tích miền tần số, Hybrid/XAI; có chỉ ra khoảng trống nghiên cứu; có quy trình xây dựng dữ liệu; có mô tả pipeline huấn luyện; có benchmark nhiều mô hình; có web demo và Grad-CAM. Đây là điểm rất đáng ghi nhận.
Tuy nhiên, báo cáo hiện có xu hướng viết khá mạnh về kết quả, trong khi một số phần vẫn cần diễn đạt thận trọng hơn. Đặc biệt, tập OOD chỉ có 182 ảnh, nhóm cũng đã tự ghi nhận khoảng tin cậy còn lớn, nên không nên khẳng định quá chắc về khả năng tổng quát hóa. Ngoài ra, tên đề tài ghi là “bằng CNN”, nhưng nội dung lại huấn luyện cả ViT-Small/16 và Swin-Tiny, vốn không phải CNN. Điểm này cần chỉnh để thống nhất.
2. Ưu điểm của báo cáo
2.1. Đề tài có tính thời sự và ý nghĩa thực tiễn cao
Bối cảnh AI-generated image, deepfake, fake news, lừa đảo trực tuyến là vấn đề rất thời sự. Phần mở đầu đã nêu được nguy cơ xã hội của ảnh giả và nhu cầu xây dựng hệ thống phát hiện ảnh tổng hợp. Đây là cách đặt vấn đề tốt, giúp đề tài không chỉ là bài toán kỹ thuật mà còn có ý nghĩa xã hội.
2.2. Tổng quan nghiên cứu có chiều sâu hơn nhiều báo cáo sinh viên
Báo cáo không chỉ liệt kê tài liệu mà đã phân nhóm các hướng nghiên cứu:
•	CNN trong miền không gian.
•	Transformer và Attention.
•	Phân tích miền tần số.
•	Hybrid và Explainable AI.
Cách phân loại này hợp lý và giúp làm rõ khoảng trống nghiên cứu. Đặc biệt, phần giải thích vì sao detector huấn luyện trên GAN có thể suy giảm khi gặp ảnh Diffusion là một điểm tốt, thể hiện nhóm có hiểu bản chất vấn đề chứ không chỉ chạy mô hình.
2.3. Có mục tiêu/KPI rõ ràng
Báo cáo đặt ra các KPI cụ thể như dataset ≥ 20.000 ảnh, ID AUC ≥ 0,92, OOD AUC ≥ 0,85, web demo latency ≤ 2 giây, có Grad-CAM. Việc lượng hóa mục tiêu như vậy giúp việc đánh giá kết quả rõ ràng hơn.
2.4. Pipeline kỹ thuật khá hoàn chỉnh
Báo cáo mô tả được nhiều thành phần kỹ thuật quan trọng:
•	Manifest JSON để quản lý dữ liệu.
•	Stratified split.
•	Train/Val/Test ID/Test OOD.
•	Transfer Learning.
•	WeightedRandomSampler.
•	AdamW, Cosine Annealing, Early Stopping.
•	BCEWithLogitsLoss.
•	Evaluator tính AUC, Accuracy, F1, Precision, Recall.
•	Web demo bằng Gradio.
•	ONNX để tối ưu inference.
•	Grad-CAM để giải thích mô hình.
Đây là điểm mạnh đáng kể. So với nhiều báo cáo NCKH sinh viên chỉ mô tả mô hình sơ sài, báo cáo này có chất lượng triển khai tốt hơn.
2.5. Có tự nhận diện hạn chế
Báo cáo đã ghi nhận các hạn chế như nguy cơ trùng lặp giữa các nguồn dữ liệu, chưa kiểm tra duplicate bằng perceptual hashing, tập OOD chỉ có 182 ảnh và kết luận tổng quát hóa chỉ mang tính chỉ báo sơ bộ. Đây là điểm rất tốt vì thể hiện thái độ khoa học, không che giấu điểm yếu.
3. Những vấn đề cần chỉnh sửa nghiêm túc
3.1. Tên đề tài chưa khớp hoàn toàn với nội dung
Tên hiện tại là:
Xây dựng hệ thống phát hiện ảnh tổng hợp bằng Mạng nơ-ron tích chập (CNN)
Tuy nhiên, báo cáo huấn luyện và so sánh cả ViT-Small/16 và Swin-Tiny, đây là các mô hình Transformer, không phải CNN. Vì vậy, tên đề tài có thể bị hội đồng hỏi:
Nếu đề tài là CNN, tại sao lại có ViT và Swin Transformer?
Có hai hướng xử lý.
Phương án 1 — giữ tên hiện tại:
Khi đó nên trình bày ViT/Swin chỉ là mô hình đối chứng, còn đóng góp chính là EfficientNet-B0/ResNet-18 thuộc nhóm CNN.
Phương án 2 — đổi tên đề tài cho bao quát hơn:
Nên đổi thành:
Xây dựng hệ thống phát hiện ảnh tổng hợp sử dụng học sâu và giải thích mô hình
Hoặc:
Xây dựng hệ thống phát hiện ảnh tổng hợp bằng học sâu kết hợp Grad-CAM
Tên này chính xác hơn vì bao quát cả CNN, Transformer, Transfer Learning và XAI.
Nếu nộp chính thức, tôi nghiêng về phương án đổi tên để tránh bị bắt lỗi về thuật ngữ.
3.2. Tập OOD quá nhỏ, không nên kết luận mạnh
Báo cáo dùng Test OOD gồm 182 ảnh, trong đó 94 real và 88 fake. Nhóm có ghi nhận khoảng tin cậy khoảng ±6%, đây là nhận xét đúng. Tuy nhiên, trong phần kết quả vẫn có các câu như “vượt KPI OOD AUC” hoặc “tổng quát hóa tốt” hơi mạnh.
Nên sửa theo hướng thận trọng:
Kết quả OOD AUC = 0,896 cho thấy tín hiệu tích cực ban đầu về khả năng tổng quát hóa, tuy nhiên do tập OOD chỉ gồm 182 ảnh, kết luận này cần được xác nhận thêm trên tập OOD lớn hơn và đa dạng hơn.
Không nên viết:
Mô hình có khả năng tổng quát hóa tốt trên dữ liệu ngoài miền.
Nên viết:
Mô hình bước đầu cho thấy khả năng tổng quát hóa khả quan trên tập OOD thử nghiệm.
3.3. Cần kiểm tra trùng lặp dữ liệu giữa các nguồn
Báo cáo tự ghi nhận nguy cơ trùng lặp dữ liệu giữa các nguồn Kaggle nhưng chưa thực hiện perceptual hashing hoặc feature-level deduplication. Đây là điểm rất quan trọng.
Nếu có ảnh trùng hoặc gần trùng giữa Train/Val/Test ID, kết quả ID AUC 0,998 có thể bị lạc quan. Đặc biệt, khi tổng hợp nhiều dataset công khai, khả năng trùng ảnh là có thật.
Nên bổ sung một bước kiểm tra:
•	Dùng perceptual hash: pHash, dHash hoặc aHash.
•	Xác định ảnh trùng/gần trùng theo Hamming distance.
•	Loại bỏ ảnh trùng giữa train/val/test.
•	Báo cáo số lượng ảnh bị loại nếu có.
Nếu chưa kịp làm, cần ghi rõ trong hạn chế:
Đề tài chưa thực hiện kiểm tra trùng lặp ảnh bằng perceptual hashing, do đó kết quả ID có thể bị ảnh hưởng nếu tồn tại ảnh trùng hoặc gần trùng giữa các split.
3.4. So sánh baseline có thể chưa thật sự công bằng
Báo cáo so sánh mô hình của nhóm với CNNDetection, UniversalFakeDetect và DeepfakeBench trên cùng bộ dữ liệu, nhưng có ghi chú baseline dùng pre-trained weights gốc, zero-shot, không retrain trên Dataset v2.
Điều này có giá trị tham khảo, nhưng không nên kết luận rằng mô hình của nhóm “vượt baseline quốc tế” theo nghĩa tuyệt đối. Vì baseline không được fine-tune trên cùng dữ liệu, còn mô hình của nhóm được train trực tiếp trên Dataset v2. So sánh như vậy có thể bị hội đồng hỏi là chưa công bằng.
Nên diễn đạt lại:
Trong điều kiện thí nghiệm của đề tài, các mô hình được huấn luyện trên Dataset v2 đạt kết quả cao hơn các baseline quốc tế khi sử dụng trọng số gốc ở chế độ zero-shot. Kết quả này cho thấy lợi ích của việc xây dựng dữ liệu phù hợp với nguồn ảnh GAN và Diffusion hiện đại, nhưng chưa thể kết luận mô hình đề xuất vượt trội tuyệt đối so với các phương pháp baseline nếu các baseline được huấn luyện lại trên cùng dữ liệu.
Cần thêm một cột trong bảng benchmark:
Phương pháp	Có retrain trên Dataset v2 không?	Ghi chú
CNNDetection	Không	Zero-shot
UniversalFakeDetect	Không	Zero-shot
DeepfakeBench	Không	Zero-shot
EfficientNet-B0	Có	Train trên Dataset v2
Như vậy sẽ minh bạch hơn.
3.5. JPEG augmentation được nhấn quá mạnh nhưng chưa có ablation study
Báo cáo nêu JPEG augmentation làm OOD AUC tăng từ 0,440 lên 0,896, tương đương +103,6%. Đây là một quan sát rất đáng chú ý, nhưng nhóm cũng ghi rõ chỉ dựa trên single seed và chưa có ablation study tách biệt.
Vì vậy, không nên viết như một kết luận chắc chắn rằng JPEG augmentation là nguyên nhân duy nhất.
Nên viết:
JPEG augmentation có tương quan mạnh với sự cải thiện OOD AUC trong thí nghiệm hiện tại. Tuy nhiên, do chưa có ablation study kiểm soát các yếu tố khác và mới chạy một seed, kết quả này nên được xem là quan sát thực nghiệm ban đầu.
Nếu có thời gian, nhóm nên làm thêm ablation:
Thí nghiệm	JPEG Aug	Blur/Noise	ColorJitter	OOD AUC
Baseline	Không	Không	Không	...
+ JPEG	Có	Không	Không	...
+ JPEG + Blur	Có	Có	Không	...
Full Aug	Có	Có	Có	...
Chỉ cần thêm bảng này, phần đóng góp về JPEG augmentation sẽ mạnh hơn nhiều.
3.6. Grad-CAM với ViT/Swin cần cẩn thận
Báo cáo ghi Grad-CAM tự động xác định target layer cho EfficientNet, ResNet, ViT và Swin. Tuy nhiên, Grad-CAM vốn phù hợp tự nhiên hơn với CNN vì có feature map không gian rõ. Với Transformer, đặc biệt ViT/Swin, cần xử lý reshape token/attention map hoặc dùng biến thể phù hợp.
Nếu web demo thật sự dùng ResNet-18 ONNX thì nên nói rõ Grad-CAM demo đang áp dụng cho ResNet-18 hoặc EfficientNet-B0, tránh gây cảm giác tất cả mô hình đều được giải thích như nhau.
Nên sửa:
Trong web demo hiện tại, Grad-CAM được ưu tiên áp dụng cho mô hình CNN/ResNet do cấu trúc convolutional phù hợp với bản đồ kích hoạt không gian. Việc mở rộng giải thích cho ViT/Swin cần các kỹ thuật chuyên biệt như attention rollout hoặc reshape transformer tokens.
Điểm này giúp báo cáo chặt chẽ hơn.
3.7. Web demo dùng ResNet-18 nhưng kết quả tốt nhất là EfficientNet-B0
Báo cáo ghi mô hình tốt nhất là EfficientNet-B0 v9, nhưng phần web demo lại ghi dùng ResNet-18 ONNX với latency ~1,5 giây/ảnh. Điều này cần giải thích rõ.
Hội đồng có thể hỏi:
Vì sao mô hình tốt nhất là EfficientNet-B0 nhưng demo lại dùng ResNet-18?
Nếu do export ONNX thuận tiện hơn hoặc ResNet-18 ổn định hơn khi tạo Grad-CAM, cần nói rõ.
Đề xuất bổ sung một câu:
Web demo hiện tại sử dụng ResNet-18 ONNX do quá trình export và tích hợp Grad-CAM ổn định hơn trong giai đoạn PoC. Trong phiên bản tiếp theo, nhóm sẽ chuyển sang EfficientNet-B0 v9 để đồng bộ với mô hình đạt kết quả tốt nhất.
Hoặc nếu EfficientNet-B0 đã chạy được demo thì nên sửa lại cho thống nhất.
3.8. Kết quả Swin-Tiny hơi bất thường, cần phân tích thận trọng
Bảng kết quả cho thấy Swin-Tiny có Val AUC 0,6198, best epoch = 0, nhưng OOD AUC lại 0,811. Điều này khá lạ: mô hình training thất bại nhưng OOD AUC lại không quá thấp. Cần kiểm tra lại:
•	Có dùng checkpoint epoch 0 không?
•	OOD AUC có tính đúng không?
•	Nhãn Real/Fake có bị đảo không?
•	AUC có dùng logits đúng chiều không?
•	Có hiện tượng phản tương quan nhưng AUC được tính sau đảo chiều không?
•	Tập OOD nhỏ nên AUC dao động mạnh?
Không nên gọi đây là “Swin thất bại” rồi vẫn đưa OOD AUC 0,811 như kết quả có ý nghĩa. Nên viết:
Kết quả Swin-Tiny không ổn định và không được xem là mô hình hợp lệ để so sánh chính, do quá trình fine-tuning không hội tụ. Các chỉ số OOD của Swin-Tiny chỉ mang tính tham khảo.
4. Góp ý theo từng phần báo cáo
4.1. Trang bìa và hành chính
Trang bìa tương đối đầy đủ, có tên trường, viện, tên đề tài, sinh viên, giảng viên hướng dẫn và ngày tháng. Tuy nhiên cần chỉnh:
•	“Viện Công Nghệ Số” nên thống nhất thành Viện Công nghệ số.
•	Địa danh nên thống nhất Thành phố Hồ Chí Minh, không để cuối mẫu là “Bình Dương”.
•	Lời cam đoan đang viết ở ngôi “tôi”, trong khi đề tài có 2 sinh viên. Nên sửa thành “nhóm chúng em” hoặc “đại diện nhóm”.
•	Cụm “nghiên cứu và phát triển độc lập của tôi” không phù hợp nếu có 2 thành viên. Nên viết “do nhóm thực hiện”.
•	Nên bổ sung Lời cảm ơn nếu mẫu yêu cầu.
4.2. Mục lục, danh mục bảng, danh mục hình
Mục lục nhìn khá chi tiết, nhưng có lỗi định dạng như số trang dính vào tiêu đề: “iLỜI CAM ĐOAN”, “ivDANH MỤC BẢNG”, “1MỞ ĐẦU”. Cần cập nhật lại mục lục tự động trong Word.
Danh mục bảng và hình khá đầy đủ, nhưng cần kiểm tra:
•	Bảng 0.1 nằm ở phần Mở đầu, đánh số “0.1” có thể không phù hợp. Có thể đổi thành Bảng MĐ.1 hoặc Bảng 1.
•	Hình ảnh minh họa GAN, Diffusion, Stable Diffusion, Midjourney cần có nguồn trích dẫn rõ nếu lấy từ Internet.
•	Danh mục từ viết tắt khá tốt, nhưng “DALL-E — tên mô hình của OpenAI” có thể giữ nguyên.
4.3. Mở đầu
Phần mở đầu tốt, có tính cấp thiết, tổng quan và khoảng trống nghiên cứu. Tuy nhiên, một số câu cần giảm mức khẳng định.
Câu:
chưa có nghiên cứu nào đồng thời giải quyết cả ba vấn đề...
Nên sửa thành:
trong phạm vi khảo sát của nhóm, chưa ghi nhận nhiều nghiên cứu đồng thời giải quyết đầy đủ cả ba vấn đề...
Cách viết này an toàn hơn, vì rất khó khẳng định “chưa có nghiên cứu nào” trong một lĩnh vực đang phát triển nhanh.
Ngoài ra, câu “theo thông tin từ Bộ Khoa học và Công nghệ” cần có tài liệu tham khảo cụ thể hoặc đường dẫn chính thức. Nếu không có nguồn, nên bỏ hoặc viết chung hơn.
4.4. Chương 2: Cơ sở lý thuyết
Chương 2 viết khá tốt, có giải thích GAN, Diffusion, CNN, EfficientNet, ResNet, ViT, Swin, Transfer Learning, Grad-CAM và chỉ số đánh giá.
Tuy nhiên, cần chú ý:
•	Nếu đề tài tên là CNN, phần ViT/Swin nên trình bày như mô hình đối chứng, không đặt ngang như đóng góp chính.
•	Phần tần số DCT/DFT có trong danh mục viết tắt nhưng chưa thấy phương pháp của đề tài sử dụng rõ ràng. Nếu không dùng, nên nói đây là hướng tham khảo trong tổng quan, không phải phương pháp chính.
•	Các hình minh họa cần có nguồn.
4.5. Chương 3: Phương pháp và xây dựng hệ thống
Đây là chương mạnh. Nhóm mô tả dữ liệu, manifest JSON, split, model architecture, augmentation, sampler, optimizer, trainer, evaluator, web demo, AWS deployment. Cấu trúc rất tốt.
Tuy nhiên, cần chỉnh:
•	Bảng nguồn dữ liệu ghi “Tổng Train ~28.220” nhưng tổng toàn dataset là 35.454. Cần tránh gây hiểu nhầm.
•	Phần “5 nguồn Kaggle” nhưng bảng có 6 dòng, gồm Deepfake Collection Real. Cần giải thích Deepfake Collection Real là nguồn thứ 6 hay là subset bổ sung từ Diffusion Fakes.
•	Camera vs AI có 400 ảnh nhưng train ghi 218 + OOD 182. Cần trình bày rõ ngay trong bảng.
•	Cần bổ sung giấy phép/dẫn nguồn dataset.
•	Cần bổ sung quy trình lọc ảnh lỗi, ảnh hỏng, ảnh quá nhỏ, ảnh duplicate nếu có.
4.6. Chương 4: Kết quả thực nghiệm
Kết quả ấn tượng, nhưng cần diễn giải thận trọng. ID AUC 0,998 rất cao, có thể đúng nhưng cũng dễ gây nghi ngờ overfitting/dataset leakage. Vì vậy cần tăng độ tin cậy bằng:
•	Deduplication.
•	Đánh giá nhiều seed.
•	Kiểm tra robustness với JPEG q=60, crop/resize thật sự.
•	Báo cáo confidence interval cho OOD.
•	Báo cáo confusion matrix.
•	Báo cáo per-source OOD.
•	Có ví dụ ảnh dự đoán sai.
Phần benchmark với baseline cần nhấn mạnh zero-shot để tránh so sánh quá mức.
5. Các điểm hội đồng có thể hỏi
Nhóm cần chuẩn bị trả lời các câu sau:
1.	Vì sao tên đề tài là CNN nhưng lại có ViT và Swin Transformer?
2.	Dataset có bị trùng ảnh giữa train/test không?
3.	Nhóm đã kiểm tra duplicate bằng phương pháp nào?
4.	Vì sao Test OOD chỉ có 182 ảnh?
5.	OOD AUC 0,896 có đủ tin cậy thống kê không?
6.	Các baseline có được retrain trên cùng dataset không?
7.	Nếu baseline không retrain thì so sánh có công bằng không?
8.	JPEG augmentation có thật sự là nguyên nhân tăng OOD AUC không?
9.	Đã chạy ablation study chưa?
10.	Đã chạy nhiều seed chưa?
11.	Tại sao EfficientNet-B0 nhỏ nhất lại tốt nhất?
12.	Vì sao Swin-Tiny best epoch = 0?
13.	Web demo dùng mô hình nào? Vì sao không dùng mô hình tốt nhất?
14.	Grad-CAM có thật sự giải thích được ảnh fake không hay chỉ là minh họa attention?
15.	Mô hình có phát hiện được ảnh AI mới như DALL-E 3, Midjourney v6, Stable Diffusion XL không?
16.	Mô hình có bị đánh lừa bởi ảnh bị chụp màn hình, nén mạng xã hội, crop, blur không?
17.	Có xử lý ảnh video/deepfake không hay chỉ ảnh tĩnh?
18.	Hệ thống có thể triển khai thực tế ở đâu?
19.	Có vấn đề đạo đức nào khi phát hiện ảnh AI không?
20.	Đề tài có đóng góp thuật toán mới không hay chủ yếu là hệ thống tích hợp?
6. Các lỗi cần sửa ngay trước khi nộp
1.	Cân nhắc đổi tên đề tài thành “Xây dựng hệ thống phát hiện ảnh tổng hợp bằng học sâu kết hợp Grad-CAM”.
2.	Sửa “Viện Công Nghệ Số” thành Viện Công nghệ số.
3.	Thống nhất địa danh: Thành phố Hồ Chí Minh, không để “Bình Dương” ở phần ký.
4.	Lời cam đoan chuyển từ “tôi” sang “nhóm chúng em” hoặc “đại diện nhóm”.
5.	Cập nhật lại mục lục để tránh lỗi dính số trang vào tiêu đề.
6.	Làm rõ số nguồn dữ liệu: 5 hay 6 nguồn.
7.	Bổ sung nguồn/giấy phép dataset.
8.	Bổ sung hoặc ghi rõ chưa thực hiện kiểm tra duplicate bằng perceptual hashing.
9.	Diễn giải lại benchmark baseline: baseline zero-shot, không retrain.
10.	Giảm mức khẳng định về “vượt baseline quốc tế”.
11.	Giảm mức khẳng định về OOD do tập OOD chỉ 182 ảnh.
12.	Làm rõ web demo dùng ResNet-18 hay EfficientNet-B0.
13.	Làm rõ Grad-CAM áp dụng cho mô hình nào.
14.	Bổ sung ablation JPEG augmentation nếu có thể.
15.	Bổ sung nhiều seed hoặc ít nhất ghi đây là single-seed experiment.
16.	Kiểm tra lại kết quả Swin-Tiny và cách diễn giải.
17.	Bổ sung bảng confusion matrix hoặc kết quả sai/dự đoán nhầm.
18.	Bổ sung hạn chế về việc chưa kiểm thử trên ảnh AI mới nhất hoặc ảnh qua mạng xã hội.
19.	Rà soát format bảng biểu, dấu chấm phẩy, thống nhất dấu thập phân 0,998 hoặc 0.998.
20.	Kiểm tra tài liệu tham khảo theo một chuẩn thống nhất IEEE/APA.
7. Đánh giá mức độ hoàn thiện
Tiêu chí	Nhận xét	Mức độ
Ý tưởng đề tài	Rất thời sự, có ý nghĩa xã hội và kỹ thuật	Tốt
Tổng quan nghiên cứu	Có phân nhóm, có phân tích khoảng trống	Tốt
Bộ dữ liệu	Quy mô khá lớn, đa nguồn, có ID/OOD	Khá tốt nhưng cần dedup
Phương pháp	Transfer Learning, augmentation, sampler, evaluator rõ	Tốt
Thực nghiệm	Có nhiều mô hình, nhiều chỉ số, benchmark	Tốt nhưng cần diễn giải thận trọng
OOD evaluation	Có nhưng tập OOD nhỏ	Cần củng cố
Web demo	Có tính ứng dụng, tích hợp Grad-CAM	Khá tốt
Tính mới	Tích hợp dữ liệu mới + XAI + mô hình nhẹ, không phải thuật toán mới	Khá
Hình thức	Còn lỗi hành chính, mục lục, địa danh, chủ thể cam đoan	Cần chỉnh
Khả năng nghiệm thu	Có thể nghiệm thu tốt nếu chỉnh lại cách diễn giải và hình thức	Tốt sau khi sửa

8. Gợi ý nhận xét chính thức cho sinh viên
Thầy/cô có thể dùng đoạn sau để góp ý trực tiếp cho nhóm:
Báo cáo đã thể hiện nỗ lực nghiên cứu nghiêm túc của nhóm trong lĩnh vực phát hiện ảnh tổng hợp do AI tạo ra. Đề tài có tính thời sự cao, phù hợp với bối cảnh phát triển mạnh của các mô hình tạo sinh như GAN và Diffusion, đồng thời có ý nghĩa thực tiễn trong việc hỗ trợ kiểm chứng nội dung số, hạn chế tin giả và deepfake. Nhóm đã xây dựng được bộ dữ liệu đa nguồn, huấn luyện và so sánh nhiều kiến trúc học sâu, đánh giá trên cả tập nội miền và ngoài miền, đồng thời tích hợp Grad-CAM và web demo nhằm tăng tính minh bạch của hệ thống.
Tuy nhiên, báo cáo cần tiếp tục hoàn thiện để tăng độ tin cậy khoa học. Nhóm cần làm rõ sự phù hợp giữa tên đề tài “CNN” và việc sử dụng thêm các kiến trúc Transformer như ViT, Swin; đồng thời cần diễn giải thận trọng hơn đối với kết quả OOD do tập ngoài miền còn nhỏ. Phần so sánh với các baseline quốc tế cần nêu rõ các baseline đang được đánh giá ở chế độ zero-shot, không được huấn luyện lại trên Dataset v2, do đó không nên kết luận vượt trội tuyệt đối. Ngoài ra, nhóm cần bổ sung hoặc thừa nhận rõ việc chưa kiểm tra trùng lặp dữ liệu bằng perceptual hashing, làm rõ mô hình dùng trong web demo, và chuẩn hóa lại các lỗi hình thức như tên đơn vị, địa danh, mục lục và lời cam đoan.
Nhìn chung, đây là một báo cáo có chất lượng kỹ thuật tốt, có sản phẩm demo và có khả năng phát triển tiếp. Nếu nhóm bổ sung thêm kiểm tra dữ liệu, diễn giải kết quả thận trọng hơn và hoàn thiện hình thức trình bày, đề tài có thể được đánh giá ở mức tốt đối với nghiên cứu khoa học sinh viên.
9. Kết luận ngắn gọn
Đây là một trong những báo cáo có hàm lượng kỹ thuật tốt, đặc biệt ở phần dữ liệu, huấn luyện mô hình, benchmark và demo. Điểm cần chỉnh không phải là ý tưởng hay năng lực triển khai, mà là cách diễn giải kết quả sao cho chặt chẽ và an toàn khoa học. Ba điểm cần sửa gấp là: tên đề tài CNN chưa khớp với ViT/Swin, tập OOD nhỏ nên không kết luận quá mạnh, và baseline zero-shot nên không so sánh như một cuộc thi công bằng tuyệt đối. Sau khi chỉnh, báo cáo này có thể đạt mức tốt khi nghiệm thu.

