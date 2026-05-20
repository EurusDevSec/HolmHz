# KHUNG SLIDE NGHIỆM THU — HolmHz
> **Đề tài**: Xây dựng hệ thống phát hiện ảnh tổng hợp bằng Mạng nơ-ron tích chập (CNN)  
> **Thời lượng đề xuất**: 10–12 phút trình bày + 5–7 phút Q&A  
> **Số slide**: ~14 slide chính

---

## SLIDE 1 — TRANG BÌA

**Nội dung:**
- Tên đề tài: **Xây dựng hệ thống phát hiện ảnh tổng hợp bằng Mạng nơ-ron tích chập (CNN)**
- Nhóm: Lê Văn Hoàng (D22CNTT02) — Ngô Huỳnh Bảo Luân (D25CNTT10)
- GVHD: ThS. Nguyễn Trung Kiệt
- Viện Công nghệ số — Trường ĐH Thủ Dầu Một — 2025–2026

**Hình ảnh gợi ý:** Logo TDMU + ảnh minh hoạ deepfake (1 ảnh thật / 1 ảnh fake đặt cạnh nhau)
- Ảnh thật mẫu: `imgs/Real/106.jpg`
- Ảnh fake mẫu: `imgs/Fake_AI_generated/Image_1.jpg`

---

## SLIDE 2 — VẤN ĐỀ VÀ TÍNH CẤP THIẾT

**Nội dung:**
- Deepfake / AI-generated images đang bùng nổ: Midjourney, DALL-E, Stable Diffusion
- Hậu quả: Tin giả, lừa đảo trực tuyến, bôi nhọ danh dự
- Thống kê tại Việt Nam (Bộ Công an: số vụ lừa đảo deepfake tăng X%/năm)
- Câu hỏi nghiên cứu: **Mô hình CNN nào phát hiện hiệu quả nhất trên cả ảnh GAN lẫn Diffusion?**

**Hình ảnh:**
- Bảng so sánh: `docs/final_report.md` → **Bảng MĐ.1** (So sánh các nghiên cứu tiêu biểu)
- Layout: 2 cột — Trái: ảnh minh hoạ fake (chân dung AI), Phải: bullet vấn đề

---

## SLIDE 3 — MỤC TIÊU & KPI ĐỀ TÀI

**Nội dung:**
- 5 mục tiêu cụ thể → 5 KPI đo được

| KPI | Mục tiêu | Đạt được | Trạng thái |
|-----|----------|----------|------------|
| Dataset ≥ 20.000 ảnh | 20.000 | **28.220** | ✅ +41% |
| ID AUC ≥ 0,92 | 0,92 | **0,998** | ✅ |
| ID Accuracy ≥ 90% | 90% | **98,4%** | ✅ |
| OOD AUC ≥ 0,85 | 0,85 | **0,896** | ✅ |
| Web demo ≤ 2s/ảnh | 2s | **~1,5s** | ✅ |

> 📌 **Nguồn bảng**: `docs/final_report.md` → **Bảng 4.7** (Đánh giá KPI đề tài)

---

## SLIDE 4 — BỘ DỮ LIỆU HolmHz-v2

**Nội dung trái (thống kê):**
- **35.454 ảnh** tổng cộng từ 5 nguồn Kaggle
- 8+ loại AI generator
- Split: Train 28.220 / Val 3.526 / Test ID 3.526 / Test OOD 182

**Nội dung phải (bảng nguồn):**

| # | Nguồn | Generator | Số lượng |
|---|-------|-----------|:--------:|
| 1 | RVF10K | StyleGAN | 8.000 |
| 2 | DeepDetect-2025 | Diffusion mixed | 8.000 |
| 3 | Diffusion Fakes | DALL-E, MJ, SD... | 4.024 |
| 4 | CIPLab Faces | Face manipulation | 3.266 |
| 5 | Camera vs AI | Mixed AI | 400 |

> 📌 **Nguồn bảng**: `docs/final_report.md` → **Bảng 3.1** (Nguồn dữ liệu v2)  
> 📌 **Nguồn thống kê**: `docs/final_report.md` → **Bảng 3.2** (Chia tập dữ liệu)

**Điểm nhấn:** Tập Test OOD (182 ảnh Camera vs AI) **hoàn toàn chưa thấy trong training** → đánh giá khả năng tổng quát hoá thực sự.

---

## SLIDE 5 — KIẾN TRÚC MÔ HÌNH

**Nội dung:** Sơ đồ pipeline Backbone + Head

```
Input [224×224]
  → Backbone (pretrained ImageNet)  →  [feature_dim]
  → Dropout(0.3)
  → Linear(feature_dim → 1)         →  P(Fake) ∈ [0,1]
```

**Bảng feature dimension:**

| Backbone | Tham số | Feature Dim |
|----------|:-------:|:-----------:|
| EfficientNet-B0 | 4M | 1.280 |
| ResNet-18 | 11M | 512 |
| ViT-Small/16 | 22M | 384 |
| Swin-Tiny | 28M | 768 |

> 📌 **Nguồn bảng**: `docs/final_report.md` → **Bảng 3.3** (Feature dimension theo backbone)

**Điểm nhấn kỹ thuật:**
- Registry Pattern → swap model chỉ bằng đổi 1 dòng config YAML
- JPEG Augmentation (quality 50–95) → kỹ thuật **then chốt** cải thiện OOD

---

## SLIDE 6 — PHƯƠNG PHÁP HUẤN LUYỆN

**Nội dung:**

| Hyperparameter | Giá trị |
|----------------|---------|
| Optimizer | AdamW (lr = 3×10⁻⁴, wd = 0.01) |
| Scheduler | Cosine Annealing |
| Loss | BCEWithLogitsLoss (pos_weight = 1.0) |
| Epochs | 30 (Early Stopping patience = 7) |
| Batch Size | 32 (CNN) / 16 (Transformer) |
| Sampler | WeightedRandomSampler |
| Platform | Kaggle T4 × 2 (DataParallel) |

> 📌 **Nguồn bảng**: `docs/final_report.md` → **Bảng 4.1** (Môi trường và tham số thực nghiệm)

**Điểm nhấn:** Augmentation pipeline — JPEG compression ngẫu nhiên bắt model học đặc trưng **bền vững**, không phụ thuộc compression artifacts

> 📌 **Nguồn bảng aug**: `docs/final_report.md` → **Bảng 3.4** (Training augmentation pipeline)

---

## SLIDE 7 — KẾT QUẢ HUẤN LUYỆN 4 MÔ HÌNH

**Bảng:**

| Mô hình | Tham số | Best Epoch | Val AUC | Checkpoint |
|---------|:-------:|:----------:|:-------:|:----------:|
| **EfficientNet-B0 v9** | **4M** | 25/30 | **0,9993** | best_v9.pt (46MB) |
| ResNet-18 | 11M | 28/30 | 0,9956 | best_resnet18_v2.pt |
| ViT-Small/16 | 22M | 29/30 | 0,9735 | best_vit_small_v2.pt |
| Swin-Tiny† | 28M | **0/30** | 0,6198 | ❌ Training thất bại |

> 📌 **Nguồn bảng**: `docs/final_report.md` → **Bảng 4.4** (Kết quả huấn luyện 4 mô hình)

**Điểm nhấn:** Swin-Tiny (mô hình LỚN NHẤT) thất bại hoàn toàn — lr=3×10⁻⁴ quá cao cho Transformer → **bài học fine-tuning**

---

## SLIDE 8 — BENCHMARK 7 MÔ HÌNH ⭐ (Slide quan trọng nhất)

**Biểu đồ cột:**

> 🖼️ **Nguồn hình**: `outputs/benchmark/final_benchmark/id_vs_ood_auc.png`

**Bảng đầy đủ:**

| Nhóm | Phương pháp | Tham số | ID AUC | OOD AUC |
|:-----|:-----------|:-------:|:------:|:-------:|
| Baseline | CNNDetection (Wang 2020) | ~23M | 0,662 | 0,325 |
| Baseline | UniversalFakeDetect (Ojha 2023) | ~304M | 0,722 | 0,486 |
| Baseline | DeepfakeBench (Yan 2023) | ~19M | 0,439 | 0,536 |
| **Ours** | **EfficientNet-B0 v9** | **4M** | **0,998** | **0,896** |
| **Ours** | ResNet-18 | 11M | 0,995 | 0,865 |
| Ours | ViT-Small/16 | 22M | 0,974 | 0,833 |
| Ours | Swin-Tiny† | 28M | 0,620 | 0,811 |

> 📌 **Nguồn bảng**: `docs/final_report.md` → **Bảng 4.5** (Benchmark 7 mô hình)  
> 📌 **Nguồn file JSON gốc**: `outputs/benchmark/v2_benchmark_results.json`

**Key message:** Mô hình **4M tham số** của chúng tôi vượt mô hình SOTA **304M tham số** (+0,41 OOD AUC).  
Cả 3 baseline thất bại vì được thiết kế cho GAN cũ, không hoạt động với Diffusion hiện đại.

---

## SLIDE 9 — BIỂU ĐỒ RADAR ĐA CHỈ SỐ

**Biểu đồ radar:**

> 🖼️ **Nguồn hình**: `outputs/benchmark/final_benchmark/radar_comparison.png`

**Cách đọc:**
- 5 trục: ID Acc, ID AUC, OOD AUC, OOD Acc, ID F1
- Diện tích lớn + đều = mô hình cân bằng tốt
- **EfficientNet-B0** → diện tích lớn nhất, gần tròn nhất
- **UniversalFakeDetect** (304M) → méo lệch nặng — chỉ tốt 1–2 chỉ số

---

## SLIDE 10 — PHÂN TÍCH OOD PER-SOURCE

**Heatmap:**

> 🖼️ **Nguồn hình**: `outputs/benchmark/final_benchmark/ood_heatmap.png`

**Bảng OOD per-source:**

| Mô hình | camera_ai (88 Fake) | camera_real (94 Real) | Nhận xét |
|---------|:-------------------:|:--------------------:|---------:|
| **EfficientNet-B0 v9** | **83,0%** | 73,4% | Phát hiện fake tốt |
| **ResNet-18** | 79,5% | **80,8%** | Cân bằng nhất |
| CNNDetection | 2,3% | 98,9% | Predict REAL cho mọi ảnh |
| UniversalFakeDetect | 4,5% | 98,9% | Predict REAL cho mọi ảnh |

> 📌 **Nguồn bảng**: `docs/final_report.md` → **Bảng 4.6** (Độ chính xác OOD per-source)

**Key message:** CNNDetection nhận đúng 98,9% ảnh thật nhưng chỉ phát hiện **2,3% ảnh fake** → vô dụng trong thực tế.

---

## SLIDE 11 — PHÁT HIỆN KHOA HỌC: JPEG AUGMENTATION

**Layout 2 cột:**

**Trái — Trước (v7, không JPEG aug):**
- OOD AUC = **0,440** (dưới random)
- camera_real accuracy = **0%** (predict fake cho mọi ảnh thật)

**Phải — Sau (v9, JPEG aug v3):**
- OOD AUC = **0,896** (+103,6%)
- camera_real accuracy = **73,4%**

**Bảng so sánh:**

| Phiên bản | OOD AUC | Kỹ thuật JPEG |
|-----------|:-------:|:-------------:|
| EfficientNet-B0 **v7** | 0,440 | ❌ Không có |
| EfficientNet-B0 **v9** | **0,896** | ✅ quality 50–95, p=0.7 |

> 📌 **Nguồn**: `docs/final_report.md` → mục 4.6 (Phân tích EfficientNet-B0)

**Kết luận:** **Chiến lược dữ liệu** quan trọng hơn kích thước mô hình.

---

## SLIDE 12 — ĐƯỜNG CONG ROC & MA TRẬN NHẦM LẪN

**Layout 2 hình cạnh nhau:**

| Hình trái | Hình phải |
|-----------|-----------|
| Đường cong ROC (EfficientNet-B0 v9) | Ma trận nhầm lẫn — ID test |

> 🖼️ **Nguồn hình ROC**: `outputs/evaluation_v9_benchmark/roc_curve.png`  
> 🖼️ **Nguồn hình CM (ID)**: `outputs/evaluation_v9_benchmark/confusion_matrix_id.png`  
> 🖼️ **Nguồn hình CM (OOD)**: `outputs/evaluation_v9_benchmark/confusion_matrix_ood.png`

**Số liệu highlight:**
- ID: TP=1684/1707 Fake đúng | FP=40/1819 Real nhầm thành Fake
- AUC = 0,9984 → gần như phân biệt hoàn hảo

---

## SLIDE 13 — EXPLAINABLE AI: GRAD-CAM

**Layout: 3–4 cặp ảnh (gốc + heatmap)**

**Ảnh FAKE được phát hiện (màu đỏ = vùng nghi ngờ):**
> 🖼️ `outputs/xai_gallery/gradcam_flux_hf_004.png` ← Flux AI fake
> 🖼️ `outputs/xai_gallery/gradcam_sd15_fake_sd15_00396.png` ← Stable Diffusion fake
> 🖼️ `outputs/xai_gallery/gradcam_tristanzhang_fake_0194.png` ← Mixed fake
> 🖼️ `outputs/xai_gallery/gradcam_Gemini_Generated_Image_h2x4b6h2x4b6h2x4.png` ← Gemini AI fake

**Ảnh REAL (heatmap phân tán, không tập trung):**
> 🖼️ `outputs/xai_gallery/gradcam_real_pexels_0046.png` ← Ảnh thật Pexels
> 🖼️ `outputs/xai_gallery/gradcam_real_camera_real_J2UHUZR5W6k.png` ← Camera thật

**Điểm nhấn:**
- Ảnh Fake: Grad-CAM **tập trung** vào khu vực da mặt, viền tóc, mắt → nơi AI để lại dấu vết
- Ảnh Real: Grad-CAM **phân tán** → không có pattern bất thường rõ ràng

---

## SLIDE 14 — WEB DEMO & KIẾN TRÚC AWS ĐỀ XUẤT

**Phần A — Web Demo (Gradio):**
- Framework: Gradio + **EfficientNet-B0 ONNX** + EXIF Analyzer
- Mode: `EfficientNet + EXIF` (fallback về EfficientNet-only nếu thiếu model phụ)
- Latency: ~1,5 giây/ảnh trên CPU (KPI ≤ 2s ✅)
- Tính năng: Upload ảnh → Real/Fake (%) + Grad-CAM heatmap + phân tích EXIF metadata

**Phần B — Kiến trúc AWS đề xuất:**
> 🖼️ **Nguồn hình**: `outputs/benchmark/final_benchmark/holmHz2_Architecture.png`

```
[User] → [Route 53] → [CloudFront + WAF]
                              ↓
                    [API Gateway] → [Lambda]
                                        ↓
                                   [S3 Heatmap]
                    [ECR] ← GitHub Actions CI/CD
                    [Secrets/Systems Manager]
                    [CloudWatch Monitoring]
```

- Chi phí ước tính: **< $5/tháng** (Serverless, ~1.000 req/ngày)
- IaC: Terraform + GitHub Actions

> 📌 **Nguồn**: `docs/AWS_ARCHITECTURE_REVIEW.md` + `docs/final_report.md` → mục 3.8

---

## SLIDE 15 — KẾT LUẬN & HƯỚNG PHÁT TRIỂN

**Kết luận — Đạt 5/5 KPI:**
- Dataset v2: 35.454 ảnh, 8+ generators (GAN + Diffusion)
- EfficientNet-B0 v9 (4M params): ID AUC **0,998**, OOD AUC **0,896**
- Vượt 3 nghiên cứu SOTA quốc tế (lớn hơn 5–75×)
- JPEG Augmentation: +103,6% OOD AUC
- Web demo Gradio: ~1,5s/ảnh CPU

**Hướng phát triển:**
1. Mở rộng sang **phát hiện video deepfake** (temporal consistency)
2. **Ensemble** EfficientNet-B0 + ResNet-18 (voting/stacking)
3. Cập nhật generator mới: Midjourney v7, Sora, Flux
4. **Triển khai cloud** theo kiến trúc AWS đã đề xuất
5. Plugin trình duyệt cảnh báo ảnh nghi ngờ

---

## PHỤ LỤC SLIDE (Q&A BACKUP)

### Backup A — Bảng per-source accuracy chi tiết (ID test)

> 📌 **Nguồn**: `docs/final_report.md` → Phụ lục A  
> 🖼️ **Hình**: `outputs/evaluation_v9_benchmark/per_source_accuracy.png`

### Backup B — Ma trận nhầm lẫn OOD

> 🖼️ **Nguồn hình**: `outputs/evaluation_v9_benchmark/confusion_matrix_ood.png`

### Backup C — Lý do Swin-Tiny thất bại

> 📌 **Nguồn**: `docs/final_report.md` → mục 4.7

| Bằng chứng | Chi tiết |
|------------|---------|
| Best epoch = 0 | Không cải thiện qua bất kỳ epoch nào |
| ID AUC = 0,620 | Dưới mức ngẫu nhiên cho nhiều nguồn |
| Recall 0,826 vs Precision 0,513 | Predict FAKE cho mọi ảnh |
| Nguyên nhân | LR = 3×10⁻⁴ quá cao; Transformer cần lr ≤ 5×10⁻⁵ + warmup |

### Backup D — Training config chi tiết

> 📌 **Nguồn**: `configs/train_v9.yaml`

---

## GHI CHÚ CHO NGƯỜI TRÌNH BÀY

```
Slide 1        → 30 giây (giới thiệu nhanh)
Slide 2–3      → 1,5 phút (vấn đề + mục tiêu)
Slide 4–5      → 2 phút (dữ liệu + kiến trúc)
Slide 6        → 1 phút (phương pháp training)
Slide 7–8      → 2 phút (kết quả + benchmark) ← TRỌNG TÂM
Slide 9–11     → 2 phút (phân tích sâu)
Slide 12–13    → 1 phút (ROC + Grad-CAM)
Slide 14–15    → 1 phút (demo + kết luận)
─────────────────────────────────────────
Tổng           ≈ 11 phút + backup Q&A
```

### Câu hỏi hội đồng thường gặp & cách trả lời:

| Câu hỏi dự kiến | Slide trả lời |
|-----------------|---------------|
| Tại sao chọn EfficientNet-B0 thay vì mô hình lớn hơn? | Slide 8, 11 |
| Tập OOD 182 ảnh có đủ để kết luận không? | Backup C + Slide 10 |
| Swin-Tiny thất bại có ảnh hưởng đến kết luận không? | Backup C |
| JPEG augmentation lấy cảm hứng từ đâu? | Slide 11 (Wang et al. 2020) |
| Hệ thống có thể phát hiện video không? | Slide 15 (hướng phát triển) |
| Chi phí triển khai thực tế? | Slide 14 (< $5/tháng) |
