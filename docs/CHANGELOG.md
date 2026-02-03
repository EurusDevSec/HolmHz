# 📋 CHANGELOG: So sánh PROJECT_PLAN.md vs plan.md gốc

> **Mục đích**: Tài liệu này ghi lại tất cả sự khác biệt giữa kế hoạch đã được hội đồng duyệt (`plan.md`) và bản PROJECT_PLAN.md được tạo để quản lý dự án.  
> **Ngày tạo**: 02/02/2026  
> **Tác giả**: AI Assistant

---

## 📊 Tổng quan thay đổi

| Loại                                 | Số lượng |
| ------------------------------------ | -------- |
| 🔴 Thay đổi quan trọng (cần xem xét) | 5        |
| 🟡 Thay đổi nhỏ (có thể chấp nhận)   | 4        |
| 🟢 Giữ nguyên/Bổ sung tốt            | 6        |

---

## 🔴 THAY ĐỔI QUAN TRỌNG (Cần xem xét kỹ)

### 1. KPI - AUC Out-of-Domain (OOD)

|                | plan.md (Gốc) | PROJECT_PLAN.md |
| -------------- | ------------- | --------------- |
| **Giá trị**    | ≥ **0.85**    | ≥ **0.75**      |
| **Chênh lệch** | -             | **Giảm 10%**    |

**Lý do thay đổi**:

- Dựa trên phân tích CRITICAL_ANALYSIS.md, SOTA hiện tại chỉ đạt 0.70-0.82 cho OOD
- Wang et al. (2020) đạt 0.78, UniversalFakeDetect (2023) đạt 0.82

**Khuyến nghị**:

- [ ] **Giữ nguyên 0.85** nếu hội đồng đã duyệt và không cho phép thay đổi
- [ ] Hoặc đề xuất điều chỉnh với hội đồng kèm tài liệu chứng minh

---

### 2. KPI - AUC In-Domain

|                | plan.md (Gốc) | PROJECT_PLAN.md |
| -------------- | ------------- | --------------- |
| **Giá trị**    | ≥ **0.92**    | ≥ **0.90**      |
| **Chênh lệch** | -             | **Giảm 2%**     |

**Lý do thay đổi**: Tạo buffer an toàn

**Khuyến nghị**:

- [ ] **Giữ nguyên 0.92** - đây là mức hoàn toàn khả thi với EfficientNet

---

### 3. KPI - Accuracy

|                | plan.md (Gốc) | PROJECT_PLAN.md |
| -------------- | ------------- | --------------- |
| **Giá trị**    | ≥ **90%**     | ≥ **88%**       |
| **Chênh lệch** | -             | **Giảm 2%**     |

**Khuyến nghị**:

- [ ] **Giữ nguyên 90%**

---

### 4. Kiến trúc mô hình

|                     | plan.md (Gốc)                   | PROJECT_PLAN.md                           |
| ------------------- | ------------------------------- | ----------------------------------------- |
| **Thiết kế**        | **Dual-branch BẮT BUỘC**        | Single-branch trước, dual-branch optional |
| **Nhánh Spatial**   | EfficientNet-B0 / ConvNeXt-Tiny | EfficientNet-B0                           |
| **Nhánh Frequency** | SRM + DCT/FFT **BẮT BUỘC**      | Optional (Phase 1.5)                      |
| **Fusion**          | **Attention-based Fusion**      | Concat + FC (đơn giản)                    |

**Lý do thay đổi**:

- Đơn giản hóa để đảm bảo hoàn thành đúng hạn
- Baseline trước, nâng cấp sau

**Khuyến nghị**:

- [ ] **Khôi phục dual-branch bắt buộc** theo đúng kế hoạch gốc
- [ ] Hoặc đề xuất phân chia: Giai đoạn 2 = baseline, Giai đoạn 3 = fusion

---

### 5. Thành viên nhóm

|                  | plan.md (Gốc)              | PROJECT_PLAN.md |
| ---------------- | -------------------------- | --------------- |
| **Số lượng**     | **2 người**                | **1 người**     |
| **Thành viên 1** | Lê Văn Hoàng (Nhóm trưởng) | Lê Văn Hoàng    |
| **Thành viên 2** | Ngô Huỳnh Bảo Luân         | ❌ **THIẾU**    |

**Lý do thay đổi**: User yêu cầu chỉ focus vào bản thân

**Khuyến nghị**:

- [ ] **Thêm lại Ngô Huỳnh Bảo Luân** vào PROJECT_PLAN.md
- [ ] Phân công task rõ ràng cho từng người

---

## 🟡 THAY ĐỔI NHỎ (Có thể chấp nhận)

### 6. Web Framework

|             | plan.md (Gốc) | PROJECT_PLAN.md |
| ----------- | ------------- | --------------- |
| **Backend** | Flask         | FastAPI         |

**Đánh giá**: ✅ Chấp nhận được - FastAPI hiện đại hơn, async support

---

### 7. Backbone options

|             | plan.md (Gốc)                      | PROJECT_PLAN.md     |
| ----------- | ---------------------------------- | ------------------- |
| **Options** | ConvNeXt-Tiny hoặc EfficientNet-B0 | Chỉ EfficientNet-B0 |

**Đánh giá**: ✅ Chấp nhận được - EfficientNet-B0 nhẹ hơn, phù hợp demo

---

### 8. Số lượng ảnh dự kiến

|               | plan.md (Gốc) | PROJECT_PLAN.md |
| ------------- | ------------- | --------------- |
| **Tối thiểu** | 20,000 ảnh    | ~45,000 ảnh     |

**Đánh giá**: ✅ Tốt hơn - tăng số lượng

---

### 9. Robustness - JPEG

|                   | plan.md (Gốc) | PROJECT_PLAN.md                   |
| ----------------- | ------------- | --------------------------------- |
| **Mức chấp nhận** | AUC giảm ≤ 5% | AUC giảm ≤ 8% (q=60), ≤ 3% (q=80) |

**Đánh giá**: ⚠️ Chi tiết hơn nhưng nới lỏng hơn một chút

---

## 🟢 GIỮ NGUYÊN / BỔ SUNG TỐT

### 10. Timeline

| Hạng mục                | Trạng thái                                 |
| ----------------------- | ------------------------------------------ |
| Thời gian thực hiện     | ✅ Giữ nguyên: 7 tháng (11/2025 - 05/2026) |
| Số giai đoạn            | ✅ Giữ nguyên: 6 giai đoạn/sprints         |
| Nội dung từng giai đoạn | ✅ Tương đương                             |

---

### 11. Loại hình nghiên cứu

| Hạng mục  | Trạng thái                         |
| --------- | ---------------------------------- |
| Loại hình | ✅ Giữ nguyên: Nghiên cứu ứng dụng |

---

### 12. XAI - Grad-CAM

| Hạng mục    | Trạng thái                           |
| ----------- | ------------------------------------ |
| Phương pháp | ✅ Giữ nguyên: Grad-CAM              |
| Mục tiêu    | ✅ Giữ nguyên: Heatmap visualization |

---

### 13. Web Demo

| Hạng mục       | Trạng thái              |
| -------------- | ----------------------- |
| Latency target | ✅ Giữ nguyên: ≤ 2s/ảnh |
| Input size     | ✅ Giữ nguyên: 512x512  |

---

### 14. Bổ sung mới (Tốt)

PROJECT_PLAN.md bổ sung thêm:

| Bổ sung                 | Mô tả                                                   | Đánh giá       |
| ----------------------- | ------------------------------------------------------- | -------------- |
| **Prior Art Table**     | Bảng so sánh với Wang et al., UniversalFakeDetect, etc. | ✅ Tốt         |
| **Dataset Sources**     | Links trực tiếp đến FFHQ, GenImage, DFFD                | ✅ Rất hữu ích |
| **Evaluation Protocol** | Template so sánh methods                                | ✅ Tốt         |
| **Hướng mở rộng**       | Phase 2 (Video), Phase 3 (Audio)                        | ✅ Tốt         |
| **Disclaimer**          | Nêu rõ giới hạn dự án                                   | ✅ Cần thiết   |

---

### 15. Cấu trúc thư mục

| Hạng mục         | Trạng thái                          |
| ---------------- | ----------------------------------- |
| Folder structure | ✅ Bổ sung chi tiết hơn plan.md gốc |

---

## 📝 CHECKLIST HÀNH ĐỘNG

### Bắt buộc sửa (để khớp với kế hoạch đã duyệt):

- [ ] **KPI In-domain AUC**: Đổi từ 0.90 → **0.92**
- [ ] **KPI OOD AUC**: Đổi từ 0.75 → **0.85**
- [ ] **KPI Accuracy**: Đổi từ 88% → **90%**
- [ ] **Kiến trúc**: Đưa dual-branch (Spatial + Frequency) thành **bắt buộc**
- [ ] **Thành viên**: Thêm **Ngô Huỳnh Bảo Luân** và phân công task

### Tùy chọn (có thể giữ):

- [ ] FastAPI thay Flask - OK
- [ ] EfficientNet-B0 làm backbone chính - OK
- [ ] Bổ sung Prior Art, Dataset Sources - OK
- [ ] Hướng mở rộng Video/Audio - OK

---

## 🔗 Tham chiếu files

| File                                         | Mô tả                               |
| -------------------------------------------- | ----------------------------------- |
| [plan.md](plan.md)                           | Kế hoạch gốc đã được hội đồng duyệt |
| [PROJECT_PLAN.md](PROJECT_PLAN.md)           | Bản working plan chi tiết           |
| [CRITICAL_ANALYSIS.md](CRITICAL_ANALYSIS.md) | Phân tích Devil's Advocate          |

---

## 📌 Ghi chú

1. **plan.md** là tài liệu pháp lý với hội đồng - **KHÔNG ĐƯỢC thay đổi**
2. **PROJECT_PLAN.md** là bản internal - có thể điều chỉnh nhưng KPI phải >= plan.md
3. Nếu muốn điều chỉnh KPI, phải đề xuất chính thức với GVHD và hội đồng

---

_Cập nhật lần cuối: 02/02/2026_
