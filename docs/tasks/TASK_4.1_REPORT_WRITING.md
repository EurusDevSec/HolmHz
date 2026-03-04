## 💡 Context

> **Task ID**: S4-001  
> **Phase**: Phase 2 - Web Application & Report  
> **Sprint**: Sprint 4 - Documentation & Defense Prep  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: 30/04/2026 (giữ nguyên — Luân bắt đầu viết Ch1-2 từ 29/03)  
> **Assignee**: Hoàng (Chương 3-5) + Luân (Chương 1-2)  
> **Blocked by**: S2-002 (benchmark results), S3-002 (web demo)  
> **Blocks**: S4-002 (Defense cần báo cáo)

> Viết báo cáo nghiên cứu khoa học. Luân viết Chương 1-2 theo outline, Hoàng viết phần kỹ thuật.

---

## 🤖 AI Refined

> **User Story:**

> As a **Research Student**, I want to **write a complete scientific report with proper experimental methodology** so that **the defense committee can evaluate the research rigorously.**

**Acceptance Criteria:**

- [ ] **Chương 1 - Mở đầu** (Luân): Bối cảnh, mục tiêu, phạm vi, đóng góp
- [ ] **Chương 2 - Tổng quan** (Luân): Related work, lý thuyết CNN, GAN, Diffusion
- [ ] **Chương 3 - Phương pháp** (Hoàng): Architecture, data pipeline, training strategy
- [ ] **Chương 4 - Kết quả** (Hoàng): Bảng benchmark, ROC curves, XAI gallery, analysis
- [ ] **Chương 5 - Kết luận** (Hoàng): Tóm tắt, hạn chế, hướng phát triển
- [ ] Format theo mẫu trường
- [ ] Review GVHD ít nhất 1 lần
- [ ] Bảng, biểu đồ, hình ảnh đầy đủ

---

## 🛠️ Implementation

### Subtasks

**Luân (Chương 1-2):**

- [ ] 4.1.1 Chương 1: Mở đầu (theo mẫu Hoàng cung cấp)
- [ ] 4.1.2 Chương 2: Tổng quan lý thuyết (theo tài liệu reference)
- [ ] 4.1.3 Gửi Hoàng review → sửa feedback

**Hoàng (Chương 3-4-5):**

- [ ] 4.1.4 Chương 3: Phương pháp (kiến trúc, data, training)
- [ ] 4.1.5 Chương 4: Kết quả thực nghiệm (bảng, biểu đồ từ Sprint 2)
- [ ] 4.1.6 Chương 5: Kết luận và hướng phát triển
- [ ] 4.1.7 Tạo bảng, biểu đồ, hình ảnh minh họa

**Tổng hợp:**

- [ ] 4.1.8 Merge Chương 1-2 của Luân
- [ ] 4.1.9 Format theo mẫu trường
- [ ] 4.1.10 Review với GVHD

### Branch & PR

- [ ] Branch: `docs/report`
- [ ] Google Docs link shared (collaborative editing)
- [ ] GVHD review passed

---

## 📝 Notes

> **Outline Chương 4 (phần quan trọng nhất cho hội đồng):**
>
> 1. Thiết lập thí nghiệm (dataset, hardware, hyperparams)
> 2. Kết quả In-domain (bảng AUC, accuracy)
> 3. Kết quả OOD (per-source breakdown)
> 4. So sánh với 3 SOTA (bảng từ S2-002)
> 5. Phân tích XAI (heatmap gallery, interpretability)
> 6. Ablation studies (nếu có: freeze vs fine-tune, augmentation effect)
> 7. Discussion: tại sao HolmHz works/fails trên từng source

> **Deadline cứng:** GVHD cần nhận bản draft trước 30/04 để review.
