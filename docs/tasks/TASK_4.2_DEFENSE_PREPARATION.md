## 💡 Context

> **Task ID**: S4-002  
> **Phase**: Phase 2 - Web Application & Report  
> **Sprint**: Sprint 4 - Documentation & Defense Prep  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: 15/05/2026  
> **Assignee**: Hoàng + Luân  
> **Blocked by**: S4-001 (báo cáo), S3-002 (web demo)  
> **Blocks**: Không — **ĐÂY LÀ TASK CUỐI CÙNG**  
> **Milestone**: ✅ M4 — Defense Ready (Full Package)

> Chuẩn bị mọi thứ cho buổi bảo vệ: slide, video demo, Q&A, đóng gói.

---

## 🤖 AI Refined

> **User Story:**

> As a **Research Student**, I want to **prepare a complete defense package including slides, demo video, Q&A preparation, and packaged source code** so that **I can confidently present the research to the scientific committee.**

**Acceptance Criteria:**

- [ ] Slide thuyết trình (15-20 slides): Problem → Method → Results → Demo → Conclusion
- [ ] Video demo (2-3 phút): quay màn hình web demo chạy real-time
- [ ] Q&A preparation: ≥20 câu hỏi dự kiến + câu trả lời
- [ ] Source code đóng gói: README, requirements, Makefile, model weights
- [ ] Hồ sơ nghiệm thu đầy đủ theo mẫu trường
- [ ] Luyện tập thuyết trình ít nhất 2 lần (Hoàng + Luân)

---

## 🛠️ Implementation

### Subtasks

**Hoàng:**

- [ ] 4.2.1 Tạo slide thuyết trình (Google Slides / PowerPoint)
- [ ] 4.2.2 Quay video demo (OBS / screen recording)
- [ ] 4.2.3 Chuẩn bị Q&A document (câu hỏi kỹ thuật + trả lời)
- [ ] 4.2.4 Đóng gói source code (clean, documented)
- [ ] 4.2.5 Tạo README.md hướng dẫn setup + chạy
- [ ] 4.2.6 Export model weights (.pt + .onnx) vào `weights/`
- [ ] 4.2.7 Chuẩn bị hồ sơ nghiệm thu

**Luân:**

- [ ] 4.2.8 Chuẩn bị ảnh test demo đa dạng (selfie, AI generated, memes)
- [ ] 4.2.9 Luyện tập thuyết trình cùng Hoàng (≥2 sessions)

### Branch & PR

- [ ] Branch: `release/v1.0`
- [ ] Tag: `v1.0.0`
- [ ] All deliverables uploaded

---

## 📝 Notes

> **Slide outline (15-20 slides):**
>
> 1. Title + Team
> 2. Problem statement (deepfake threat)
> 3. Research objectives
> 4. Related work (3 SOTA + results)
>    5-6. Methodology (architecture, data, training)
>    7-8. Results (bảng benchmark, ROC curves)
>    9-10. XAI analysis (heatmap gallery)
> 5. Live demo / video demo
> 6. Contributions
> 7. Limitations + Future work
> 8. Q&A

> **Q&A categories dự kiến:**
>
> - "Tại sao chọn EfficientNet-B0 mà không phải model khác?"
> - "Làm sao model handle ảnh chưa từng thấy (OOD)?"
> - "XAI có giúp improve model không hay chỉ để giải thích?"
> - "So với CLIP approach, CNN approach có ưu điểm gì?"
> - "Nếu có thêm thời gian, bạn sẽ làm gì khác?"

> **Final checklist trước defense:**
>
> - [ ] Báo cáo in + đóng bìa
> - [ ] USB chứa source code + weights + video
> - [ ] Laptop demo sẵn sàng (đã test trước 1 ngày)
> - [ ] Backup slide trên Google Drive
