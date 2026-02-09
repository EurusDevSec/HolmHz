## 💡 Context

> [!note]+ Mô tả vấn đề cho AI

> _Viết 2-3 câu, gửi cho AI để refine thành User Story_

> Tôi cần chạy thử các dự án có sẵn để hiểu trước kết quả họ làm ra, rồi tổng hợp kết quả rút ra phân tích, kinh nghiệm.

---

## 🤖 AI Refined

> **User Story:**  

> As a **DevOps Engineer / AI Researcher**, I want to **set up and execute existing deepfake detection projects (e.g., CNNDetection, UniversalFakeDetect)** so that **I can understand the practical workflow, validate system architecture decisions, and establish a performance baseline for the HolmHz project.**

**Acceptance Criteria:**

- [ ] **Environment Setup**: Successfully create a local python environment and install dependencies for at least one target project (CNNDetection or UniversalFakeDetect).
- [ ] **Data Prep**: Collect a small dataset of Real (selfie, unplash) and Fake (thispersondoesnotexist) images for testing.
- [ ] **Execution**: Successfully run the inference/demo script and obtain prediction results (Real/Fake + Confidence score).
- [ ] **Analysis Report**: Update `docs/RUN_EXISTING_PROJECTS.md` or create a log with findings regarding:
  - Input/Output formats (Image size, return types).
  - Project folder structure and configuration methods.
  - Performance observations (speed, accuracy on test set).
- [ ] **Integration Plan**: List specific "Action Items" or patterns (e.g., Gradio structure, config handling) to be adopted into HolmHz.

---

## 🛠️ Implementation

- [ ] Branch: `chore/research/run-existing-projects`
- [ ] PR Created
- [ ] Tests Passed (Manual verification of scripts)
- [ ] Deployed (Analysis Doc Updated)

---

## 📝 Notes

> _Quick notes, links, snippets_

- **Reference Doc**: [docs/RUN_EXISTING_PROJECTS.md](../RUN_EXISTING_PROJECTS.md) - Contains full detailed instructions.
- **Repositories**:
  - `https://github.com/PeterWang512/CNNDetection` (Easier start)
  - `https://github.com/Yuheng-Li/UniversalFakeDetect` (SOTA approach)
- **Key Focus**: Don't just run it, analyze _how_ they built it (folder structure, utils, model loading).
