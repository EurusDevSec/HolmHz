## 💡 Context

> **Task ID**: P0-001  
> **Phase**: Phase 0 - Research & Benchmarking  
> **Sprint**: N/A (Research phase)  
> **Status**: ✅ DONE  
> **Created**: 01/2026  
> **Completed**: 10/02/2026  
> **Assignee**: Hoàng

> [!note]+ Mô tả vấn đề cho AI

> _Viết 2-3 câu, gửi cho AI để refine thành User Story_

> Tôi cần chạy thử các dự án có sẵn để hiểu trước kết quả họ làm ra, rồi tổng hợp kết quả rút ra phân tích, kinh nghiệm.

---

## 🤖 AI Refined

> **User Story:**

> As a **DevOps Engineer / AI Researcher**, I want to **set up and execute existing deepfake detection projects (e.g., CNNDetection, UniversalFakeDetect, DeepfakeBench)** so that **I can understand the practical workflow, validate system architecture decisions, and establish a performance baseline for the HolmHz project.**

**Acceptance Criteria:**

- [x] **Environment Setup**: Successfully create a local python environment and install dependencies for at least one target project (CNNDetection or UniversalFakeDetect).
- [x] **Data Prep**: Collect a small dataset of Real (selfie, unplash) and Fake (thispersondoesnotexist) images for testing.
- [x] **Execution**: Successfully run the inference/demo script and obtain prediction results (Real/Fake + Confidence score).
- [x] **Analysis Report**: Update `docs/RUN_EXISTING_PROJECTS.md` or create a log with findings regarding:
  - Input/Output formats (Image size, return types).
  - Project folder structure and configuration methods.
  - Performance observations (speed, accuracy on test set).
- [x] **Integration Plan**: List specific "Action Items" or patterns (e.g., Gradio structure, config handling) to be adopted into HolmHz.

---

## 🛠️ Implementation

- [x] Branch: `chore/research/run-existing-projects`
- [x] PR Created
- [x] Tests Passed (Manual verification of scripts)
- [x] Deployed (Analysis Doc Updated)

---

## 📝 Notes

> _Quick notes, links, snippets_

- **Reference Doc**: [docs/RUN_EXISTING_PROJECTS.md](../RUN_EXISTING_PROJECTS.md) - Contains full detailed instructions.
- **Repositories**:
  - `https://github.com/PeterWang512/CNNDetection` (Easier start)
  - `https://github.com/Yuheng-Li/UniversalFakeDetect` (SOTA approach)
  - `https://github.com/SCLBD/DeepfakeBench` (Benchmark framework)
- **Key Focus**: Don't just run it, analyze _how_ they built it (folder structure, utils, model loading).

---

## 📊 Kết quả (10/02/2026)

| Project             | Kết quả GAN | Kết quả Diffusion (Gemini/Flux) | Deep Dive                                  |
| ------------------- | ----------- | ------------------------------- | ------------------------------------------ |
| CNNDetection        | ✅ 94.6%    | ❌ 6%                           | `research/CNNDetection_DeepDive.md`        |
| UniversalFakeDetect | ✅ 100%     | ❌ <10%                         | `research/UniversalFakeDetect_DeepDive.md` |
| DeepfakeBench       | ⚠️ 50.7%    | ⚠️ 50.7% (đoán mò)              | `research/DeepfakeBench_DeepDive.md`       |

**Bài học chính**: Training data > Architecture. Tất cả SOTA fail trên Diffusion hiện đại.
