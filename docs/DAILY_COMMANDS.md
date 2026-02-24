# 📋 Lệnh Kiểm Tra Thường Xuyên — HolmHz

> Bookmark file này. Mỗi ngày làm việc chạy theo thứ tự từ trên xuống.

---

## 🌅 Buổi sáng — Trước khi code

```bash
# 1. Di chuyển vào project
cd R:/_Projects/Eurus_Workspace/HolmHz

# 2. Kích hoạt virtual environment (BẮT BUỘC mỗi lần mở terminal mới)
.venv\Scripts\activate
# Kết quả đúng: thấy (.venv) ở đầu dòng terminal

# 3. Pull code mới nhất từ remote
git pull

# 4. Xem đang ở branch nào + file nào thay đổi
git status
```

---

## ✅ Kiểm tra môi trường

```bash
# Kiểm tra tất cả import chính
python -c "import torch, timm, holmhz, albumentations, wandb; print('All imports OK')"

# Kiểm tra GPU (kết quả mong đợi: True + tên GPU)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Kiểm tra phiên bản holmhz
python -c "import holmhz; print(f'holmhz v{holmhz.__version__}')"

# Kiểm tra phiên bản PyTorch
python -c "import torch; print(torch.__version__)"
```

---

## 🔍 Kiểm tra code quality

```bash
# Lint — kiểm tra lỗi (chạy trước mỗi commit)
ruff check .

# Lint + tự sửa các lỗi có thể fix tự động
ruff check . --fix

# Format — tự động sửa format code
ruff format .

# Hoặc dùng Makefile shortcut
make lint
make format
```

---

## 🧪 Chạy tests

```bash
# Chạy toàn bộ test suite
pytest tests/ -v

# Chạy 1 file test cụ thể
pytest tests/test_backbones.py -v

# Chạy với coverage report
pytest tests/ --cov=src/holmhz --cov-report=term-missing

# Makefile shortcut
make test
```

---

## 🌿 Git workflow thường ngày

```bash
# Xem status
git status

# Xem lịch sử commit gần nhất
git log --oneline -10

# Xem diff những gì vừa thay đổi
git diff

# Stage và commit
git add .
git commit -m "feat: mô tả thay đổi"   # tính năng mới
git commit -m "fix: mô tả bug đã sửa"  # bug fix
git commit -m "docs: cập nhật tài liệu" # docs

# Push lên remote
git push

# Push lần đầu cho branch mới
git push -u origin tên-branch
```

### Quy tắc đặt tên commit message

| Prefix      | Dùng khi              |
| ----------- | --------------------- |
| `feat:`     | Thêm tính năng mới    |
| `fix:`      | Sửa bug               |
| `docs:`     | Cập nhật tài liệu     |
| `refactor:` | Cấu trúc lại code     |
| `test:`     | Thêm/sửa test         |
| `chore:`    | Cài đặt, config, deps |

---

## 📦 Kiểm tra packages

```bash
# Xem tất cả packages đã cài trong venv
pip list

# Kiểm tra package cụ thể
pip show torch
pip show holmhz

# Tìm package có chứa từ khóa
pip list | grep torch
pip list | grep ruff
```

---

## 🚀 Makefile shortcuts (nhanh nhất)

```bash
make help      # xem tất cả targets có sẵn
make install   # cài đầy đủ deps (GPU)
make lint      # ruff check .
make format    # ruff format .
make check     # lint + test
make test      # pytest tests/
make train     # python scripts/train.py
make serve     # uvicorn app.api:app
make clean     # xóa cache, build artifacts
```

---

## 🌙 Cuối ngày — Trước khi tắt máy

```bash
# 1. Lint lần cuối
ruff check .

# 2. Stage và commit tất cả thay đổi
git add .
git commit -m "chore: end of day — mô tả"

# 3. Push lên remote (backup code)
git push

# 4. Xác nhận push thành công
git log --oneline -3
```

---

## ⚠️ Lỗi hay gặp & cách fix

| Lỗi                          | Nguyên nhân             | Fix                                                                                          |
| ---------------------------- | ----------------------- | -------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: torch` | Chưa activate venv      | `.venv\Scripts\activate`                                                                     |
| `ruff: command not found`    | Chưa activate venv      | `.venv\Scripts\activate`                                                                     |
| `ImportError: holmhz`        | Chưa install editable   | `pip install -e . --no-deps`                                                                 |
| `git push` bị từ chối        | Branch chưa có upstream | `git push -u origin <branch>`                                                                |
| `CUDA: False`                | PyTorch CPU version     | Cài lại PyTorch CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
