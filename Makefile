# ============================================
# HolmHz Makefile
# ============================================
# Usage: make <target>
#   make train    — Train model
#   make test     — Run evaluation
#   make serve    — Start API server
#   make lint     — Lint & format code
#   make help     — Show all targets

PYTHON = .venv/Scripts/python.exe
PIP = $(PYTHON) -m pip

.PHONY: help train test serve lint format check install clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# === Development ===

install: ## Install all dependencies (editable mode)
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	$(PIP) install -e ".[dev]"

install-cpu: ## Install dependencies (CPU only)
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e ".[dev]"

# === Training & Evaluation ===

train: ## Train model (default config)
	$(PYTHON) scripts/train.py --config configs/train.yaml

test: ## Run evaluation on test set
	$(PYTHON) scripts/test.py --config configs/test.yaml

predict: ## Run prediction on single image
	$(PYTHON) scripts/predict.py --image $(IMAGE)

export: ## Export model to ONNX
	$(PYTHON) scripts/export_onnx.py --config configs/export.yaml

# === Web / API ===

serve: ## Start FastAPI server
	$(PYTHON) -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

gradio: ## Start Gradio UI
	$(PYTHON) -m app.gradio_ui

# === Code Quality ===

lint: ## Lint code with ruff
	$(PYTHON) -m ruff check src/ tests/ scripts/ app/

format: ## Format code with ruff
	$(PYTHON) -m ruff format src/ tests/ scripts/ app/

check: lint ## Run lint + tests
	$(PYTHON) -m pytest tests/ -v

# === Cleanup ===

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/
