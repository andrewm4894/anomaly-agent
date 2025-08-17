.PHONY: requirements-compile
.PHONY: requirements-install
.PHONY: requirements-dev
.PHONY: build publish tests
.PHONY: pre-commit pre-commit-install pre-commit-fix
.PHONY: examples venv

# Virtual environment activation
VENV_ACTIVATE = source venv/bin/activate &&

# Check if venv exists, create if not
venv:
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python -m venv venv; \
		$(VENV_ACTIVATE) pip install --upgrade pip; \
	fi

requirements-compile: venv
	$(VENV_ACTIVATE) pip-compile requirements.compile

requirements-install: venv
	$(VENV_ACTIVATE) pip install -r requirements.txt

requirements-dev: venv
	$(VENV_ACTIVATE) pip install -r requirements-dev.txt

build: venv
	$(VENV_ACTIVATE) python -m build

publish: venv
	$(VENV_ACTIVATE) python -c "import glob; import os; files = glob.glob('dist/*.whl') + glob.glob('dist/*.tar.gz'); latest = max(files, key=os.path.getctime) if files else exit(1); print(f'Publishing: {latest}'); input('Continue? (y/n) ') == 'y' or exit(1); exit(os.system(f'twine upload {latest}'))"

tests: venv
	$(VENV_ACTIVATE) pytest tests/ -v --cov=anomaly_agent --cov-report=term-missing

test: tests

pre-commit-install: venv
	$(VENV_ACTIVATE) pip install pre-commit
	$(VENV_ACTIVATE) pre-commit install

pre-commit: venv
	$(VENV_ACTIVATE) pre-commit run --all-files

pre-commit-fix: venv
	$(VENV_ACTIVATE) pre-commit run --all-files --fix

examples: venv
	$(VENV_ACTIVATE) python examples/examples.py $(ARGS)
