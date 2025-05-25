.PHONY: requirements-compile
.PHONY: requirements-install
.PHONY: requirements-dev
.PHONY: build publish tests
.PHONY: pre-commit pre-commit-install pre-commit-fix
.PHONY: examples


requirements-compile:
	pip-compile requirements.compile

requirements-install:
	pip install -r requirements.txt

requirements-dev:
	pip install -r requirements-dev.txt

build:
	@python -m build

publish:
	@python -c "import glob; import os; files = glob.glob('dist/*.whl') + glob.glob('dist/*.tar.gz'); latest = max(files, key=os.path.getctime) if files else exit(1); print(f'Publishing: {latest}'); input('Continue? (y/n) ') == 'y' or exit(1); exit(os.system(f'twine upload {latest}'))"

tests:
	pytest tests/ -v --cov=anomaly_agent --cov-report=term-missing

test: tests
pre-commit-install:
	pip install pre-commit
	pre-commit install

pre-commit:
	pre-commit run --all-files

pre-commit-fix:
	pre-commit run --all-files --fix

examples:
	@python examples/examples.py $(ARGS)
