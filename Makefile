.PHONY: sync sync-dev install install-dev
.PHONY: build publish tests
.PHONY: pre-commit pre-commit-install pre-commit-fix
.PHONY: examples

# uv commands for dependency management
sync:
	uv sync

sync-dev:
	uv sync --group dev

install:
	uv sync

install-dev:
	uv sync --group dev

# Legacy aliases for backward compatibility
requirements-install: sync

requirements-dev: sync-dev

build:
	uv build

publish:
	@uv run python -c "import glob; import os; files = glob.glob('dist/*.whl') + glob.glob('dist/*.tar.gz'); latest = max(files, key=os.path.getctime) if files else exit(1); print(f'Publishing: {latest}'); input('Continue? (y/n) ') == 'y' or exit(1); exit(os.system(f'uv run twine upload {latest}'))"

tests:
	uv run pytest tests/ -v --cov=anomaly_agent --cov-branch --cov-report=term-missing --cov-report=xml

test: tests

pre-commit-install:
	uv sync --group dev
	uv run pre-commit install

pre-commit:
	uv run pre-commit run --all-files

pre-commit-fix:
	uv run pre-commit run --all-files --fix

examples:
	@uv run python examples/examples.py $(ARGS)

# Additional uv-specific commands
add:
	uv add $(PACKAGE)

add-dev:
	uv add --group dev $(PACKAGE)

remove:
	uv remove $(PACKAGE)

lock:
	uv lock

update:
	uv sync --upgrade
