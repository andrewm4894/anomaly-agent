.PHONY: sync sync-dev install install-dev
.PHONY: build publish tests release
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

# Release via GitHub (triggers trusted publishing to PyPI)
# Usage: make release VERSION=0.11.6
release:
ifndef VERSION
	$(error VERSION is required. Usage: make release VERSION=x.y.z)
endif
	@echo "Current version: $$(grep 'version = ' pyproject.toml | head -1 | cut -d'"' -f2)"
	@echo "New version: $(VERSION)"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	@sed -i '' 's/version = ".*"/version = "$(VERSION)"/' pyproject.toml
	@git add pyproject.toml
	@git commit -m "chore: Bump version to $(VERSION)"
	@git push
	@gh release create v$(VERSION) --title "v$(VERSION)" --generate-notes
	@echo "Release v$(VERSION) created! GitHub Actions will publish to PyPI."

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
