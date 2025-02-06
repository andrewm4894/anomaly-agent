.PHONY: requirements-compile
.PHONY: requirements-install
.PHONY: requirements-dev
.PHONY: build publish


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