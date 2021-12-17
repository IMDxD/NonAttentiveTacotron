requirements:
	pip install -r requirements-dev.txt

start: requirements

test:
	pytest --cov=src --cov-fail-under 80 --blockage  --cov-report term-missing

coverage-collect:
	coverage run -m pytest

coverage-report:
	coverage html

coverage: coverage-collect coverage-report

mypy:
	mypy src

flake8:
	flake8 .

isort:
	isort .

check: isort flake8 mypy test
