install:
	poetry install --with dev

lint:
	poetry run flake8 juditha --count --select=E9,F63,F7,F82 --show-source --statistics
	poetry run flake8 juditha --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

pre-commit:
	poetry run pre-commit install
	poetry run pre-commit run -a

typecheck:
	poetry run mypy --strict juditha

test:
	poetry run pytest -v --capture=sys --cov=juditha --cov-report lcov -v

build:
	poetry run build

clean:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

api:
	uvicorn juditha.api:app --reload --port 5000
