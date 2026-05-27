.PHONY: test lint type-check coverage security install clean docker-build docker-run

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -q --tb=short

test-verbose:
	python -m pytest tests/ -v --tb=short

coverage:
	python -m pytest tests/ --cov=sunset_ecosystem --cov-report=term-missing --cov-fail-under=75

lint:
	ruff check sunset_ecosystem/ swarm/ nexus/ logos/ a2a/ nerve/

format:
	ruff format sunset_ecosystem/ swarm/ nexus/ logos/ a2a/ nerve/

type-check:
	mypy sunset_ecosystem/ --ignore-missing-imports

security:
	bandit -r sunset_ecosystem/ swarm/ nexus/ logos/
	pip-audit --desc .

dev: install lint type-check test

docker-build:
	docker build -t sunset-ecosystem:latest .

docker-run:
	docker run -p 8080:8080 -e NEXUS_IP=localhost sunset-ecosystem:latest

docker-compose-up:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
