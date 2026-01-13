# Makefile for Neural Machine Translation System

.PHONY: help install install-dev test test-coverage lint format clean run-demo run-web run-cli generate-data

# Default target
help:
	@echo "Neural Machine Translation System - Available Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test         Run all tests"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  lint         Run code linting (flake8)"
	@echo "  format       Format code with black"
	@echo "  clean        Clean temporary files and caches"
	@echo ""
	@echo "Data & Models:"
	@echo "  generate-data Generate synthetic datasets"
	@echo ""
	@echo "Running:"
	@echo "  run-demo     Run the demo script"
	@echo "  run-web      Start the Streamlit web application"
	@echo "  run-cli      Show CLI help"
	@echo ""
	@echo "Examples:"
	@echo "  make install-dev && make test"
	@echo "  make generate-data && make run-web"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

# Testing
test:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Code Quality
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ --line-length=100
	isort src/ tests/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf logs/*.log

# Data Generation
generate-data:
	python src/dataset_generator.py

# Running Applications
run-demo:
	python demo.py

run-web:
	streamlit run web_app/app.py

run-cli:
	python cli.py --help

# Quick start
quick-start: install-dev generate-data
	@echo "Quick start completed! You can now:"
	@echo "  - Run tests: make test"
	@echo "  - Start web app: make run-web"
	@echo "  - Run demo: make run-demo"

# Development workflow
dev-setup: install-dev generate-data test
	@echo "Development environment setup complete!"

# Production build
build: clean install test
	@echo "Production build complete!"

# Docker commands (if using Docker)
docker-build:
	docker build -t neural-machine-translation .

docker-run:
	docker run -p 8501:8501 neural-machine-translation

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "See README.md for comprehensive documentation"

# Performance testing
perf-test:
	python -m pytest tests/test_performance.py -v

# Security check
security-check:
	bandit -r src/ -f json -o security-report.json || true
	@echo "Security check completed. See security-report.json for details."
