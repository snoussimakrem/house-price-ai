.PHONY: help install test train api dashboard docker-build docker-up clean

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  train       Train the model"
	@echo "  api         Start the API server"
	@echo "  dashboard   Start the dashboard"
	@echo "  test        Run tests"
	@echo "  docker-build Build Docker images"
	@echo "  docker-up   Start services with Docker"
	@echo "  clean       Clean generated files"

install:
	pip install -r requirements.txt
	pip install -r requirements_dev.txt

train:
	python src/models/train_model.py

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run app/dashboard.py --server.port 8501

test:
	pytest tests/ -v

docker-build:
	docker build -f docker/Dockerfile.api -t house-price-api .
	docker build -f docker/Dockerfile.dashboard -t house-price-dashboard .

docker-up:
	docker-compose -f docker/docker-compose.yml up

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf model/*.pkl
	rm -rf data/processed/*
	rm -rf mlruns/