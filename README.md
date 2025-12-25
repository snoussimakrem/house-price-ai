# 🏠 End-to-End House Price Prediction System

A complete, production-ready AI system for predicting house prices using machine learning. This project demonstrates the full lifecycle of an AI application from data to deployment.

## ✨ Features

- **Complete ML Pipeline**: Data collection → preprocessing → training → evaluation
- **RESTful API**: FastAPI backend with validation and documentation
- **Interactive Dashboard**: Streamlit web interface for predictions
- **Docker Support**: Containerized deployment
- **MLflow Integration**: Experiment tracking and model registry
- **Automated Testing**: Comprehensive test suite
- **CI/CD Ready**: GitHub Actions workflow included

## 🏗️ System Architecture
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Data Source │───▶│ Preprocessing │───▶│ Model Training│
└─────────────────┘ └─────────────────┘ └─────────────────┘
│ │ │
▼ ▼ ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ FastAPI Server │◀───│ Saved Model │◀───│ Model Registry │
└─────────────────┘ └─────────────────┘ └─────────────────┘
│
▼
┌─────────────────┐
│ Streamlit UI │
└─────────────────┘

text

## 🛠️ Tech Stack

- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Backend**: FastAPI, Pydantic, Uvicorn
- **Frontend**: Streamlit, Plotly
- **DevOps**: Docker, Docker Compose, GitHub Actions
- **MLOps**: MLflow, Joblib
- **Testing**: Pytest, TestClient

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (optional)
- Git

### Option 1: Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/house-price-ai.git