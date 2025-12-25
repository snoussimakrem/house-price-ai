#  End-to-End House Price Prediction System

A complete, production-ready AI system for predicting house prices using machine learning. This project demonstrates the full lifecycle of an AI application from data to deployment.


- **Complete ML Pipeline**: Data collection → preprocessing → training → evaluation
- **RESTful API**: FastAPI backend with validation and documentation
- **Interactive Dashboard**: Streamlit web interface for predictions
- **Docker Support**: Containerized deployment
- **MLflow Integration**: Experiment tracking and model registry
- **Automated Testing**: Comprehensive test suite
- **CI/CD Ready**: GitHub Actions workflow included

 System Architecture
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

