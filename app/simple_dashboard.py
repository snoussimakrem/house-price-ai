import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objects as go

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background-color: #F0F9FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏠 House Price Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Predict", "About", "API Status"])
    
    st.markdown("---")
    st.title("Quick Info")
    st.info("This model predicts house prices based on 18 features.")
    
    if st.button("🔄 Reset Form"):
        st.rerun()

# Main content
if page == "Predict":
    st.subheader("Enter House Details")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Information**")
        bedrooms = st.slider("Bedrooms", 1, 10, 3)
        bathrooms = st.slider("Bathrooms", 0.5, 8.0, 2.0, 0.5)
        floors = st.slider("Floors", 1.0, 3.5, 1.0, 0.5)
        condition = st.slider("Condition (1-5)", 1, 5, 3)
        grade = st.slider("Grade (1-13)", 1, 13, 7)
    
    with col2:
        st.write("**Size & Area**")
        sqft_living = st.number_input("Living Area (sqft)", 500, 10000, 1500)
        sqft_lot = st.number_input("Lot Area (sqft)", 500, 50000, 4000)
        sqft_above = st.number_input("Above Ground (sqft)", 500, 8000, 1500)
        sqft_basement = st.number_input("Basement (sqft)", 0, 4000, 0)
    
    # More features
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Location**")
        zipcode = st.number_input("Zip Code", 98001, 98199, 98178)
        lat = st.number_input("Latitude", 47.0, 48.0, 47.5112, 0.0001, format="%.4f")
        long = st.number_input("Longitude", -123.0, -121.0, -122.257, 0.0001, format="%.4f")
    
    with col4:
        st.write("**Quality & Other**")
        waterfront = st.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        view = st.slider("View Quality (0-4)", 0, 4, 0)
        yr_built = st.number_input("Year Built", 1900, 2024, 1995)
        yr_renovated = st.number_input("Year Renovated (0 if never)", 0, 2024, 0)
    
    # Estimated neighbor values
    sqft_living15 = sqft_living + 100
    sqft_lot15 = sqft_lot + 1000
    
    # Predict button
    if st.button("🔮 Predict Price", type="primary"):
        # Prepare data
        house_data = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "waterfront": waterfront,
            "view": view,
            "condition": condition,
            "grade": grade,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement,
            "yr_built": yr_built,
            "yr_renovated": yr_renovated,
            "zipcode": zipcode,
            "lat": lat,
            "long": long,
            "sqft_living15": sqft_living15,
            "sqft_lot15": sqft_lot15
        }
        
        # Show loading
        with st.spinner("Predicting price..."):
            try:
                # Make API call
                response = requests.post(
                    "http://localhost:8000/predict",
                    json=house_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display prediction
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    
                    col_pred1, col_pred2 = st.columns([2, 1])
                    
                    with col_pred1:
                        st.markdown(f"## {result['prediction_formatted']}")
                        st.caption("Predicted House Price")
                    
                    with col_pred2:
                        st.metric("Price per sqft", f"${result['prediction']/sqft_living:,.0f}")
                        st.metric("Total Rooms", bedrooms + bathrooms)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show feature summary
                    with st.expander("📋 View House Summary"):
                        summary_col1, summary_col2 = st.columns(2)
                        
                        with summary_col1:
                            st.write("**Basic Info:**")
                            st.write(f"- Bedrooms: {bedrooms}")
                            st.write(f"- Bathrooms: {bathrooms}")
                            st.write(f"- Floors: {floors}")
                            st.write(f"- Living Area: {sqft_living:,} sqft")
                            st.write(f"- Lot Area: {sqft_lot:,} sqft")
                        
                        with summary_col2:
                            st.write("**Quality & Location:**")
                            st.write(f"- Condition: {condition}/5")
                            st.write(f"- Grade: {grade}/13")
                            st.write(f"- View: {view}/4")
                            st.write(f"- Year Built: {yr_built}")
                            st.write(f"- Waterfront: {'Yes' if waterfront == 1 else 'No'}")
                
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.json(response.json())
                    
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot connect to API server. Make sure it's running!")
                st.info("Start the API with: `uvicorn src.api.simple_api:app --reload`")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

elif page == "About":
    st.subheader("About This Project")
    
    st.markdown("""
    ## 🏠 House Price Prediction System
    
    This is an end-to-end machine learning project that predicts house prices based on various features.
    
    ### 📊 Model Information
    - **Model Type**: Random Forest Regressor
    - **Features Used**: 18 different house characteristics
    - **Training Data**: 500 samples (3 real + 497 synthetic)
    - **Current R² Score**: -0.1152 (needs improvement with real data)
    
    ### 🔧 How to Improve Accuracy
    1. **Add more real data** - The model currently has only 3 real samples
    2. **Collect better features** - More detailed house information
    3. **Feature engineering** - Create better derived features
    4. **Hyperparameter tuning** - Optimize model parameters
    
    ### 🚀 Getting Started
    1. The API server should be running on port 8000
    2. This dashboard runs on port 8501
    3. You can test predictions using the form
    
    ### 📁 Project Structure
    ```
    house-price-ai/
    ├── data/           # Data files
    ├── model/          # Trained model
    ├── src/            # Source code
    └── app/            # Dashboard
    ```
    """)

elif page == "API Status":
    st.subheader("API Server Status")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            status_data = response.json()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if status_data.get("status") == "healthy":
                    st.success("✅ API Online")
                else:
                    st.warning("⚠️ API Issues")
            
            with col2:
                if status_data.get("model_loaded"):
                    st.success("✅ Model Loaded")
                else:
                    st.error("❌ Model Not Loaded")
            
            with col3:
                st.info("📍 Port 8000")
            
            st.json(status_data)
            
            # Test prediction button
            if st.button("🧪 Test Prediction with Sample Data"):
                sample_data = {
                    "bedrooms": 3,
                    "bathrooms": 2.0,
                    "sqft_living": 1500,
                    "sqft_lot": 4000,
                    "floors": 1.0,
                    "waterfront": 0,
                    "view": 0,
                    "condition": 3,
                    "grade": 7,
                    "sqft_above": 1500,
                    "sqft_basement": 0,
                    "yr_built": 1995,
                    "yr_renovated": 0,
                    "zipcode": 98178,
                    "lat": 47.5112,
                    "long": -122.257,
                    "sqft_living15": 1600,
                    "sqft_lot15": 5000
                }
                
                with st.spinner("Testing..."):
                    try:
                        test_response = requests.post(
                            "http://localhost:8000/predict",
                            json=sample_data,
                            timeout=10
                        )
                        
                        if test_response.status_code == 200:
                            result = test_response.json()
                            st.success(f"✅ Test successful! Predicted: {result['prediction_formatted']}")
                        else:
                            st.error(f"Test failed: {test_response.status_code}")
                    except Exception as e:
                        st.error(f"Test failed: {str(e)}")
        
        else:
            st.error(f"❌ API returned status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API server!")
        st.info("""
        To start the API server:
        
        ```bash
        # In the project directory
        uvicorn src.api.simple_api:app --reload --host 0.0.0.0 --port 8000
        ```
        
        Or use the simple API script:
        
        ```bash
        python src/api/simple_api.py
        ```
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🏠 House Price Prediction System | Built with FastAPI & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)