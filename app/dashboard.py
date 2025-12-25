import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #F0F9FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .feature-importance {
        background-color: #FEFCE8;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# API configuration
API_URL = "http://localhost:8000"  # Change if deployed

class Dashboard:
    def __init__(self):
        self.api_url = API_URL
    
    def check_api_health(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.ok else None
        except:
            return False, None
    
    def make_prediction(self, features):
        """Make prediction through API"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=features,
                timeout=10
            )
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_sample_data(self):
        """Get sample house data"""
        return {
            "bedrooms": 3,
            "bathrooms": 2.5,
            "sqft_living": 1800,
            "sqft_lot": 5000,
            "floors": 2,
            "waterfront": 0,
            "view": 2,
            "condition": 3,
            "grade": 7,
            "sqft_above": 1800,
            "sqft_basement": 0,
            "yr_built": 1995,
            "yr_renovated": 2010,
            "zipcode": 98115,
            "lat": 47.6829,
            "long": -122.293,
            "sqft_living15": 1900,
            "sqft_lot15": 5200
        }

# Initialize dashboard
dashboard = Dashboard()

# Sidebar
with st.sidebar:
    st.title("🏠 Navigation")
    
    page = st.radio(
        "Go to",
        ["Home", "Single Prediction", "Batch Prediction", "Market Analysis", "History"]
    )
    
    st.markdown("---")
    
    # API Status
    st.subheader("API Status")
    api_healthy, api_info = dashboard.check_api_health()
    
    if api_healthy:
        st.success("✅ API Connected")
        if api_info:
            st.caption(f"Model loaded: {api_info.get('model_loaded', 'N/A')}")
    else:
        st.error("❌ API Not Connected")
        st.caption("Make sure the API server is running on port 8000")
    
    st.markdown("---")
    
    # Quick Actions
    st.subheader("Quick Actions")
    
    if st.button("🔄 Reset Form"):
        st.session_state.clear()
        st.rerun()
    
    if st.button("📊 Load Sample Data"):
        sample_data = dashboard.get_sample_data()
        for key, value in sample_data.items():
            if key not in st.session_state:
                st.session_state[key] = value
        st.rerun()

# Main content based on page selection
if page == "Home":
    st.markdown('<h1 class="main-header">🏡 House Price Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to the House Price Prediction System
        
        This AI-powered dashboard helps you:
        
        🔹 **Predict** house prices based on features
        🔹 **Analyze** market trends
        🔹 **Compare** multiple properties
        🔹 **Track** prediction history
        
        ### How to use:
        1. Go to **Single Prediction** to predict price for one house
        2. Use **Batch Prediction** for multiple houses
        3. Check **Market Analysis** for insights
        4. View **History** for past predictions
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/489/489870.png", width=200)
    
    # Quick stats
    st.markdown("---")
    st.subheader("📈 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(st.session_state.predictions_history))
    
    with col2:
        avg_price = np.mean([p.get('prediction', 0) for p in st.session_state.predictions_history]) if st.session_state.predictions_history else 0
        st.metric("Average Price", f"${avg_price:,.0f}")
    
    with col3:
        if api_healthy:
            st.metric("API Status", "Online", delta="Connected")
        else:
            st.metric("API Status", "Offline", delta="Disconnected", delta_color="inverse")
    
    with col4:
        st.metric("System Version", "1.0.0")

elif page == "Single Prediction":
    st.title("🏠 Single House Prediction")
    
    # Create form for house features
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Basic Information")
            bedrooms = st.slider("Bedrooms", 1, 10, 
                                value=st.session_state.get('bedrooms', 3))
            bathrooms = st.slider("Bathrooms", 0.5, 8.0, 0.5,
                                 value=st.session_state.get('bathrooms', 2.0))
            floors = st.slider("Floors", 1.0, 3.5, 0.5,
                              value=st.session_state.get('floors', 1.0))
        
        with col2:
            st.subheader("Size & Area")
            sqft_living = st.number_input("Living Area (sqft)", 500, 10000,
                                         value=st.session_state.get('sqft_living', 1500))
            sqft_lot = st.number_input("Lot Area (sqft)", 500, 100000,
                                      value=st.session_state.get('sqft_lot', 4000))
            sqft_above = st.number_input("Above Ground (sqft)", 500, 10000,
                                        value=st.session_state.get('sqft_above', 1500))
            sqft_basement = st.number_input("Basement (sqft)", 0, 5000,
                                           value=st.session_state.get('sqft_basement', 0))
        
        with col3:
            st.subheader("Quality & Location")
            grade = st.slider("Grade", 1, 13, value=st.session_state.get('grade', 7))
            condition = st.slider("Condition", 1, 5, value=st.session_state.get('condition', 3))
            view = st.slider("View", 0, 4, value=st.session_state.get('view', 0))
            waterfront = st.selectbox("Waterfront", [0, 1], 
                                     index=st.session_state.get('waterfront', 0),
                                     format_func=lambda x: "Yes" if x == 1 else "No")
        
        # Additional features
        st.subheader("Additional Details")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            yr_built = st.number_input("Year Built", 1900, 2024,
                                      value=st.session_state.get('yr_built', 1995))
            yr_renovated = st.number_input("Year Renovated", 0, 2024,
                                          value=st.session_state.get('yr_renovated', 0))
        
        with col5:
            lat = st.number_input("Latitude", 47.0, 48.0, 47.5,
                                 format="%.4f",
                                 value=st.session_state.get('lat', 47.5112))
            long = st.number_input("Longitude", -123.0, -121.0, -122.0,
                                  format="%.4f",
                                  value=st.session_state.get('long', -122.257))
        
        with col6:
            zipcode = st.number_input("Zip Code", 98001, 98199,
                                     value=st.session_state.get('zipcode', 98178))
        
        # Submit button
        submit_col, reset_col = st.columns([1, 1])
        
        with submit_col:
            submit_button = st.form_submit_button("🔮 Predict Price", type="primary")
        
        with reset_col:
            if st.form_submit_button("🔄 Reset"):
                for key in st.session_state.keys():
                    if key != 'predictions_history':
                        del st.session_state[key]
                st.rerun()
    
    # Handle form submission
    if submit_button:
        # Prepare features dictionary
        features = {
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
            "sqft_living15": sqft_living + 100,  # Estimate
            "sqft_lot15": sqft_lot + 1000        # Estimate
        }
        
        # Store in session state
        for key, value in features.items():
            st.session_state[key] = value
        
        # Show loading
        with st.spinner("🤖 Predicting price..."):
            success, result = dashboard.make_prediction(features)
        
        if success:
            # Display prediction
            st.markdown("---")
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("💰 Prediction Result")
                prediction = result.get('prediction', 0)
                formatted_price = result.get('prediction_formatted', '$0')
                
                st.markdown(f"### {formatted_price}")
                st.caption(f"Predicted on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Add to history
                history_entry = {
                    **features,
                    'prediction': prediction,
                    'timestamp': time.time(),
                    'formatted_price': formatted_price
                }
                st.session_state.predictions_history.append(history_entry)
            
            with col2:
                # Quick stats
                st.metric("Price per sqft", f"${prediction/sqft_living:,.0f}")
                st.metric("Rooms", bedrooms + bathrooms)
                st.metric("Age", 2024 - yr_built)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature importance
            top_features = result.get('top_features', {})
            if top_features:
                st.markdown("---")
                st.subheader("📊 Feature Importance")
                
                # Create dataframe for visualization
                importance_df = pd.DataFrame(
                    list(top_features.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=True)
                
                # Plot
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top Influential Features",
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # House summary
            st.markdown("---")
            st.subheader("🏡 House Summary")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write("**Basic Info:**")
                st.write(f"- Bedrooms: {bedrooms}")
                st.write(f"- Bathrooms: {bathrooms}")
                st.write(f"- Total Area: {sqft_living:,} sqft")
                st.write(f"- Lot Area: {sqft_lot:,} sqft")
                st.write(f"- Floors: {floors}")
            
            with summary_col2:
                st.write("**Quality & Location:**")
                st.write(f"- Grade: {grade}/13")
                st.write(f"- Condition: {condition}/5")
                st.write(f"- View: {view}/4")
                st.write(f"- Waterfront: {'Yes' if waterfront == 1 else 'No'}")
                st.write(f"- Year Built: {yr_built}")
                if yr_renovated > 0:
                    st.write(f"- Renovated: {yr_renovated}")
            
        else:
            st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

elif page == "Batch Prediction":
    st.title("📊 Batch Prediction")
    
    st.markdown("""
    Upload a CSV file with multiple houses to get predictions for all of them at once.
    
    **Required columns:** bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, 
    view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, 
    zipcode, lat, long
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully: {len(df)} rows")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Check required columns
            required_columns = [
                'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                'floors', 'waterfront', 'view', 'condition', 'grade',
                'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                'zipcode', 'lat', 'long'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
            else:
                # Convert to list of dictionaries
                houses = df[required_columns].to_dict('records')
                
                if st.button("🚀 Predict All Houses", type="primary"):
                    with st.spinner(f"Predicting {len(houses)} houses..."):
                        # Make batch prediction
                        batch_data = {"houses": houses}
                        
                        try:
                            response = requests.post(
                                f"{API_URL}/batch-predict",
                                json=batch_data,
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                results = response.json()
                                
                                # Display results
                                st.success(f"✅ Predictions complete!")
                                
                                # Add predictions to dataframe
                                predictions = [r.get('prediction', 0) for r in results['results']]
                                df['predicted_price'] = predictions
                                df['predicted_price_formatted'] = [f"${p:,.0f}" for p in predictions]
                                
                                # Show results table
                                st.subheader("Prediction Results")
                                display_cols = ['bedrooms', 'bathrooms', 'sqft_living', 
                                              'predicted_price_formatted']
                                st.dataframe(df[display_cols].head(20))
                                
                                # Download button
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv,
                                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                                # Statistics
                                st.subheader("📈 Batch Statistics")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Average Price", f"${np.mean(predictions):,.0f}")
                                
                                with col2:
                                    st.metric("Minimum Price", f"${np.min(predictions):,.0f}")
                                
                                with col3:
                                    st.metric("Maximum Price", f"${np.max(predictions):,.0f}")
                                
                                with col4:
                                    st.metric("Total Value", f"${np.sum(predictions):,.0f}")
                                
                                # Distribution chart
                                fig = px.histogram(
                                    df, 
                                    x='predicted_price',
                                    title="Price Distribution",
                                    nbins=20,
                                    labels={'predicted_price': 'Predicted Price'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            else:
                                st.error(f"❌ API Error: {response.status_code}")
                                
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    else:
        # Show sample CSV structure
        st.info("💡 Need a sample CSV? Download the template below:")
        
        # Create sample data
        sample_data = {
            'bedrooms': [3, 4, 2],
            'bathrooms': [2.0, 2.5, 1.0],
            'sqft_living': [1500, 2200, 950],
            'sqft_lot': [4000, 7500, 3000],
            'floors': [1.0, 2.0, 1.0],
            'waterfront': [0, 0, 0],
            'view': [0, 2, 0],
            'condition': [3, 4, 3],
            'grade': [7, 8, 6],
            'sqft_above': [1500, 2200, 950],
            'sqft_basement': [0, 0, 0],
            'yr_built': [1995, 2005, 1985],
            'yr_renovated': [2010, 0, 0],
            'zipcode': [98115, 98105, 98178],
            'lat': [47.6829, 47.6616, 47.5112],
            'long': [-122.293, -122.313, -122.257]
        }
        
        sample_df = pd.DataFrame(sample_data)
        csv = sample_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Download Sample CSV",
            data=csv,
            file_name="sample_houses.csv",
            mime="text/csv"
        )

elif page == "Market Analysis":
    st.title("📈 Market Analysis")
    
    if not st.session_state.predictions_history:
        st.info("No prediction history yet. Make some predictions first!")
    else:
        # Convert history to dataframe
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Summary statistics
        st.subheader("📊 Prediction History Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(history_df))
        
        with col2:
            avg_price = history_df['prediction'].mean()
            st.metric("Average Price", f"${avg_price:,.0f}")
        
        with col3:
            total_value = history_df['prediction'].sum()
            st.metric("Total Value", f"${total_value:,.0f}")
        
        with col4:
            avg_sqft_price = (history_df['prediction'] / history_df['sqft_living']).mean()
            st.metric("Avg Price/sqft", f"${avg_sqft_price:,.0f}")
        
        # Price distribution
        st.subheader("💰 Price Distribution")
        fig1 = px.histogram(
            history_df, 
            x='prediction',
            nbins=20,
            title="Distribution of Predicted Prices"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Price vs Size
        st.subheader("📏 Price vs Living Area")
        fig2 = px.scatter(
            history_df,
            x='sqft_living',
            y='prediction',
            color='bedrooms',
            size='bathrooms',
            hover_data=['grade', 'condition'],
            title="Price vs Living Area"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔗 Feature Correlations")
        
        # Select numerical columns for correlation
        numerical_cols = ['prediction', 'bedrooms', 'bathrooms', 'sqft_living', 
                         'sqft_lot', 'grade', 'condition', 'view']
        
        corr_df = history_df[numerical_cols].corr()
        
        fig3 = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig3, use_container_width=True)

elif page == "History":
    st.title("📋 Prediction History")
    
    if not st.session_state.predictions_history:
        st.info("No predictions made yet. Go to Single Prediction to get started!")
    else:
        # Convert to dataframe
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Format timestamp
        if 'timestamp' in history_df.columns:
            history_df['date'] = pd.to_datetime(history_df['timestamp'], unit='s')
            history_df['date_str'] = history_df['date'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Display table
        st.subheader("All Predictions")
        
        # Select columns to display
        display_cols = ['date_str', 'bedrooms', 'bathrooms', 'sqft_living', 
                       'prediction', 'formatted_price']
        
        # Filter to available columns
        available_cols = [col for col in display_cols if col in history_df.columns]
        
        st.dataframe(
            history_df[available_cols].sort_values('date', ascending=False),
            use_container_width=True
        )
        
        # Download option
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Clear history option
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.predictions_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🏠 House Price Prediction System v1.0 | Built with ❤️ using Streamlit & FastAPI
    </div>
    """,
    unsafe_allow_html=True
)