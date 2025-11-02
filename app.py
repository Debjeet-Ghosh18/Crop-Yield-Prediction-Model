import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Crop Yield AI Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FFD700, #32CD32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border-left: 5px solid #32CD32;
        margin: 1rem 0;
    }
    .demo-card {
        background: rgba(255, 243, 205, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border-left: 5px solid #FFC107;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #FFA500, #FF8C00);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load Model and Data with Better Error Handling
@st.cache_resource
def load_model():
    """Load model files with fallback to demo mode"""
    try:
        # Check if model files exist
        if not os.path.exists('model/crop_yield_model.pkl'):
            st.sidebar.warning("🔍 Model files not found. Running in DEMO mode.")
            return None, None, None
            
        with open('model/crop_yield_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('model/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        st.sidebar.success("✅ Model loaded successfully!")
        return model, preprocessor, model_info
        
    except Exception as e:
        st.sidebar.warning(f"🔍 Using DEMO mode: {e}")
        return None, None, None

@st.cache_data
def load_data():
    """Load data with intelligent fallback"""
    try:
        # Try to load sample data first
        if os.path.exists('model/sample_data.csv'):
            df = pd.read_csv('model/sample_data.csv')
            st.sidebar.success("✅ Data loaded successfully!")
            return df
        elif os.path.exists('data/yield_df.csv'):
            df = pd.read_csv('data/yield_df.csv')
            st.sidebar.success("✅ Original data loaded!")
            return df
        else:
            raise FileNotFoundError("No data files found")
            
    except Exception as e:
        st.sidebar.info("📊 Using demo data for display")
        # Create realistic demo data based on actual dataset structure
        np.random.seed(42)
        n_samples = 1000
        
        demo_data = {
            'Year': np.random.randint(1990, 2023, n_samples),
            'average_rain_fall_mm_per_year': np.random.normal(1485, 300, n_samples),
            'pesticides_tonnes': np.random.normal(121, 30, n_samples),
            'avg_temp': np.random.normal(16.37, 3, n_samples),
            'Area': np.random.choice(['Albania', 'Argentina', 'Armenia', 'Australia', 'Brazil', 'China', 'India', 'USA'], n_samples),
            'Item': np.random.choice(['Maize', 'Potatoes', 'Rice', 'Wheat', 'Soybeans', 'Barley', 'Sorghum'], n_samples),
            'hg/ha_yield': np.random.normal(25000, 15000, n_samples)
        }
        
        # Add some realistic correlations
        df = pd.DataFrame(demo_data)
        df['hg/ha_yield'] = df['hg/ha_yield'] + (df['average_rain_fall_mm_per_year'] - 800) * 8
        df['hg/ha_yield'] = df['hg/ha_yield'] + (df['avg_temp'] - 20) * 500
        df['hg/ha_yield'] = np.abs(df['hg/ha_yield'])  # Ensure positive values
        
        return df

# Create Demo Prediction Function
def demo_predict_yield(year, rainfall, pesticides, temperature, area, item):
    """Demo prediction function that simulates realistic yields"""
    # Base yield based on crop type
    crop_base_yields = {
        'Maize': 30000, 'Potatoes': 25000, 'Rice': 35000, 
        'Wheat': 28000, 'Soybeans': 22000, 'Barley': 20000, 'Sorghum': 18000
    }
    
    base_yield = crop_base_yields.get(item, 25000)
    
    # Adjust based on environmental factors (simplified model)
    rain_factor = (rainfall - 800) / 800 * 0.3  # ±30% based on rainfall
    temp_factor = (temperature - 20) / 20 * 0.2  # ±20% based on temperature
    pest_factor = (pesticides - 50) / 50 * 0.1   # ±10% based on pesticides
    year_factor = (year - 2000) / 23 * 0.2       # ±20% based on year (improvement over time)
    
    # Calculate final yield
    predicted_yield = base_yield * (1 + rain_factor + temp_factor + pest_factor + year_factor)
    
    # Add some random variation
    predicted_yield *= np.random.normal(1, 0.1)
    
    return max(5000, predicted_yield)  # Ensure minimum yield

# Initialize
model, preprocessor, model_info = load_model()
df = load_data()

# Main App
st.markdown('<h1 class="main-header">🌾 AI Crop Yield Predictor</h1>', unsafe_allow_html=True)

# Sidebar with Export Instructions
st.sidebar.title("🌾 Navigation")
page = st.sidebar.radio("Choose Page", ["🏠 Home", "🎯 Prediction"])

# Show export instructions in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Setup Instructions")
if model is None:
    st.sidebar.markdown("""
    **To use real AI model:**
    1. Open your notebook
    2. Run the export cell
    3. Restart this app
    
    **Current mode:** 🎭 **DEMO** (Using simulated predictions)
    """)
else:
    st.sidebar.markdown("""
    **Current mode:** 🤖 **AI MODEL** (Using trained ML model)
    """)

if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if model is None:
            st.markdown("""
            <div class="demo-card">
                <h2>🎭 Demo Mode Active</h2>
                <p>Currently running with <b>simulated predictions</b>. For real AI predictions, export your model from the notebook.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="prediction-card">
            <h2>🚀 Welcome to Crop Yield Predictor</h2>
            <p>Predict crop yields using environmental data and machine learning.</p>
            
            <h3>🎯 What You Can Do:</h3>
            <ul>
                <li>📈 <b>Predict</b> crop yields for any region</li>
                <li>🌦️ <b>Analyze</b> impact of weather conditions</li>
                <li>📊 <b>Compare</b> crop performance globally</li>
                <li>🤖 <b>Understand</b> AI model predictions</li>
            </ul>
            
            <h3>💡 How It Works:</h3>
            <ol>
                <li>Select your crop and location</li>
                <li>Input environmental conditions</li>
                <li>Get instant yield prediction</li>
                <li>Analyze results and insights</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("🌍 Countries", df['Area'].nunique())
        st.metric("🌾 Crops", df['Item'].nunique())
        st.metric("📊 Records", f"{len(df):,}")
        
        if model is None:
            st.metric("🔧 Mode", "DEMO")
        else:
            st.metric("🔧 Mode", "AI MODEL")
            if model_info and 'performance' in model_info:
                st.metric("🎯 Accuracy", f"{model_info['performance']['r2_score']:.1%}")

elif page == "🎯 Prediction":
    st.markdown('<h1 class="main-header">🎯 Crop Yield Prediction</h1>', unsafe_allow_html=True)
    
    # Show mode indicator
    if model is None:
        st.markdown("""
        <div class="demo-card">
            <h3>🎭 Demo Mode</h3>
            <p>Using <b>simulated predictions</b> based on crop science formulas. For AI predictions, export your trained model.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="prediction-card">
            <h3>🤖 AI Model Active</h3>
            <p>Using <b>trained machine learning model</b> for accurate predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Input Form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="prediction-card">
            <h3>📍 Location & Crop</h3>
        </div>
        """, unsafe_allow_html=True)
        
        area = st.selectbox("🌍 Country", sorted(df['Area'].unique()))
        item = st.selectbox("🌾 Crop Type", sorted(df['Item'].unique()))
        year = st.slider("📅 Year", 1990, 2030, 2023)
    
    with col2:
        st.markdown("""
        <div class="prediction-card">
            <h3>🌦️ Environmental Factors</h3>
        </div>
        """, unsafe_allow_html=True)
        
        rainfall = st.slider("💧 Rainfall (mm/year)", 0, 3000, 1485, 
                           help="Annual rainfall in millimeters")
        pesticides = st.slider("🧪 Pesticides (tonnes)", 0, 500, 121,
                             help="Pesticide usage in tonnes per hectare")
        temperature = st.slider("🌡️ Temperature (°C)", -10, 40, 16,
                              help="Average annual temperature in Celsius")
    
    # Unified Prediction Function
    def predict_yield(year, rainfall, pesticides, temperature, area, item):
        if model is not None:
            # Use real AI model
            try:
                input_data = pd.DataFrame({
                    'Year': [year],
                    'average_rain_fall_mm_per_year': [rainfall],
                    'pesticides_tonnes': [pesticides],
                    'avg_temp': [temperature],
                    'Area': [area],
                    'Item': [item]
                })
                
                transformed_data = preprocessor.transform(input_data)
                prediction = model.predict(transformed_data)[0]
                return prediction, "AI Model"
                
            except Exception as e:
                st.error(f"AI Model Error: {e}")
                # Fallback to demo
                prediction = demo_predict_yield(year, rainfall, pesticides, temperature, area, item)
                return prediction, "Demo (Fallback)"
        else:
            # Use demo prediction
            prediction = demo_predict_yield(year, rainfall, pesticides, temperature, area, item)
            return prediction, "Demo Model"
    
    # Prediction Button
    if st.button("🚀 PREDICT YIELD", use_container_width=True):
        with st.spinner("🔮 Analyzing data and predicting yield..."):
            prediction, model_type = predict_yield(year, rainfall, pesticides, temperature, area, item)
            
            st.success(f"✅ Prediction Complete! ({model_type})")
            
            # Display Results
            st.markdown("""
            <div class="prediction-card">
                <h2 style="text-align: center; color: #2E8B57;">📊 Prediction Results</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Results in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Predicted Yield", f"{prediction:,.0f} hg/ha")
            with col2:
                st.metric("In Tons/Hectare", f"{prediction/10000:,.1f} t/ha")
            with col3:
                avg_yield = df[df['Item'] == item]['hg/ha_yield'].mean()
                difference = ((prediction - avg_yield) / avg_yield * 100)
                st.metric("Vs Average", f"{difference:+.1f}%")
            with col4:
                st.metric("Model Type", model_type)
            
            # Input Summary
            st.subheader("📋 Input Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                input_df = pd.DataFrame({
                    'Parameter': ['Country', 'Crop', 'Year'],
                    'Value': [area, item, str(year)]
                })
                st.table(input_df)
            
            with col2:
                input_df_env = pd.DataFrame({
                    'Parameter': ['Rainfall', 'Pesticides', 'Temperature'],
                    'Value': [f"{rainfall} mm/year", f"{pesticides} tonnes", f"{temperature}°C"]
                })
                st.table(input_df_env)
            
            # Yield Gauge Chart
            st.subheader("📈 Yield Prediction Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                title={'text': f"Yield Prediction for {item} in {area}", 'font': {'size': 20}},
                delta={'reference': avg_yield, 'relative': True, 'valueformat': '.1%'},
                gauge={
                    'axis': {'range': [None, max(prediction*1.3, avg_yield*1.5)], 'tickwidth': 1},
                    'bar': {'color': "darkgreen"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, avg_yield*0.6], 'color': "lightcoral"},
                        {'range': [avg_yield*0.6, avg_yield*0.9], 'color': "lightyellow"},
                        {'range': [avg_yield*0.9, avg_yield*1.2], 'color': "lightgreen"},
                        {'range': [avg_yield*1.2, avg_yield*1.5], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': avg_yield
                    }
                }
            ))
            
            fig.update_layout(height=400, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional Insights
            st.subheader("💡 Insights & Recommendations")
            
            if prediction > avg_yield * 1.1:
                st.success(f"**Excellent Conditions!** Yield is {((prediction-avg_yield)/avg_yield*100):.1f}% above average for {item}.")
            elif prediction > avg_yield * 0.9:
                st.info(f"**Good Conditions** Yield is close to average for {item}.")
            else:
                st.warning(f"**Challenging Conditions** Yield is {((avg_yield-prediction)/avg_yield*100):.1f}% below average. Consider adjusting inputs.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌾 <b>Crop Yield Prediction System</b> | Built with Streamlit & Machine Learning</p>
    <p>💡 Making agriculture smarter with data-driven insights</p>
</div>
""", unsafe_allow_html=True)