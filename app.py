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
    page_title="Crop Yield Predictor Pro",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with vibrant styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        color: white;
        margin: 1rem 0;
        border: 3px solid #FFD93D;
    }
    .demo-card {
        background: linear-gradient(135deg, #FF9A8B 0%, #FF6A88 55%, #FF99AC 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        color: white;
        margin: 1rem 0;
        border: 3px solid #FFD93D;
    }
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        color: white;
        margin: 1rem 0;
        border: 3px solid #FFD93D;
    }
    .input-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 3px solid #FFD93D;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: 2px solid #FF6B6B;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 30px;
        font-weight: bold;
        font-size: 1.2rem;
        width: 100%;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #FF8E8E, #6BE8E1, #5AC7E0);
    }
    .mode-indicator {
        padding: 15px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
        font-size: 1.1rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .ai-mode {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        border: 3px solid #FFD93D;
    }
    .demo-mode {
        background: linear-gradient(135deg, #ff9a9e, #fad0c4);
        color: white;
        border: 3px solid #FFD93D;
    }
    .tab-content {
        padding: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 15px 15px 0 0;
        gap: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    /* Custom metric styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border: 2px solid #FF6B6B;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    /* Custom slider styling */
    .stSlider [data-testid="stThumbValue"] {
        color: #FF6B6B;
        font-weight: bold;
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
            'Item': np.random.choice(['Maize', 'Potatoes', 'Rice', 'Wheat', 'Soybeans', 'Barley', 'Sorghum', 'Cassava'], n_samples),
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
        'Wheat': 28000, 'Soybeans': 22000, 'Barley': 20000, 
        'Sorghum': 18000, 'Cassava': 26000
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
st.markdown('<h1 class="main-header">🌾 Crop Yield Predictor Pro</h1>', unsafe_allow_html=True)

# Sidebar with Export Instructions
st.sidebar.markdown("""
<div class="info-card">
    <h2 style="text-align: center; margin: 0;">🌍 Navigation</h2>
</div>
""", unsafe_allow_html=True)
page = st.sidebar.radio("Choose Page", ["🏠 Home", "🎯 Prediction"], label_visibility="collapsed")

# Show export instructions in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Setup Instructions")

# Mode indicator
if model is None:
    mode_class = "demo-mode"
    mode_text = "🎭 DEMO MODE (Using simulated predictions)"
    st.sidebar.markdown(f"""
    <div class="mode-indicator {mode_class}">{mode_text}</div>
    
    **To use advanced model:**
    1. Open your notebook
    2. Run the export cell
    3. Restart this app
    """, unsafe_allow_html=True)
else:
    mode_class = "ai-mode"
    mode_text = "🚀 ADVANCED MODEL (Using trained prediction model)"
    st.sidebar.markdown(f"""
    <div class="mode-indicator {mode_class}">{mode_text}</div>
    
    **Model Status:** ✅ Ready
    """, unsafe_allow_html=True)

# Add quick stats to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🌍</h3>
        <h4>{df['Area'].nunique()}</h4>
        <p>Countries</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🌾</h3>
        <h4>{df['Item'].nunique()}</h4>
        <p>Crops</p>
    </div>
    """, unsafe_allow_html=True)

if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Mode indicator at the top
        if model is None:
            st.markdown(f"""
            <div class="demo-card">
                <h2>🎭 Demo Mode Active</h2>
                <p>Currently running with <b>simulated predictions</b>. For advanced predictions, export your model from the notebook.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🚀 Advanced Model Active</h2>
                <p>Using <b>trained prediction model</b> for accurate yield forecasts.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="prediction-card">
            <h2>🚀 Welcome to Crop Yield Predictor Pro</h2>
            <p>Predict crop yields using environmental data and advanced analytics.</p>
            
            <h3>🎯 What You Can Do:</h3>
            <ul>
                <li>📈 <b>Predict</b> crop yields for any region</li>
                <li>🌦️ <b>Analyze</b> impact of weather conditions</li>
                <li>📊 <b>Compare</b> crop performance globally</li>
                <li>🔮 <b>Understand</b> model predictions</li>
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
        
        # Data Preview
        st.markdown("""
        <div class="info-card">
            <h3>📋 Data Overview</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>📈 Key Metrics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("🌍 Countries", df['Area'].nunique())
        st.metric("🌾 Crops", df['Item'].nunique())
        st.metric("📊 Records", f"{len(df):,}")
        
        if model is None:
            st.metric("🔧 Mode", "DEMO")
        else:
            st.metric("🔧 Mode", "ADVANCED")
            if model_info and 'performance' in model_info:
                st.metric("🎯 Accuracy", f"{model_info['performance']['r2_score']:.1%}")
        
        # Top crops visualization
        st.markdown("""
        <div class="info-card">
            <h3>🌾 Top Crops by Yield</h3>
        </div>
        """, unsafe_allow_html=True)
        
        top_crops = df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False).head(5)
        fig = px.bar(
            x=top_crops.values, 
            y=top_crops.index,
            orientation='h',
            title="Average Yield by Crop",
            labels={'x': 'Yield (hg/ha)', 'y': 'Crop'},
            color=top_crops.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=20, color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Prediction":
    st.markdown('<h1 class="main-header">🎯 Crop Yield Prediction</h1>', unsafe_allow_html=True)
    
    # Show mode indicator
    if model is None:
        st.markdown(f"""
        <div class="demo-card">
            <h3>🎭 Demo Mode</h3>
            <p>Using <b>simulated predictions</b> based on crop science formulas. For advanced predictions, export your trained model.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-card">
            <h3>🚀 Advanced Model Active</h3>
            <p>Using <b>trained prediction model</b> for accurate yield forecasts.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Input Form with Tabs for different input methods
    st.markdown("""
    <div class="input-card">
        <h3>🔧 Input Methods</h3>
        <p>Choose how you want to input data:</p>
    </div>
    """, unsafe_allow_html=True)
    
    input_tab1, input_tab2 = st.tabs(["🎯 Quick Input (Sliders)", "⌨️ Manual Input (Numbers)"])
    
    with input_tab1:
        st.markdown("""
        <div class="tab-content">
            <h4>Use sliders for quick adjustments</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="prediction-card">
                <h3>📍 Location & Crop</h3>
            </div>
            """, unsafe_allow_html=True)
            
            area = st.selectbox("🌍 Country", sorted(df['Area'].unique()), 
                              index=0 if 'Albania' in df['Area'].unique() else 0,
                              key="area_slider")
            item = st.selectbox("🌾 Crop Type", sorted(df['Item'].unique()), 
                              index=list(df['Item'].unique()).index('Cassava') if 'Cassava' in df['Item'].unique() else 0,
                              key="item_slider")
            year = st.slider("📅 Year", 1990, 2050, 2023, key="year_slider")
        
        with col2:
            st.markdown("""
            <div class="prediction-card">
                <h3>🌦️ Environmental Factors</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create columns for better layout
            col2a, col2b = st.columns(2)
            
            with col2a:
                rainfall = st.slider("💧 Rainfall (mm/year)", 0, 5000, 1485, 
                                   help="Annual rainfall in millimeters",
                                   key="rainfall_slider")
                pesticides = st.slider("🧪 Pesticides (tonnes)", 0, 1000, 121,
                                     help="Pesticide usage in tonnes per hectare",
                                     key="pesticides_slider")
            
            with col2b:
                temperature = st.slider("🌡️ Temperature (°C)", -20, 50, 16,
                                      help="Average annual temperature in Celsius",
                                      key="temperature_slider")
                
                # Add a visual indicator for temperature
                temp_indicators = {
                    '❄️ Cool': temperature < 10,
                    '🌤️ Moderate': 10 <= temperature < 20,
                    '☀️ Warm': 20 <= temperature < 30,
                    '🔥 Hot': temperature >= 30
                }
                temp_status = next((status for status, condition in temp_indicators.items() if condition), "🌡️ Normal")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          padding: 15px; border-radius: 15px; text-align: center; color: white; 
                          border: 2px solid #FFD93D;">
                    <h4>Temperature Status</h4>
                    <h2>{temp_status}</h2>
                    <p>{temperature}°C</p>
                </div>
                """, unsafe_allow_html=True)
    
    with input_tab2:
        st.markdown("""
        <div class="tab-content">
            <h4>Enter exact values for precise control</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="prediction-card">
                <h3>📍 Location & Crop</h3>
            </div>
            """, unsafe_allow_html=True)
            
            area_manual = st.selectbox("🌍 Country", sorted(df['Area'].unique()), 
                                     index=0 if 'Albania' in df['Area'].unique() else 0,
                                     key="area_manual")
            item_manual = st.selectbox("🌾 Crop Type", sorted(df['Item'].unique()), 
                                     index=list(df['Item'].unique()).index('Cassava') if 'Cassava' in df['Item'].unique() else 0,
                                     key="item_manual")
            year_manual = st.number_input("📅 Year", min_value=1990, max_value=2050, value=2023, step=1,
                                        help="Enter the year for prediction",
                                        key="year_manual")
        
        with col2:
            st.markdown("""
            <div class="prediction-card">
                <h3>🌦️ Environmental Factors</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col2a, col2b = st.columns(2)
            
            with col2a:
                rainfall_manual = st.number_input("💧 Rainfall (mm/year)", 
                                                min_value=0, 
                                                max_value=10000, 
                                                value=1485,
                                                step=10,
                                                help="Annual rainfall in millimeters",
                                                key="rainfall_manual")
                
                pesticides_manual = st.number_input("🧪 Pesticides (tonnes)", 
                                                  min_value=0.0, 
                                                  max_value=2000.0, 
                                                  value=121.0,
                                                  step=0.1,
                                                  help="Pesticide usage in tonnes per hectare",
                                                  key="pesticides_manual")
            
            with col2b:
                temperature_manual = st.number_input("🌡️ Temperature (°C)", 
                                                   min_value=-50.0, 
                                                   max_value=60.0, 
                                                   value=16.37,
                                                   step=0.1,
                                                   help="Average annual temperature in Celsius",
                                                   key="temperature_manual")
                
                # Add a visual indicator for temperature
                temp_indicators_manual = {
                    '❄️ Cool': temperature_manual < 10,
                    '🌤️ Moderate': 10 <= temperature_manual < 20,
                    '☀️ Warm': 20 <= temperature_manual < 30,
                    '🔥 Hot': temperature_manual >= 30
                }
                temp_status_manual = next((status for status, condition in temp_indicators_manual.items() if condition), "🌡️ Normal")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          padding: 15px; border-radius: 15px; text-align: center; color: white; 
                          border: 2px solid #FFD93D;">
                    <h4>Temperature Status</h4>
                    <h2>{temp_status_manual}</h2>
                    <p>{temperature_manual}°C</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Determine which input method to use
    use_manual_input = input_tab2  # This will be True if manual tab is active
    
    if use_manual_input:
        # Use manual input values
        final_area = area_manual
        final_item = item_manual
        final_year = year_manual
        final_rainfall = rainfall_manual
        final_pesticides = pesticides_manual
        final_temperature = temperature_manual
        input_method = "Manual Input"
    else:
        # Use slider values
        final_area = area
        final_item = item
        final_year = year
        final_rainfall = rainfall
        final_pesticides = pesticides
        final_temperature = temperature
        input_method = "Quick Input"
    
    # Display current input values
    st.markdown("""
    <div class="info-card">
        <h3>📋 Current Input Values</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🌍 Country</h4>
            <h3>{final_area}</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <h4>🌾 Crop</h4>
            <h3>{final_item}</h3>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📅 Year</h4>
            <h3>{final_year}</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <h4>💧 Rainfall</h4>
            <h3>{final_rainfall} mm</h3>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🧪 Pesticides</h4>
            <h3>{final_pesticides} t</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <h4>🌡️ Temperature</h4>
            <h3>{final_temperature}°C</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Unified Prediction Function
    def predict_yield(year, rainfall, pesticides, temperature, area, item):
        if model is not None:
            # Use real model
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
                return prediction, "Advanced Model"
                
            except Exception as e:
                st.error(f"Model Error: {e}")
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
            prediction, model_type = predict_yield(final_year, final_rainfall, final_pesticides, final_temperature, final_area, final_item)
            
            st.balloons()
            st.success(f"✅ Prediction Complete! ({model_type})")
            
            # Display Results
            st.markdown("""
            <div class="prediction-card">
                <h2 style="text-align: center; color: white;">📊 Prediction Results</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Results in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Predicted Yield", f"{prediction:,.0f} hg/ha")
            with col2:
                st.metric("In Tons/Hectare", f"{prediction/10000:,.1f} t/ha")
            with col3:
                avg_yield = df[df['Item'] == final_item]['hg/ha_yield'].mean()
                difference = ((prediction - avg_yield) / avg_yield * 100)
                st.metric("Vs Average", f"{difference:+.1f}%")
            with col4:
                st.metric("Input Method", input_method)
            
            # Yield Gauge Chart
            st.subheader("📈 Yield Prediction Gauge")
            
            # Calculate min and max for gauge
            min_yield = df[df['Item'] == final_item]['hg/ha_yield'].min()
            max_yield = df[df['Item'] == final_item]['hg/ha_yield'].max()
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                title={'text': f"Yield Prediction for {final_item} in {final_area}", 'font': {'size': 20, 'color': 'white'}},
                delta={'reference': avg_yield, 'relative': True, 'valueformat': '.1%'},
                gauge={
                    'axis': {'range': [min_yield, max_yield], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#FFD93D"},
                    'bgcolor': "rgba(0,0,0,0.3)",
                    'borderwidth': 2,
                    'bordercolor': "white",
                    'steps': [
                        {'range': [min_yield, avg_yield*0.6], 'color': "#FF6B6B"},
                        {'range': [avg_yield*0.6, avg_yield*0.9], 'color': "#FFD93D"},
                        {'range': [avg_yield*0.9, avg_yield*1.2], 'color': "#4ECDC4"},
                        {'range': [avg_yield*1.2, max_yield], 'color': "#45B7D1"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': avg_yield
                    }
                }
            ))
            
            fig.update_layout(
                height=400, 
                margin=dict(t=50, b=10, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional Visualizations
            st.subheader("📊 Additional Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Historical trend for selected crop and country
                st.markdown("**Historical Trend**")
                historical_data = df[(df['Item'] == final_item) & (df['Area'] == final_area)]
                if not historical_data.empty:
                    fig_trend = px.line(
                        historical_data.groupby('Year')['hg/ha_yield'].mean().reset_index(),
                        x='Year',
                        y='hg/ha_yield',
                        title=f"{final_item} Yield Trend in {final_area}",
                        color_discrete_sequence=['#FFD93D']
                    )
                    fig_trend.add_hline(y=prediction, line_dash="dash", line_color="#FF6B6B", 
                                       annotation_text="Prediction")
                    fig_trend.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        title_font=dict(color='white')
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("No historical data available for this crop and country combination")
            
            with col2:
                # Crop comparison
                st.markdown("**Crop Comparison**")
                crop_comparison = df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False).head(10)
                fig_comparison = px.bar(
                    crop_comparison.reset_index(),
                    x='hg/ha_yield',
                    y='Item',
                    orientation='h',
                    title="Top Crops by Average Yield",
                    color='hg/ha_yield',
                    color_continuous_scale='viridis'
                )
                # Highlight the selected crop
                if final_item in crop_comparison.index:
                    fig_comparison.update_traces(
                        marker_color=['#FF6B6B' if crop == final_item else px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] 
                        for i, crop in enumerate(crop_comparison.index)]
                    )
                fig_comparison.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    title_font=dict(color='white')
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Additional Insights
            st.subheader("💡 Insights & Recommendations")
            
            if prediction > avg_yield * 1.1:
                st.markdown(f"""
                <div class="info-card">
                    <h3>🎉 Excellent Conditions!</h3>
                    <p>Yield is <b>{((prediction-avg_yield)/avg_yield*100):.1f}% above average</b> for {final_item}.</p>
                    <p><b>Recommendations:</b> Continue current practices. Consider scaling production.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction > avg_yield * 0.9:
                st.markdown(f"""
                <div class="info-card">
                    <h3>👍 Good Conditions</h3>
                    <p>Yield is close to average for {final_item}.</p>
                    <p><b>Recommendations:</b> Maintain current practices. Monitor for any changes in conditions.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="demo-card">
                    <h3>⚠️ Challenging Conditions</h3>
                    <p>Yield is <b>{((avg_yield-prediction)/avg_yield*100):.1f}% below average</b>. Consider adjusting inputs.</p>
                    <p><b>Recommendations:</b> Review irrigation, fertilization, and pest control strategies.</p>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌾 <b>Crop Yield Prediction System</b> | Advanced Agricultural Analytics</p>
    <p>💡 Making agriculture smarter with data-driven insights</p>
</div>
""", unsafe_allow_html=True)