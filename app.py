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
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern, Premium CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 2rem 0 3rem 0;
        letter-spacing: -2px;
        animation: fadeInDown 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, rgba(168, 237, 234, 0.1) 0%, rgba(254, 214, 227, 0.1) 100%);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 3rem;
        border: 2px solid rgba(168, 237, 234, 0.3);
        box-shadow: 0 20px 60px rgba(168, 237, 234, 0.2);
        margin: 2rem 0;
        animation: scaleIn 0.5s ease-out;
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .metric-box {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 16px;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .mode-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        margin: 1rem 0;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    .ai-mode {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    }
    
    .demo-mode {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(240, 147, 251, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.03);
        padding: 8px;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.6);
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.9);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 16px;
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        background: rgba(255, 255, 255, 0.06);
        transform: translateY(-2px);
    }
    
    .stSelectbox, .stSlider, .stNumberInput {
        color: white;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
    }
    
    .info-badge {
        background: rgba(168, 237, 234, 0.1);
        border: 1px solid rgba(168, 237, 234, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #a8edea;
        font-weight: 500;
    }
    
    .warning-badge {
        background: rgba(245, 87, 108, 0.1);
        border: 1px solid rgba(245, 87, 108, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #f5576c;
        font-weight: 500;
    }
    
    .success-badge {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #667eea;
        font-weight: 500;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    p, span, div {
        color: rgba(255, 255, 255, 0.8);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, rgba(15, 12, 41, 0.95) 0%, rgba(48, 43, 99, 0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
    }
    
    .dataframe {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: white;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(168, 237, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load Model and Data with Better Error Handling
@st.cache_resource
def load_model():
    """Load model files with fallback to demo mode"""
    try:
        if not os.path.exists('model/crop_yield_model.pkl'):
            return None, None, None
            
        with open('model/crop_yield_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('model/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        return model, preprocessor, model_info
        
    except Exception as e:
        return None, None, None

@st.cache_data
def load_data():
    """Load data with intelligent fallback"""
    try:
        if os.path.exists('model/sample_data.csv'):
            df = pd.read_csv('model/sample_data.csv')
            return df
        elif os.path.exists('data/yield_df.csv'):
            df = pd.read_csv('data/yield_df.csv')
            return df
        else:
            raise FileNotFoundError("No data files found")
            
    except Exception as e:
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
        
        df = pd.DataFrame(demo_data)
        df['hg/ha_yield'] = df['hg/ha_yield'] + (df['average_rain_fall_mm_per_year'] - 800) * 8
        df['hg/ha_yield'] = df['hg/ha_yield'] + (df['avg_temp'] - 20) * 500
        df['hg/ha_yield'] = np.abs(df['hg/ha_yield'])
        
        return df

def demo_predict_yield(year, rainfall, pesticides, temperature, area, item):
    """Demo prediction function"""
    crop_base_yields = {
        'Maize': 30000, 'Potatoes': 25000, 'Rice': 35000, 
        'Wheat': 28000, 'Soybeans': 22000, 'Barley': 20000, 
        'Sorghum': 18000, 'Cassava': 26000
    }
    
    base_yield = crop_base_yields.get(item, 25000)
    rain_factor = (rainfall - 800) / 800 * 0.3
    temp_factor = (temperature - 20) / 20 * 0.2
    pest_factor = (pesticides - 50) / 50 * 0.1
    year_factor = (year - 2000) / 23 * 0.2
    
    predicted_yield = base_yield * (1 + rain_factor + temp_factor + pest_factor + year_factor)
    predicted_yield *= np.random.normal(1, 0.1)
    
    return max(5000, predicted_yield)

# Initialize
model, preprocessor, model_info = load_model()
df = load_data()

# Sidebar
with st.sidebar:
    st.markdown('<h2 style="text-align: center; margin-bottom: 2rem;">🌾 Navigation</h2>', unsafe_allow_html=True)
    
    page = st.radio(
        "Choose Page",
        ["🏠 Home", "🎯 Prediction"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if model is None:
        st.markdown("""
        <div class="mode-badge demo-mode">
            🎭 DEMO MODE
        </div>
        <div class="info-badge">
            Using simulated predictions. Export your trained model for advanced analytics.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="mode-badge ai-mode">
            🚀 AI MODEL ACTIVE
        </div>
        <div class="success-badge">
            Using trained ML model for accurate predictions.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<p class="section-header" style="font-size: 1.2rem;">📊 Dataset Stats</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Countries</div>
            <div class="metric-value">{df['Area'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Crops</div>
            <div class="metric-value">{df['Item'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

# Main Content
st.markdown('<h1 class="main-header">🌾 Crop Yield Predictor</h1>', unsafe_allow_html=True)

if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="margin-top: 0;">Welcome to Advanced Agricultural Analytics</h2>', unsafe_allow_html=True)
        st.markdown("""
        Harness the power of machine learning to predict crop yields with precision. 
        Our system analyzes environmental factors, historical trends, and agricultural data 
        to provide accurate forecasts for informed decision-making.
        
        ### 🎯 Key Features
        
        - **Smart Predictions**: ML-powered yield forecasting
        - **Real-time Analysis**: Instant insights from environmental data
        - **Historical Trends**: Compare with past performance
        - **Global Coverage**: Data from multiple countries and crops
        
        ### 🚀 How It Works
        
        1. **Select** your crop and location
        2. **Input** environmental parameters
        3. **Get** instant yield predictions
        4. **Analyze** comprehensive insights
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">📊 Data Preview</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Sample Data**")
            st.dataframe(df.head(10), use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Statistical Summary**")
            st.dataframe(df.describe(), use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin-top: 0;">📈 Key Metrics</h3>', unsafe_allow_html=True)
        
        st.metric("🌍 Countries", df['Area'].nunique())
        st.metric("🌾 Crop Types", df['Item'].nunique())
        st.metric("📊 Total Records", f"{len(df):,}")
        
        if model is not None and model_info and 'performance' in model_info:
            st.metric("🎯 Model Accuracy", f"{model_info['performance']['r2_score']:.1%}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin-top: 0;">🏆 Top Performers</h3>', unsafe_allow_html=True)
        
        top_crops = df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False).head(5)
        fig = go.Figure(go.Bar(
            y=top_crops.index,
            x=top_crops.values,
            orientation='h',
            marker=dict(
                color=top_crops.values,
                colorscale='viridis',
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            )
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "🎯 Prediction":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="margin-top: 0;">Configure Prediction Parameters</h2>', unsafe_allow_html=True)
    
    input_tab1, input_tab2 = st.tabs(["🎚️ Slider Input", "⌨️ Manual Input"])
    
    with input_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📍 Location & Crop")
            area = st.selectbox("Country", sorted(df['Area'].unique()), key="area_slider")
            item = st.selectbox("Crop Type", sorted(df['Item'].unique()), key="item_slider")
            year = st.slider("Year", 1990, 2050, 2023, key="year_slider")
        
        with col2:
            st.markdown("### 🌦️ Environmental Factors")
            rainfall = st.slider("💧 Rainfall (mm/year)", 0, 5000, 1485, key="rainfall_slider")
            pesticides = st.slider("🧪 Pesticides (tonnes)", 0, 1000, 121, key="pesticides_slider")
            temperature = st.slider("🌡️ Temperature (°C)", -20, 50, 16, key="temperature_slider")
        
        final_area, final_item, final_year = area, item, year
        final_rainfall, final_pesticides, final_temperature = rainfall, pesticides, temperature
    
    with input_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📍 Location & Crop")
            area_m = st.selectbox("Country", sorted(df['Area'].unique()), key="area_manual")
            item_m = st.selectbox("Crop Type", sorted(df['Item'].unique()), key="item_manual")
            year_m = st.number_input("Year", 1990, 2050, 2023, key="year_manual")
        
        with col2:
            st.markdown("### 🌦️ Environmental Factors")
            rainfall_m = st.number_input("💧 Rainfall (mm/year)", 0, 10000, 1485, key="rainfall_manual")
            pesticides_m = st.number_input("🧪 Pesticides (tonnes)", 0.0, 2000.0, 121.0, key="pesticides_manual")
            temperature_m = st.number_input("🌡️ Temperature (°C)", -50.0, 60.0, 16.37, key="temperature_manual")
        
        final_area, final_item, final_year = area_m, item_m, year_m
        final_rainfall, final_pesticides, final_temperature = rainfall_m, pesticides_m, temperature_m
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Current inputs display
    st.markdown('<p class="section-header">📋 Current Configuration</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Country</div>
            <div class="metric-value" style="font-size: 1.5rem;">{final_area}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Year</div>
            <div class="metric-value">{final_year}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Crop</div>
            <div class="metric-value" style="font-size: 1.5rem;">{final_item}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Rainfall</div>
            <div class="metric-value">{final_rainfall}<span style="font-size: 0.8rem;"> mm</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Temperature</div>
            <div class="metric-value">{final_temperature}<span style="font-size: 0.8rem;">°C</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Pesticides</div>
            <div class="metric-value">{final_pesticides}<span style="font-size: 0.8rem;"> t</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction Button
    if st.button("🚀 PREDICT YIELD", use_container_width=True):
        with st.spinner("🔮 Analyzing data..."):
            if model is not None:
                try:
                    input_data = pd.DataFrame({
                        'Year': [final_year],
                        'average_rain_fall_mm_per_year': [final_rainfall],
                        'pesticides_tonnes': [final_pesticides],
                        'avg_temp': [final_temperature],
                        'Area': [final_area],
                        'Item': [final_item]
                    })
                    
                    transformed_data = preprocessor.transform(input_data)
                    prediction = model.predict(transformed_data)[0]
                    model_type = "Advanced Model"
                except:
                    prediction = demo_predict_yield(final_year, final_rainfall, final_pesticides, final_temperature, final_area, final_item)
                    model_type = "Demo Model"
            else:
                prediction = demo_predict_yield(final_year, final_rainfall, final_pesticides, final_temperature, final_area, final_item)
                model_type = "Demo Model"
            
            st.balloons()
            
            st.markdown(f'<div class="prediction-result">', unsafe_allow_html=True)
            st.markdown(f'<h2 style="text-align: center; margin-bottom: 2rem;">🎉 Prediction Complete</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Predicted Yield</div>
                    <div class="metric-value">{prediction:,.0f}</div>
                    <div class="metric-label">hg/ha</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">In Tonnes</div>
                    <div class="metric-value">{prediction/10000:,.1f}</div>
                    <div class="metric-label">t/ha</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_yield = df[df['Item'] == final_item]['hg/ha_yield'].mean()
                difference = ((prediction - avg_yield) / avg_yield * 100)
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Vs Average</div>
                    <div class="metric-value">{difference:+.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Model Type</div>
                    <div class="metric-value" style="font-size: 1.2rem;">{model_type}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Gauge Chart
            min_yield = df[df['Item'] == final_item]['hg/ha_yield'].min()
            max_yield = df[df['Item'] == final_item]['hg/ha_yield'].max()
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                title={'text': f"Yield for {final_item}", 'font': {'size': 24, 'color': 'white'}},
                delta={'reference': avg_yield, 'relative': True, 'valueformat': '.1%'},
                gauge={
                    'axis': {'range': [min_yield, max_yield], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.5)"},
                    'bar': {'color': "#a8edea"},
                    'bgcolor': "rgba(0,0,0,0.3)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.2)",
                    'steps': [
                        {'range': [min_yield, avg_yield*0.7], 'color': "rgba(245, 87, 108, 0.3)"},
                        {'range': [avg_yield*0.7, avg_yield*1.1], 'color': "rgba(168, 237, 234, 0.3)"},
                        {'range': [avg_yield*1.1, max_yield], 'color': "rgba(102, 126, 234, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "#fed6e3", 'width': 4},
                        'thickness': 0.75,
                        'value': avg_yield
                    }
                }
            ))
            
            fig.update_layout(
                height=400,
                margin=dict(t=80, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional Insights
            st.markdown('<p class="section-header">📊 Additional Analysis</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**Historical Trend**")
                
                historical_data = df[(df['Item'] == final_item) & (df['Area'] == final_area)]
                if not historical_data.empty:
                    trend_data = historical_data.groupby('Year')['hg/ha_yield'].mean().reset_index()
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=trend_data['Year'],
                        y=trend_data['hg/ha_yield'],
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='#a8edea', width=3),
                        marker=dict(size=8, color='#fed6e3')
                    ))
                    
                    fig_trend.add_hline(
                        y=prediction,
                        line_dash="dash",
                        line_color="#667eea",
                        annotation_text="Prediction",
                        annotation_position="right"
                    )
                    
                    fig_trend.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=20, b=0),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=10),
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                        showlegend=False
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("No historical data available")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**Global Comparison**")
                
                crop_comparison = df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False).head(10)
                
                colors = ['#667eea' if crop == final_item else 'rgba(168, 237, 234, 0.5)' 
                         for crop in crop_comparison.index]
                
                fig_comparison = go.Figure(go.Bar(
                    y=crop_comparison.index,
                    x=crop_comparison.values,
                    orientation='h',
                    marker=dict(
                        color=colors,
                        line=dict(color='rgba(255,255,255,0.2)', width=1)
                    )
                ))
                
                fig_comparison.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=20, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=10),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                    showlegend=False
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommendations
            st.markdown('<p class="section-header">💡 Insights & Recommendations</p>', unsafe_allow_html=True)
            
            if prediction > avg_yield * 1.15:
                st.markdown(f"""
                <div class="success-badge">
                    <h3 style="margin: 0 0 0.5rem 0;">🎉 Excellent Conditions Detected</h3>
                    <p style="margin: 0;">Predicted yield is <strong>{((prediction-avg_yield)/avg_yield*100):.1f}% above average</strong> for {final_item}.</p>
                    <p style="margin: 0.5rem 0 0 0;"><strong>Recommendation:</strong> Optimal conditions for maximizing production. Consider scaling operations and maintaining current practices.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction > avg_yield * 0.9:
                st.markdown(f"""
                <div class="info-badge">
                    <h3 style="margin: 0 0 0.5rem 0;">👍 Good Growing Conditions</h3>
                    <p style="margin: 0;">Yield is within normal range for {final_item}.</p>
                    <p style="margin: 0.5rem 0 0 0;"><strong>Recommendation:</strong> Maintain current agricultural practices and monitor environmental changes closely.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-badge">
                    <h3 style="margin: 0 0 0.5rem 0;">⚠️ Below Average Conditions</h3>
                    <p style="margin: 0;">Yield is <strong>{((avg_yield-prediction)/avg_yield*100):.1f}% below average</strong>. Review input parameters.</p>
                    <p style="margin: 0.5rem 0 0 0;"><strong>Recommendation:</strong> Consider adjusting irrigation, fertilization strategies, or exploring alternative crops better suited to current conditions.</p>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <p style="color: rgba(255,255,255,0.6); font-size: 0.9rem; margin: 0;">
        🌾 <strong>Crop Yield Predictor</strong> | Powered by Machine Learning
    </p>
    <p style="color: rgba(255,255,255,0.4); font-size: 0.8rem; margin: 0.5rem 0 0 0;">
        Making agriculture smarter through data-driven insights
    </p>
</div>
""", unsafe_allow_html=True)