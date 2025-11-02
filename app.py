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

# Custom CSS with improved styling
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
    .info-card {
        background: rgba(173, 216, 230, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #4169E1;
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
    .mode-indicator {
        padding: 10px 15px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    .ai-mode {
        background-color: #e6f7e9;
        color: #2e7d32;
        border: 2px solid #4caf50;
    }
    .demo-mode {
        background-color: #fff8e1;
        color: #f57c00;
        border: 2px solid #ffb300;
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
st.markdown('<h1 class="main-header">🌾 AI Crop Yield Predictor</h1>', unsafe_allow_html=True)

# Sidebar with Export Instructions
st.sidebar.title("🌾 Navigation")
page = st.sidebar.radio("Choose Page", ["🏠 Home", "🎯 Prediction"])

# Show export instructions in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Setup Instructions")

# Mode indicator
if model is None:
    mode_class = "demo-mode"
    mode_text = "🎭 DEMO MODE (Using simulated predictions)"
    st.sidebar.markdown(f"""
    <div class="mode-indicator {mode_class}">{mode_text}</div>
    
    **To use real AI model:**
    1. Open your notebook
    2. Run the export cell
    3. Restart this app
    """, unsafe_allow_html=True)
else:
    mode_class = "ai-mode"
    mode_text = "🤖 AI MODEL (Using trained ML model)"
    st.sidebar.markdown(f"""
    <div class="mode-indicator {mode_class}">{mode_text}</div>
    
    **Model Status:** ✅ Ready
    """, unsafe_allow_html=True)

# Add quick stats to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("🌍 Countries", df['Area'].nunique())
with col2:
    st.metric("🌾 Crops", df['Item'].nunique())

if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Mode indicator at the top
        if model is None:
            st.markdown(f"""
            <div class="demo-card">
                <h2>🎭 Demo Mode Active</h2>
                <p>Currently running with <b>simulated predictions</b>. For real AI predictions, export your model from the notebook.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🤖 AI Model Active</h2>
                <p>Using <b>trained machine learning model</b> for accurate predictions.</p>
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
            st.metric("🔧 Mode", "AI MODEL")
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
            labels={'x': 'Yield (hg/ha)', 'y': 'Crop'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Prediction":
    st.markdown('<h1 class="main-header">🎯 Crop Yield Prediction</h1>', unsafe_allow_html=True)
    
    # Show mode indicator
    if model is None:
        st.markdown(f"""
        <div class="demo-card">
            <h3>🎭 Demo Mode</h3>
            <p>Using <b>simulated predictions</b> based on crop science formulas. For AI predictions, export your trained model.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
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
        
        area = st.selectbox("🌍 Country", sorted(df['Area'].unique()), index=0 if 'Albania' in df['Area'].unique() else 0)
        item = st.selectbox("🌾 Crop Type", sorted(df['Item'].unique()), index=list(df['Item'].unique()).index('Cassava') if 'Cassava' in df['Item'].unique() else 0)
        year = st.slider("📅 Year", 1990, 2030, 2023)
    
    with col2:
        st.markdown("""
        <div class="prediction-card">
            <h3>🌦️ Environmental Factors</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for better layout
        col2a, col2b = st.columns(2)
        
        with col2a:
            rainfall = st.slider("💧 Rainfall (mm/year)", 0, 3000, 1485, 
                               help="Annual rainfall in millimeters")
            pesticides = st.slider("🧪 Pesticides (tonnes)", 0, 500, 121,
                                 help="Pesticide usage in tonnes per hectare")
        
        with col2b:
            temperature = st.slider("🌡️ Temperature (°C)", -10, 40, 16,
                                  help="Average annual temperature in Celsius")
            
            # Add a visual indicator for temperature
            if temperature < 10:
                temp_status = "❄️ Cool"
            elif temperature < 20:
                temp_status = "🌤️ Moderate"
            elif temperature < 30:
                temp_status = "☀️ Warm"
            else:
                temp_status = "🔥 Hot"
                
            st.markdown(f"**Temperature Status:** {temp_status}")
    
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
            
            # Calculate min and max for gauge
            min_yield = df[df['Item'] == item]['hg/ha_yield'].min()
            max_yield = df[df['Item'] == item]['hg/ha_yield'].max()
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                title={'text': f"Yield Prediction for {item} in {area}", 'font': {'size': 20}},
                delta={'reference': avg_yield, 'relative': True, 'valueformat': '.1%'},
                gauge={
                    'axis': {'range': [min_yield, max_yield], 'tickwidth': 1},
                    'bar': {'color': "darkgreen"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [min_yield, avg_yield*0.6], 'color': "lightcoral"},
                        {'range': [avg_yield*0.6, avg_yield*0.9], 'color': "lightyellow"},
                        {'range': [avg_yield*0.9, avg_yield*1.2], 'color': "lightgreen"},
                        {'range': [avg_yield*1.2, max_yield], 'color': "green"}
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
            
            # Additional Visualizations
            st.subheader("📊 Additional Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Historical trend for selected crop and country
                st.markdown("**Historical Trend**")
                historical_data = df[(df['Item'] == item) & (df['Area'] == area)]
                if not historical_data.empty:
                    fig_trend = px.line(
                        historical_data.groupby('Year')['hg/ha_yield'].mean().reset_index(),
                        x='Year',
                        y='hg/ha_yield',
                        title=f"{item} Yield Trend in {area}"
                    )
                    fig_trend.add_hline(y=prediction, line_dash="dash", line_color="red", 
                                       annotation_text="Prediction")
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
                    title="Top Crops by Average Yield"
                )
                # Highlight the selected crop
                if item in crop_comparison.index:
                    fig_comparison.update_traces(
                        marker_color=['red' if crop == item else 'blue' for crop in crop_comparison.index]
                    )
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Additional Insights
            st.subheader("💡 Insights & Recommendations")
            
            if prediction > avg_yield * 1.1:
                st.success(f"**Excellent Conditions!** Yield is {((prediction-avg_yield)/avg_yield*100):.1f}% above average for {item}.")
                st.info("**Recommendations:** Continue current practices. Consider scaling production.")
            elif prediction > avg_yield * 0.9:
                st.info(f"**Good Conditions** Yield is close to average for {item}.")
                st.info("**Recommendations:** Maintain current practices. Monitor for any changes in conditions.")
            else:
                st.warning(f"**Challenging Conditions** Yield is {((avg_yield-prediction)/avg_yield*100):.1f}% below average. Consider adjusting inputs.")
                st.info("**Recommendations:** Review irrigation, fertilization, and pest control strategies.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌾 <b>Crop Yield Prediction System</b> | Built with Streamlit & Machine Learning</p>
    <p>💡 Making agriculture smarter with data-driven insights</p>
</div>
""", unsafe_allow_html=True)