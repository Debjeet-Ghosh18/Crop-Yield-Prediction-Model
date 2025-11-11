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

# High Contrast, Readable CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #0d1117;
    }
    
    .app-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin: 2rem 0 0.5rem 0;
        letter-spacing: -1px;
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #c9d1d9;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    .card {
        background: #161b22;
        border-radius: 12px;
        padding: 1.8rem;
        margin: 1rem 0;
        border: 1px solid #30363d;
    }
    
    .card h3, .card h4 {
        color: #ffffff !important;
        margin-top: 0;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .card p, .card li {
        color: #c9d1d9 !important;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .metric-container {
        background: #161b22;
        border-radius: 10px;
        padding: 1.3rem;
        text-align: center;
        border: 1px solid #30363d;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.3rem 0;
    }
    
    .metric-small {
        font-size: 1.6rem;
    }
    
    .stButton>button {
        background: #238636;
        color: #ffffff;
        border: none;
        padding: 0.85rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
        border: 1px solid #2ea043;
    }
    
    .stButton>button:hover {
        background: #2ea043;
        border-color: #3fb950;
        transform: translateY(-1px);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-active {
        background: #238636;
        color: #ffffff;
        border: 1px solid #2ea043;
    }
    
    .status-demo {
        background: #9e6a03;
        color: #ffffff;
        border: 1px solid #bb7f0a;
    }
    
    .info-box {
        background: #161b22;
        border-left: 4px solid #58a6ff;
        padding: 1.2rem 1.5rem;
        border-radius: 6px;
        margin: 1.5rem 0;
    }
    
    .info-box strong {
        color: #ffffff;
        font-size: 1.05rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .info-box p {
        color: #c9d1d9;
        margin: 0;
        line-height: 1.6;
    }
    
    .success-box {
        background: #161b22;
        border-left: 4px solid #3fb950;
        padding: 1.2rem 1.5rem;
        border-radius: 6px;
        margin: 1.5rem 0;
    }
    
    .success-box strong {
        color: #ffffff;
        font-size: 1.05rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .success-box p {
        color: #c9d1d9;
        margin: 0;
        line-height: 1.6;
    }
    
    .warning-box {
        background: #161b22;
        border-left: 4px solid #d29922;
        padding: 1.2rem 1.5rem;
        border-radius: 6px;
        margin: 1.5rem 0;
    }
    
    .warning-box strong {
        color: #ffffff;
        font-size: 1.05rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .warning-box p {
        color: #c9d1d9;
        margin: 0;
        line-height: 1.6;
    }
    
    .section-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2.5rem 0 1.2rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #161b22;
        padding: 8px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        color: #8b949e;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: #238636;
        color: #ffffff;
    }
    
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    p, span, div, label {
        color: #c9d1d9;
    }
    
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stSelectbox > div > div {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        color: #ffffff;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #58a6ff;
    }
    
    input[type="number"] {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    input[type="number"]:focus {
        border-color: #58a6ff !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: #30363d;
    }
    
    .stSlider > div > div > div > div {
        background: #238636;
    }
    
    [data-testid="stThumbValue"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    .stDataFrame {
        background: #161b22;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        color: #c9d1d9;
    }
    
    hr {
        border-color: #30363d;
        margin: 2.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: #0d1117;
    }
    
    [data-testid="stSidebar"] {
        background: #0d1117;
    }
    
    /* Radio button styling */
    .stRadio > label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    .stRadio > div {
        color: #c9d1d9;
    }
    
    .stRadio [role="radiogroup"] label {
        color: #c9d1d9 !important;
    }
    
    /* Caption styling */
    .stCaption {
        color: #8b949e !important;
    }
    
    /* Loading spinner customization */
    .stSpinner > div {
        border-color: #238636 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load Functions
@st.cache_resource
def load_model():
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
    except:
        return None, None, None

@st.cache_data
def load_data():
    try:
        if os.path.exists('model/sample_data.csv'):
            return pd.read_csv('model/sample_data.csv')
        elif os.path.exists('data/yield_df.csv'):
            return pd.read_csv('data/yield_df.csv')
        else:
            raise FileNotFoundError()
    except:
        np.random.seed(42)
        n = 1000
        
        data = {
            'Year': np.random.randint(1990, 2023, n),
            'average_rain_fall_mm_per_year': np.random.normal(1485, 300, n),
            'pesticides_tonnes': np.random.normal(121, 30, n),
            'avg_temp': np.random.normal(16.37, 3, n),
            'Area': np.random.choice(['Albania', 'Argentina', 'Australia', 'Brazil', 'China', 'India', 'USA'], n),
            'Item': np.random.choice(['Maize', 'Potatoes', 'Rice', 'Wheat', 'Soybeans', 'Barley', 'Cassava'], n),
            'hg/ha_yield': np.random.normal(25000, 15000, n)
        }
        
        df = pd.DataFrame(data)
        df['hg/ha_yield'] = df['hg/ha_yield'] + (df['average_rain_fall_mm_per_year'] - 800) * 8
        df['hg/ha_yield'] = df['hg/ha_yield'] + (df['avg_temp'] - 20) * 500
        df['hg/ha_yield'] = np.abs(df['hg/ha_yield'])
        
        return df

def improved_predict_yield(year, rainfall, pesticides, temp, area, item):
    # Realistic base yields
    crop_base_yields = {
        'Maize': 45000, 'Potatoes': 35000, 'Rice': 52000, 'Wheat': 38000,
        'Soybeans': 28000, 'Barley': 32000, 'Cassava': 42000
    }
    
    base = crop_base_yields.get(item, 35000)
    
    # Smart environmental factors
    rain_optimal = 1200
    rain_factor = np.tanh((rainfall - rain_optimal) / 800) * 0.25
    
    temp_optimal = 22 if item in ['Maize', 'Rice'] else 18
    temp_factor = -0.01 * (temp - temp_optimal) ** 2
    
    pest_factor = np.log1p(pesticides) * 0.08 - 0.15
    
    # Yearly improvement
    year_factor = (year - 2000) * 0.005
    
    # Calculate yield
    predicted = base * (1 + rain_factor + temp_factor + pest_factor + year_factor)
    
    # Add realistic noise
    predicted *= np.random.normal(1, 0.08)
    
    return max(15000, min(80000, predicted))

# Initialize
model, preprocessor, model_info = load_model()
df = load_data()

# Sidebar
with st.sidebar:
    st.markdown('<h2 style="color: white; margin-bottom: 1.5rem; font-size: 1.4rem;">Navigation</h2>', unsafe_allow_html=True)
    page = st.radio("", ["🏠 Home", "🎯 Predict"], label_visibility="collapsed")
    
    st.markdown("---")
    
    if model is None:
        st.markdown('<span class="status-badge status-demo">Demo Mode</span>', unsafe_allow_html=True)
        st.caption("Using simulation model")
    else:
        st.markdown('<span class="status-badge status-active">Model Ready</span>', unsafe_allow_html=True)
        st.caption("Using trained model")
    
    st.markdown("---")
    st.markdown("### Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Countries</div>
            <div class="metric-value metric-small">{df['Area'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Crops</div>
            <div class="metric-value metric-small">{df['Item'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-container" style="margin-top: 1rem;">
        <div class="metric-label">Total Records</div>
        <div class="metric-value metric-small">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

# Main Content
if page == "🏠 Home":
    st.markdown('<h1 class="app-title">Crop Yield Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict agricultural yields using environmental data</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### About")
        st.markdown("""
        This tool helps predict crop yields based on environmental conditions and historical data. 
        Enter your parameters to get yield estimates for different crops and regions.
        """)
        
        st.markdown("### Features")
        st.markdown("• Predict yields for multiple crops")
        st.markdown("• Analyze environmental impact")
        st.markdown("• Compare historical trends")
        st.markdown("• Get actionable insights")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<h3 class="section-title">Data Overview</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Sample Records**")
        st.dataframe(df.head(10), use_container_width=True, height=350)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Quick Stats")
        
        st.metric("Countries", df['Area'].nunique())
        st.metric("Crop Types", df['Item'].nunique())
        st.metric("Records", f"{len(df):,}")
        
        if model is not None and model_info and 'performance' in model_info:
            st.metric("Accuracy", f"{model_info['performance']['r2_score']:.1%}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Top Crops")
        
        top = df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False).head(6)
        
        fig = go.Figure(go.Bar(
            y=top.index,
            x=top.values,
            orientation='h',
            marker=dict(color='#238636')
        ))
        
        fig.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=5, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9', size=11),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:  # Prediction page
    st.markdown('<h1 class="app-title">Yield Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Configure parameters and get instant predictions</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Inputs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Location & Crop")
        area = st.selectbox("Country", sorted(df['Area'].unique()), key="a1")
        item = st.selectbox("Crop", sorted(df['Item'].unique()), key="i1")
        year = st.number_input("Year", 1990, 2050, 2023, key="y1")
    
    with col2:
        st.markdown("#### Environmental Data")
        rainfall = st.number_input("Rainfall (mm/year)", 0, 10000, 1485, key="r1")
        pesticides = st.number_input("Pesticides (tonnes)", 0.0, 2000.0, 121.0, key="p1")
        temp = st.number_input("Temperature (°C)", -50.0, 60.0, 16.37, key="t1")
    
    # Validate inputs
    if rainfall == 0:
        st.warning("⚠️ Rainfall cannot be zero for crop growth")
    if temp < -10 or temp > 50:
        st.warning("⚠️ Temperature outside typical growing range")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display current values
    st.markdown('<h3 class="section-title">Current Settings</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Country</div>
            <div class="metric-value metric-small">{area[:8]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Crop</div>
            <div class="metric-value metric-small">{item[:8]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Year</div>
            <div class="metric-value metric-small">{year}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Rainfall</div>
            <div class="metric-value metric-small">{int(rainfall)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Pesticides</div>
            <div class="metric-value metric-small">{int(pesticides)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Temp (°C)</div>
            <div class="metric-value metric-small">{int(temp)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Predict Button
    if st.button("Calculate Prediction", use_container_width=True):
        with st.spinner("Analyzing data and generating prediction..."):
            if model is not None:
                try:
                    input_df = pd.DataFrame({
                        'Year': [year],
                        'average_rain_fall_mm_per_year': [rainfall],
                        'pesticides_tonnes': [pesticides],
                        'avg_temp': [temp],
                        'Area': [area],
                        'Item': [item]
                    })
                    
                    transformed = preprocessor.transform(input_df)
                    pred = model.predict(transformed)[0]
                    mode = "Trained Model"
                except Exception as e:
                    st.error(f"Model error: {e}")
                    pred = improved_predict_yield(year, rainfall, pesticides, temp, area, item)
                    mode = "Demo Mode"
            else:
                pred = improved_predict_yield(year, rainfall, pesticides, temp, area, item)
                mode = "Demo Mode"
            
            st.success("✅ Prediction complete!")
            
            # Results
            st.markdown('<h3 class="section-title">Results</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Predicted Yield</div>
                    <div class="metric-value">{pred:,.0f}</div>
                    <div class="metric-label">hg/ha</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">In Tonnes</div>
                    <div class="metric-value">{pred/10000:,.1f}</div>
                    <div class="metric-label">t/ha</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Source</div>
                    <div class="metric-value metric-small">{mode.split()[0]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                title={'text': f"{item} Yield Prediction", 'font': {'size': 22, 'color': '#ffffff'}},
                number={'font': {'color': '#ffffff', 'size': 40}},
                gauge={
                    'axis': {'range': [0, 80000], 'tickcolor': "#c9d1d9", 'tickfont': {'color': '#c9d1d9'}},
                    'bar': {'color': "#238636"},
                    'bgcolor': "#161b22",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 20000], 'color': "#da3633"},
                        {'range': [20000, 40000], 'color': "#d29922"},
                        {'range': [40000, 60000], 'color': "#238636"},
                        {'range': [60000, 80000], 'color': "#3fb950"}
                    ]
                }
            ))
            
            fig.update_layout(
                height=350,
                margin=dict(t=70, b=10, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # New Visualizations Section
            st.markdown('<h3 class="section-title">Yield Analysis</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Environmental Impact**")
                
                # Calculate impact scores
                factors = ['Rainfall', 'Temperature', 'Pesticides']
                
                # Calculate optimal ranges
                rain_optimal = 1200
                temp_optimal = 22 if item in ['Maize', 'Rice'] else 18
                pest_optimal = 150
                
                rain_score = max(0, 100 - abs(rainfall - rain_optimal) / 20)
                temp_score = max(0, 100 - abs(temp - temp_optimal))
                pest_score = max(0, min(100, pesticides / 2))
                
                impact_scores = [rain_score, temp_score, pest_score]
                
                fig = go.Figure(go.Bar(
                    x=impact_scores,
                    y=factors,
                    orientation='h',
                    marker=dict(color=['#58a6ff', '#d29922', '#3fb950'])
                ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c9d1d9', size=12),
                    xaxis=dict(
                        range=[0, 100],
                        title="Optimality Score",
                        showgrid=True,
                        gridcolor='#30363d'
                    ),
                    yaxis=dict(showgrid=False),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Optimal Range Analysis**")
                
                # Create gauge indicators for each factor
                fig = go.Figure()
                
                # Add optimal range indicators
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=rainfall,
                    title={'text': "Rainfall", 'font': {'size': 14}},
                    number={'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [0, 3000]},
                        'bar': {'color': "#58a6ff"},
                        'steps': [
                            {'range': [800, 1600], 'color': "rgba(88, 166, 255, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 2},
                            'thickness': 0.75,
                            'value': 1200
                        }
                    },
                    domain={'row': 0, 'column': 0}
                ))
                
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=temp,
                    title={'text': "Temperature", 'font': {'size': 14}},
                    number={'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [0, 40]},
                        'bar': {'color': "#d29922"},
                        'steps': [
                            {'range': [15, 25], 'color': "rgba(210, 153, 34, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 2},
                            'thickness': 0.75,
                            'value': 20
                        }
                    },
                    domain={'row': 0, 'column': 1}
                ))
                
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=pesticides,
                    title={'text': "Pesticides", 'font': {'size': 14}},
                    number={'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [0, 300]},
                        'bar': {'color': "#3fb950"},
                        'steps': [
                            {'range': [50, 200], 'color': "rgba(63, 185, 80, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 2},
                            'thickness': 0.75,
                            'value': 125
                        }
                    },
                    domain={'row': 1, 'column': 0}
                ))
                
                fig.update_layout(
                    grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
                    height=400,
                    margin=dict(t=50, b=10, l=10, r=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c9d1d9')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional Analysis
            st.markdown('<h3 class="section-title">Additional Insights</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Yield Distribution**")
                
                # Create yield distribution for the selected crop
                crop_data = df[df['Item'] == item]['hg/ha_yield']
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=crop_data,
                    nbinsx=20,
                    marker_color='#238636',
                    opacity=0.7,
                    name='Historical Yields'
                ))
                
                fig.add_vline(x=pred, line_dash="dash", line_color="#ff7b72", line_width=3,
                            annotation_text="Your Prediction")
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c9d1d9', size=12),
                    xaxis=dict(title="Yield (hg/ha)", gridcolor='#30363d'),
                    yaxis=dict(title="Frequency", gridcolor='#30363d'),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Input Sensitivity**")
                
                # Show how changing inputs affects yield
                factors = ['+20% Rainfall', '+2°C Temp', '+50% Pesticides']
                changes = [1.15, 1.08, 1.12]  # Example impact factors
                
                new_yields = [pred * change for change in changes]
                
                fig = go.Figure(go.Bar(
                    x=new_yields,
                    y=factors,
                    orientation='h',
                    marker=dict(color=['#58a6ff', '#d29922', '#3fb950']),
                    text=[f'+{int((change-1)*100)}%' for change in changes],
                    textposition='auto'
                ))
                
                fig.add_vline(x=pred, line_dash="dash", line_color="#c9d1d9", line_width=2,
                            annotation_text="Current")
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c9d1d9', size=12),
                    xaxis=dict(title="Yield (hg/ha)", gridcolor='#30363d'),
                    yaxis=dict(showgrid=False),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1.5rem; color: #8b949e;">
    <p style="margin: 0; font-size: 0.95rem; font-weight: 500;">Crop Yield Predictor | Data-Driven Agriculture</p>
</div>
""", unsafe_allow_html=True)

















