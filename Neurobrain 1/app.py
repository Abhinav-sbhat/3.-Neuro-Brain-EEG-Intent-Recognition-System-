import os
os.environ["STREAMLIT_SERVER_PORT"] = "5000"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Configure page
st.set_page_config(
    page_title=" NeuroBrain: EEG Intent Recognition System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'eeg_data' not in st.session_state:
    st.session_state.eeg_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_type' not in st.session_state: 
    st.session_state.model_type = 'Random Forest'

def main():
    # Custom CSS for professional white theme
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: #ffffff;
        font-family: 'Inter', sans-serif;
        color: #333333;
    }
    
    /* All text color */
    .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader, .stButton, 
    .stSelectbox, .stSlider, .stTextInput, .stNumberInput, .stTextArea,
    .stCheckbox, .stRadio, .stFileUploader, .stProgress, .stExpander,
    .stTabs, .stDataFrame, .stTable, .stJson, .stMetric, .stAlert {
        color: #333333 !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #dee2e6;
        margin-bottom: 2rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0d6efd 0%, #0dcaf0 50%, #198754 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color;
        text-align: center;
        margin: 0;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
        border-right: 1px solid #dee2e6;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8f9fa;
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid #dee2e6;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #6c757d;
        font-weight: 500;
        padding: 12px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0d6efd 0%, #0a58ca 100%);
        color: white !important;
        box-shadow: 0 4px 15px rgba(13, 110, 253, 0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
    }
    
    /* Status indicators */
    .status-positive { color: #198754; }
    .status-negative { color: #dc3545; }
    .status-warning { color: #ffc107; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0d6efd 0%, #0a58ca 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(13, 110, 253, 0.15);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(13, 110, 253, 0.2);
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div {
        background: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 8px;
        color: #333333 !important;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #0d6efd 0%, #198754 100%);
        border-radius: 10px;
    }
    
    /* Charts and plots */
    .js-plotly-plot {
        border-radius: 12px;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        color: #333333 !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: #f8f9fa;
        border: 2px dashed #ced4da;
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .pulse { animation: pulse 2s infinite; }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Modern header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🧠 NeuroBrain: A Real-Time Electroencephalogram (EEG)-Based Intention Recognition System for Immobile Patients Using Artificial Intelligence</h1>
        <p class="main-subtitle">Advanced Brain-Computer Interface for Motor Imagery and Imagined Speech Classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar navigation
    st.sidebar.markdown("""
    <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #dee2e6;">
        <h2 style="color: #0d6efd; margin: 0; font-size: 1.5rem;">🎛️ Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced system status with modern cards
    st.sidebar.markdown("""
    <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #dee2e6;">
        <h3 style="color: #198754; margin: 0 0 1rem 0; font-size: 1.2rem;">📊 System Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Status indicators with enhanced styling
    data_status = ("🟢 Loaded", "status-positive") if st.session_state.eeg_data is not None else ("🔴 No Data", "status-negative")
    processed_status = ("🟢 Processed", "status-positive") if st.session_state.processed_data is not None else ("🟠 Not Processed", "status-warning")
    model_status = ("🟢 Trained", "status-positive") if st.session_state.trained_model is not None else ("🔴 Not Trained", "status-negative")
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <p><strong>📁 Data:</strong> <span class="{data_status[1]}">{data_status[0]}</span></p>
        <p><strong>⚙️ Processing:</strong> <span class="{processed_status[1]}">{processed_status[0]}</span></p>
        <p><strong>🎯 Model:</strong> <span class="{model_status[1]}">{model_status[0]}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Information panel
    st.sidebar.markdown("""
    <div style="padding: 1rem; background: #d1e7dd; border-radius: 10px; margin-top: 1rem; border: 1px solid #badbcc;">
        <h4 style="color: #0f5132; margin: 0 0 0.5rem 0;">ℹ️ About NeuroBrain</h4>
        <p style="color: #0f5132; font-size: 0.9rem; margin: 0;">Advanced AI system for EEG-based intention recognition. Supports motor imagery (Left /Right/Up/Down) and imagined speech (Yes/No/Start/Stop/Help) classification for immobile patients.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced main tabs with modern styling
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Data Upload", 
        "⚙️ Preprocessing", 
        "🎯 Model Training", 
        "⚡ Real-time Prediction", 
        "📊 Visualization"
    ])
    
    with tab1:
        data_upload_page()
    
    with tab2:
        preprocessing_page()
    
    with tab3:
        training_page()
    
    with tab4:
        real_time_page()
    
    with tab5:
        visualization_page()

def data_upload_page():
    """Data upload and validation page"""
    from pages.data_upload import render_data_upload
    render_data_upload()

def preprocessing_page():
    """Signal preprocessing page"""
    from pages.preprocessing import render_preprocessing
    render_preprocessing()

def training_page():
    """Model training page"""
    from pages.training import render_training
    render_training()

def real_time_page():
    """Real-time prediction page"""
    from pages.real_time import render_real_time
    render_real_time()

def visualization_page():
    """Advanced visualization page"""
    from pages.visualization import render_visualization
    render_visualization()

if __name__ == "__main__":
    main()