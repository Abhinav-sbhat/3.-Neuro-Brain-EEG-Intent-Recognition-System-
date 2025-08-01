import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Configure page
st.set_page_config(
    page_title="🧠 NeuroBrain: EEG Intent Recognition System",
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
    # Custom CSS for modern dark theme
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #475569;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 50%, #fbbf24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
        text-shadow: 0 0 30px rgba(96, 165, 250, 0.3);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid #334155;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        padding: 12px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #475569;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 0.5rem 0;
    }
    
    /* Status indicators */
    .status-positive { color: #10b981; }
    .status-negative { color: #ef4444; }
    .status-warning { color: #f59e0b; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #475569;
        border-radius: 8px;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #3b82f6 0%, #10b981 100%);
        border-radius: 10px;
    }
    
    /* Charts and plots */
    .js-plotly-plot {
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 8px;
        border: 1px solid #475569;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 2px dashed #475569;
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
        background: #0f172a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
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
    <div style="padding: 1rem; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: #60a5fa; margin: 0; font-size: 1.5rem;">🎛️ Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced system status with modern cards
    st.sidebar.markdown("""
    <div style="padding: 1rem; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 10px; margin-bottom: 1rem; border: 1px solid #475569;">
        <h3 style="color: #34d399; margin: 0 0 1rem 0; font-size: 1.2rem;">📊 System Status</h3>
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
    <div style="padding: 1rem; background: linear-gradient(135deg, #065f46 0%, #047857 100%); border-radius: 10px; margin-top: 1rem; border: 1px solid #059669;">
        <h4 style="color: #d1fae5; margin: 0 0 0.5rem 0;">ℹ️ About NeuroBrain</h4>
        <p style="color: #a7f3d0; font-size: 0.9rem; margin: 0;">Advanced AI system for EEG-based intention recognition. Supports motor imagery (Left/Right/Up/Down) and imagined speech (Yes/No/Start/Stop/Help) classification for immobile patients.</p>
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
