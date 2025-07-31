import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Configure page
st.set_page_config(
    page_title="NeuroBrain - EEG Intent Recognition",
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
    # Header
    st.title("🧠 NeuroBrain - EEG Intent Recognition System")
    st.markdown("### Advanced Brain-Computer Interface for Motor Imagery and Imagined Speech Classification")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    data_status = "✅ Loaded" if st.session_state.eeg_data is not None else "❌ No Data"
    processed_status = "✅ Processed" if st.session_state.processed_data is not None else "❌ Not Processed"
    model_status = "✅ Trained" if st.session_state.trained_model is not None else "❌ Not Trained"
    
    st.sidebar.markdown(f"**Data:** {data_status}")
    st.sidebar.markdown(f"**Processing:** {processed_status}")
    st.sidebar.markdown(f"**Model:** {model_status}")
    
    # Navigation tabs
    st.sidebar.markdown("---")
    
    # Main tabs
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
