# NeuroBrain - EEG Intent Recognition System

## Overview

NeuroBrain is a comprehensive EEG (Electroencephalography) intent recognition system built with Streamlit. The application provides a complete Brain-Computer Interface (BCI) for motor imagery and imagined speech classification, enabling users to upload, process, analyze, and classify EEG signals in real-time. The system is designed for research and development in neurotechnology applications.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application
- **Navigation**: Multi-page application using Streamlit's page routing
- **State Management**: Session-based state management for data persistence across pages
- **Visualization**: Plotly for interactive graphs and Matplotlib for static plots

### Backend Architecture
- **Signal Processing**: Scipy-based signal processing pipeline with custom EEGSignalProcessor class
- **Machine Learning**: Scikit-learn models (Random Forest, SVM) with custom EEGClassifier wrapper
- **Feature Extraction**: Custom feature extraction pipeline supporting time-domain and frequency-domain features
- **Real-time Processing**: Event-driven processing for live EEG classification

### Data Processing Pipeline
- **Input**: Supports multiple EEG file formats (.mat, .csv, .edf)
- **Preprocessing**: Artifact removal, bandpass filtering, and signal enhancement
- **Feature Extraction**: Windowed feature extraction with configurable overlap
- **Classification**: Multi-class classification for motor imagery and speech intents

## Key Components

### Core Modules

1. **app.py**: Main application entry point with Streamlit configuration and session state management
2. **Signal Processing (`utils/signal_processing.py`)**: 
   - EEG data loading from multiple formats
   - Filtering and artifact removal
   - Signal quality assessment
3. **Feature Extraction (`utils/feature_extraction.py`)**:
   - Time-domain features (mean, std, variance, skewness, kurtosis)
   - Frequency-domain features (power spectral density, band power)
   - Windowing with configurable overlap
4. **Machine Learning (`utils/ml_models.py`)**:
   - Multiple classifier support (Random Forest, SVM)
   - Cross-validation and hyperparameter tuning
   - Intent mapping for motor imagery and speech classification
5. **Visualization (`utils/visualization.py`)**:
   - Raw signal plotting
   - Frequency domain analysis
   - Time-frequency spectrograms
   - Feature visualization

### Page Components

1. **Data Upload (`pages/data_upload.py`)**: File upload interface with validation
2. **Preprocessing (`pages/preprocessing.py`)**: Signal processing configuration and execution
3. **Training (`pages/training.py`)**: Model training interface with feature extraction
4. **Visualization (`pages/visualization.py`)**: Comprehensive visualization dashboard
5. **Real-time (`pages/real_time.py`)**: Live classification interface

## Data Flow

1. **Data Ingestion**: Users upload EEG files through the web interface
2. **Preprocessing**: Raw signals undergo filtering, artifact removal, and quality assessment
3. **Feature Extraction**: Windowed features are extracted from preprocessed signals
4. **Model Training**: Machine learning models are trained on extracted features
5. **Classification**: Trained models classify new EEG segments into intent categories
6. **Real-time Processing**: Live EEG streams are processed and classified continuously

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **NumPy/SciPy**: Numerical computing and signal processing
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static plotting

### Signal Processing
- **Scipy.signal**: Digital signal processing
- **Scipy.io**: File I/O for MATLAB files

### Machine Learning
- **Scikit-learn**: Classification algorithms and preprocessing
- **Joblib**: Model serialization

### Optional Dependencies
- **pyttsx3**: Text-to-speech functionality (referenced in attached assets)
- **MNE**: EDF file support (mentioned but not implemented)

## Deployment Strategy

### Development Environment
- **Platform**: Replit-compatible Python environment
- **Configuration**: Streamlit app with multi-page architecture
- **File Structure**: Modular organization with utilities and pages separation

### Production Considerations
- **Scalability**: Session-based state management suitable for single-user applications
- **Performance**: Optimized for real-time processing with configurable parameters
- **Memory Management**: Efficient handling of large EEG datasets through windowing

### Configuration
- **Sampling Rate**: Default 250 Hz (configurable)
- **Window Size**: Default 2 seconds with 50% overlap
- **Model Parameters**: Configurable through UI
- **File Formats**: Support for .mat, .csv, and .edf files

The system is designed as a research tool for EEG analysis and BCI development, with emphasis on ease of use, visualization, and real-time processing capabilities.