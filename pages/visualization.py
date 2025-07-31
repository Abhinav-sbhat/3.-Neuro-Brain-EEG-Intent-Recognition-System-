import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from utils.visualization import EEGVisualizer
from utils.signal_processing import EEGSignalProcessor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_visualization():
    st.header("📊 Advanced EEG Visualization & Analysis")
    
    if st.session_state.eeg_data is None:
        st.warning("⚠️ Please upload EEG data first in the Data Upload tab.")
        return
    
    visualizer = EEGVisualizer()
    processor = EEGSignalProcessor()
    
    # Visualization options
    st.subheader("Visualization Options")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        options=[
            "Raw Signal Analysis",
            "Frequency Domain Analysis", 
            "Time-Frequency Analysis",
            "Feature Analysis",
            "Model Performance",
            "EEG Topography"
        ]
    )
    
    if viz_type == "Raw Signal Analysis":
        render_raw_signal_analysis(visualizer)
    
    elif viz_type == "Frequency Domain Analysis":
        render_frequency_analysis(visualizer, processor)
    
    elif viz_type == "Time-Frequency Analysis":
        render_time_frequency_analysis(visualizer, processor)
    
    elif viz_type == "Feature Analysis":
        render_feature_analysis(visualizer)
    
    elif viz_type == "Model Performance":
        render_model_performance(visualizer)
    
    elif viz_type == "EEG Topography":
        render_topography_analysis(visualizer)

def render_raw_signal_analysis(visualizer):
    """Raw signal visualization and analysis"""
    st.subheader("Raw Signal Analysis")
    
    eeg_data = st.session_state.eeg_data
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Display Options")
        
        # Channel selection
        n_channels = eeg_data.shape[1] if eeg_data.ndim > 1 else 1
        if n_channels > 1:
            channels_to_plot = st.multiselect(
                "Select Channels",
                options=[f"Channel {i+1}" for i in range(n_channels)],
                default=[f"Channel {i+1}" for i in range(min(4, n_channels))]
            )
            
            selected_indices = [int(ch.split()[-1]) - 1 for ch in channels_to_plot]
        else:
            selected_indices = [0]
            channels_to_plot = ["Channel 1"]
        
        # Time range
        duration = eeg_data.shape[0] / 250  # Assuming 250 Hz
        
        time_start = st.slider(
            "Start Time (s)",
            min_value=0.0,
            max_value=max(0.0, duration - 1.0),
            value=0.0,
            step=0.5
        )
        
        time_duration = st.slider(
            "Display Duration (s)",
            min_value=1.0,
            max_value=min(30.0, duration - time_start),
            value=min(10.0, duration - time_start),
            step=0.5
        )
        
        # Signal properties
        show_envelope = st.checkbox("Show Signal Envelope")
        show_statistics = st.checkbox("Show Statistics", value=True)
    
    with col2:
        # Plot raw signals
        if selected_indices:
            start_sample = int(time_start * 250)
            duration_samples = int(time_duration * 250)
            end_sample = start_sample + duration_samples
            
            if eeg_data.ndim > 1:
                data_to_plot = eeg_data[start_sample:end_sample, selected_indices]
            else:
                data_to_plot = eeg_data[start_sample:end_sample].reshape(-1, 1)
            
            # Create time axis
            time_axis = np.arange(data_to_plot.shape[0]) / 250 + time_start
            
            # Create subplot figure
            n_channels_plot = len(selected_indices)
            fig = make_subplots(
                rows=n_channels_plot, 
                cols=1,
                shared_xaxes=True,
                subplot_titles=channels_to_plot,
                vertical_spacing=0.02
            )
            
            for i, ch_idx in enumerate(selected_indices):
                signal = data_to_plot[:, i] if data_to_plot.ndim > 1 else data_to_plot.flatten()
                
                # Main signal
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=signal,
                        mode='lines',
                        name=f'Ch {ch_idx + 1}',
                        line=dict(width=1, color='blue')
                    ),
                    row=i+1, col=1
                )
                
                # Signal envelope
                if show_envelope:
                    from scipy.signal import hilbert
                    analytic_signal = hilbert(signal)
                    amplitude_envelope = np.abs(analytic_signal)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=amplitude_envelope,
                            mode='lines',
                            name=f'Ch {ch_idx + 1} Envelope',
                            line=dict(width=1, color='red', dash='dash')
                        ),
                        row=i+1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=-amplitude_envelope,
                            mode='lines',
                            name=f'Ch {ch_idx + 1} -Envelope',
                            line=dict(width=1, color='red', dash='dash'),
                            showlegend=False
                        ),
                        row=i+1, col=1
                    )
            
            fig.update_layout(
                title="EEG Signal Visualization",
                height=200 * n_channels_plot,
                showlegend=False,
                template="plotly_white"
            )
            
            fig.update_xaxes(title_text="Time (s)", row=n_channels_plot, col=1)
            
            for i in range(n_channels_plot):
                fig.update_yaxes(title_text="Amplitude (µV)", row=i+1, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal statistics
            if show_statistics:
                st.markdown("### Signal Statistics")
                
                stats_data = []
                for i, ch_idx in enumerate(selected_indices):
                    signal = data_to_plot[:, i] if data_to_plot.ndim > 1 else data_to_plot.flatten()
                    
                    stats_data.append({
                        'Channel': f'Channel {ch_idx + 1}',
                        'Mean': f"{np.mean(signal):.3f}",
                        'Std': f"{np.std(signal):.3f}",
                        'Min': f"{np.min(signal):.3f}",
                        'Max': f"{np.max(signal):.3f}",
                        'RMS': f"{np.sqrt(np.mean(signal**2)):.3f}"
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

def render_frequency_analysis(visualizer, processor):
    """Frequency domain analysis"""
    st.subheader("Frequency Domain Analysis")
    
    # Use processed data if available, otherwise raw data
    data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.eeg_data
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Analysis Options")
        
        # Frequency range
        max_freq = st.slider(
            "Maximum Frequency (Hz)",
            min_value=10,
            max_value=100,
            value=50,
            step=5
        )
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            options=["Power Spectral Density", "Frequency Bands", "Both"]
        )
        
        # Window parameters
        window_length = st.slider(
            "Window Length (s)",
            min_value=1.0,
            max_value=8.0,
            value=4.0,
            step=0.5
        )
    
    with col2:
        if analysis_type in ["Power Spectral Density", "Both"]:
            # Compute and plot PSD
            nperseg = int(window_length * 250)  # Assuming 250 Hz
            freqs, psd = processor.compute_psd(data, nperseg=nperseg)
            
            psd_fig = visualizer.plot_power_spectrum(data, freqs, psd, max_freq=max_freq)
            st.plotly_chart(psd_fig, use_container_width=True)
        
        if analysis_type in ["Frequency Bands", "Both"]:
            # Compute and plot frequency bands
            band_powers = processor.extract_frequency_bands(data)
            
            if band_powers:
                bands_fig = visualizer.plot_frequency_bands(band_powers)
                st.plotly_chart(bands_fig, use_container_width=True)
                
                # Band power details
                st.markdown("### Frequency Band Powers")
                
                band_data = []
                for band, powers in band_powers.items():
                    band_data.append({
                        'Band': band.capitalize(),
                        'Mean Power': f"{np.mean(powers):.2e}",
                        'Std Power': f"{np.std(powers):.2e}",
                        'Relative Power': f"{np.mean(powers) / np.sum([np.mean(p) for p in band_powers.values()]) * 100:.1f}%"
                    })
                
                band_df = pd.DataFrame(band_data)
                st.dataframe(band_df, use_container_width=True)

def render_time_frequency_analysis(visualizer, processor):
    """Time-frequency analysis"""
    st.subheader("Time-Frequency Analysis")
    
    data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.eeg_data
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Spectrogram Options")
        
        # Channel selection for spectrogram
        n_channels = data.shape[1] if data.ndim > 1 else 1
        if n_channels > 1:
            channel_for_spec = st.selectbox(
                "Select Channel",
                options=[f"Channel {i+1}" for i in range(n_channels)],
                index=0
            )
            ch_idx = int(channel_for_spec.split()[-1]) - 1
        else:
            ch_idx = 0
        
        # Spectrogram parameters
        nperseg = st.slider(
            "Window Size (samples)",
            min_value=64,
            max_value=512,
            value=256,
            step=64
        )
        
        max_freq_spec = st.slider(
            "Max Frequency (Hz)",
            min_value=20,
            max_value=100,
            value=50,
            step=5
        )
    
    with col2:
        # Compute spectrogram
        if data.ndim > 1:
            signal_for_spec = data[:, ch_idx]
        else:
            signal_for_spec = data
        
        freqs, times, Sxx = processor.compute_spectrogram(signal_for_spec, nperseg=nperseg)
        
        # Plot spectrogram
        spec_fig = visualizer.plot_spectrogram(freqs, times, Sxx, max_freq=max_freq_spec)
        st.plotly_chart(spec_fig, use_container_width=True)
        
        # Time-frequency statistics
        st.markdown(f"### Spectrogram Statistics ({channel_for_spec if n_channels > 1 else 'EEG Signal'})")
        
        # Find peak time-frequency
        freq_mask = freqs <= max_freq_spec
        Sxx_masked = Sxx[freq_mask, :]
        freqs_masked = freqs[freq_mask]
        
        peak_idx = np.unravel_index(np.argmax(Sxx_masked), Sxx_masked.shape)
        peak_freq = freqs_masked[peak_idx[0]]
        peak_time = times[peak_idx[1]]
        peak_power = Sxx_masked[peak_idx]
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Peak Frequency", f"{peak_freq:.1f} Hz")
        
        with col_b:
            st.metric("Peak Time", f"{peak_time:.1f} s")
        
        with col_c:
            st.metric("Peak Power", f"{10 * np.log10(peak_power):.1f} dB")

def render_feature_analysis(visualizer):
    """Feature analysis and visualization"""
    st.subheader("Feature Analysis")
    
    if st.session_state.features is None:
        st.warning("⚠️ Please extract features first in the Model Training tab.")
        return
    
    features = st.session_state.features
    feature_names = st.session_state.feature_names
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Feature Selection")
        
        # Feature category selection
        time_features = [name for name in feature_names if any(stat in name for stat in ['mean', 'std', 'var', 'median', 'ptp', 'skew', 'kurtosis', 'energy', 'zcr', 'mobility', 'complexity'])]
        freq_features = [name for name in feature_names if any(band in name for band in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'peak_freq', 'spectral'])]
        
        feature_category = st.selectbox(
            "Feature Category",
            options=["All Features", "Time Domain", "Frequency Domain"]
        )
        
        if feature_category == "Time Domain":
            available_features = time_features
        elif feature_category == "Frequency Domain":
            available_features = freq_features
        else:
            available_features = feature_names
        
        selected_features = st.multiselect(
            "Select Features to Analyze",
            options=available_features,
            default=available_features[:min(6, len(available_features))]
        )
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            options=["Feature Distribution", "Feature Correlation", "Feature Importance"]
        )
    
    with col2:
        if selected_features:
            if analysis_type == "Feature Distribution":
                # Feature distribution plot
                fig = go.Figure()
                
                for feature in selected_features:
                    feature_idx = feature_names.index(feature)
                    feature_values = features[:, feature_idx]
                    
                    fig.add_trace(go.Box(
                        y=feature_values,
                        name=feature,
                        boxpoints='outliers'
                    ))
                
                fig.update_layout(
                    title="Feature Value Distributions",
                    yaxis_title="Feature Value",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature statistics table
                st.markdown("### Feature Statistics")
                
                stats_data = []
                for feature in selected_features:
                    feature_idx = feature_names.index(feature)
                    feature_values = features[:, feature_idx]
                    
                    stats_data.append({
                        'Feature': feature,
                        'Mean': f"{np.mean(feature_values):.3f}",
                        'Std': f"{np.std(feature_values):.3f}",
                        'Min': f"{np.min(feature_values):.3f}",
                        'Max': f"{np.max(feature_values):.3f}",
                        'Skewness': f"{scipy_stats.skew(feature_values):.3f}"
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
            
            elif analysis_type == "Feature Correlation":
                # Feature correlation heatmap
                import plotly.express as px
                
                feature_indices = [feature_names.index(f) for f in selected_features]
                selected_feature_data = features[:, feature_indices]
                
                correlation_matrix = np.corrcoef(selected_feature_data.T)
                
                fig = px.imshow(
                    correlation_matrix,
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=selected_features,
                    y=selected_features,
                    color_continuous_scale='RdBu',
                    aspect="auto"
                )
                
                fig.update_layout(
                    title="Feature Correlation Matrix",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Feature Importance":
                # Feature importance from trained model
                if st.session_state.trained_model is not None:
                    importance = st.session_state.trained_model.get_feature_importance()
                    
                    if importance is not None:
                        # Filter importance for selected features
                        filtered_importance = {f: importance.get(f, 0) for f in selected_features if f in importance}
                        
                        if filtered_importance:
                            importance_fig = visualizer.plot_feature_importance(filtered_importance, top_n=len(filtered_importance))
                            st.plotly_chart(importance_fig, use_container_width=True)
                        else:
                            st.warning("No importance data available for selected features.")
                    else:
                        st.warning("Feature importance not available for this model type.")
                else:
                    st.warning("Train a model first to see feature importance.")

def render_model_performance(visualizer):
    """Model performance visualization"""
    st.subheader("Model Performance Analysis")
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first in the Model Training tab.")
        return
    
    if st.session_state.training_results is None:
        st.warning("⚠️ No training results available.")
        return
    
    results = st.session_state.training_results
    
    # Performance metrics overview
    st.markdown("### Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Accuracy", f"{results['train_accuracy']:.3f}")
    
    with col2:
        st.metric("Test Accuracy", f"{results['test_accuracy']:.3f}")
    
    with col3:
        st.metric("CV Mean", f"{results['cv_mean']:.3f}")
    
    with col4:
        st.metric("CV Std", f"±{results['cv_std']:.3f}")
    
    # Confusion matrix
    if results['confusion_matrix'] is not None:
        st.markdown("### Confusion Matrix")
        
        classifier = st.session_state.trained_model
        intent_type = getattr(st.session_state, 'intent_type', 'motor_imagery')
        
        if intent_type == 'motor_imagery':
            labels = list(classifier.motor_imagery_intents.values())
        else:
            labels = list(classifier.speech_intents.values())
        
        cm = results['confusion_matrix']
        labels_subset = labels[:cm.shape[0]]  # Match matrix size
        
        cm_fig = visualizer.plot_confusion_matrix(cm, labels_subset)
        st.plotly_chart(cm_fig, use_container_width=True)
    
    # Feature importance
    if results['feature_importance'] is not None:
        st.markdown("### Feature Importance")
        
        importance_fig = visualizer.plot_feature_importance(results['feature_importance'])
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
    
    # Real-time prediction performance
    if st.session_state.real_time_predictions:
        st.markdown("### Real-time Performance")
        
        predictions = st.session_state.real_time_predictions
        
        # Confidence timeline
        confidence_fig = visualizer.plot_prediction_confidence(predictions)
        st.plotly_chart(confidence_fig, use_container_width=True)
        
        # Performance statistics
        reliable_count = sum(1 for p in predictions if p['reliable'])
        total_count = len(predictions)
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", total_count)
        
        with col2:
            reliability_rate = (reliable_count / total_count) * 100 if total_count > 0 else 0
            st.metric("Reliability Rate", f"{reliability_rate:.1f}%")
        
        with col3:
            st.metric("Average Confidence", f"{avg_confidence:.3f}")

def render_topography_analysis(visualizer):
    """EEG topography visualization"""
    st.subheader("EEG Topography")
    
    data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.eeg_data
    
    if data is None:
        st.warning("⚠️ No EEG data available.")
        return
    
    n_channels = data.shape[1] if data.ndim > 1 else 1
    
    if n_channels == 1:
        st.info("Topography visualization requires multiple channels.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Topography Options")
        
        # Time point selection
        duration = data.shape[0] / 250
        time_point = st.slider(
            "Time Point (s)",
            min_value=0.0,
            max_value=duration,
            value=duration / 2,
            step=0.1
        )
        
        # Analysis type
        topo_type = st.selectbox(
            "Analysis Type",
            options=["Instantaneous Amplitude", "Band Power", "RMS Power"]
        )
        
        if topo_type == "Band Power":
            frequency_band = st.selectbox(
                "Frequency Band",
                options=["Delta (0.5-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-12 Hz)", "Beta (13-30 Hz)", "Gamma (30-50 Hz)"]
            )
    
    with col2:
        sample_idx = int(time_point * 250)
        
        if topo_type == "Instantaneous Amplitude":
            # Use instantaneous amplitude at selected time point
            if sample_idx < data.shape[0]:
                values = data[sample_idx, :]
            else:
                values = data[-1, :]
        
        elif topo_type == "Band Power":
            # Extract band power for each channel
            band_map = {
                "Delta (0.5-4 Hz)": (0.5, 4),
                "Theta (4-8 Hz)": (4, 8),
                "Alpha (8-12 Hz)": (8, 12),
                "Beta (13-30 Hz)": (13, 30),
                "Gamma (30-50 Hz)": (30, 50)
            }
            
            low, high = band_map[frequency_band]
            processor = EEGSignalProcessor()
            filtered_data = processor.apply_bandpass_filter(data, low_freq=low, high_freq=high)
            values = np.mean(filtered_data**2, axis=0)
        
        elif topo_type == "RMS Power":
            # RMS power for each channel
            values = np.sqrt(np.mean(data**2, axis=0))
        
        # Create topography plot
        topo_fig = visualizer.plot_eeg_topography(values)
        st.plotly_chart(topo_fig, use_container_width=True)
        
        # Channel values table
        st.markdown("### Channel Values")
        
        topo_data = []
        for i, value in enumerate(values):
            topo_data.append({
                'Channel': f'Channel {i+1}',
                'Value': f"{value:.3f}",
                'Normalized': f"{(value - np.min(values)) / (np.max(values) - np.min(values)):.3f}"
            })
        
        topo_df = pd.DataFrame(topo_data)
        st.dataframe(topo_df, use_container_width=True)
