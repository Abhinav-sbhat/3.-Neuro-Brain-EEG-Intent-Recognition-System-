import numpy as np
import scipy.io
from scipy.signal import butter, lfilter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import os
import pyttsx3  # For text-to-speech
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuration
FS = 250  # Hz
WIN_SIZE = FS * 2  # 2-second windows (500 samples)
STEP_SIZE = FS // 2  # 50% overlap (125 samples)

# Enhanced intent mapping
INTENT_MAP = {
    0: "Yes",
    1: "No", 
    2: "Emergency",
    3: "Hungry",
    4: "I'm fine",
    5: "Help",
    6: "Thank you"
}

class EEGIntentSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=len(INTENT_MAP), random_state=42)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Slower speech rate
        
    def speak(self, text):
        """Convert text to speech"""
        print(f"SYSTEM: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        
    def load_data(self, filepath):
        """Load and validate EEG data with voice feedback"""
        filename = os.path.basename(filepath)
        self.speak(f"Loading EEG data from file {filename}")
        
        try:
            mat = scipy.io.loadmat(filepath)
            
            # Handle different MATLAB structures
            if 'data' in mat:
                data_struct = mat['data'][0,0]
                eeg_data = data_struct['X'][0]
                
                if isinstance(eeg_data, np.ndarray) and eeg_data.dtype == object:
                    eeg_data = np.vstack([ch.reshape(-1, 1) for ch in eeg_data])
                else:
                    eeg_data = eeg_data.reshape(-1, 1)
            else:
                for key in mat.keys():
                    if not key.startswith('__') and isinstance(mat[key], np.ndarray):
                        eeg_data = mat[key]
                        if eeg_data.shape[0] < eeg_data.shape[1]:
                            eeg_data = eeg_data.T
                        break
                else:
                    raise ValueError("No EEG data array found")
            
            print(f"Loaded EEG data: {eeg_data.shape[0]} samples, {eeg_data.shape[1]} channels")
            self.speak(f"Successfully loaded {eeg_data.shape[0]} samples with {eeg_data.shape[1]} channels")
            return eeg_data
            
        except Exception as e:
            self.speak("Error loading data file. Please check the file format.")
            raise ValueError(f"Data loading failed: {str(e)}")

    def preprocess(self, eeg_data):
        """Preprocessing with voice feedback"""
        self.speak("Starting data preprocessing")
        
        # Remove DC offset
        processed = eeg_data - np.mean(eeg_data, axis=0)
        
        # Bandpass filter if enough data
        if len(processed) > 3 * WIN_SIZE:
            try:
                b, a = butter(5, [8/(FS/2), 30/(FS/2)], btype='band')
                processed = lfilter(b, a, processed, axis=0)
                self.speak("Applied bandpass filter from 8 to 30 Hertz")
            except Exception as e:
                self.speak("Warning: Could not apply filter. Using unfiltered data.")
        
        # Normalize
        processed = self.scaler.fit_transform(processed)
        self.speak("Data normalization complete")
        return processed

    def extract_features(self, data):
        """Feature extraction with progress feedback"""
        n_samples = data.shape[0]
        n_windows = max(1, (n_samples - WIN_SIZE) // STEP_SIZE + 1)
        features = np.zeros((n_windows, 5))
        
        self.speak(f"Extracting features from {n_windows} time windows")
        
        for i in range(n_windows):
            start = i * STEP_SIZE
            window = data[start:start + WIN_SIZE]
            
            features[i, 0] = np.mean(window)
            features[i, 1] = np.std(window)
            features[i, 2] = np.median(window)
            features[i, 3] = np.ptp(window)
            features[i, 4] = np.sum(np.abs(np.diff(window)))
        
        return features

    def train(self, features):
        """Training with voice feedback"""
        self.speak(f"Training intent classifier with {len(features)} feature vectors")
        
        if len(features) < len(INTENT_MAP):
            self.speak("Error: Not enough data for training")
            raise ValueError(f"Need at least {len(INTENT_MAP)} feature vectors")
            
        self.model.fit(features)
        
        # Ensure balanced clusters
        while True:
            counts = Counter(self.model.labels_)
            if len(counts) == len(INTENT_MAP) and min(counts.values()) > 5:
                break
            self.model = KMeans(n_clusters=len(INTENT_MAP), 
                              random_state=np.random.randint(100))
            self.model.fit(features)
        
        self.speak("Training completed successfully")
        return self.model.labels_

    def visualize_results(self, counts):
        """Enhanced visualization with voice description"""
        plt.figure(figsize=(12, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        plt.bar(INTENT_MAP.values(), [counts[i] for i in range(len(INTENT_MAP))])
        plt.title('Detected Intent Distribution', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Number of Detections')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie([counts[i] for i in range(len(INTENT_MAP))], 
                labels=INTENT_MAP.values(),
                autopct='%1.1f%%')
        plt.title('Intent Proportions')
        
        plt.tight_layout()
        
        # Save figure with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"intent_results_{timestamp}.png"
        plt.savefig(plot_file)
        plt.show()
        
        self.speak(f"Results visualization saved as {plot_file}")

    def save(self, filename):
        """Save model with voice confirmation"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'config': {
                'fs': FS,
                'win_size': WIN_SIZE,
                'step_size': STEP_SIZE
            }
        }, filename)
        self.speak(f"Model successfully saved to {filename}")

    def predict(self, eeg_window):
        """Predict with voice output"""
        if len(eeg_window) < WIN_SIZE:
            self.speak("Error: Insufficient data for prediction")
            raise ValueError(f"Need {WIN_SIZE} samples, got {len(eeg_window)}")
            
        processed = self.preprocess(eeg_window.reshape(-1, 1))
        features = self.extract_features(processed)
        intent = INTENT_MAP[self.model.predict(features)[0]]
        
        self.speak(f"Detected intent: {intent}")
        return intent

def main():
    print("=== Enhanced EEG Communication System ===")
    system = EEGIntentSystem()
    
    try:
        # Get and announce current time
        current_time = datetime.now().strftime("%H:%M")
        system.speak(f"Starting system at {current_time}")
        
        # Load data
        data_file = r"C:\Users\Abhinav S  Bhat\OneDrive\Desktop\Neuro Brain\Dataset\A03E.mat"
        system.speak(f"Processing data file: {os.path.basename(data_file)}")
        eeg_data = system.load_data(data_file)
        
        # Preprocess
        processed = system.preprocess(eeg_data)
        
        # Feature extraction
        features = system.extract_features(processed)
        
        # Train
        labels = system.train(features)
        
        # Results
        counts = Counter(labels)
        system.speak("Training results:")
        for cl in sorted(counts.keys()):
            system.speak(f"{INTENT_MAP[cl]}: {counts[cl]} detections")
        
        # Visualization
        system.visualize_results(counts)
        
        # Save model
        model_file = "eeg_communication_system.pkl"
        system.save(model_file)
        
        # Test prediction
        system.speak("Running test prediction")
        test_pred = system.predict(processed[:WIN_SIZE])
        print(f"\nTest prediction: {test_pred}")
        
        # System summary
        system.speak("System ready for real-time use. You may now connect your EEG headset.")
        
    except Exception as e:
        system.speak("An error occurred during processing")
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify your .mat file contains proper EEG data")
        print("2. Ensure the file has more rows (samples) than columns (channels)")
        print(f"3. Try reducing WIN_SIZE (current: {WIN_SIZE} samples)")
        print("4. Check for NaN/infinite values in your data")

if __name__ == "__main__":
    main()