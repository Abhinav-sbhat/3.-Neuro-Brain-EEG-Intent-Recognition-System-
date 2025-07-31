🧠 NeuroBrain – EEG-Based Intent Recognition System
NeuroBrain is an advanced AI-powered EEG communication system designed to decode human intentions (like “Yes”, “No”, “Emergency”, etc.) using brainwave data. This project bridges neuroscience and machine learning to enable intuitive, hands-free communication — especially beneficial for individuals with physical disabilities.

📥 Dataset – BCI Horizon 4
To run this project, download publicly available EEG data from the BCI Competition IV Dataset A (Graz Dataset A):

🔗 Download Dataset (BCI Competition IV - Dataset A)

After downloading:

Unzip the dataset.

Place any .mat file (e.g., A03T.mat) into the Dataset/ directory.

Update the path in app.py accordingly:

python
Copy
Edit
data_file = r"Dataset/A03T.mat"
🚀 Features
💬 Voice Feedback: Converts predicted user intents into speech using pyttsx3.

🧠 EEG Signal Processing: Bandpass filtering, DC offset removal, and normalization.

🔍 Feature Extraction: Windowed signal analysis to derive statistical features.

🤖 Unsupervised Learning: Uses KMeans clustering for intent classification.

📊 Visualization: Bar chart and pie chart summaries of detected intent distribution.

💾 Model Saving: Easily save and load trained models for future use.

🧰 Tech Stack
Languages: Python

Libraries: NumPy, SciPy, scikit-learn, Pyttsx3, Matplotlib, Joblib

Data Format: MATLAB .mat EEG files

ML Techniques: KMeans Clustering, Signal Preprocessing, Feature Engineering

