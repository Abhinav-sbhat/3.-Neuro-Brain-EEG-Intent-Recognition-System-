# 🧠 NeuroBrain: A Real-Time EEG-Based Intention Recognition System for Immobile Patients Using AI

## 🔍 Overview
**NeuroBrain** is a real-time Brain-Computer Interface (BCI) system that utilizes EEG (Electroencephalography) signals to detect and classify human intentions such as motor imagery and imagined speech. The system is designed especially for **immobile or paralyzed individuals**, enabling them to communicate or control devices using only their brain activity.

📥 Dataset – BCI Horizon 4
To run this project, download publicly available EEG data from the BCI Competition IV Dataset A (Graz Dataset A):

🔗 Download Dataset (BCI Competition IV - Dataset A)

After downloading:

Unzip the dataset.

## 🎯 Key Features
- ✅ Real-time EEG signal acquisition and classification
- 🧠 Supports motor imagery and imagined speech intent detection
- 🧼 Preprocessing pipeline: artifact removal, bandpass filtering
- 🧪 Feature extraction: time-domain & frequency-domain features
- 🧠 Machine learning: SVM, Random Forest, PSO-LDA, etc.
- 📈 Visual analytics using Plotly & Matplotlib
- 💾 Supports `.mat`, `.csv`, `.edf` EEG file formats
- 🧩 Modular and extensible architecture for BCI research
- 🌐 Built using Streamlit for interactive web-based usage

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python (NumPy, Pandas, SciPy, Scikit-learn)
- **Visualization:** Plotly, Matplotlib
- **ML Models:** Random Forest, SVM, PSO-LDA
- **EEG Formats Supported:** `.mat`, `.csv`, `.edf`

## 💡 Use Cases
- Assistive communication for **paralyzed or coma patients**
- Real-time intent detection for **smart prosthetics**
- Hands-free control for **smart home devices**
- Research in **BCI, neurofeedback, and cognitive computing**

## 🧪 How to Run
```bash
# Clone the repo
git clone https://github.com/Abhinav-sbhat/3.-Neuro-Brain-EEG-Intent-Recognition-System.git
cd NeuroBrain

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py



