# 🌍 Spatio-Temporal Earthquake Risk Prediction Using CNN and LSTM

> AI-powered earthquake risk prediction using Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) for analyzing seismic waveform and time-series data.

---

## 📖 Overview

This project is a **Final Year B.Tech Computer Science & Engineering Project** that predicts earthquake risk by combining spatial and temporal features extracted from seismic data.

The application utilizes a hybrid **CNN-LSTM** architecture to classify earthquake risk levels and provides an interactive web interface built with **Streamlit** for visualization and prediction.

---

## ✨ Features

- 🌍 Earthquake Risk Prediction
- 🧠 Hybrid CNN-LSTM Deep Learning Model
- 📊 Seismic Waveform Analysis
- 📈 Training Curve Visualization
- 📉 Confusion Matrix Visualization
- ⚡ Streamlit Web Application
- 📂 Upload and Process Seismic Data
- 🔍 Explainable AI using SHAP
- 📑 Metadata-based Dataset Handling

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Machine Learning | Scikit-learn |
| Data Processing | NumPy, Pandas, SciPy |
| Visualization | Matplotlib |
| Explainability | SHAP |
| Web Framework | Streamlit |
| Model Storage | HDF5 (.h5) |

---

## 📁 Project Structure

```text
spatio-temporal-earthquake-risk-prediction-cnn-lstm
│
├── dataset_earthquakes/
├── dataset_noise/
├── app.py
├── main.py
├── requirements.txt
├── model.h5
├── scaler.pkl
├── create_dummy_hdf5.py
├── inspect_hdf5.py
├── confusion_matrix.png
├── training_curves.png
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Janani0511/spatio-temporal-earthquake-risk-prediction-cnn-lstm.git
```

Navigate to the project directory

```bash
cd spatio-temporal-earthquake-risk-prediction-cnn-lstm
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📊 Model Architecture

```
Seismic Data
      │
      ▼
Data Preprocessing
      │
      ▼
CNN
(Spatial Feature Extraction)
      │
      ▼
LSTM
(Temporal Feature Learning)
      │
      ▼
Dense Layers
      │
      ▼
Earthquake Risk Prediction
```

---

## 📈 Output

The application provides:

- Earthquake Risk Prediction
- Prediction Confidence
- Waveform Visualization
- Training Curves
- Confusion Matrix
- SHAP Explainability

---

## 📂 Dataset

The project uses seismic waveform and metadata for earthquake risk prediction.


---

## 🔮 Future Enhancements

- Real-time seismic data integration
- Earthquake early warning system
- Cloud deployment
- Mobile application support
- Improved deep learning models
- Interactive GIS-based visualization

---

## 👩‍💻 Author

**Janani Appanabhotla**

🎓 B.Tech Graduate in Computer Science & Engineering

- GitHub: https://github.com/Janani0511
- LinkedIn: https://www.linkedin.com/in/janani-appanabhotla-409626302

---

## 📜 License

This project is intended for educational and research purposes.





## 📜 License

This project is intended for educational and research purposes.
