# 🌊 Drain Guard AI

**Next-Gen Urban Drainage Monitoring System**

Drain Guard AI is a smart, real-time monitoring dashboard designed to predict and prevent drain blockages in urban environments. Powered by Machine Learning, it analyzes sensor data to detect critical obstructions and ensure optimal water flow.

## 🚀 Features

- **Real-Time Prediction**: Instantly classifies drain status as `NORMAL` or `BLOCKED` based on sensor inputs.
- **Smart Sensor Simulation**:
  - 🌧️ Rain Detection
  - 🌡️ Temperature Monitoring
  - 📏 Water Level / Distance
  - ⚠️ Toxic Gas Detection
  - 🌊 Water Flow Status
- **Modern UI**: Features a high-contrast Dark Mode with Glassmorphism elements for a premium user experience.
- **Critical Alerts**: Visual and textual warnings for immediate maintenance actions.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/drain-guard-ai.git
   cd drain-guard-ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run frontend.py
   ```

## 📂 Project Structure

- `frontend.py`: The main application dashboard.
- `drain_status_model.pkl`: Pre-trained Machine Learning model for status prediction.
- `scaler.pkl`: Data scaler to normalize inputs for the model.
- `requirements.txt`: List of Python dependencies.
- `drain_prediction-1.ipynb`: (Optional) Jupyter notebook used for model training.

## 🧠 Model Info

The system uses a Machine Learning classifier trained on environmental data to predict blockage risks. It processes inputs such as water distance, temperature, and flow sensors to determine if maintenance is required.

---

*Developed by Edunet Foundation*
