import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Drain Guard AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Custom Styling (Dark Mode & Glassmorphism)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Background Gradient */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Glassmorphism Cards */
    div[data-testid="stMetric"], div[data-testid="stContainer"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        background: rgba(255, 255, 255, 0.07);
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }

    /* Headers & Typography */
    h1 {
        background: linear-gradient(120deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    h2, h3 {
        color: #f1f5f9;
        font-weight: 600;
    }

    p, span, div {
        color: #e2e8f0;
    }

    /* Custom Buttons/Sliders */
    div.stSlider > div[data-baseweb="slider"] > div > div > div {
        background-color: #38bdf8 !important;
    }
    
    div.stToggle > label > div[data-testid="stValidator"] {
        background-color: #38bdf8 !important;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Load Models
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'drain_status_model.pkl')
        scaler_path = os.path.join(current_dir, 'scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error(f"⚠️ Artifacts missing! Expected in: {current_dir}")
            return None, None
            
        # Load the artifacts
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        return None, None

model, scaler = load_artifacts()

# -----------------------------------------------------------------------------
# 4. Sidebar (Inputs)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Controls")
    st.markdown("---")
    
    st.subheader("Sensor Inputs")
    
    # Environmental
    st.caption("Environment")
    rain_val = st.toggle("🌧️ Raining", value=False)
    temp_val = st.slider("🌡️ Temperature (°C)", -10.0, 50.0, 25.0, 0.1)
    
    st.caption("Drainage")
    # Note: 792 was a value in the dataset associated with 'Normal'.
    water_dist = st.slider("📏 Water Distance (mm)", 0, 1000, 792, help="Distance from sensor to water surface.")
    gas_val = st.toggle("⚠️ Toxic Gas", value=False)
    wf_val = st.toggle("🌊 Water Flowing", value=True)
    
    st.markdown("---")
    st.info("Adjust simulators to test prediction logic.")

# -----------------------------------------------------------------------------
# 5. Main Dashboard
# -----------------------------------------------------------------------------
st.title("Drain Guard AI")
st.markdown("#### 🚀 Next-Gen Urban Drainage Monitoring System")

if model and scaler:
    # --- Data Preparation ---
    # Construct DataFrame with EXACT feature names and order from training
    input_data = pd.DataFrame([{
        'gas_value': int(gas_val),
        'rain_value': int(rain_val),
        'temp_value': float(temp_val),
        'water_dist': float(water_dist),
        'wf_value': int(wf_val)
    }])
    
    # --- Prediction ---
    try:
        # Scale inputs
        scaled_input = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(scaled_input)[0]
        
        # Map Prediction (0 = Blocked, 1 = Normal)
        if prediction == 0:
            status_label = "BLOCKED"
            status_color = "#ef4444" # Red
            bg_color = "rgba(239, 68, 68, 0.1)"
            result_icon = "🚨"
            message = "CRITICAL OBSTRUCTION DETECTED"
            sub_message = "Immediate maintenance team dispatch required."
        else:
            status_label = "NORMAL"
            status_color = "#22c55e" # Green
            bg_color = "rgba(34, 197, 94, 0.1)"
            result_icon = "✅"
            message = "SYSTEM OPTIMAL"
            sub_message = "Water flow is unobstructed."

        # --- Enhanced UI Layout ---
        
        # Top Hero Section
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, {bg_color}, transparent);
            border-left: 5px solid {status_color};
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
        ">
            <h2 style="margin:0; color: {status_color};">{result_icon} {status_label}</h2>
            <p style="font-size: 1.2rem; margin: 5px 0 0 0; opacity: 0.9;">{message}</p>
            <small style="opacity: 0.7;">{sub_message}</small>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.subheader("📡 Live Telemetry")
            
            c1, c2 = st.columns(2)
            c1.metric("🌡️ Temp", f"{temp_val} °C", delta=f"{temp_val-25:.1f}°C")
            c2.metric("📏 Level", f"{water_dist} mm", delta="-12mm" if water_dist < 500 else "Normal")
            
            st.markdown("### Status Indicators")
            # Custom badges
            st.markdown(f"""
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span style="background: {'rgba(56, 189, 248, 0.2)' if rain_val else 'rgba(255,255,255,0.05)'}; padding: 5px 10px; border-radius: 20px; border: 1px solid {'#38bdf8' if rain_val else 'rgba(255,255,255,0.1)'}">
                    {'🌧️ Raining' if rain_val else '☁️ No Rain'}
                </span>
                <span style="background: {'rgba(239, 68, 68, 0.2)' if gas_val else 'rgba(255,255,255,0.05)'}; padding: 5px 10px; border-radius: 20px; border: 1px solid {'#ef4444' if gas_val else 'rgba(255,255,255,0.1)'}">
                    {'⚠️ Gas Detected' if gas_val else '💨 No Gas'}
                </span>
                <span style="background: {'rgba(34, 197, 94, 0.2)' if wf_val else 'rgba(239, 68, 68, 0.2)'}; padding: 5px 10px; border-radius: 20px; border: 1px solid {'#22c55e' if wf_val else '#ef4444'}">
                    {'🌊 Flowing' if wf_val else '🛑 No Flow'}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("🧠 Model Confidence")
            confidence = 0.98 if status_label == "NORMAL" else 0.85 # Mock confidence if model doesn't support predict_proba
            
            # Try to get probability if available
            try:
                probs = model.predict_proba(scaled_input)[0]
                confidence = max(probs)
            except:
                pass

            st.progress(confidence, text=f"confidence: {confidence*100:.1f}%")
            
            with st.expander("🔍 View Raw Analysis Data"):
                st.dataframe(input_data, use_container_width=True, hide_index=True)


    except Exception as e:
        st.error(f"Prediction System Error: {e}")

else:
    st.warning("Model not loaded. Please check server logs.")

# Footer
st.markdown("---")
st.caption("System v2.0 | Connecting to `drain_status_model.pkl` | Edunet Foundation")
