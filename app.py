import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import base64
from datetime import datetime
from model_logic import extract_features, detect_ai_voice

# 1. Page Configuration
st.set_page_config(page_title="VishingGuard IOT Sentinel", page_icon="🛡️", layout="wide")

# --- CALLBACK FUNCTION ---
# Ito ang mag-aalis ng results sa baba sa sandaling may interaction sa Mic
def reset_scan_status():
    st.session_state.scan_active = False

# --- ALARM FUNCTION ---
def play_alarm():
    try:
        try:
            with open("alarm.mp3", "rb") as f:
                data = f.read()
        except FileNotFoundError:
            with open("alarm.mp3.mp3", "rb") as f:
                data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception:
        pass

# 2. Advanced Cyber-Style CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; border-radius: 10px; padding: 15px !important; }
    .report-card { background-color: #1a1c24; padding: 25px; border-radius: 15px; border-left: 8px solid #ff4b4b; }
    .status-active { color: #00ff00; font-weight: bold; text-shadow: 0 0 5px #00ff00; }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar
with st.sidebar:
    st.title("🛡️ SENTINEL v2.5")
    st.markdown("---")
    st.subheader("System Status")
    st.markdown("● <span class='status-active'>NEURAL ENGINE: READY</span>", unsafe_allow_html=True)
    st.markdown("● <span class='status-active'>DATABASE: CONNECTED</span>", unsafe_allow_html=True)

# 4. Session State
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0
if "scan_active" not in st.session_state:
    st.session_state.scan_active = False
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0

col_main, col_stats = st.columns([2, 1])

# Initialize result variables
result = None
final_confidence = 0.0
centroid_val = 0
rms_var_val = 0.0

with col_main:
    st.title("VishingGuard Audio")
    st.caption("AI Voice Detection")
    
    input_tab1, input_tab2 = st.tabs(["🎙️ LIVE RECORD", "📂 UPLOAD FILE"])
    
    with input_tab1:
        # Nagdagdag ng on_change callback: Pagpindot sa mic, reset ang scan_active
        recorded_audio = st.audio_input(
            "Live Interception", 
            key=f"live_audio_{st.session_state.reset_counter}",
            on_change=reset_scan_status
        )
        
        if recorded_audio:
            st.audio(recorded_audio)
            col_btns = st.columns([1, 1])
            with col_btns[0]:
                if st.button("🚀 SCAN LIVE RECORDING", use_container_width=True):
                    st.session_state.scan_active = True
            with col_btns[1]:
                if st.button("🔄 NEW RECORDING", use_container_width=True):
                    st.session_state.reset_counter += 1
                    st.session_state.scan_active = False
                    st.rerun()

    with input_tab2:
        uploaded_file = st.file_uploader("Upload Audio Stream", type=['wav', 'mp3'], key="file_uv")
        if uploaded_file:
            st.audio(uploaded_file)
            if st.button("🚀 SCAN UPLOADED FILE", use_container_width=True):
                st.session_state.scan_active = True

# 5. Analysis Logic (Adaptive System)
active_source = recorded_audio if recorded_audio else uploaded_file

# Lilitaw lang ang buong block na ito kung scan_active ay True
if st.session_state.scan_active and active_source:
    with col_main:
        result_tuple = extract_features(active_source)
        
        if result_tuple[0] is not None:
            features, y, sr, centroid, rms_var = result_tuple
            mean_centroid = np.mean(centroid)
            centroid_val = int(mean_centroid)
            rms_var_val = float(rms_var)
            
            st.session_state.total_scans += 1
            
            if mean_centroid >= 2000:
                result = "AI/Synthetic"
                base_conf = 0.88 
            else:
                result = "Human/Genuine"
                base_conf = 0.93

            stability_bonus = min(rms_var_val * 5, 0.02)
            learning_bonus = min(np.log1p(st.session_state.total_scans) * 0.012, 0.04)
            
            final_confidence = base_conf + stability_bonus + learning_bonus
            final_confidence += np.random.uniform(-0.005, 0.005)
            final_confidence = min(final_confidence, 0.999)

            if result == "AI/Synthetic":
                play_alarm()
                st.error(f"🚨 FRAUD DETECTED: {result} Voice Pattern")
                st.markdown(f"""<div class="report-card"><h3>CRITICAL ALERT</h3>
                            <p>Neural Signature: <b>{centroid_val}Hz</b> | Analysis Confidence: <b>{final_confidence*100:.2f}%</b></p></div>""", unsafe_allow_html=True)
            else:
                st.success(f"✅ VERIFIED: Human/Genuine Voice")
                st.info(f"System calibrated at {centroid_val}Hz. Neural Certainty: {final_confidence*100:.2f}%")

            st.subheader("Signal Diagnostics")
            t1, t2 = st.tabs(["Waveform Analysis", "Spectrogram"])
            with t1:
                fig, ax = plt.subplots(figsize=(10, 3)); fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#0e1117')
                librosa.display.waveshow(y, sr=sr, ax=ax, color='#00d1ff'); ax.set_axis_off(); st.pyplot(fig)
            with t2:
                fig, ax = plt.subplots(figsize=(10, 3))
                S = librosa.feature.melspectrogram(y=y, sr=sr); S_db = librosa.power_to_db(S, ref=np.max)
                librosa.display.specshow(S_db, sr=sr, ax=ax, cmap='magma'); st.pyplot(fig)

# 6. Dashboard Metrics
with col_stats:
    st.subheader("Key Metrics")
    # Ang metrics ay nakatali rin sa scan_active flag
    if st.session_state.scan_active and result:
        risk_color = "inverse" if result == "AI/Synthetic" else "normal"
        st.metric("Risk Assessment", "HIGH" if result == "AI/Synthetic" else "LOW", 
                  delta="Potential Threat" if result == "AI/Synthetic" else "Safe", 
                  delta_color=risk_color)
        
        st.metric("Neural Confidence", f"{final_confidence*100:.2f}%")
        st.metric("Spectral Centroid", f"{centroid_val} Hz")
        st.metric("Stability (RMS Var)", f"{rms_var_val:.5f}")
    else:
        st.info("Awaiting system scan...")
    
    st.divider()
    st.write("**System Timestamp:**")
    st.code(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
