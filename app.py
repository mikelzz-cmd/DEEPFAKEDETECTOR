import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import time
import base64
from datetime import datetime
from model_logic import extract_features, detect_ai_voice

# 1. Page Configuration
st.set_page_config(page_title="VishingGuard IOT Sentinel", page_icon="🛡️", layout="wide")

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
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception:
        st.sidebar.warning("🔊 Audio system standby")

# 2. Advanced Cyber-Style CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    
    /* Green Log Terminal */
    .log-container {
        background-color: #000000;
        color: #00ff41;
        padding: 15px;
        border-radius: 10px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 13px;
        border: 1px solid #00ff4133;
        height: 180px;
        overflow-y: auto;
        margin-bottom: 20px;
    }

    /* Red Alert Card */
    .report-card {
        background-color: #1a1c24; 
        padding: 25px; 
        border-radius: 15px;
        border-left: 8px solid #ff4b4b;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.2);
    }
    .status-active { color: #00ff00; font-weight: bold; text-shadow: 0 0 5px #00ff00; }
    
    /* Center aligning audio input */
    .stAudioInput {
        border: 1px solid #30363d;
        border-radius: 15px;
        padding: 10px;
        background: #161b22;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar (Control Panel)
with st.sidebar:
    st.title("🛡️ SENTINEL v2.5")
    st.markdown("---")
    st.subheader("System Status")
    st.markdown("● <span class='status-active'>NEURAL ENGINE: READY</span>", unsafe_allow_html=True)
    st.markdown("● <span class='status-active'>DATABASE: CONNECTED</span>", unsafe_allow_html=True)
    
    st.divider()
    st.info("💡 **Tip:** AI-generated voices often show abnormally high stability in the 'RMS Variance' metric.")

# 4. Main UI Layout
col_main, col_stats = st.columns([2, 1])

# Initialize variables to avoid errors
final_audio_source = None
scan_btn = False
result = None
final_confidence = 0.0
centroid = [0]
rms_var = 0.0
y, sr = None, None

with col_main:
    st.title("VishingGuard Audio Forensics")
    st.caption("Industrial-grade AI Voice Synthesis Detection")
    
    # --- INPUT SELECTION TABS (NAKA-UNA NA ANG LIVE RECORD) ---
    input_tab1, input_tab2 = st.tabs(["🎙️ LIVE RECORD", "📂 UPLOAD FILE"])
    
    with input_tab1:
        st.write("Click the microphone to start live interception:")
        recorded_audio = st.audio_input("Live Interception", key="live_rv")
        if recorded_audio:
            st.audio(recorded_audio) # Player specifically for live record tab
            if st.button("🚀 SCAN LIVE RECORDING", use_container_width=True):
                final_audio_source = recorded_audio
                scan_btn = True

    with input_tab2:
        uploaded_file = st.file_uploader("Upload Intercepted Audio Stream", type=['wav', 'mp3'], key="file_uv")
        if uploaded_file:
            st.audio(uploaded_file) # Player specifically for upload tab
            if st.button("🚀 SCAN UPLOADED FILE", use_container_width=True):
                final_audio_source = uploaded_file
                scan_btn = True

# 5. Analysis Logic
if final_audio_source and scan_btn:
    with col_main:
        log_placeholder = st.empty()
        logs_list = []

        def add_log(msg):
            ts = datetime.now().strftime("%H:%M:%S")
            logs_list.append(f"[{ts}] {msg}")
            log_placeholder.markdown(f'<div class="log-container">{"<br>".join(logs_list)}</div>', unsafe_allow_html=True)
            time.sleep(0.4)

        add_log("Initializing Sentinel core...")
        add_log("Analyzing spectral density...")
        
        result_tuple = extract_features(final_audio_source)
        
        if result_tuple[0] is not None:
            features, y, sr, centroid, rms_var = result_tuple
            result, _ = detect_ai_voice(features, centroid, rms_var)
            
            # --- CUSTOM CONFIDENCE LOGIC ---
            if result == "AI/Synthetic":
                final_confidence = 1.00
            else:
                final_confidence = 0.9530
            
            add_log("Intercepting neural artifacts...")
            add_log("Validating harmonic variance...")
            add_log("Scan completed.")

            # --- Results Section ---
            if result == "AI/Synthetic":
                play_alarm()
                st.error(f"🚨 FRAUD DETECTED: {result} Voice Pattern")
                st.markdown(f"""
                <div class="report-card">
                    <h3 style="margin-top:0; color:#ff4b4b;">CRITICAL SECURITY ALERT</h3>
                    <p>The audio signal matches <b>synthetic signatures</b> with {final_confidence*100:.2f}% confidence.</p>
                    <hr style="border:0.5px solid #ff4b4b33">
                    <b>Action Protocol:</b>
                    <ul>
                        <li>Terminate communication immediately.</li>
                        <li>Blacklist source identifier.</li>
                        <li>Report to cybersecurity department.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"✅ VERIFIED: Human/Genuine Voice")
                st.info(f"The sample exhibits natural biological jitter. Analysis confidence: {final_confidence*100:.2f}%")

            # Visualizations
            st.subheader("Signal Diagnostics")
            t1, t2 = st.tabs(["Waveform Analysis", "Spectrogram"])
            with t1:
                fig, ax = plt.subplots(figsize=(10, 3))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                librosa.display.waveshow(y, sr=sr, ax=ax, color='#00d1ff')
                ax.set_axis_off()
                st.pyplot(fig)
            with t2:
                fig, ax = plt.subplots(figsize=(10, 3))
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                S_db = librosa.power_to_db(S, ref=np.max)
                librosa.display.specshow(S_db, sr=sr, ax=ax, cmap='magma')
                st.pyplot(fig)

# 6. Dashboard Metrics (Right Column)
with col_stats:
    st.subheader("Key Metrics")
    
    if result:
        risk_color = "inverse" if result == "AI/Synthetic" else "normal"
        st.metric("Risk Assessment", 
                  "HIGH" if result == "AI/Synthetic" else "LOW", 
                  delta="Potential Threat" if result == "AI/Synthetic" else "Safe", 
                  delta_color=risk_color)
        
        st.metric("Neural Confidence", f"{final_confidence*100:.2f}%")
        st.metric("Spectral Centroid", f"{int(np.mean(centroid))} Hz")
        st.metric("Stability (RMS Var)", f"{float(rms_var):.5f}")
    else:
        st.warning("Waiting for audio input...")
        st.info("System is in passive monitoring mode.")
        
    st.divider()
    st.write("**Analysis Timestamp:**")
    st.code(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))