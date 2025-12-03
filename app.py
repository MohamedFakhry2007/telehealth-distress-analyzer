import os
import shutil
import subprocess
import streamlit as st
import torch
import torchaudio
import soundfile as sf
import uuid

# --- WINDOWS FIX 1: DISABLE SYMLINKS ---
# os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
WORKSPACE_DIR = os.path.join(BASE_DIR, "temp_workspace")

# --- SURGICAL MONKEY PATCH (V2) ---
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

def surgical_load_audio(filepath, **kwargs):
    """
    Smart Audio Loader that fixes 'Doubled Paths' on Windows.
    """
    path_str = str(filepath).strip()
    
    # --- PATH SANITIZATION LOGIC ---
    # Fix duplication bug: e.g., "E:\Project\E:\Project\file.wav"
    # We look for the last occurrence of a drive letter (e.g., "E:")
    if ":" in path_str:
        last_colon_index = path_str.rfind(":")
        if last_colon_index > 1:
            # If colon isn't at start (C:...), assume duplication and take the last part
            # Take 1 char before colon (the drive letter)
            path_str = path_str[last_colon_index - 1:]
    
    # Resolve to absolute if needed, but prioritize existing path
    if os.path.exists(path_str):
        final_path = path_str
    else:
        final_path = os.path.abspath(path_str)
        
    # Final Validation
    if not os.path.exists(final_path):
        raise FileNotFoundError(f"Audio file missing at sanitized path: {final_path}")
    
    if os.path.getsize(final_path) == 0:
        raise ValueError("Audio file is empty.")

    # Read Audio
    try:
        data, sample_rate = sf.read(final_path, dtype='float32')
    except Exception as e:
        # Fallback for locked files
        with open(final_path, 'rb') as f:
            data, sample_rate = sf.read(f, dtype='float32')

    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(1)
    
    return waveform, sample_rate

# Apply Patch
torchaudio.load = surgical_load_audio

from moviepy.editor import VideoFileClip
from speechbrain.inference.interfaces import foreign_class

# --- Clinical AI Configuration ---
MODEL_ID = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
PAGE_TITLE = "Telehealth Distress Analyzer"
PAGE_ICON = "🩺"
ANALYSIS_DURATION_SECONDS = 30 

CLINICAL_MAP = {
    "ang": {"label": "High Distress (Agitation)", "color": "red", "priority": "Urgent"},
    "sad": {"label": "Depressive Symptoms / Low Mood", "color": "orange", "priority": "Review Needed"},
    "hap": {"label": "Stable / Positive Affect", "color": "green", "priority": "Routine"},
    "neu": {"label": "Neutral / Baseline", "color": "blue", "priority": "Routine"}
}

# --- Core Functions ---

@st.cache_resource
def load_model():
    return foreign_class(
        source=MODEL_ID,
        pymodule_file="custom_interface.py", 
        classname="CustomEncoderWav2vec2Classifier"
    )

def setup_workspace():
    if os.path.exists(WORKSPACE_DIR):
        try:
            shutil.rmtree(WORKSPACE_DIR)
        except:
            pass
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    return WORKSPACE_DIR

def download_video_with_yt_dlp(url, directory, filename="input_video.mp4"):
    try:
        filepath = os.path.join(directory, filename)
        command = [
            "yt-dlp", "-f", "bestaudio/best", "--remux-video", "mp4", 
            "--force-overwrites", "-o", filepath, url
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', timeout=120)
        return filepath if os.path.exists(filepath) else None
    except Exception as e:
        st.error(f"Download Error: {e}")
        return None

def extract_audio(video_path, audio_path, max_duration_sec):
    try:
        # Simple FFmpeg command
        command = [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", "-y"
        ]
        # Duration check manually to avoid MoviePy locks if possible
        command.extend(["-t", str(max_duration_sec)])
        command.append(audio_path)
        
        subprocess.run(command, check=True, capture_output=True, text=True)
        return os.path.exists(audio_path) and os.path.getsize(audio_path) > 100
    except Exception as e:
        st.error(f"Extraction Error: {e}")
        return False

def analyze_patient_audio(audio_path, classifier):
    # Pass path directly, let surgical_load_audio handle sanitization
    out_prob, score, index, text_lab = classifier.classify_file(audio_path)
    return text_lab[0], round(out_prob.max().item() * 100, 2)

# --- UI ---
def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")
    
    # Custom CSS for Medical Dashboard Look
    st.markdown("""
        <style>
        .medical-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            border-left: 5px solid #2e86de;
        }
        .metric-container {
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-label { font-size: 0.85em; color: #666; font-weight: 600; text-transform: uppercase; }
        .metric-value { font-size: 1.1em; color: #333; font-weight: 700; margin-top: 5px; word-wrap: break-word; }
        </style>
    """, unsafe_allow_html=True)

    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("**Clinical Decision Support System (CDSS)** | Emotion-based Triage")
    st.warning("⚠️ **Disclaimer:** POC for educational use. Not a medical device.")

    with st.spinner("Initializing Models..."):
        try:
            classifier = load_model()
        except Exception as e:
            st.error(f"Init Failed: {e}")
            return

    with st.form("main_form"):
        video_url = st.text_input("Telehealth URL", placeholder="https://...")
        submit = st.form_submit_button("Analyze Vitals")

    if submit and video_url:
        with st.spinner("Processing..."):
            ws_dir = setup_workspace()
            
            # Generate a unique session ID
            session_id = str(uuid.uuid4())[:8]
            
            # Use unique filenames
            video_filename = f"video_{session_id}.mp4"
            audio_filename = f"audio_{session_id}.wav"
            
            # Download video with unique filename
            video_path = download_video_with_yt_dlp(video_url, ws_dir, filename=video_filename)
            
            if video_path:
                audio_path = os.path.join(ws_dir, audio_filename)
                if extract_audio(video_path, audio_path, ANALYSIS_DURATION_SECONDS):
                    try:
                        state, conf = analyze_patient_audio(audio_path, classifier)
                        result = CLINICAL_MAP.get(state, {"label": "Unknown", "color": "gray", "priority": "Assess"})
                        
                        st.divider()
                        st.subheader("🎯 Triage Assessment")
                        
                        # New Layout using Custom HTML for full visibility
                        cols = st.columns(3)
                        
                        # Column 1: Detected State
                        with cols[0]:
                            st.markdown(f"""
                            <div class="metric-container" style="border-bottom: 4px solid {result['color']};">
                                <div class="metric-label">Detected State</div>
                                <div class="metric-value">{result['label']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Column 2: Confidence
                        with cols[1]:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value">{conf}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Column 3: Priority
                        with cols[2]:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">Priority</div>
                                <div class="metric-value">{result['priority']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Detailed Finding (Full Width)
                        st.markdown(f"""
                        <div class="medical-card">
                            <b>Clinical Interpretation:</b> Patient acoustic biomarkers suggest a 
                            <span style="color:{result['color']};font-weight:bold;">{result['label']}</span> state. 
                            Recommended Triage Action: <b>{result['priority']}</b>.
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Analysis Error: {e}")

if __name__ == "__main__":
    main()