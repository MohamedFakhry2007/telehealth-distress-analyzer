# 🩺 Telehealth Distress Analyzer (Clinical AI POC)

### AI-Powered Acoustic Triage for Telemedicine

**Author:** Dr. Mohamed Fakhry (Clinical AI Engineer & MD)

Click the image to watch the demo:

[![Watch the Demo](https://img.youtube.com/vi/LrsGSmyUY4w/maxresdefault.jpg)](https://youtu.be/LrsGSmyUY4w)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-SpeechBrain-red)
![Domain](https://img.shields.io/badge/Domain-Clinical%20AI-green)

## 📋 Executive Summary

In high-volume telehealth operations, identifying at-risk patients requires more than just transcript analysis. The **Telehealth Distress Analyzer** is a Clinical Decision Support System (CDSS) prototype engineered to detect vocal biomarkers of distress, agitation, or depressive states from patient audio.

Unlike standard sentiment analysis tools, this system focuses on **acoustic triage**, prioritizing patients based on the emotional urgency detected in their voice (e.g., Agitation vs. Calmness) before a clinician reviews the case.

## 🏥 Clinical Use Case

- **Problem:** Telehealth providers struggle to prioritize asynchronous patient video messages/voicemails efficiently.
- **Solution:** Automated acoustic analysis to flag "High Distress" or "Agitated" communications for immediate review.
- **Impact:** Reduces time-to-intervention for critical mental health or behavioral cases.

## ⚙️ Technical Architecture

The system operates on a robust pipeline designed for Windows compatibility and ease of deployment:

1. **Ingestion:** Fetches telehealth session recordings via `yt-dlp`.
2. **Preprocessing:** Extracts 16 kHz mono audio waveforms using `ffmpeg` (with path handling for Windows).
3. **Inference:** Utilizes **SpeechBrain's Wav2Vec2-IEMOCAP** model to map acoustic features to clinical states.
4. **Triage Logic:** Maps raw model outputs (Anger, Sadness, etc.) to clinical priority levels (Urgent, Routine).

## 🚀 Installation & Usage

### Prerequisites

- Python 3.10+
- FFmpeg installed and added to system `PATH`.

### Setup

```bash
# Clone the repository
git clone https://github.com/MohamedFakhry2007/Telehealth-Distress-Analyzer.git
cd Telehealth-Distress-Analyzer

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Running the system

```bash
streamlit run app.py
```

## 🧪 Clinical Validation & Limitations

During the engineering and testing phase, several observations were made regarding the model's performance in a clinical context:

1. High-Arousal Conflation (The "Scream" Test)

   - Observation: The model occasionally misclassified "Agitated Screaming" as "Positive Affect".
   - Root cause: High-arousal states (Anger/Fear) can share spectral energy and pitch profiles with high-energy excitement (Joy); dataset bias (IEMOCAP) contributes to overlap.
   - Mitigation: In production, any "High Arousal" signal should trigger a manual review flag regardless of the label.

2. Low-Arousal Ambiguity (The "Calm" Test)

   - Observation: Calm, quiet speech was sometimes flagged as "Depressive/Sad".
   - Root cause: Calmness and depression both exhibit low valence and arousal (slow tempo, low volume).
   - Clinical insight: Longitudinal analysis (comparing a patient against their baseline) reduces false positives.

## 🛠️ Engineering Challenges & Fixes

- Dependency conflicts between `torchaudio` and `speechbrain` were resolved with targeted compatibility fixes.
- Path sanitization logic was added to handle Windows-specific absolute/relative path mixing in audio libraries.

## 🔮 Future Roadmap

- Multimodal fusion: Integrate Whisper ASR to analyze text (what is said) alongside audio (how it is said).
- Patient baseline: Implement a vector database to store patient "voice prints" for deviation detection.

---

Engineered by Dr. Mohamed Fakhry — Bridging Medicine & Technology.
