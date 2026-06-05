# 🩺 Telehealth Distress Analyzer (Clinical AI POC)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telehealth-distress-analyzer-9rv6xsyezw4apdi8nwe4zs.streamlit.app)

### Audio-based Clinical Triage with Guardrails & Evaluation Infrastructure
**Try the Live Demo:** [Click Here to Open App](https://telehealth-distress-analyzer-9rv6xsyezw4apdi8nwe4zs.streamlit.app)

**Author:** Dr. Mohamed Fakhry (Clinical AI Engineer & MD)

Click the image to watch the demo:

[![Watch the Demo](https://img.youtube.com/vi/LrsGSmyUY4w/maxresdefault.jpg)](https://youtu.be/LrsGSmyUY4w)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-SpeechBrain-red)
![Guardrails](https://img.shields.io/badge/Guardrails-VLM--Guard-8A2BE2)
![Domain](https://img.shields.io/badge/Domain-Clinical%20AI-green)
![Tests](https://img.shields.io/badge/Tests-22%20passing-brightgreen)

## 📋 Executive Summary

The **Telehealth Distress Analyzer** is a Clinical Decision Support System (CDSS) prototype that detects vocal biomarkers of distress, agitation, or depressive states from patient audio — with inference-time safety guardrails, an auditable verification layer, and a documented evaluation framework.

This project demonstrates acoustic ML engineering (not LLM/RAG): audio preprocessing, Wav2Vec2-based emotion classification, clinical triage mapping, and production-grade safety infrastructure via [**VLM-Guard**](https://github.com/MohamedFakhry2007/vlm-guard). Every prediction is validated by composable rules before reaching the clinician, and all decisions are recorded in an auditable trail.

## 🏥 Clinical Use Case

- **Problem:** Telehealth providers struggle to prioritize asynchronous patient video messages/voicemails efficiently.
- **Solution:** Automated acoustic analysis to flag "High Distress" or "Agitated" communications for immediate review.
- **Impact:** Reduces time-to-intervention for critical mental health or behavioral cases.

## ⚙️ Technical Architecture

The system operates on a robust pipeline designed for Windows compatibility and ease of deployment:

1. **Ingestion:** User uploads a telehealth recording (video or audio file).
2. **Preprocessing:** Extracts 16 kHz mono audio waveforms using `ffmpeg` (with path handling for Windows).
3. **Inference:** Utilizes **SpeechBrain's Wav2Vec2-IEMOCAP** model to map acoustic features to clinical states.
4. **Guardrails:** [**VLM-Guard**](https://github.com/MohamedFakhry2007/vlm-guard) composable rule engine validates the prediction (confidence thresholds, known model biases, triage consistency) before it reaches the clinician.
5. **Triage Logic:** Maps validated model outputs (Anger, Sadness, etc.) to clinical priority levels (Urgent, Routine).

## 🚀 Installation & Usage

### Prerequisites

- Python 3.10+
- FFmpeg installed and added to system `PATH`.

### Setup

```bash
git clone https://github.com/MohamedFakhry2007/Telehealth-Distress-Analyzer.git
cd Telehealth-Distress-Analyzer
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

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

## 🛡️ Safety Guardrails (VLM-Guard)

Inference-time safety is enforced via [**VLM-Guard**](https://github.com/MohamedFakhry2007/vlm-guard), a composable rule engine for auditable LLM verification. Three rules run on every prediction:

| Rule | Trigger | Action | Clinical Rationale |
|---|---|---|---|
| **Confidence Threshold** | Confidence < 45% | **BLOCK** — prediction rejected | Low-confidence classifications are unreliable for clinical decisions |
| **High-Arousal Ambiguity** | "Positive Affect" predicted at ≥ 90% confidence | **FLAG** — manual review recommended | IEMOCAP dataset bias: high-arousal states (anger/fear) share spectral profiles with excitement |
| **Triage Consistency** | "Urgent" priority at < 60% confidence | **FLAG** — manual verification required | Urgent clinical actions demand higher certainty |

Every rule firing is recorded in an **auditable trail** with before/after snapshots, available in the UI via the Safety Audit Trail expander.

## 🧪 Test Suite

22 tests covering guardrail rules, triage mapping, rule ordering, and end-to-end orchestrator behavior:

```bash
pytest tests/ -v
```

| Area | Tests | What's verified |
|---|---|---|
| **Clinical Map** | 5 | All 4 emotion mappings + unknown fallback |
| **Adapter** | 2 | Analysis object creation, confidence level mapping |
| **ConfidenceThresholdRule** | 2 | Blocks below floor, passes above |
| **HighArousalAmbiguityRule** | 3 | Flags hap≥90%, skips low-conf hap, skips non-hap |
| **TriageConsistencyRule** | 3 | Flags urgent<60%, passes ≥60%, skips non-urgent |
| **Rule Ordering** | 1 | Blocked status prevents subsequent rule firing |
| **Integration** | 5 | Full orchestrator path: pass, block, flag (arousal), flag (inconsistency), unknown emotion |
| **Audit Trail** | 1 | Entries populated on action |

## 🛠️ Engineering Challenges & Fixes

- Path sanitization logic was added to handle Windows-specific absolute/relative path mixing in audio libraries.
- Dependency conflicts between `torchaudio` and `speechbrain` were resolved with targeted compatibility fixes.

## 🔮 Future Roadmap

- Multimodal fusion: Integrate Whisper ASR to analyze text (what is said) alongside audio (how it is said).
- Patient baseline: Implement a vector database to store patient "voice prints" for deviation detection.

---

Engineered by Dr. Mohamed Fakhry — Bridging Medicine & Technology.
