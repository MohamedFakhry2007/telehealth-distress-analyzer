from vlm_guard import Analysis, BaseRule, RuleResult, GuardrailEngine, AuditTrail

CLINICAL_MAP = {
    "ang": {"label": "High Distress (Agitation)", "color": "red", "priority": "Urgent"},
    "sad": {"label": "Depressive Symptoms / Low Mood", "color": "orange", "priority": "Review Needed"},
    "hap": {"label": "Stable / Positive Affect", "color": "green", "priority": "Routine"},
    "neu": {"label": "Neutral / Baseline", "color": "blue", "priority": "Routine"},
}


def emotion_to_analysis(emotion: str, confidence_pct: float, session_id: str) -> Analysis:
    clinical = CLINICAL_MAP.get(emotion, {"label": "Unknown", "priority": "Assess"})
    conf_level = "High" if confidence_pct >= 80 else "Medium" if confidence_pct >= 50 else "Low"
    return Analysis(
        label=emotion,
        domain="telehealth_distress",
        claim_type="other",
        claim_text=f"Patient acoustic biomarkers indicate {clinical['label']}",
        confidence=conf_level,
        evidence=f"SpeechBrain Wav2Vec2-IEMOCAP confidence: {confidence_pct:.1f}%",
        recommendation=f"Triage priority: {clinical['priority']}",
        metadata={
            "emotion": emotion,
            "clinical_label": clinical["label"],
            "confidence_pct": confidence_pct,
            "priority": clinical["priority"],
            "session_id": session_id,
        },
    )


CONFIDENCE_THRESHOLD_FLOOR = 45.0
HIGH_CONFIDENCE_SUSPICION_THRESHOLD = 90.0


class ConfidenceThresholdRule(BaseRule):
    name = "confidence_threshold"
    description = "Blocks predictions below the minimum confidence floor for clinical triage"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.metadata.get("confidence_pct", 0) < CONFIDENCE_THRESHOLD_FLOOR

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        conf = analysis.metadata.get("confidence_pct", 0)
        analysis.validation_status = "blocked"
        analysis.validation_message = (
            f"Confidence {conf:.1f}% is below the {CONFIDENCE_THRESHOLD_FLOOR}% threshold "
            "required for clinical triage. Suggest re-recording or manual review."
        )
        return analysis, RuleResult(
            action_taken=True,
            action_type="block",
            message=analysis.validation_message,
            severity="error",
            modified_fields={
                "validation_status": "blocked",
                "confidence_pct": conf,
            },
        )


class HighArousalAmbiguityRule(BaseRule):
    name = "high_arousal_ambiguity"
    description = (
        "Flags 'Stable/Positive' predictions that may be high-arousal misclassifications "
        "(known IEMOCAP bias: anger/fear can share spectral profiles with excitement)"
    )

    def condition(self, analysis: Analysis, context: dict) -> bool:
        if analysis.validation_status == "blocked":
            return False
        label = analysis.metadata.get("emotion", "")
        conf = analysis.metadata.get("confidence_pct", 0)
        return label == "hap" and conf >= HIGH_CONFIDENCE_SUSPICION_THRESHOLD

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        conf = analysis.metadata.get("confidence_pct", 0)
        analysis.validation_status = "flagged"
        analysis.validation_message = (
            f"High-arousal ambiguity detected: '{analysis.label}' predicted at {conf:.1f}%. "
            "High-confidence 'Positive Affect' may be a high-arousal misclassification "
            "(IEMOCAP known limitation). Manual clinician review recommended."
        )
        return analysis, RuleResult(
            action_taken=True,
            action_type="flag",
            message=analysis.validation_message,
            severity="warning",
            modified_fields={"validation_status": "flagged"},
        )


class TriageConsistencyRule(BaseRule):
    name = "triage_consistency"
    description = "Flags cases where triage priority contradicts confidence level"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        if analysis.validation_status == "blocked":
            return False
        priority = analysis.metadata.get("priority", "")
        conf = analysis.metadata.get("confidence_pct", 0)
        return priority == "Urgent" and conf < 60

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        conf = analysis.metadata.get("confidence_pct", 0)
        analysis.validation_status = "flagged"
        analysis.validation_message = (
            f"Triage inconsistency: priority is 'Urgent' but confidence is only {conf:.1f}%. "
            "Urgent clinical decisions require higher confidence. Recommend manual verification."
        )
        return analysis, RuleResult(
            action_taken=True,
            action_type="flag",
            message=analysis.validation_message,
            severity="warning",
            modified_fields={"validation_status": "flagged"},
        )


def make_guardrail_engine() -> GuardrailEngine:
    engine = GuardrailEngine()
    engine.register(ConfidenceThresholdRule())
    engine.register(HighArousalAmbiguityRule())
    engine.register(TriageConsistencyRule())
    return engine


def guardrail_orchestrator(
    engine: GuardrailEngine,
    emotion: str,
    confidence_pct: float,
    session_id: str,
) -> tuple[Analysis, AuditTrail]:
    analysis = emotion_to_analysis(emotion, confidence_pct, session_id)
    context = {
        "session_id": session_id,
        "emotion": emotion,
        "confidence_pct": confidence_pct,
    }
    return engine.apply_with_audit(analysis, context)
