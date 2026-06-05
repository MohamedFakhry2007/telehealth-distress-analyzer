from telehealth_guardrails import (
    CLINICAL_MAP,
    emotion_to_analysis,
    ConfidenceThresholdRule,
    HighArousalAmbiguityRule,
    TriageConsistencyRule,
    make_guardrail_engine,
    guardrail_orchestrator,
    CONFIDENCE_THRESHOLD_FLOOR,
    HIGH_CONFIDENCE_SUSPICION_THRESHOLD,
)


def test_clinical_map_has_all_emotions():
    assert set(CLINICAL_MAP.keys()) == {"ang", "sad", "hap", "neu"}


def test_clinical_map_unknown_fallback():
    result = CLINICAL_MAP.get("unknown", {"label": "Unknown", "color": "gray", "priority": "Assess"})
    assert result["label"] == "Unknown"


def test_clinical_map_urgent_mapping():
    assert CLINICAL_MAP["ang"]["priority"] == "Urgent"


def test_clinical_map_review_mapping():
    assert CLINICAL_MAP["sad"]["priority"] == "Review Needed"


def test_clinical_map_routine_mapping():
    assert CLINICAL_MAP["hap"]["priority"] == "Routine"
    assert CLINICAL_MAP["neu"]["priority"] == "Routine"


def test_emotion_to_analysis_creates_correct_object():
    analysis = emotion_to_analysis("ang", 88.5, "session-1")
    assert analysis.label == "ang"
    assert analysis.domain == "telehealth_distress"
    assert analysis.metadata["confidence_pct"] == 88.5
    assert analysis.metadata["session_id"] == "session-1"
    assert analysis.metadata["priority"] == "Urgent"


def test_emotion_to_analysis_confidence_levels():
    high = emotion_to_analysis("ang", 85.0, "s")
    assert high.confidence == "High"
    med = emotion_to_analysis("ang", 65.0, "s")
    assert med.confidence == "Medium"
    low = emotion_to_analysis("ang", 30.0, "s")
    assert low.confidence == "Low"


def test_confidence_threshold_blocks_below_floor():
    rule = ConfidenceThresholdRule()
    analysis = emotion_to_analysis("ang", CONFIDENCE_THRESHOLD_FLOOR - 5, "s")
    assert rule.condition(analysis, {}) is True


def test_confidence_threshold_passes_above_floor():
    rule = ConfidenceThresholdRule()
    analysis = emotion_to_analysis("ang", CONFIDENCE_THRESHOLD_FLOOR + 5, "s")
    assert rule.condition(analysis, {}) is False


def test_high_arousal_flags_hap_at_high_confidence():
    rule = HighArousalAmbiguityRule()
    analysis = emotion_to_analysis("hap", HIGH_CONFIDENCE_SUSPICION_THRESHOLD, "s")
    assert rule.condition(analysis, {}) is True


def test_high_arousal_does_not_flag_low_confidence_hap():
    rule = HighArousalAmbiguityRule()
    analysis = emotion_to_analysis("hap", HIGH_CONFIDENCE_SUSPICION_THRESHOLD - 1, "s")
    assert rule.condition(analysis, {}) is False


def test_high_arousal_does_not_flag_non_hap():
    rule = HighArousalAmbiguityRule()
    for emotion in ("ang", "sad", "neu"):
        analysis = emotion_to_analysis(emotion, 95.0, "s")
        assert rule.condition(analysis, {}) is False, f"Should not flag {emotion}"


def test_triage_consistency_flags_urgent_low_confidence():
    rule = TriageConsistencyRule()
    analysis = emotion_to_analysis("ang", 55.0, "s")
    assert rule.condition(analysis, {}) is True


def test_triage_consistency_passes_urgent_high_confidence():
    rule = TriageConsistencyRule()
    analysis = emotion_to_analysis("ang", 75.0, "s")
    assert rule.condition(analysis, {}) is False


def test_triage_consistency_passes_non_urgent():
    rule = TriageConsistencyRule()
    for emotion in ("sad", "hap", "neu"):
        analysis = emotion_to_analysis(emotion, 30.0, "s")
        assert rule.condition(analysis, {}) is False


def test_blocked_prevents_subsequent_rules():
    rule_high = HighArousalAmbiguityRule()
    rule_triage = TriageConsistencyRule()
    analysis = emotion_to_analysis("hap", 95.0, "s")
    analysis.validation_status = "blocked"
    assert rule_high.condition(analysis, {}) is False
    assert rule_triage.condition(analysis, {}) is False


def test_guardrail_orchestrator_passes_normal():
    engine = make_guardrail_engine()
    analysis, audit = guardrail_orchestrator(engine, "neu", 85.0, "session-p")
    assert analysis.validation_status == "passed"


def test_guardrail_orchestrator_blocks_low_confidence():
    engine = make_guardrail_engine()
    analysis, audit = guardrail_orchestrator(engine, "ang", 30.0, "session-b")
    assert analysis.validation_status == "blocked"
    assert any(e.action_type == "block" for e in audit.entries if e.action_type == "block")


def test_guardrail_orchestrator_flags_high_arousal():
    engine = make_guardrail_engine()
    analysis, audit = guardrail_orchestrator(engine, "hap", 92.0, "session-f")
    assert analysis.validation_status == "flagged"


def test_guardrail_orchestrator_flags_inconsistent_triage():
    engine = make_guardrail_engine()
    analysis, audit = guardrail_orchestrator(engine, "ang", 55.0, "session-f2")
    assert analysis.validation_status == "flagged"


def test_guardrail_orchestrator_unknown_emotion():
    engine = make_guardrail_engine()
    analysis, audit = guardrail_orchestrator(engine, "unknown", 99.0, "session-u")
    assert analysis.metadata["priority"] == "Assess"


def test_audit_trail_is_populated_on_action():
    engine = make_guardrail_engine()
    analysis, audit = guardrail_orchestrator(engine, "hap", 95.0, "session-a")
    assert len(audit.entries) >= 1
    summary = audit.summary()
    assert len(summary) >= 1
    entry = summary[0]
    assert "high_arousal_ambiguity" in entry["rule"]
    assert entry["action"] == "flag"
