from benchmark_review.engine.models import ReviewRecord, ReviewStatus


def test_review_record_defaults_to_pending():
    rec = ReviewRecord(
        candidate_id="qc_50.1_cit_0",
        unit_id="50.1",
        query="What does 10 CFR 50.1 say?",
        query_class="citation_lookup",
        difficulty="easy",
        source_citations=("10 CFR 50.1",),
        evidence_span_ids=("50.1_0",),
        is_valid=False,
        validation_flags=("missing_snapshot_id",),
        critical_evidence=(),
        supporting_evidence=(),
        contextual_evidence=(),
        is_unanswerable=False,
        unanswerable_reason=None,
    )
    assert rec.review_status == ReviewStatus.PENDING
    assert rec.reviewed_by is None
    assert rec.reviewed_at is None
    assert rec.revision_notes is None
    assert rec.rejection_note is None


def test_review_status_values():
    assert ReviewStatus.PENDING.value == "pending"
    assert ReviewStatus.APPROVED.value == "approved"
    assert ReviewStatus.REJECTED.value == "rejected"
    assert ReviewStatus.NEEDS_REVISION.value == "needs_revision"
