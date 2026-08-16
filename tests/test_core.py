"""Unit tests for the pure-Python transformation logic.

These matter more than usual in this project: the Spark job calls exactly these
functions inside its UDFs, so a green suite here is real evidence about the
pipeline, not just about a helper module.
"""

from __future__ import annotations

import math

import pytest

from tickets.core import kpis as kpi_module
from tickets.core.classify import classify_priority, classify_queue
from tickets.core.enrich import enrich_ticket
from tickets.core.pii import assert_safe_for_external, redact
from tickets.core.schema import Ticket, collect_tags, is_valid, make_ticket_id, parse_row
from tickets.core.sentiment import score_sentiment
from tickets.core.textproc import build_text, detect_language, normalize, tokenize, truncate


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def test_parse_row_handles_missing_subject():
    """13% of real rows have a null subject; that is data, not an error."""
    ticket = parse_row({"subject": None, "body": "Server is down", "queue": "IT Support"})
    assert ticket.subject == ""
    assert ticket.body == "Server is down"


@pytest.mark.parametrize("null_value", [None, "nan", "NaN", "None", "null", float("nan")])
def test_parse_row_normalises_every_null_spelling(null_value):
    """pandas yields NaN, Spark yields the string 'nan'; both must become ''."""
    ticket = parse_row({"subject": null_value, "body": "text"})
    assert ticket.subject == ""


def test_ticket_id_is_content_addressed_and_stable():
    row = {"subject": "a", "body": "b", "answer": "c", "queue": "IT Support", "language": "en"}
    assert make_ticket_id(row) == make_ticket_id(dict(row))


def test_ticket_id_changes_with_content():
    base = {"subject": "a", "body": "b", "answer": "c", "queue": "IT Support", "language": "en"}
    other = dict(base, body="different")
    assert make_ticket_id(base) != make_ticket_id(other)


def test_collect_tags_deduplicates_and_drops_nulls():
    row = {"tag_1": "IT", "tag_2": "IT", "tag_3": None, "tag_4": "nan", "tag_5": "Bug"}
    assert collect_tags(row) == ["IT", "Bug"]


def test_version_survives_a_float_string():
    assert parse_row({"body": "x", "version": "400.0"}).version == 400


def test_version_falls_back_to_zero_when_unparseable():
    assert parse_row({"body": "x", "version": "not-a-number"}).version == 0


def test_validation_rejects_empty_body():
    ok, reason = is_valid(parse_row({"body": "   "}))
    assert not ok and reason == "empty_body"


def test_validation_rejects_unknown_queue():
    ok, reason = is_valid(parse_row({"body": "hello", "queue": "Made Up Queue"}))
    assert not ok and "unknown_queue" in reason


def test_validation_accepts_a_real_row():
    ok, _ = is_valid(
        parse_row(
            {"body": "hello", "queue": "IT Support", "priority": "high", "type": "Incident"}
        )
    )
    assert ok


# --------------------------------------------------------------------------- #
# textproc
# --------------------------------------------------------------------------- #


def test_normalize_strips_literal_escape_sequences():
    """The CSV stores newlines as the characters backslash-n."""
    assert normalize("Dear Team,\\n\\nPlease help") == "Dear Team, Please help"


def test_tokenize_drops_urls_and_placeholders():
    tokens = tokenize("contact <name> at https://example.com/path about billing")
    assert "name" not in tokens
    assert "https" not in tokens
    assert "billing" in tokens


def test_build_text_omits_a_missing_subject():
    assert build_text("", "body only") == "body only"
    assert build_text("Subject", "body") == "Subject body"


def test_detect_language_english():
    assert detect_language("Dear support team, I would like to request assistance with this") == "en"


def test_detect_language_german():
    assert detect_language("Sehr geehrtes Team, ich möchte ein Problem melden und bitte um Hilfe") == "de"


def test_detect_language_unknown_on_empty_input():
    assert detect_language("") == "unknown"


def test_truncate_does_not_split_a_word():
    result = truncate("alpha beta gamma delta epsilon", 14)
    assert result.endswith("...")
    assert "gam..." not in result


# --------------------------------------------------------------------------- #
# pii
# --------------------------------------------------------------------------- #


def test_redact_email_and_phone():
    result = redact("Reach me at jane.doe@example.com or +49 170 1234567")
    assert "jane.doe@example.com" not in result.text
    assert "[EMAIL]" in result.text
    assert "email" in result.kinds
    assert "phone" in result.kinds


def test_redact_leaves_small_numbers_alone():
    """A quantity must not be mistaken for a phone number."""
    result = redact("We ordered 25 units and 3 spares")
    assert "25" in result.text
    assert "phone" not in result.kinds


def test_redact_detects_publisher_placeholders():
    result = redact("Please call <tel_num> to reach <name>")
    assert "placeholder" in result.kinds
    assert "<tel_num>" not in result.text


def test_assert_safe_for_external_blocks_unredacted_pii():
    with pytest.raises(ValueError, match="unredacted PII"):
        assert_safe_for_external("my card is 4111 1111 1111 1111")


def test_assert_safe_for_external_allows_clean_text():
    assert_safe_for_external("The server returned a 500 error on the checkout page")


def test_redaction_is_idempotent():
    once = redact("mail me at a@b.com").text
    assert redact(once).text == once


# --------------------------------------------------------------------------- #
# sentiment
# --------------------------------------------------------------------------- #


def test_sentiment_negative():
    assert score_sentiment("This is broken and completely unacceptable, urgent failure").label == "negative"


def test_sentiment_positive():
    assert score_sentiment("Thank you, excellent and very helpful support, perfect").label == "positive"


def test_sentiment_neutral_on_empty():
    result = score_sentiment("")
    assert result.label == "neutral" and result.score == 0.0


def test_sentiment_handles_negation():
    """'not good' must not score as positive."""
    assert score_sentiment("this is not good").score <= 0


def test_sentiment_score_stays_in_range():
    for text in ["terrible awful broken failure error", "great perfect excellent wonderful"]:
        assert -1.0 <= score_sentiment(text).score <= 1.0


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #


def test_classify_queue_billing():
    assert classify_queue("My invoice shows the wrong payment amount, please refund").label == (
        "Billing and Payments"
    )


def test_classify_queue_german_billing():
    assert classify_queue("Meine Rechnung ist falsch, ich bitte um Erstattung der Zahlung").label == (
        "Billing and Payments"
    )


def test_classify_queue_falls_back_to_majority_class():
    prediction = classify_queue("aaa bbb ccc")
    assert prediction.label == "Technical Support"
    assert prediction.confidence == 0.0


def test_classify_priority_high():
    assert classify_priority("URGENT: critical production outage, escalate immediately").label == "high"


def test_classify_priority_defaults_to_medium():
    assert classify_priority("some text with no priority signal at all").label == "medium"


def test_classify_confidence_is_a_share():
    prediction = classify_queue("invoice payment billing refund charge")
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.evidence


# --------------------------------------------------------------------------- #
# enrichment
# --------------------------------------------------------------------------- #


def test_enrich_populates_every_derived_field():
    ticket = parse_row(
        {
            "subject": "Cannot log in",
            "body": "I forgot my password and cannot access the VPN. Contact me at a@b.com",
            "queue": "IT Support",
            "priority": "high",
            "type": "Incident",
            "language": "en",
            "tag_1": "Access",
        }
    )
    enriched = enrich_ticket(ticket)

    assert enriched.text
    assert "a@b.com" not in enriched.text_redacted
    assert "email" in enriched.pii_found
    assert enriched.detected_language == "en"
    assert enriched.sentiment in {"negative", "neutral", "positive"}
    assert enriched.predicted_queue
    assert enriched.ingest_ts and enriched.ingest_date
    assert enriched.enrichment_source == "rules"


def test_enrich_flags_a_language_mismatch():
    """A row labelled German whose body is English must be flagged."""
    ticket = parse_row(
        {
            "body": "Dear support team, I would like to request assistance with this issue",
            "language": "de",
        }
    )
    enriched = enrich_ticket(ticket)
    assert enriched.detected_language == "en"
    assert enriched.language_mismatch is True


def test_enrich_does_not_flag_a_matching_language():
    ticket = parse_row(
        {"body": "Sehr geehrtes Team, ich habe ein Problem mit der Anwendung", "language": "de"}
    )
    assert enrich_ticket(ticket).language_mismatch is False


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #


def _sample_tickets() -> list[Ticket]:
    rows = [
        {"body": "invoice wrong", "queue": "Billing and Payments", "priority": "high",
         "type": "Incident", "language": "en", "tag_1": "Billing"},
        {"body": "server outage urgent", "queue": "Technical Support", "priority": "high",
         "type": "Incident", "language": "en", "tag_1": "Outage"},
        {"body": "thanks for the great help", "queue": "Customer Service", "priority": "low",
         "type": "Request", "language": "en", "tag_1": "Feedback"},
        {"body": "password reset please", "queue": "IT Support", "priority": "medium",
         "type": "Request", "language": "en", "tag_1": "Access"},
    ]
    return [enrich_ticket(parse_row(r)) for r in rows]


def test_volume_shares_sum_to_one():
    rows = kpi_module.volume_by(_sample_tickets(), "queue")
    assert math.isclose(sum(r["value"] for r in rows), 1.0, rel_tol=1e-6)


def test_routing_accuracy_is_reported_overall_and_per_queue():
    rows = kpi_module.routing_accuracy(_sample_tickets())
    assert any(r["dimension"] == "overall" for r in rows)
    assert any(r["dimension"] == "queue" for r in rows)
    for row in rows:
        assert 0.0 <= row["value"] <= 1.0


def test_summarize_reports_expected_keys():
    summary = kpi_module.summarize(_sample_tickets())
    assert summary["total_tickets"] == 4
    for key in ("routing_accuracy_overall", "language_mismatch_overall", "negative_share"):
        assert key in summary


def test_compute_all_emits_the_shared_long_format():
    for row in kpi_module.compute_all(_sample_tickets()):
        assert set(row) == {"kpi", "dimension", "bucket", "value", "count", "computed_at"}


def test_kpis_accept_dicts_as_well_as_dataclasses():
    tickets = _sample_tickets()
    assert kpi_module.summarize(tickets) ["total_tickets"] == kpi_module.summarize(
        [t.to_dict() for t in tickets]
    )["total_tickets"]


def test_kpis_on_empty_input_do_not_crash():
    assert kpi_module.routing_accuracy([]) == []
    assert kpi_module.summarize([])["total_tickets"] == 0
