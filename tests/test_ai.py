"""Tests for the AI layer: embeddings, search, routing, RAG and evaluation.

Everything here runs on the offline embedding backend and a stub LLM, so the
suite needs no model download, no API key and no running services.
"""

from __future__ import annotations

import numpy as np
import pytest

from tickets.ai.evaluate import accuracy, evaluate_retrieval, evaluate_routing, macro_f1
from tickets.ai.llm import Enrichment, NullProvider, _extract_json, enrich_with_llm
from tickets.ai.rag import _strip_ungrounded_citations, answer_question
from tickets.ai.router import CentroidRouter, KnnRouter
from tickets.ai.search import BM25, InMemoryIndex
from tickets.core.embeddings import OfflineTfidfSvdBackend, cosine_similarity
from tickets.core.enrich import attach_embeddings, enrich_ticket
from tickets.core.schema import parse_row

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_ROWS = [
    ("Invoice is wrong", "The invoice shows a wrong payment amount and I want a refund",
     "Billing and Payments", "en", ["Billing", "Payment"]),
    ("Double charged", "I was charged twice for my subscription payment, please refund the invoice",
     "Billing and Payments", "en", ["Billing", "Payment"]),
    ("Server outage", "The production server is down and the whole service is unavailable",
     "Service Outages and Maintenance", "en", ["Outage", "Disruption"]),
    ("System unavailable", "Complete downtime of the platform, the service outage is critical",
     "Service Outages and Maintenance", "en", ["Outage", "Disruption"]),
    ("Password reset", "I cannot log in and need my password reset for VPN access",
     "IT Support", "en", ["Access", "IT"]),
    ("Cannot access account", "Login fails, my account password does not work on the VPN",
     "IT Support", "en", ["Access", "IT"]),
    ("Rechnung falsch", "Meine Rechnung ist falsch und ich bitte um Erstattung der Zahlung",
     "Billing and Payments", "de", ["Billing", "Payment"]),
    ("Server ausgefallen", "Der Server ist ausgefallen und die Störung betrifft alle Nutzer",
     "Service Outages and Maintenance", "de", ["Outage", "Disruption"]),
    ("Passwort vergessen", "Ich kann mich nicht anmelden und brauche ein neues Passwort",
     "IT Support", "de", ["Access", "IT"]),
    ("Feature request", "Could you document the integration and add a configuration guide",
     "Product Support", "en", ["Feature", "Documentation"]),
    ("Documentation gap", "The integration guide is missing configuration details and setup steps",
     "Product Support", "en", ["Feature", "Documentation"]),
    ("Return item", "I want to return the damaged product and get a replacement under warranty",
     "Returns and Exchanges", "en", ["Return", "Warranty"]),
]


@pytest.fixture(scope="module")
def tickets():
    out = []
    for subject, body, queue, language, tags in _ROWS:
        row = {"subject": subject, "body": body, "queue": queue, "language": language,
               "priority": "medium", "type": "Incident"}
        for i, tag in enumerate(tags, start=1):
            row[f"tag_{i}"] = tag
        out.append(enrich_ticket(parse_row(row)))
    return out


@pytest.fixture(scope="module")
def backend(tickets):
    # min_df in the real backend assumes a large corpus; a 12-document fixture
    # needs the defaults relaxed, which fit() handles via its n_components clamp.
    return OfflineTfidfSvdBackend(dim=8).fit([t.text for t in tickets])


@pytest.fixture(scope="module")
def embedded(tickets, backend):
    attach_embeddings(tickets, backend)
    return tickets


@pytest.fixture(scope="module")
def index(embedded, backend):
    return InMemoryIndex(embedded, backend)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


def test_offline_backend_emits_unit_vectors(backend, tickets):
    vectors = backend.encode([t.text for t in tickets])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_offline_backend_is_deterministic(backend):
    first = backend.encode(["the invoice is wrong"])
    second = backend.encode(["the invoice is wrong"])
    assert np.allclose(first, second)


def test_encode_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        OfflineTfidfSvdBackend(dim=8).encode(["text"])


def test_fit_on_empty_corpus_raises():
    with pytest.raises(ValueError, match="empty corpus"):
        OfflineTfidfSvdBackend(dim=8).fit([])


def test_similar_texts_score_higher_than_unrelated(backend):
    vectors = backend.encode(
        [
            "the invoice shows a wrong payment amount",   # query
            "I was charged twice on my invoice payment",  # related
            "the production server is down",              # unrelated
        ]
    )
    similarities = cosine_similarity(vectors[0], vectors[1:])
    assert similarities[0] > similarities[1]


def test_save_and_load_roundtrip(backend, tmp_path):
    path = tmp_path / "model.joblib"
    backend.save(path)
    restored = OfflineTfidfSvdBackend.load(path)
    assert np.allclose(backend.encode(["invoice"]), restored.encode(["invoice"]))


# --------------------------------------------------------------------------- #
# BM25 and search
# --------------------------------------------------------------------------- #


def test_bm25_ranks_the_matching_document_first():
    corpus = [
        "the invoice payment is wrong".split(),
        "the server is down".split(),
        "password reset request".split(),
    ]
    scores = BM25(corpus).scores("invoice payment")
    assert int(np.argmax(scores)) == 0


def test_bm25_returns_zeros_for_unknown_terms():
    scores = BM25([["alpha"], ["beta"]]).scores("zzzz")
    assert np.allclose(scores, 0.0)


def test_semantic_search_returns_k_hits(index):
    assert len(index.semantic("refund my invoice", k=3)) == 3


def test_keyword_search_finds_the_literal_term(index):
    hits = index.keyword("refund invoice payment", k=3)
    assert any("invoice" in h.text.lower() for h in hits)


def test_hybrid_search_returns_results(index):
    assert len(index.hybrid("server outage", k=3)) == 3


def test_search_filter_restricts_the_result_set(index):
    for hit in index.semantic("problem", k=5, language="de"):
        assert hit.language == "de"


def test_index_rejects_missing_embeddings(backend):
    bare = [enrich_ticket(parse_row({"body": "no embedding attached"}))]
    with pytest.raises(ValueError, match="embedding"):
        InMemoryIndex(bare, backend)


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #


def test_knn_router_recovers_training_labels(embedded):
    matrix = np.asarray([t.embedding for t in embedded], dtype=np.float32)
    labels = [t.queue for t in embedded]
    router = KnnRouter(k=1).fit(matrix, labels)
    # k=1 against the training set is a sanity check on the wiring, not a
    # generalisation claim: each point's nearest neighbour is itself.
    assert [p.label for p in router.predict(matrix)] == labels


def test_centroid_router_predicts_a_known_class(embedded):
    matrix = np.asarray([t.embedding for t in embedded], dtype=np.float32)
    labels = [t.queue for t in embedded]
    router = CentroidRouter().fit(matrix, labels)
    prediction = router.predict_one(matrix[0])
    assert prediction.label in set(labels)
    assert 0.0 <= prediction.confidence <= 1.0


def test_router_rejects_mismatched_input_lengths():
    with pytest.raises(ValueError, match="same length"):
        KnnRouter().fit(np.zeros((3, 4), dtype=np.float32), ["a", "b"])


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        KnnRouter().predict_one(np.zeros(4, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_accuracy_is_exact():
    assert accuracy(["a", "b", "c"], ["a", "b", "x"]) == pytest.approx(2 / 3)


def test_macro_f1_is_perfect_on_perfect_predictions():
    assert macro_f1(["a", "b"], ["a", "b"]) == pytest.approx(1.0)


def test_macro_f1_punishes_majority_class_guessing():
    """The reason macro-F1 is the headline metric rather than accuracy."""
    true = ["a"] * 9 + ["b"]
    predicted = ["a"] * 10
    assert accuracy(true, predicted) == pytest.approx(0.9)
    assert macro_f1(true, predicted) < 0.5


def test_evaluate_routing_needs_enough_data(embedded):
    with pytest.raises(ValueError, match="at least 50"):
        evaluate_routing(embedded[:10])


def test_evaluate_retrieval_reports_every_method(index, embedded):
    results = evaluate_retrieval(embedded, index, n_queries=5, k=3)
    assert {r.method for r in results} == {"semantic", "keyword", "hybrid"}
    for result in results:
        assert 0.0 <= result.queue_purity_at_k <= 1.0
        assert 0.0 <= result.mrr <= 1.0


# --------------------------------------------------------------------------- #
# LLM layer and RAG
# --------------------------------------------------------------------------- #


class StubProvider:
    """Deterministic stand-in for a hosted model."""

    name = "stub"

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str, system: str = "", max_tokens: int = 512) -> str:
        self.prompts.append(prompt)
        return self.response


def test_extract_json_survives_surrounding_prose():
    parsed = _extract_json('Sure! Here you go:\n```json\n{"queue": "IT Support"}\n```')
    assert parsed["queue"] == "IT Support"


def test_extract_json_raises_without_an_object():
    with pytest.raises(ValueError, match="no JSON object"):
        _extract_json("there is no json here")


def test_null_provider_gives_rule_based_enrichment():
    result = enrich_with_llm("invoice payment refund", provider=NullProvider())
    assert isinstance(result, Enrichment)
    assert result.source == "rules"
    assert result.queue == "Billing and Payments"


def test_llm_enrichment_uses_a_valid_model_label():
    provider = StubProvider('{"queue": "IT Support", "priority": "high", '
                            '"sentiment": "negative", "summary": "s", "entities": ["VPN"]}')
    result = enrich_with_llm("cannot connect to vpn", provider=provider)
    assert result.queue == "IT Support"
    assert result.source == "llm:stub"


def test_llm_enrichment_rejects_an_invented_label():
    """A hallucinated queue must not reach the index."""
    provider = StubProvider('{"queue": "Totally Made Up", "priority": "urgent", '
                            '"sentiment": "furious", "summary": "s", "entities": []}')
    result = enrich_with_llm("invoice payment refund charge", provider=provider)
    assert result.queue == "Billing and Payments"   # fell back to the rule label
    assert result.priority in {"low", "medium", "high"}
    assert result.sentiment in {"negative", "neutral", "positive"}


def test_llm_enrichment_survives_a_provider_error():
    class Broken:
        name = "broken"

        def complete(self, *args, **kwargs):
            raise RuntimeError("upstream is down")

    result = enrich_with_llm("invoice refund", provider=Broken())
    assert result.source == "rules"


def test_strip_ungrounded_citations_removes_unknown_ids():
    cleaned, dropped = _strip_ungrounded_citations("See [abc123] and [deadbeef].", {"abc123"})
    assert "abc123" in cleaned
    assert "deadbeef" not in cleaned
    assert dropped == ["deadbeef"]


def test_rag_without_an_llm_is_extractive_and_grounded(index):
    answer = answer_question("problem with my invoice", index, k=3, provider=NullProvider())
    assert answer.source == "extractive"
    assert answer.hits
    assert answer.answer


def test_rag_strips_hallucinated_citations(index):
    provider = StubProvider("Based on the excerpts [ffffffff] this is a billing issue.")
    answer = answer_question("invoice problem", index, k=3, provider=provider)
    assert "ffffffff" not in answer.answer
    assert answer.dropped_citations == ["ffffffff"]


def test_rag_falls_back_when_generation_fails(index):
    class Broken:
        name = "broken"

        def complete(self, *args, **kwargs):
            raise RuntimeError("timeout")

    answer = answer_question("invoice", index, k=2, provider=Broken())
    assert answer.source == "extractive-fallback"
    assert answer.hits
