"""Retrieval-augmented question answering over the ticket corpus.

Flow: embed the question -> retrieve the k most similar tickets (semantic, or
hybrid when the question contains product names that BM25 handles better) ->
build a prompt containing *only* the retrieved text -> ask the LLM to answer
strictly from that context, citing ticket ids.

Two properties are worth defending in the viva:

1. **Grounding is enforced, not requested.** The answer is post-checked: if the
   model cites an id that was not retrieved, the citation is stripped. A prompt
   instruction alone is not a control.
2. **It degrades to extractive.** With ``LLM_PROVIDER=none`` the same retrieval
   runs and the answer becomes the highest-scoring passages plus an aggregate
   over the retrieved set. Less fluent, still grounded, and it means the demo
   works with no API key.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..config import AIConfig, get_settings
from ..core.textproc import truncate
from .llm import LLMProvider, NullProvider, build_provider
from .search import SearchHit, SearchIndex

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You answer questions about a customer support ticket corpus. "
    "Use ONLY the numbered ticket excerpts provided. "
    "If they do not contain the answer, say so plainly. "
    "Cite the ticket ids you used in square brackets, e.g. [a1b2c3d4]. "
    "Never invent ticket ids."
)

_TEMPLATE = """Question: {question}

Ticket excerpts:
{context}

Answer the question using only the excerpts above. Be concise and cite ticket ids."""


@dataclass
class RagAnswer:
    question: str
    answer: str
    hits: list[SearchHit] = field(default_factory=list)
    source: str = "extractive"
    dropped_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "source": self.source,
            "dropped_citations": self.dropped_citations,
            "citations": [h.ticket_id for h in self.hits],
            "hits": [h.to_dict() for h in self.hits],
        }


def _format_context(hits: list[SearchHit], max_chars: int = 700) -> str:
    return "\n\n".join(
        f"[{hit.ticket_id}] (queue={hit.queue or 'n/a'}, priority={hit.priority or 'n/a'}, "
        f"lang={hit.language or 'n/a'})\n{truncate(hit.text, max_chars)}"
        for hit in hits
    )


def _extractive_answer(question: str, hits: list[SearchHit]) -> str:
    """Answer without a model: describe what was retrieved.

    Deliberately states aggregate facts about the retrieved set rather than
    pretending to reason, so it is never misleading about what it is.
    """
    if not hits:
        return "No tickets in the corpus match that question."

    queues = Counter(h.queue for h in hits if h.queue)
    priorities = Counter(h.priority for h in hits if h.priority)
    lead = hits[0]

    parts = [
        f"Found {len(hits)} related tickets.",
        (
            "Most common queue: "
            f"{queues.most_common(1)[0][0]} ({queues.most_common(1)[0][1]}/{len(hits)})."
            if queues
            else ""
        ),
        (
            "Priority mix: "
            + ", ".join(f"{p}={c}" for p, c in priorities.most_common())
            + "."
            if priorities
            else ""
        ),
        f"Closest match [{lead.ticket_id}]: {truncate(lead.text, 400)}",
    ]
    return " ".join(p for p in parts if p)


_CITATION = re.compile(r"\[([0-9a-f]{6,32})\]")


def _strip_ungrounded_citations(answer: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Remove citations that point at tickets we never retrieved."""
    dropped: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        ticket_id = match.group(1)
        if ticket_id in allowed:
            return match.group(0)
        dropped.append(ticket_id)
        return ""

    cleaned = _CITATION.sub(_replace, answer)
    return re.sub(r"\s{2,}", " ", cleaned).strip(), dropped


def answer_question(
    question: str,
    index: SearchIndex,
    k: int = 5,
    provider: LLMProvider | None = None,
    ai: AIConfig | None = None,
    mode: str = "hybrid",
    **filters,
) -> RagAnswer:
    """Retrieve, then generate a grounded answer."""
    ai = ai or get_settings().ai
    provider = provider or build_provider(ai)

    retrieve = {"semantic": index.semantic, "keyword": index.keyword}.get(mode, index.hybrid)
    hits = retrieve(question, k=k, **filters)

    if isinstance(provider, NullProvider):
        return RagAnswer(question, _extractive_answer(question, hits), hits, "extractive")

    prompt = _TEMPLATE.format(question=question, context=_format_context(hits))
    try:
        raw = provider.complete(prompt, system=_SYSTEM, max_tokens=ai.llm_max_tokens)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG generation failed (%s); returning extractive answer", exc)
        return RagAnswer(question, _extractive_answer(question, hits), hits, "extractive-fallback")

    allowed = {hit.ticket_id for hit in hits}
    cleaned, dropped = _strip_ungrounded_citations(raw.strip(), allowed)
    if dropped:
        logger.warning("stripped %d ungrounded citation(s): %s", len(dropped), dropped)

    return RagAnswer(question, cleaned, hits, f"llm:{provider.name}", dropped)


def narrate_insights(summary: dict[str, Any], provider: LLMProvider | None = None, ai: AIConfig | None = None) -> str:
    """Option (g) from the brief: an LLM-written narrative over computed KPIs.

    The brief requires that generated text be clearly separated from computed
    numbers, so the caller is expected to render this under its own heading. The
    KPIs themselves are never produced by the model -- only the prose about them.
    """
    ai = ai or get_settings().ai
    provider = provider or build_provider(ai)
    if isinstance(provider, NullProvider):
        return (
            "[No LLM configured - narrative skipped. Set LLM_PROVIDER to generate this section. "
            "All KPI values above are computed, not generated.]"
        )

    import json

    prompt = (
        "These are computed KPIs from a customer support ticket pipeline. "
        "Write 3-5 short bullet points of operational insight and recommendation. "
        "Do not invent numbers; refer only to the values given.\n\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
    )
    try:
        return provider.complete(
            prompt,
            system="You are a data analyst writing for a support operations manager.",
            max_tokens=ai.llm_max_tokens,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("insight narration failed: %s", exc)
        return "[Narrative generation failed; see computed KPIs above.]"
