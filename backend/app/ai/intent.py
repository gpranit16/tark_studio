"""TARK Studio — Zero-Hardcoding LLM Intent Checker for Agentic Web Routing.

Uses a fast Groq LPU structured JSON classification call with current date awareness
and conversation context to decide whether a query requires live external web search
or can be answered directly by the LLM with zero search latency.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field

from app.config import settings
from app.ai.providers import get_groq_client

logger = logging.getLogger(__name__)

_INTENT_TIMEOUT_SEC = 4.0
_intent_groq_client = None
_intent_groq_key: Optional[str] = None


def _get_pooled_intent_client():
    """Reuse pooled AsyncGroq client so intent classification avoids per-call TLS handshakes."""
    global _intent_groq_client, _intent_groq_key
    if _intent_groq_client is None or _intent_groq_key != settings.groq_api_key:
        _intent_groq_client = get_groq_client()
        _intent_groq_key = settings.groq_api_key
    return _intent_groq_client


async def close_intent_client() -> None:
    """Close pooled intent client gracefully during shutdown."""
    global _intent_groq_client, _intent_groq_key
    if _intent_groq_client is not None:
        try:
            await _intent_groq_client.close()
        except Exception:
            pass
        finally:
            _intent_groq_client = None
            _intent_groq_key = None


def _sanitize_target_domains(raw_domains: Any) -> List[str]:
    """Validate and normalize LLM-suggested authoritative root domains (max 4)."""
    import re
    from urllib.parse import urlparse

    if not isinstance(raw_domains, list):
        return []
    cleaned: List[str] = []
    seen: set[str] = set()
    blocked_social = ("facebook.com", "linkedin.com", "instagram.com", "pinterest.com", "tiktok.com")
    for item in raw_domains:
        val = str(item or "").strip().lower()
        if not val:
            continue
        if "://" in val:
            val = urlparse(val).netloc.lower()
        val = val.split("/")[0].split(":")[0].strip()
        if val.startswith("www."):
            val = val[4:]
        if not val or val in seen or any(b in val for b in blocked_social):
            continue
        if re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", val) and len(val) <= 80:
            seen.add(val)
            cleaned.append(val)
            if len(cleaned) >= 4:
                break
    return cleaned


class IntentDecision(BaseModel):
    """Structured routing decision emitted by the LLM intent classifier."""

    needs_web_search: bool = False
    search_queries: List[str] = Field(default_factory=list, max_length=2)
    target_domains: List[str] = Field(default_factory=list, max_length=4)
    topic: Literal["general", "news", "finance"] = "general"
    reason: str = ""
    latency_ms: float = 0.0


def _build_intent_system_prompt(today_iso: str, month_year: str) -> str:
    return (
        "You are an agentic query-routing and search-planning engine for an AI assistant.\n"
        f"Current UTC date: {today_iso} ({month_year}).\n\n"
        "Decide whether answering the user's latest message accurately and completely requires "
        "live/real-time/post-cutoff external web search.\n\n"
        "Set needs_web_search=true ONLY when:\n"
        "- The query asks about current events, recent news, live prices/stocks/weather/sports, "
        "recent software/AI releases/versions, current officeholders/leaders, or recent facts up to "
        f"{today_iso}.\n"
        "- The query asks for mutable/evolving statistics, career totals, records, awards, or active entity status "
        'that can change over time (especially when phrased with "till today", "to date", "so far", "currently", or "latest").\n'
        "- The query asks to look up, search, check, or verify specific current external websites, announcements, "
        'sources, or changing data (including imperative lookups like "search on X and tell" or "check reliable info").\n\n'
        "Set needs_web_search=false when:\n"
        "- The query asks for timeless programming/debugging help, math, algorithms, architecture design, "
        "writing/editing, translation, brainstorming, greetings, or well-established general knowledge.\n\n"
        "Choose `topic` carefully:\n"
        '- "news": ONLY for breaking/recent news headlines, announcements, match recaps, or unfolding events.\n'
        '- "finance": ONLY for live stock prices, crypto/FX rates, market caps, or quarterly earnings.\n'
        '- "general": DEFAULT for factual lookups, cumulative statistics/records/totals, rankings, software/AI '
        'model releases & specs, documentation, or encyclopedic answers (even when the user says "till today" or "latest").\n\n'
        "Planning rules when needs_web_search=true:\n"
        "- `search_queries`: Emit 2 complementary standalone descriptive phrases (3-7 words each; never a single generic word). "
        f'When asking for recent/latest news, releases, or dated events, anchor Query 1 with "{month_year}" (plus parent organization if only a product name is given); '
        'for cumulative career totals or biographical records, use "[subject] total [metric] career stats" without over-constraining to a single month. '
        "Make Query 2 target official documentation, release notes, or primary reference/encyclopedic records.\n"
        "- `target_domains`: Dynamically list 2-4 authoritative primary domains (root domains only, e.g. vendor docs/newsroom, "
        "official governing body, `wikipedia.org` for verified cumulative records/bios, or major wire) best suited to verify THIS specific query—never hardcode unrelated domains.\n"
        "- Resolve pronouns (it, they, that, its), elliptic follow-ups (\"check official site\", \"what about price\"), "
        "and likely entity/source misspellings against the active conversation subject established in prior turns.\n"
        "- If recent turns show prior results were insufficient or low quality, formulate fresh authoritative reference "
        "queries rather than repeating the same weak phrasing.\n\n"
        "Respond strictly with valid JSON matching:\n"
        '{"needs_web_search": bool, "search_queries": ["...", "..."], "target_domains": ["..."], "topic": "general"|"news"|"finance", "reason": "brief"}'
    )


def _is_obvious_non_web_query(query: str) -> bool:
    """Zero-latency fast path skipping LLM intent check for greetings, code blocks, and pure math."""
    import re
    q = (query or "").strip()
    if not q:
        return True
    if "```" in q or "traceback (most recent call last)" in q.lower():
        return True
    if re.fullmatch(r"(?i)(hi|hello|hey|yo|thanks|thank you|ok|okay|cool|great|bye|good morning|good night)[!.\s]*", q):
        return True
    if re.fullmatch(r"[\d\s+\-*/^=().,%xXyY]+", q) and len(q) <= 40:
        return True
    return False


async def classify_web_intent(
    query: str,
    recent_turns: Optional[List[dict]] = None,
    today_iso: Optional[str] = None,
) -> IntentDecision:
    """Classify whether `query` needs live web search using Groq structured JSON output.

    Fails open to needs_web_search=False on timeout or error so normal chat is never blocked.
    """
    t0 = time.perf_counter()
    cleaned_q = (query or "").strip()
    if not cleaned_q or not settings.groq_api_key:
        return IntentDecision(needs_web_search=False, reason="empty_or_unconfigured")
    if _is_obvious_non_web_query(cleaned_q):
        return IntentDecision(needs_web_search=False, reason="fast_path_non_web", latency_ms=0.1)

    now_dt = datetime.now(timezone.utc)
    date_str = today_iso or now_dt.strftime("%Y-%m-%d")
    try:
        month_year = datetime.strptime(date_str[:10], "%Y-%m-%d").strftime("%B %Y")
    except Exception:
        month_year = now_dt.strftime("%B %Y")

    # Compact recent conversational context for pronoun resolution (max last 4 non-system turns)
    context_lines: List[str] = []
    if recent_turns:
        tail = [m for m in recent_turns[:-1] if m.get("role") in ("user", "assistant")][-4:]
        for m in tail:
            role_label = "User" if m.get("role") == "user" else "Assistant"
            snippet = (m.get("content") or "").strip()[:300]
            if snippet:
                context_lines.append(f"{role_label}: {snippet}")

    user_prompt = ""
    if context_lines:
        user_prompt += "Recent conversation context:\n" + "\n".join(context_lines) + "\n\n"
    user_prompt += f"Latest user message:\n{cleaned_q}"

    try:
        client = _get_pooled_intent_client()
        extra_kwargs = {}
        if "gpt-oss" in settings.intent_router_model.lower():
            extra_kwargs["reasoning_effort"] = "low"
        completion = await client.chat.completions.create(
            model=settings.intent_router_model,
            messages=[
                {"role": "system", "content": _build_intent_system_prompt(date_str, month_year)},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=260,
            response_format={"type": "json_object"},
            timeout=_INTENT_TIMEOUT_SEC,
            **extra_kwargs,
        )
        raw_content = (completion.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            # Recover valid JSON object if wrapped in markdown fences or preamble
            start = raw_content.find("{")
            end = raw_content.rfind("}")
            if start != -1 and end > start:
                parsed = json.loads(raw_content[start : end + 1])
            else:
                raise

        needs_web = bool(parsed.get("needs_web_search", False))
        raw_queries = parsed.get("search_queries") or []
        if isinstance(raw_queries, str):
            raw_queries = [raw_queries]
        clean_queries = [str(q).strip() for q in raw_queries if str(q).strip()][:2]
        if needs_web and not clean_queries:
            clean_queries = [cleaned_q]
        clean_domains = _sanitize_target_domains(parsed.get("target_domains")) if needs_web else []

        topic = parsed.get("topic", "general")
        if topic not in ("general", "news", "finance"):
            topic = "general"

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return IntentDecision(
            needs_web_search=needs_web,
            search_queries=clean_queries if needs_web else [],
            target_domains=clean_domains,
            topic=topic,
            reason=str(parsed.get("reason", ""))[:200],
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.warning(f"[IntentRouter] Falling back to direct LLM response ({exc}) in {latency_ms}ms.")
        return IntentDecision(
            needs_web_search=False,
            search_queries=[],
            reason=f"fail_open: {exc}",
            latency_ms=latency_ms,
        )
