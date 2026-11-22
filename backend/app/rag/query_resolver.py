"""TARK Studio — Conversation-Aware Query Resolution & Reference Resolution.

Resolves conversational follow-up references (pronouns, relative questions, and user corrections)
using conversation context, while strictly enforcing the Source-of-Truth Hierarchy:
1. The conversation history provides SUBJECT ENTITY and QUERY INTENT.
2. The uploaded document provides the AUTHORITATIVE FACTS.
3. Prior assistant answers or conversation turns NEVER substitute for original source evidence.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PRONOUN_PATTERNS = [
    r"\b(he|him|his|she|her|hers|they|them|their)\b",
    r"\b(this person|the person|the candidate|this candidate|the student|this student|the author)\b",
    r"\b(who is he|who is she|where is he|where is she|where from|what about him|what about her)\b",
    r"\b(earlier|previous|before|you said|you told|different location|why did you|why you told)\b",
]

INTENT_MAP = [
    (
        r"\b(location|where.{0,20}from|where.{0,20}is|where.{0,20}live|city|address|country|state|place|origin|hometown|resid)\b",
        "location address city country place origin residence",
    ),
    (
        r"\b(url|link|hyperlink|website|site|profile|social|contact|email|phone|mobile|web address)\b",
        "url link hyperlink website profile contact email phone",
    ),
    (
        r"\b(internship|experience|work|job|role|complet|duration|dates?|period|when.{0,15}end|when.{0,15}start|certificate)\b",
        "internship experience date duration completion certificate organization",
    ),
    (
        r"\b(education|college|university|degree|cgpa|gpa|marks|grade|school|b\.?e\.?|btech|branch)\b",
        "education university degree grade cgpa school",
    ),
    (
        r"\b(skill|tech|stack|language|framework|tools?|library|database|software|technology)\b",
        "skills technology stack tools programming languages frameworks",
    ),
    (
        r"\b(project|platform|app|application|built|developed|created|made)\b",
        "project application platform built developed",
    ),
]


@dataclass
class ResolvedQuery:
    original_query: str
    search_query: str
    is_followup: bool
    is_correction: bool
    entity: Optional[str] = None
    intent_keywords: Optional[str] = None


def extract_entities_from_history(messages: List[Dict[str, Any]], doc_names: Optional[List[str]] = None) -> List[str]:
    """Extract candidate subject entities (people, primary topics) strictly from conversation history."""
    entities: List[str] = []

    # Inspect prior messages (reverse order: most recent assistant/user turns first)
    chat_turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    for m in reversed(chat_turns):
        content = m.get("content", "")
        # Look for explicit name patterns: "belongs to X", "resume of X", "Pranit Kumar"
        m_belongs = re.search(r"(?:belongs to|resume of|candidate is|name is|about)\s+\*?\*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\*?\*?", content, re.IGNORECASE)
        if m_belongs:
            entities.append(m_belongs.group(1).strip())

        # Look for capitalized 2-word proper nouns (e.g., 'Pranit Kumar')
        caps = re.findall(r"\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b", content)
        for c in caps:
            if not any(skip in c.lower() for skip in ("tark studio", "tark ai", "client e2e", "hybrid retrieval", "onnx reranker", "verified sources")):
                entities.append(c)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for e in entities:
        norm = e.lower()
        if norm not in seen and len(norm) > 2:
            seen.add(norm)
            deduped.append(e)

    return deduped


def resolve_conversational_query(
    query: str,
    messages: List[Dict[str, Any]],
    doc_names: Optional[List[str]] = None,
) -> ResolvedQuery:
    """Resolve pronouns, conversational references, and intent from chat history.
    
    Generates an authoritative search query for retrieval and reranking while leaving
    the user's original query untouched for generation.
    """
    clean_q = query.strip()
    q_lower = clean_q.lower()

    # Prior turns in the conversation: a follow-up requires prior assistant dialogue
    assistant_turns = [m for m in messages if m.get("role") == "assistant"]
    has_prior_dialogue = len(assistant_turns) >= 1

    # 1. Detect if query is a follow-up or has pronouns/references
    has_pronoun = any(re.search(pat, q_lower) for pat in PRONOUN_PATTERNS)
    is_short = len(q_lower.split()) <= 4 and any(w in q_lower for w in ("what", "where", "when", "why", "who", "his", "her", "email", "location", "url", "link", "profile", "website"))
    is_correction = any(w in q_lower for w in ("earlier", "previous", "before", "different", "why did you", "why you told", "mistake", "wrong", "you said", "you told", "ig ", "i think", "mentioned"))

    # A query can only be a conversational follow-up if there is prior dialogue
    is_followup = has_prior_dialogue and (has_pronoun or is_short or is_correction)

    # If standalone and not a follow-up, search as-is
    if not is_followup:
        return ResolvedQuery(
            original_query=clean_q,
            search_query=clean_q,
            is_followup=False,
            is_correction=False,
        )

    # 2. Extract entity strictly from prior conversation history
    entities = extract_entities_from_history(messages)
    primary_entity = entities[0] if entities else None

    # 3. Match informational intents
    intent_words = []
    for pat, keywords in INTENT_MAP:
        if re.search(pat, q_lower):
            intent_words.append(keywords)

    intent_str = " ".join(intent_words) if intent_words else ""

    # 4. Construct rich search query
    query_parts = []
    if primary_entity and primary_entity.lower() not in q_lower:
        query_parts.append(primary_entity)

    query_parts.append(clean_q)

    if intent_str:
        query_parts.append(intent_str)

    # For any correction, signal re-verification from the original source document
    if is_correction:
        query_parts.append("verify confirm original source document")

    search_query = " ".join(query_parts).strip()

    logger.info(
        f"[QueryResolver] Original: '{clean_q}' | Followup={is_followup} Correction={is_correction} | "
        f"Entity='{primary_entity}' | Resolved Search Query: '{search_query}'"
    )

    return ResolvedQuery(
        original_query=clean_q,
        search_query=search_query,
        is_followup=is_followup,
        is_correction=is_correction,
        entity=primary_entity,
        intent_keywords=intent_str,
    )
