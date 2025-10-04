# core/analyzer.py
import re
from typing import List, Optional, Any

import spacy
from core.rule_engine import RuleEngine

# Load the small English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")


def check_requirement_ambiguity(requirement_text: str, rule_engine: Optional[Any]) -> List[str]:
    """
    Analyzes a requirement string for ambiguity issues.

    Behavior:
    - If a RuleEngine with `check_ambiguity()` is provided and the 'ambiguity' check is enabled,
      delegate to it so JSON-configured rules (binding modal / measurability / alert triggers)
      are applied. Returns a list of issue strings.
    - Otherwise, fall back to legacy weak-word matching using RuleEngine.ambiguity.words.
    """
    text = (requirement_text or "").strip()
    if not text:
        return []

    # Prefer JSON-driven engine if available & enabled
    if isinstance(rule_engine, RuleEngine):
        try:
            if rule_engine.is_check_enabled("ambiguity") and hasattr(rule_engine, "check_ambiguity"):
                hits = rule_engine.check_ambiguity(text) or []
                return list(hits)
        except Exception:
            # Fall through to legacy behavior if the engine raises
            pass

    # ---- Legacy fallback: keyword scan ----
    found_words: List[str] = []
    lower_requirement = text.lower()
    weak_words = []
    try:
        if isinstance(rule_engine, RuleEngine) and rule_engine.is_check_enabled("ambiguity"):
            weak_words = rule_engine.get_ambiguity_words() or []
    except Exception:
        weak_words = []

    # If no configured list, keep a tiny default so it's never a no-op
    if not weak_words:
        weak_words = ["etc.", "about", "approximately", "should", "may", "optimize", "robust", "user-friendly"]

    for word in weak_words:
        if re.search(r"\b" + re.escape(word) + r"\b", lower_requirement, flags=re.IGNORECASE):
            found_words.append(word)

    # Return list (empty list means "no ambiguity issues")
    return found_words


def check_passive_voice(requirement_text: str) -> List[str]:
    """Analyzes a requirement string for passive voice using spaCy."""
    found_phrases: List[str] = []
    doc = nlp(requirement_text or "")
    for token in doc:
        if token.dep_ == "auxpass":
            # Reconstruct a simple phrase around the passive head
            verb_phrase = [child.text for child in token.head.children]
            verb_phrase.append(token.head.text)
            found_phrases.append(" ".join(sorted(verb_phrase, key=lambda x: doc.text.find(x))))
    return found_phrases


def check_incompleteness(requirement_text: str) -> bool:
    """Checks if a requirement is a full sentence by looking for a verb. Returns True if incomplete."""
    doc = nlp(requirement_text or "")
    has_verb = any(token.pos_ in ["VERB", "AUX"] for token in doc)
    return not has_verb


def check_singularity(requirement_text: str) -> List[str]:
    """
    Checks if a requirement contains multiple actions (e.g., coordinated by 'and'/'or'),
    violating the 'singular' principle. Returns a list of coordinating conjunctions found.
    """
    issues: List[str] = []
    doc = nlp(requirement_text or "")
    conjunctions = [
        token.text.lower()
        for token in doc
        if token.dep_ == "cc" and token.text.lower() in ["and", "or"]
    ]
    if conjunctions:
        issues.extend(conjunctions)
    # Return unique set (list) for UI chips
    return list(dict.fromkeys(issues))
