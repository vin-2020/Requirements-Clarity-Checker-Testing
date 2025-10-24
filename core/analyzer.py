# core/analyzer.py
import re
from typing import List, Optional, Any

# --- Try to load spaCy lazily and fall back cleanly ---------------------------
try:
    import spacy
except Exception:  # spaCy not available at all
    spacy = None

_NLP = None  # cached pipeline


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    if spacy is None:
        return None
    try:
        _NLP = spacy.load("en_core_web_sm", disable=["ner"])  # lighter
    except Exception:
        # Fallback: minimal blank English (no parser/tagger = safest no-op)
        try:
            _NLP = spacy.blank("en")
        except Exception:
            _NLP = None
    return _NLP


# --- Gates to avoid false positives on code & non-requirements ----------------
_MODAL_RE = re.compile(r"\b(shall|must|should|will)\b", re.I)

def _has_modal_language(s: str) -> bool:
    return bool(_MODAL_RE.search(s or ""))

_CODE_SIGNS = [
    r"^\s*(from\s+\w+(\.\w+)*\s+import\s+|import\s+\w+)",  # imports
    r"^\s*(class|def)\s+\w+\s*\(",                         # defs/classes
    r":\s*$",                                              # python indented blocks
    r"^\s*@\w+",                                           # decorators
    r"\breturn\b|\btry:\b|\bexcept\b|\bfinally:\b",        # keywords
    r"[{};]",                                              # non-prose punct
    r"\bself\.",                                           # OO code
    r"=\s*(re\.compile|lambda|\[|\{)",                     # assignments
    r"r?\"\"\"|r?\'\'\'",                                  # triple quotes / regex
    r"\b(?:True|False|None)\b",
    r"^\s*#\s*[-=]{3,}",                                   # block comment rulers
]
_CODE_RE = re.compile("|".join(_CODE_SIGNS), re.I)

def _looks_like_code(s: str) -> bool:
    if not s:
        return False
    t = s.strip()
    if len(t) < 3:
        return False
    sym_ratio = sum(ch in r"(){}[]:;=|\/<>.*+-_#\"'" for ch in t) / max(len(t), 1)
    snake = len(re.findall(r"\b[a-z]+_[a-z0-9_]+\b", t))
    camel = len(re.findall(r"\b[a-z]+[A-Z][A-Za-z0-9]+\b", t))
    return bool(_CODE_RE.search(t)) or sym_ratio > 0.20 or (snake + camel) >= 2


# --- Ambiguity ---------------------------------------------------------------
def check_requirement_ambiguity(requirement_text: str, rule_engine: Optional[Any]) -> List[str]:
    """
    Returns a list of ambiguous/weak words found. Empty list => no ambiguity issues.
    Gate: skip when line looks like code OR lacks modal language (not a requirement).
    """
    text = (requirement_text or "").strip()
    if not text or _looks_like_code(text) or not _has_modal_language(text):
        return []

    # Prefer JSON-driven engine if available & enabled
    try:
        from core.rule_engine import RuleEngine  # local import to avoid hard dep at import time
        if isinstance(rule_engine, RuleEngine):
            try:
                if rule_engine.is_check_enabled("ambiguity") and hasattr(rule_engine, "check_ambiguity"):
                    hits = rule_engine.check_ambiguity(text) or []
                    return list(hits)
            except Exception:
                pass
    except Exception:
        pass

    # ---- Legacy fallback: keyword scan ----
    lower_requirement = text.lower()
    weak_words = []
    try:
        # Use configured list if present
        from core.rule_engine import RuleEngine  # re-import if earlier failed
        if isinstance(rule_engine, RuleEngine) and rule_engine.is_check_enabled("ambiguity"):
            weak_words = rule_engine.get_ambiguity_words() or []
    except Exception:
        weak_words = []

    # Minimal default list (kept small on purpose)
    if not weak_words:
        weak_words = ["etc.", "about", "approximately", "optimize", "robust", "user-friendly"]

    found_words: List[str] = []
    for word in weak_words:
        if re.search(r"\b" + re.escape(word) + r"\b", lower_requirement, flags=re.IGNORECASE):
            found_words.append(word)

    return found_words


# --- Passive voice -----------------------------------------------------------
def check_passive_voice(requirement_text: str) -> List[str]:
    """
    Returns a list of detected passive constructs. Empty list => no passive found.
    Gates:
      - Skip code-like lines.
      - Skip lines without modal language (treat as non-requirements).
    If spaCy parser is unavailable, use a light heuristic: 'be' + VBN-like token.
    """
    text = (requirement_text or "")
    if not text or _looks_like_code(text) or not _has_modal_language(text):
        return []

    nlp = _get_nlp()
    if nlp is None or not getattr(nlp, "has_pipe", lambda *_: False)("parser"):
        # Heuristic fallback: look for "be" auxiliaries + past participle tokens
        # e.g., "shall be updated", "must be calibrated"
        heur = re.findall(r"\b(shall|must|should|will)\s+be\s+([a-z]+ed)\b", text, flags=re.I)
        return [f"be {v}" for _, v in heur]

    found_phrases: List[str] = []
    doc = nlp(text)
    # Prefer dependency-based detection when available
    for token in doc:
        # auxpass e.g., "be" attached to a verb head in passive constructions
        if token.dep_ == "auxpass" and token.head and token.head.pos_ == "VERB":
            # capture a small window around the head
            start = max(token.head.i - 1, 0)
            end = min(token.head.i + 2, len(doc))
            span = doc[start:end].text
            found_phrases.append(span)
    return found_phrases


# --- Incompleteness ----------------------------------------------------------
def check_incompleteness(requirement_text: str) -> bool:
    """
    True if incomplete. We treat non-requirements (no modal) as NOT incomplete—
    they simply aren't requirements to score.
    """
    text = (requirement_text or "")
    if not text or _looks_like_code(text) or not _has_modal_language(text):
        return False

    nlp = _get_nlp()
    if nlp is None or not getattr(nlp, "has_pipe", lambda *_: False)("tagger"):
        # Heuristic: look for any verb-ish word after the modal
        return not bool(re.search(r"\b(shall|must|should|will)\b\s+\w+", text, flags=re.I))

    doc = nlp(text)
    has_verb = any(t.pos_ in ("VERB", "AUX") for t in doc)
    return not has_verb


# --- Singularity -------------------------------------------------------------
def check_singularity(requirement_text: str) -> List[str]:
    """
    Returns list of coordinating conjunctions ('and', 'or') indicating multiple actions.
    Gate on code-like and non-requirements.
    Smarter heuristic: only flag when we see coordination between VERB heads (or multiple root verbs).
    """
    text = (requirement_text or "")
    if not text or _looks_like_code(text) or not _has_modal_language(text):
        return []

    nlp = _get_nlp()
    if nlp is None or not getattr(nlp, "has_pipe", lambda *_: False)("parser"):
        # Heuristic fallback: modal ... VERB ... (and|or) ... VERB
        if re.search(r"\b(shall|must|should|will)\b.*\b\w+(?:ed|ing|e|s)\b.*\b(and|or)\b.*\b\w+(?:ed|ing|e|s)\b", text, flags=re.I):
            conj = re.findall(r"\b(and|or)\b", text, flags=re.I)
            return list(dict.fromkeys([c.lower() for c in conj]))
        return []

    doc = nlp(text)
    # Count coordinated verb heads
    issues: List[str] = []
    # Gather verbs that are heads of coordinated actions
    verb_tokens = [t for t in doc if t.pos_ == "VERB"]
    has_coordination = False
    conj_words = set()

    for t in verb_tokens:
        # look for a coordinating conjunction attached to this verb
        for child in t.children:
            if child.dep_ == "cc" and child.text.lower() in ("and", "or"):
                conj_words.add(child.text.lower())
                has_coordination = True
        # also catch verb-verb conj relations (e.g., "shall detect and report")
        if t.dep_ == "conj" and t.head.pos_ == "VERB":
            has_coordination = True

    if has_coordination:
        issues.extend(sorted(conj_words))
    return issues
