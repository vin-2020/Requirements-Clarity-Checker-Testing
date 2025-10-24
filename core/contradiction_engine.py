# core/contradiction_engine.py
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

@dataclass
class Finding:
    kind: str               # "numeric_conflict", "policy_conflict", "priority_conflict", "duration_conflict"
    reason: str
    scope: Optional[str]
    a_id: str
    a_doc: str
    a_text: str
    b_id: str
    b_doc: str
    b_text: str

# ---------- Unit handling ----------
_TIME_TO_SEC = {"ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0, "": 1.0}
_LEN_TO_M   = {"cm": 0.01, "m": 1.0, "km": 1000.0, "": 1.0}
_FRAC       = {"%": 0.01, "": 1.0}

def _to_base(metric: str, value: float, unit: str) -> float:
    if value is None:
        return float("nan")
    u = (unit or "").lower()
    if metric in {"deadline", "timeslice", "duration"}:
        return value * _TIME_TO_SEC.get(u, 1.0)
    if metric in {"pos_error", "miss_distance"}:
        return value * _LEN_TO_M.get(u, 1.0)
    if metric in {"mass_fraction"}:
        return value * _FRAC.get(u, 1.0)
    # collision_prob and others are unitless
    return value

def _same(a: Optional[str], b: Optional[str]) -> bool:
    return (a or "") == (b or "")

def _same_context(a, b) -> bool:
    # strict: same metric & same object; scope equal (or both blank)
    return (a.metric == b.metric) and (a.object_key == b.object_key) and _same(a.scope, b.scope)

def _close(x: float, y: float, tol=1e-12) -> bool:
    try:
        return abs(x - y) <= tol * max(1.0, abs(x), abs(y))
    except Exception:
        return False

def _pair_key(a, b) -> Tuple[str, str, str]:
    lo, hi = sorted([a.id, b.id])
    return (lo, hi, a.metric or "")

# ---------- Numeric contradiction logic ----------
def _extract_numeric(text):
    # Extracts (operator, value, unit) from requirement text, e.g. "≤ 0.05 m"
    m = re.search(r'(≤|>=|≥|<=|<|>|=)?\s*([0-9.eE+-]+)\s*([a-zA-Z%]*)', text)
    if m:
        op = m.group(1) or "="
        val = float(m.group(2))
        unit = m.group(3) or ""
        return op, val, unit
    return None, None, None

def _numeric_conflict(a, b) -> Optional[str]:
    # Improved: Detects contradiction in numeric thresholds (e.g. "at least 20 ms" vs "5 ms")
    try:
        # Try to extract from text if not present as value/unit
        va = _to_base(a.metric, getattr(a, "value", None), getattr(a, "unit", None))
        vb = _to_base(b.metric, getattr(b, "value", None), getattr(b, "unit", None))
        if (va is None or vb is None or any(map(lambda z: z != z, [va, vb]))) or (a.metric != b.metric):
            # Try to extract from text
            opa, va2, ua = _extract_numeric(a.text)
            opb, vb2, ub = _extract_numeric(b.text)
            if va2 is not None and vb2 is not None and _same(a.metric, b.metric):
                va = _to_base(a.metric, va2, ua)
                vb = _to_base(b.metric, vb2, ub)
                # Check for contradiction in thresholds
                # e.g. "at least 20 ms" (>=20) vs "5 ms" (==5)
                if (opa in (">=", "≥") and opb in ("<=", "≤") and va > vb) or \
                   (opa in ("<=", "≤") and opb in (">=", "≥") and va < vb):
                    return f"Contradictory numeric thresholds: {a.text} vs {b.text}"
                if (opa in (">",) and opb in ("<",) and va >= vb) or \
                   (opa in ("<",) and opb in (">",) and va <= vb):
                    return f"Contradictory numeric thresholds: {a.text} vs {b.text}"
                if opa == "=" and opb == "=" and not _close(va, vb):
                    return f"Numeric mismatch: {a.metric}={va} vs {b.metric}={vb}"
            return None
        # If both are numbers, check for mismatch
        if abs(va - vb) > 1e-9:
            return f"Numeric mismatch: {a.metric}={va} vs {b.metric}={vb}"
    except Exception:
        return None
    return None

def _extract_action_object_polarity(text: str):
    # Use spaCy if available, else fallback to regex
    actions = set()
    objects = set()
    polarity = 1
    negation_words = {"not", "no", "never", "prohibit", "forbid", "deny", "without", "except", "cannot", "must not", "shall not"}
    text_l = text.lower()
    if any(w in text_l for w in negation_words):
        polarity = -1
    if _NLP:
        doc = _NLP(text)
        for token in doc:
            if token.pos_ == "VERB":
                actions.add(token.lemma_)
            if token.dep_ in {"dobj", "pobj", "nsubj", "attr"}:
                objects.add(token.lemma_)
    else:
        # Fallback: extract verbs and nouns via regex
        actions.update(re.findall(r"\b[a-z]{4,}ing\b|\b[a-z]{4,}ed\b|\b[a-z]{4,}\b", text_l))
        objects.update(re.findall(r"\b[a-z]{4,}\b", text_l))
    return actions, objects, polarity

# ---------- Policy (allow vs require vs prohibit) ----------
def _policy_conflict(a, b) -> Optional[str]:
    # Use pre-extracted actions/objects/polarity if available
    try:
        actions_a = getattr(a, "actions", None)
        objects_a = getattr(a, "objects", None)
        polarity_a = getattr(a, "polarity", None)
        actions_b = getattr(b, "actions", None)
        objects_b = getattr(b, "objects", None)
        polarity_b = getattr(b, "polarity", None)
        if actions_a is not None and objects_a is not None and polarity_a is not None and \
           actions_b is not None and objects_b is not None and polarity_b is not None:
            # If actions and objects overlap, but polarity is opposed, it's a contradiction
            if actions_a & actions_b and objects_a & objects_b and polarity_a != polarity_b:
                return "Contradictory policy: same action/object with opposite intent."
            if (actions_a & actions_b or objects_a & objects_b) and polarity_a != polarity_b:
                return "Potential contradiction: opposite intent for similar action/object."
        # fallback to text-based extraction if needed
        txta = a.text.lower()
        txtb = b.text.lower()
        prohibit_words = ["prohibit", "not allow", "no ", "never", "not ", "shall not", "must not", "cannot", "may not"]
        require_words = ["require", "shall", "must"]
        only_words = ["only"]
        except_words = ["except"]
        always_words = ["always"]
        after_words = ["after"]
        before_words = ["before"]

        def has_any(txt, words):
            return any(w in txt for w in words)

        require_a = has_any(txta, require_words)
        prohibit_a = has_any(txta, prohibit_words)
        only_a = has_any(txta, only_words)
        except_a = has_any(txta, except_words)
        always_a = has_any(txta, always_words)
        after_a = has_any(txta, after_words)
        before_a = has_any(txta, before_words)

        require_b = has_any(txtb, require_words)
        prohibit_b = has_any(txtb, prohibit_words)
        only_b = has_any(txtb, only_words)
        except_b = has_any(txtb, except_words)
        always_b = has_any(txtb, always_words)
        after_b = has_any(txtb, after_words)
        before_b = has_any(txtb, before_words)

        # Contradiction if one requires and the other prohibits
        if (require_a and prohibit_b) or (prohibit_a and require_b):
            return "Contradictory policy: one requires, the other prohibits."
        # Contradiction if both "shall" but one says "only" or "except" and the other does not
        if (only_a and not except_a and not only_b) or (only_b and not except_b and not only_a):
            return "Contradictory exclusivity in policy."
        # Contradiction if one says "always" and the other says "never" or "only after"
        if (always_a and (prohibit_b or only_b or except_b or after_b or before_b)) or \
           (always_b and (prohibit_a or only_a or except_a or after_a or before_a)):
            return "Contradictory temporal policy."
        # Contradiction if one says "accept any" and the other says "only after approval"
        if ("accept any" in txta and "only after" in txtb) or ("accept any" in txtb and "only after" in txta):
            return "Contradictory acceptance policy."
        # Contradiction if both mention the same object but one is "require" and the other is "only after" or "except"
        if (require_a and (after_b or except_b)) or (require_b and (after_a or except_a)):
            return "Contradictory conditional policy."
        # Contradiction if both mention "and" or "or" (multiple actions) and one prohibits or restricts
        if ((" and " in txta or " or " in txta) and (prohibit_a or only_a or except_a)) or \
           ((" and " in txtb or " or " in txtb) and (prohibit_b or only_b or except_b)):
            return "Contradictory multi-action policy."
        # Fallback to polarity logic
        if hasattr(a, "polarity") and hasattr(b, "polarity"):
            if a.polarity * b.polarity < 0:
                return "Opposed actions or policies detected."
    except Exception:
        pass
    return None

# ---------- Priority conflicts ----------
def _priority_conflict(a, b) -> Optional[str]:
    if a.metric != "priority" or b.metric != "priority":
        return None
    if not _same(a.scope, b.scope):
        return None
    sa = set(filter(None, [a.target_a, a.target_b]))
    sb = set(filter(None, [b.target_a, b.target_b]))
    if sa and sb and not sa.intersection(sb):
        return "Conflicting source priorities in the same scope."
    return None

# ---------- Main ----------
def run_detectors(items) -> List[Finding]:
    out: List[Finding] = []
    seen = set()

    n = len(items)
    for i in range(n):
        a = items[i]
        for j in range(i + 1, n):
            b = items[j]
            reason = None

            # Numeric conflicts (aggressive, always check)
            reason = _numeric_conflict(a, b)
            if reason:
                key = (a.id, b.id, "numeric")
                if key not in seen:
                    seen.add(key)
                    out.append(Finding(
                        kind="numeric_conflict",
                        reason=reason,
                        scope=getattr(a, "scope", None) or getattr(b, "scope", None),
                        a_id=a.id, a_doc=a.doc, a_text=a.text,
                        b_id=b.id, b_doc=b.doc, b_text=b.text,
                    ))
                continue

            # Policy conflicts (aggressive, always check)
            reason = _policy_conflict(a, b)
            if reason:
                key = (a.id, b.id, "policy")
                if key not in seen:
                    seen.add(key)
                    out.append(Finding(
                        kind="policy_conflict",
                        reason=reason,
                        scope=getattr(a, "scope", None) or getattr(b, "scope", None),
                        a_id=a.id, a_doc=a.doc, a_text=a.text,
                        b_id=b.id, b_doc=b.doc, b_text=b.text,
                    ))
                continue
    return out
