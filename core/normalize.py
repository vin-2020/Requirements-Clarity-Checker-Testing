# core/normalize.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

_ID_BULLET_PREFIX = re.compile(
    r"""^\s*(?:[A-Z]{1,8}-\d{1,6}|[A-Z]{1,3}\d{2,6}|\d+(?:\.\d+)*|(?:ID|Req|REQ)\s*[:#]?\s*\d+|[•\-–—])[\s.:)\]]*""",
    re.IGNORECASE | re.VERBOSE,
)

def _strip_leading_id_and_article(s: str) -> str:
    s = (s or "").strip()
    s = _ID_BULLET_PREFIX.sub("", s, count=1).strip()
    return re.sub(r"^(the|a|an)\s+", "", s, flags=re.I)

def _tok(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[‘’]", "'", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

@dataclass
class NormReq:
    id: str
    doc: str
    text: str
    subject_key: str
    object_key: str
    metric: Optional[str]
    rel: Optional[str]
    value: Optional[float]
    unit: Optional[str]
    polarity: int
    scope: Optional[str]
    target_a: Optional[str]
    target_b: Optional[str]
    numbers: List[Dict[str, Any]]  # generic extracted quantities anywhere in the sentence

_TIME = {"ms":"ms","millisecond":"ms","milliseconds":"ms","s":"s","sec":"s","secs":"s","second":"s","seconds":"s",
         "min":"min","mins":"min","minute":"min","minutes":"min","h":"h","hr":"h","hrs":"h","hour":"h","hours":"h",
         "day":"day","days":"day"}
_LEN  = {"m":"m","meter":"m","meters":"m","cm":"cm","centimeter":"cm","centimeters":"cm","km":"km","kilometer":"km","kilometers":"km"}

def _canon_unit(u: str) -> str:
    u = (u or "").lower()
    if u in _TIME: return _TIME[u]
    if u in _LEN:  return _LEN[u]
    if u == "percent": return "%"
    return u

def _subject_key(s: str) -> str:
    t = _tok(_strip_leading_id_and_article(s))
    m = re.search(r"\b(payload|thermal control system|tcs|autonomy module|guidance computer|propulsion subsystem|communications subsystem|spacecraft|system)\b", t)
    if m: return m.group(1)
    if " shall " in t:
        left = t.split(" shall ", 1)[0].strip()
        return left[:80] or "system"
    return t[:80] or "system"

def _object_key(s: str) -> str:
    t = _tok(_strip_leading_id_and_article(s))
    patterns = [
        (r"\bclosed[- ]loop control update\b|\bcontrol update\b", "control loop update"),
        (r"\bschedul(?:ing|er) slice(s)?\b", "cpu scheduler"),
        (r"\bnavigation(?: solution)?\b", "navigation solution"),
        (r"\bpropellant mass fraction\b", "propulsion propellant fraction"),
        (r"\bqueue\b.*\boperator\b.*\bapproval\b", "cmd apply policy"),
        (r"\baccept and apply any authenticated command\b", "cmd apply policy"),
        (r"\bprimary\b|\bprioriti[sz]e\b", "priority policy"),
    ]
    for pat, tag in patterns:
        if re.search(pat, t): return tag
    if " shall " in t:
        tail = t.split(" shall ", 1)[1]
        return " ".join(tail.split()[:6]) or "requirement"
    return " ".join(t.split()[:6]) or "requirement"

def _scope(s: str) -> Optional[str]:
    t = _tok(s)
    for sc in ["orbit insertion","emergency mode","declared catastrophic failure mode","low-power mode","science operations"]:
        if sc in t: return sc
    return None

def _polarity(s: str) -> int:
    t = _tok(s)
    if re.search(r"\b(shall not|must not|prohibit|inhibit|forbid)\b", t): return -1
    return +1

# Specific extractors (optional metric hints)
def _grab_number_unit(s: str, unit_map: dict) -> Optional[Tuple[float, str]]:
    m = re.search(r"(\d+(?:\.\d+)?(?:e-?\d+)?)\s*(" + "|".join(map(re.escape, unit_map.keys())) + r")\b", s, re.I)
    if not m: return None
    return float(m.group(1)), unit_map[m.group(2).lower()]

def _extract_closed_loop_deadline(s: str):
    if re.search(r"\bclosed[- ]loop control update deadlines?\b", s, re.I):
        m = _grab_number_unit(s, _TIME)
        if m:
            v, u = m
            return ("deadline", "<=", v, u or "ms")
    return (None, None, None, None)

def _extract_scheduler_timeslice(s: str):
    if re.search(r"\bschedul(?:ing|er) slice(s)?\b", s, re.I):
        m = re.search(
            r"\bminimum of\s+(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|second|seconds|min|mins|minute|minutes|h|hr|hrs|hour|hours)\b",
            s,
            re.I,
        )
        if m:
            return ("timeslice", ">=", float(m.group(1)), _canon_unit(m.group(2)))
    return (None, None, None, None)

def _extract_navigation_accuracy(s: str):
    if re.search(r"\bposition accuracy\b", s, re.I):
        m = re.search(r"(?:<=|≤)\s*(\d+(?:\.\d+)?)\s*(m|cm|km)\b", s, re.I)
        if m: return ("pos_error","<=", float(m.group(1)), _canon_unit(m.group(2)))
    m = re.search(r"\baccepting position errors up to\s+(\d+(?:\.\d+)?)\s*(m|cm|km)\b", s, re.I)
    if m and re.search(r"\bnavigation solution\b|\bnavigation\b", s, re.I):
        return ("pos_error","<=", float(m.group(1)), _canon_unit(m.group(2)))
    return (None,None,None,None)

def _extract_probability_pc(s: str):
    m = re.search(r"\bP[_ ]?c\s*(?:>=|≥)\s*(\d+(?:\.\d+)?(?:e-?\d+)?)\b", s, re.I)
    if m:
        try:
            return ("collision_prob", ">=", float(m.group(1)), "")
        except ValueError:
            return ("collision_prob", ">=", float(str(m.group(1)).replace(" ", "")), "")
    return (None, None, None, None)

def _extract_propellant_mass_fraction(s: str):
    m = re.search(
        r"\bpropellant mass fraction\b.*?(?:<=|≤|not exceed(?:s|ed)?|shall not exceed)\s*(\d+(?:\.\d+)?)\s*%\b",
        s,
        re.I,
    )
    if m:
        return ("mass_fraction", "<=", float(m.group(1)), "%")
    return (None, None, None, None)

def _extract_priority(s: str):
    m = re.search(r"(?:always\s+)?use\s+([a-z0-9 \-]+?)\s+as the primary", s, re.I)
    if m: return (m.group(1).strip().lower(), None)
    m = re.search(r"prioriti[sz]e\s+([a-z0-9 \-]+?)\s+over\s+([a-z0-9 \-]+)", s, re.I)
    if m: return (m.group(1).strip().lower(), m.group(2).strip().lower())
    return (None, None)

def _requires_operator_approval(s: str) -> Optional[int]:
    t = _tok(s)
    if re.search(r"\b(queue|hold|retain)\b.*\b(operator|human)\b.*\bapproval\b", t) and re.search(r"\bapply\b|\bexecute\b", t): return +1
    if re.search(r"\bshall require\b.*\bapproval\b", t): return +1
    if re.search(r"\baccept and apply any authenticated command\b", t): return -1
    if re.search(r"\bapply\b.*\bwithout\b.*\bapproval\b", t): return -1
    return None

def _is_redundancy_statement(s: str) -> bool:
    return bool(re.search(r"\btriple[- ]redundant\b|\bsingle[- ]fault tolerant\b", s, re.I))

# Generic quantity pickup (works even without metric hints)
_REL_WORDS = [
    (r"\bno more than\b|\bat most\b|\bnot exceed(?:s|ed|ing)?\b|\bshall not exceed\b|\bwithin\b|\blimit(?:ed)? to\b", "<="),
    (r"\bat least\b|\bno less than\b|\bnot earlier than\b|\bno earlier than\b|\bafter\b", ">="),
    (r"\bbetween\b", "between"),
    (r"(?:<=|≤)", "<="),
    (r"(?:>=|≥)", ">="),
    (r"(?:==|=)\b", "=="),
    (r"\bless than\b|<", "<"),
    (r"\bgreater than\b|>", ">"),
]
_UNIT_GROUP = (
    r"(?:ms|millisecond|milliseconds|s|sec|secs|second|seconds|min|mins|minute|minutes|h|hr|hrs|hour|hours|day|days|"
    r"m|meter|meters|cm|centimeter|centimeters|km|kilometer|kilometers|%|percent|)"
)

def _extract_all_quantities(text: str) -> List[Dict[str, Any]]:
    s = text
    out: List[Dict[str, Any]] = []

    # between X and Y unit
    for g in re.finditer(
        rf"\bbetween\s+(\d+(?:\.\d+)?(?:e-?\d+)?)\s*(?:and|-)\s*(\d+(?:\.\d+)?(?:e-?\d+)?)\s*({_UNIT_GROUP})\b",
        s,
        re.I,
    ):
        v1 = float(g.group(1))
        v2 = float(g.group(2))
        u = _canon_unit(g.group(3) or "")
        out.append({"rel": "between", "value": (min(v1, v2), max(v1, v2)), "unit": u, "span": g.span()})

    # verbal comparators
    for patt, rel in _REL_WORDS:
        for g in re.finditer(rf"(?:{patt})\s*(\d+(?:\.\d+)?(?:e-?\d+)?)\s*({_UNIT_GROUP})\b", s, re.I):
            out.append({"rel":rel,"value":float(g.group(1)),"unit":_canon_unit(g.group(2) or ""),"span":g.span()})

    # symbol forms (<= 5 ms, etc.)
    for sym, rel in [("<=","<="), (">=",">="), ("<","<"), (">",">"), ("==","==")]:
        for g in re.finditer(rf"{re.escape(sym)}\s*(\d+(?:\.\d+)?(?:e-?\d+)?)\s*({_UNIT_GROUP})\b", s, re.I):
            out.append({"rel":rel,"value":float(g.group(1)),"unit":_canon_unit(g.group(2) or ""),"span":g.span()})

    # normalize "percent"
    for q in out:
        if q["unit"] == "percent": q["unit"] = "%"

    # dedupe overlaps (keep longest)
    out.sort(key=lambda d: (d["span"][0], -(d["span"][1]-d["span"][0])))
    dedup: List[Dict[str, Any]] = []
    last_end = -1
    for q in out:
        start, end = q["span"]
        if start >= last_end:
            dedup.append({k:v for k,v in q.items() if k != "span"})
            last_end = end
    return dedup

# ---------------- main ----------------
def normalize_requirement(rid: str, doc: str, text: str) -> NormReq:
    raw = text or ""
    cleaned = _strip_leading_id_and_article(raw)
    t = _tok(cleaned)

    subj = _subject_key(cleaned)
    obj  = _object_key(cleaned)
    scp  = _scope(cleaned)
    pol  = _polarity(cleaned)

    metric = rel = unit = None
    value: Optional[float] = None
    ta = tb = None

    # Specific, high-confidence metric hints
    for grab in (_extract_closed_loop_deadline, _extract_scheduler_timeslice,
                 _extract_navigation_accuracy, _extract_probability_pc,
                 _extract_propellant_mass_fraction):
        m, r, v, u = grab(cleaned)
        if m:
            metric, rel, value, unit = m, r, v, u
            break

    # Priority
    pa, pb = _extract_priority(cleaned)
    if pa or pb:
        metric = "priority"; ta, tb = pa, pb

    # Approval policy
    appr = _requires_operator_approval(cleaned)
    if appr is not None:
        metric = "approval_required"
        obj = "cmd apply policy"
        rel = "require" if appr == +1 else "allow"
        pol = +1 if appr == +1 else -1
        value = None; unit = None

    # Redundancy (informational)
    if metric is None and _is_redundancy_statement(cleaned):
        metric = "redundancy"

    # Generic quantities (for interval contradictions)
    numbers = _extract_all_quantities(cleaned)

    return NormReq(
        id=rid, doc=doc, text=raw,
        subject_key=subj, object_key=obj,
        metric=metric, rel=rel, value=value, unit=(unit or ""),
        polarity=pol, scope=scp,
        target_a=ta, target_b=tb,
        numbers=numbers,
    )
