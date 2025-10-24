# core/contradiction_runner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import itertools
import json
import re

@dataclass
class Finding:
    kind: str           # "hard", "soft", "timing", "policy", etc.
    reason: str
    a_id: str
    a_doc: str
    a_text: str
    b_id: str
    b_doc: str
    b_text: str
    scope: str | None = None

_JSON_GRAB = re.compile(r'\{[\s\S]*\}', re.M)  # grab first JSON block

PROMPT = """You are a rigorous systems-engineering reviewer.
You will be given TWO requirements, A and B.
Decide if they are in contradiction.

Definition:
- "hard": cannot be simultaneously satisfied (directly opposite, or thresholds make one impossible with the other).
- "soft": plausible tension likely requiring trade-offs but not strictly impossible.
- "none": no contradiction.

Rules:
- Be conservative; prefer "hard" only when clearly impossible under the same scope/time/conditions.
- Consider implicit scope/time: if scopes differ (e.g., different modes), prefer "none" unless same scope is implied.
- If the conflict is about timing/latency/throughput/limits, use kind "timing".
- If the conflict is policy/authorization/approval (allow vs require approval), use kind "policy".
- If numeric thresholds clash (<= vs >=, min vs max) under same subject/scope, use "hard".

Return ONLY a single JSON object (no extra text) with this schema:
{
  "decision": "hard" | "soft" | "timing" | "policy" | "none",
  "reason": "short human explanation",
  "scope": "optional short scope tag or empty string"
}

A:
ID: {a_id}
DOC: {a_doc}
TEXT: {a_text}

B:
ID: {b_id}
DOC: {b_doc}
TEXT: {b_text}
"""

def _first_json_block(txt: str) -> Dict[str, Any] | None:
    if not txt:
        return None
    m = _JSON_GRAB.search(txt)
    if not m:
        return None
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        raw = raw.replace("\n", " ").replace("\r", " ")
        raw = re.sub(r",\s*}", "}", raw)
        raw = re.sub(r",\s*]", "]", raw)
        try:
            return json.loads(raw)
        except Exception:
            return None

def _pairwise_topk(items: List[Any], topk: int) -> List[Tuple[int, int]]:
    pairs = list(itertools.combinations(range(len(items)), 2))
    return pairs[:topk] if topk and topk > 0 else pairs

def run_with_ai(norm_reqs: List[Any], CTX: Dict[str, Any]) -> List[Finding]:
    """
    AI-only contradiction finder.
    `norm_reqs` is a list of objects with: .id, .doc, .text (we only use those 3).
    CTX requires:
        CTX["USE_AI_CONTRA"]=True, CTX["call_ai_contra"]=callable,
        CTX["AI_CONTRA_TOPK"], CTX["AI_CONTRA_PER_CALL_S"]
    """
    if not norm_reqs or not CTX.get("USE_AI_CONTRA"):
        return []

    call_ai = CTX.get("call_ai_contra")
    if not callable(call_ai):
        return []

    topk = int(CTX.get("AI_CONTRA_TOPK", 60))
    per_call_s = int(CTX.get("AI_CONTRA_PER_CALL_S", 12))

    pairs = _pairwise_topk(norm_reqs, topk)
    out: List[Finding] = []

    for i, j in pairs:
        a = norm_reqs[i]
        b = norm_reqs[j]
        prompt = PROMPT.format(
            a_id=getattr(a, "id", ""),
            a_doc=getattr(a, "doc", ""),
            a_text=getattr(a, "text", ""),
            b_id=getattr(b, "id", ""),
            b_doc=getattr(b, "doc", ""),
            b_text=getattr(b, "text", ""),
        )
        try:
            resp = call_ai(prompt, timeout_s=per_call_s)
        except Exception:
            resp = ""

        js = _first_json_block(resp)
        if not js:
            continue

        decision = (js.get("decision") or "none").strip().lower()
        if decision == "none":
            continue

        reason = (js.get("reason") or "").strip()
        scope  = (js.get("scope") or "").strip() or None

        if decision in ("hard", "soft", "timing", "policy"):
            kind = decision
        else:
            kind = "soft"

        out.append(Finding(
            kind=kind,
            reason=reason,
            a_id=getattr(a, "id", ""),
            a_doc=getattr(a, "doc", ""),
            a_text=getattr(a, "text", ""),
            b_id=getattr(b, "id", ""),
            b_doc=getattr(b, "doc", ""),
            b_text=getattr(b, "text", ""),
            scope=scope,
        ))

    return out
