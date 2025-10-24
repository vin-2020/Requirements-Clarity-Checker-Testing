
# ui/tabs/need_tab.py
from __future__ import annotations

import re
import time
from typing import Callable, List, Dict
import json
import pandas as pd
import streamlit as st

# -------- Streamlit rerun compatibility (new & old) --------
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ----------------- Resilient LLM wrapper -------------------
def _llm_retry(api_fn: Callable[[str], str], prompt: str, retries: int = 1, backoff: float = 0.6) -> str:
    """
    AI-only: we DO NOT fall back to local templates. If the provider fails, we surface the error.
    """
    def retriable(msg: str) -> bool:
        m = (msg or "").lower()
        # Only match real service/transport errors, not common words like "rate"
        patterns = [
            "429", "quota exceeded", "rate limit", "rate-limited",
            "deadline exceeded", "timed out", "timeout",
            "unavailable", "503", "overloaded",
            "internal server error", "server error"
        ]
        return any(p in m for p in patterns)

    last = ""
    for i in range(retries + 1):
        try:
            out = api_fn(prompt) or ""
            if retriable(out):
                raise RuntimeError(out)
            return out.strip()
        except Exception as e:
            last = str(e)
            if i < retries and retriable(last):
                time.sleep(backoff * (i + 1))
                continue
            st.error(f"AI service error: {last}")
            return ""



# ----------------- Need normalizer --------------------------
_NEED_SHALL_RX = re.compile(r"\b(shall|must|will)\b", re.I)

def _normalize_need(raw: str) -> str:
    txt = (raw or "").strip()
    if not txt:
        return ""
    if _NEED_SHALL_RX.search(txt):
        txt_no_modal = re.sub(r"\b(the\s+)?(system|uav|vehicle|spacecraft|satellite|platform)\b\s+(shall|must|will)\s+", "", txt, flags=re.I)
        txt_no_modal = re.sub(r"\b(shall|must|will)\s+", "", txt_no_modal, flags=re.I)
        txt = re.sub(r"^\s*(to\s+)?", "", txt_no_modal).strip()
        if not re.match(r"^(enable|provide|maintain|perform|achieve|support)\b", txt, re.I):
            txt = "Enable " + txt[0].lower() + txt[1:]
    txt = re.sub(r"\s+", " ", txt)
    return txt.rstrip(" .")

# ----------------- Domain/keywords inference ----------------
UNIT_RX = r"(?:ms|s|min|hr|Hz|k?Hz|MHz|GHz|°C|K|Pa|kPa|bar|m/s|km/s|m|km|deg|°|A|mA|V|W|kW|dB|%|σ|Sigma|g)"
def _infer_keywords(need: str) -> list[str]:
    low = (need or "").lower()

    lex = {
        "propulsion": ["Δv","burn","thruster","Isp","conjunction","momentum dump","keep-out zone","GN&C","engine-out"],
        "thermal": ["TVAC","radiator","heater","time-at-limit","eclipse","full sun","thermal model","sensor fault"],
        "uas": ["BVLOS","C2 link","GNSS","geo-fence","DAA","latency","endurance","RTH"],
        "comms": ["throughput","latency","availability","packet loss","bandwidth","jitter","link margin"],
        "power": ["state of charge","DoD","battery","bus voltage","power budget","load shed"],
        "safety": ["fault","FDIR","abort","safe mode","redundancy","single-fault tolerant","hazard"],
        "software": ["API","request rate","timeout","retry","telemetry","logging","audit"],
        "navigation": ["odometry","state estimation","drift","star tracker","IMU","EKF","accuracy"],
        "mechanical": ["vibration","thermal cycling","shock","mass budget","CG","modal frequency"],
    }

    bucket_hits = []
    for _, words in lex.items():
        if any(w.lower() in low for w in words):
            bucket_hits.extend(words)

    nums = re.findall(rf"\b[0-9]+(?:\.[0-9]+)?\s*(?:{UNIT_RX})\b", need or "", flags=re.I)

    seen = set()
    out = []
    for w in bucket_hits + nums:
        if w not in seen:
            out.append(w); seen.add(w)

    # Always seed with general quality terms
    for g in ["units","thresholds","modes","conditions","verification","faults","budgets","interfaces","safety"]:
        if g not in seen:
            out.append(g); seen.add(g)
    return out[:20]

# ----------------- Structured-edit parsing helpers ----------
_ACTIONS = ["maintain", "regulate", "limit", "detect", "log", "achieve", "provide", "enforce", "control", "acquire"]

def _extract_trigger(txt: str) -> str:
    m = re.match(r"\s*((?:when|if|while|during)[^,\.]+)[, ]", txt, flags=re.I)
    return (m.group(1).strip() if m else "")

def _extract_conditions(txt: str) -> str:
    m = re.search(r"\b(in (?:nominal (?:mode|conditions)|safe mode|eclipse(?: and full sun)?|full sun))\b", txt, flags=re.I)
    return (m.group(1).strip() if m else "")

def _extract_perf(txt: str) -> str:
    m = re.search(r"\b(within [^\.]+|between [^\.]+|≥ ?[^,\.]+|<= ?[^,\.]+|≤ ?[^,\.]+|>= ?[^,\.]+|±\s?[^,\.]+|no more than [^,\.]+|not exceed [^,\.]+|lasting [^,\.]+|for [^,\.]+ seconds?)", txt, flags=re.I)
    return (m.group(0).strip() if m else "")

def _extract_action(txt: str) -> str:
    for a in _ACTIONS:
        if re.search(rf"\b{a}\b", txt, flags=re.I):
            return a
    return "maintain"

def _extract_object(txt: str) -> str:
    if re.search(r"payload optics", txt, flags=re.I):
        return "payload optics temperature"
    if re.search(r"battery", txt, flags=re.I):
        return "battery temperatures"
    if re.search(r"avionics", txt, flags=re.I):
        return "avionics temperatures"
    if re.search(r"(?:component|onboard)\s+temperatures?|temperatures?\b", txt, flags=re.I):
        return "temperatures"
    if re.search(r"C2 link", txt, flags=re.I):
        return "C2 link"
    if re.search(r"geo-?fenc", txt, flags=re.I):
        return "geo-fencing"
    if re.search(r"endurance", txt, flags=re.I):
        return "endurance"
    if re.search(r"latency", txt, flags=re.I):
        return "latency"
    return "function"

def _extract_actor(txt: str) -> str:
    if re.search(r"\bthermal\b|\bheater|radiator|temperature", txt, flags=re.I):
        return "Thermal Control Subsystem"
    if re.search(r"\bbattery|power\b", txt, flags=re.I):
        return "Power Subsystem"
    if re.search(r"\bpayload\b", txt, flags=re.I):
        return "Payload"
    if re.search(r"\buav|drone\b", txt, flags=re.I):
        return "UAV"
    if re.search(r"\bsystem\b|\bspacecraft\b", txt, flags=re.I):
        return "System"
    return "System"

def _parse_req_text(txt: str) -> dict:
    return {
        "trigger": _extract_trigger(txt),
        "conditions": _extract_conditions(txt),
        "perf": _extract_perf(txt),
        "action": _extract_action(txt),
        "object": _extract_object(txt),
        "actor": _extract_actor(txt),
    }

# ----------------- AI Helpers: Questions --------------------
def _parse_questions_lines(raw: str) -> List[str]:
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln.strip()) for ln in (raw or "").splitlines() if ln.strip()]
    out: List[str] = []
    seen = set()
    for ln in lines:
        if "→ e.g." not in ln:
            if "?" not in ln:
                ln = ln.rstrip(".") + "? → e.g., (add 1–2 unit-bearing examples)."
            else:
                ln = ln + " → e.g., (add 1–2 unit-bearing examples)."
        else:
            pre, post = ln.split("→ e.g.", 1)
            if "?" not in pre:
                pre = pre.rstrip(".") + "?"
            ln = pre.strip() + " → e.g." + post
        key = ln.split("→", 1)[0].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return out

def _ai_questions(need: str, req_type: str, call_fn) -> list[str]:
    if not st.session_state.get("api_key"):
        st.error("Missing API key. Configure your AI provider.")
        return []

    hints = ", ".join(_infer_keywords(need))

    FEWSHOT = """What Δv magnitude accuracy and timing deviation thresholds apply? → e.g., 0.1 m/s (3σ); ±10 s (3σ)
What temperature limits and time-at-limit rules apply per component? → e.g., optics 0–20 °C; ≤300 s above limit/orbit
What C2 link availability and latency must be maintained? → e.g., ≥99.9% availability; ≤200 ms (95%)
What resources/budgets constrain operation? → e.g., heater ≤12 W (eclipse); CPU ≤40%; propellant ≥10% reserve
What failure cases must be detected and handled? → e.g., star tracker dropout; single-engine-out; sensor open/short
What triggers/conditions initiate action or abort? → e.g., threshold crossing; zone violation; GN&C uncertainty > X
What verification methods and levels apply? → e.g., SIL/HIL; TVAC; system demo; flight telemetry correlation
What interfaces/keep-out/compliance constraints apply? → e.g., API v2 rate ≤10 Hz; safety corridor; regulatory TBD"""

    base_prompt = f"""
You are a principal systems engineer. Produce EXACTLY 10 ultra-specific clarifying questions for the NEED below.

FORMAT (strict):
• Each line MUST be “Question? → e.g., <1–2 concrete examples with units>”.
• No bullets or numbering. No blank lines. No preface or postscript.
• Target domain terms: {hints}

NEED:
\"\"\"{(need or '').strip()}\"\"\"\n
STYLE EXAMPLES (do not copy values):
{FEWSHOT}
"""
    raw = _llm_retry(lambda p: call_fn(st.session_state.api_key, p), base_prompt)
    lines = _parse_questions_lines(raw)

    if not (8 <= len(lines) <= 12):
        # Ask the AI to repair to exactly 10 lines
        repair_prompt = f"""
You did not follow instructions. Rewrite the content BELOW into EXACTLY 10 lines.

RULES:
• Each line MUST be “Question? → e.g., <1–2 concrete examples with units>”.
• No bullets, numbering, or extra text. Output ONLY the 10 lines.

NEED:
\"\"\"{(need or '').strip()}\"\"\"\n
YOUR PRIOR OUTPUT:
\"\"\"{raw}\"\"\""""
        raw2 = _llm_retry(lambda p: call_fn(st.session_state.api_key, p), repair_prompt)
        lines = _parse_questions_lines(raw2)

    if len(lines) < 6:
        st.warning("AI returned fewer than 6 clarifying questions. Showing the best available.")
    return lines[:12]

# ----------------- AI Helpers: Requirements -----------------
def _parse_requirements_jsonl(raw: str) -> List[Dict]:
    out: List[Dict] = []
    for ln in (raw or "").splitlines():
        ln = ln.strip()
        if not ln or not ln.startswith("{"):
            continue
        try:
            row = json.loads(ln)
        except Exception:
            continue
        txt = (row.get("text") or "").strip()
        role = (row.get("role") or "").strip().title()
        vm = (row.get("verification_method") or "Test").strip().title()
        vl = (row.get("verification_level") or "Subsystem").strip().title()
        if not txt or " shall " not in f" {txt.lower()} ":
            continue
        if len(txt.split()) > 26:
            continue
        out.append({
            "ID": "",
            "ParentID": "",
            "Text": txt if txt.endswith(".") else (txt + "."),
            "Role": role if role in {"Parent","Child"} else "Child",
            "Verification": vm if vm in {"Test","Analysis","Inspection","Demo"} else "Test",
            "VerificationLevel": vl if vl in {"Unit","Subsystem","System","Mission"} else "Subsystem",
            "VerificationEvidence": "",
            "ValidationNeedID": "",
            "TestCaseIDs": "",
            "AllocatedTo": "",
            "Criticality": "Medium",
            "Status": "Draft",
        })
    return out

def _ai_requirements_raw(need: str, rationale: str, keywords: str, call_fn) -> str:
    schema = """Each output line MUST be a single JSON object with keys EXACTLY:
{
 "role": "Parent"|"Child",
 "text": "Requirement sentence using 'shall' (≤ 26 words, active, singular, verifiable).",
 "verification_method": "Test"|"Analysis"|"Inspection"|"Demo",
 "verification_level": "Unit"|"Subsystem"|"System"|"Mission"
}"""

    FEWSHOT = '''{"role":"Parent","text":"The system shall deliver the specified function within defined performance and safety constraints across operational modes.","verification_method":"Analysis","verification_level":"System"}
{"role":"Child","text":"The system shall achieve end-to-end latency ≤ 200 ms (95%) during nominal operation.","verification_method":"Test","verification_level":"System"}
{"role":"Child","text":"The Thermal Control Subsystem shall maintain payload optics between 0–20 °C during active imaging.","verification_method":"Test","verification_level":"Subsystem"}
{"role":"Child","text":"The system shall abort a burn if GN&C position uncertainty exceeds TBD prior to ignition.","verification_method":"Test","verification_level":"System"}
{"role":"Child","text":"The UAV shall enforce geo-fence error ≤ 10 m (95%) during BVLOS mission.","verification_method":"Demo","verification_level":"System"}'''

    rubric = f"""
QUALITY BAR:
- Use domain vocabulary ({keywords}) and active voice with 'shall'.
- Parent: outcome-oriented, ≤ 26 words.
- Children: one measurable idea each; prefer concrete units; crisp TBDs only if unavoidable.
- Include clear triggers/conditions when useful.

OUTPUT:
- EXACTLY 1 Parent line and 10–12 Child lines.
- JSON Lines only (no prose or code fences).
"""
    prompt = f"""
Generate requirements JSON Lines from the NEED and RATIONALE.

NEED:
\"\"\"{(need or '').strip()}\"\"\"\n
RATIONALE:
\"\"\"{(rationale or '').strip()}\"\"\"\n
{schema}

FEW-SHOT (style):
{FEWSHOT}

{rubric}
"""
    return _llm_retry(lambda p: call_fn(st.session_state.api_key, p), prompt)

def _ai_requirements_repair(raw: str, need: str, rationale: str, call_fn) -> str:
    prompt = f"""
Your previous output did not meet format/count requirements.

REPAIR TASK:
- Convert the content BELOW into valid JSON Lines with EXACTLY 1 Parent and EXACTLY 10 Child lines.
- Preserve intent, remove duplicates, keep ≤ 26 words per text.
- Keys must be: role, text, verification_method, verification_level.
- No prose or code fences. Output JSON Lines only.

NEED:
\"\"\"{(need or '').strip()}\"\"\"\n
RATIONALE:
\"\"\"{(rationale or '').strip()}\"\"\"\n
PRIOR OUTPUT:
\"\"\"{raw}\"\"\""""
    return _llm_retry(lambda p: call_fn(st.session_state.api_key, p), prompt)

def _ai_parent_from_children(need: str, children: List[str], call_fn) -> dict | None:
    """
    Ask AI to synthesize ONE parent requirement summarizing the children.
    Returns a requirement dict or None. (AI-only, no local fallback text.)
    """
    snip = "\n".join(f"- {c}" for c in children[:12])
    prompt = f"""
Create ONE parent requirement in JSON Lines format summarizing the CHILD requirements below.

RULES:
- ≤ 26 words; active voice; 'shall'; outcome-oriented; no lists inside the sentence.
- JSON object keys: role="Parent", text, verification_method, verification_level (suggest System/Analysis).
- Output ONE JSON object ONLY.

NEED:
\"\"\"{(need or '').strip()}\"\"\"\n
CHILD CANDIDATES:
{snip}
"""
    raw = _llm_retry(lambda p: call_fn(st.session_state.api_key, p), prompt)
    items = _parse_requirements_jsonl(raw)
    return items[0] if items else None

def _ai_generate_requirements(need: str, rationale: str, call_fn, need_id: str) -> list[dict]:
    if not st.session_state.get("api_key"):
        st.error("Missing API key. Configure your AI provider.")
        return []

    keywords = ", ".join(_infer_keywords(need))

    # Pass 1: ask for correct counts
    raw = _ai_requirements_raw(need, rationale, keywords, call_fn)
    items = _parse_requirements_jsonl(raw)

    # Evaluate counts
    parents = [r for r in items if r["Role"] == "Parent"]
    children = [r for r in items if r["Role"] == "Child"]

    if not (len(parents) == 1 and 8 <= len(children) <= 16):
        # Pass 2: ask model to repair into exact counts
        raw2 = _ai_requirements_repair(raw, need, rationale, call_fn)
        items2 = _parse_requirements_jsonl(raw2)
        parents = [r for r in items2 if r["Role"] == "Parent"]
        children = [r for r in items2 if r["Role"] == "Child"]

        # If still missing a Parent but we have children, ask AI to synthesize a Parent
        if len(parents) == 0 and len(children) >= 6:
            parent_ai = _ai_parent_from_children(need, [c["Text"] for c in children], call_fn)
            if parent_ai:
                parents = [parent_ai]

        items = parents[:1] + children

    # Attach need_id and defaults
    for r in items:
        r["ValidationNeedID"] = need_id or "NEED-001"
        if r["Verification"] not in {"Test","Analysis","Inspection","Demo"}:
            r["Verification"] = "Test"
        if r["VerificationLevel"] not in {"Unit","Subsystem","System","Mission"}:
            r["VerificationLevel"] = "Subsystem"

    # De-duplicate children text
    seen = set()
    final_parents = [r for r in items if r["Role"] == "Parent"][:1]
    final_children = []
    for ch in [r for r in items if r["Role"] == "Child"]:
        key = re.sub(r"\s+", " ", ch["Text"].lower())
        if key in seen:
            continue
        seen.add(key)
        final_children.append(ch)

    if len(final_parents) == 0 and len(final_children) == 0:
        st.error("AI did not produce usable requirement lines.")
        return []

    if len(final_parents) == 0 and len(final_children) > 0:
        st.warning("AI did not return a Parent; children will be shown standalone. Consider regenerating.")

    return final_parents + final_children[:20]

# ----------------- Render Tab -------------------------------
def render(st, db, rule_engine, CTX):
    """
    Need → Questions (with → e.g.) → Requirements (AI-only, resilient).
    - One-click generation with NEED sanitizer.
    - Editable IDs & text; Rewrite; Decompose only when non-singular; Structured dropdown editor; Delete.
    - Per-item V&V/traceability and CSV export (unchanged).
    """

    # Analyzer & LLM hooks
    safe_call_ambiguity = CTX.get("safe_call_ambiguity", lambda t, e=None: [])
    check_passive_voice = CTX.get("check_passive_voice", lambda t: [])
    check_incompleteness = CTX.get("check_incompleteness", lambda t: [])
    check_singularity = CTX.get("check_singularity", lambda t: [])
    get_ai_suggestion = CTX.get("get_ai_suggestion", lambda *a, **k: "")
    decompose_requirement_with_ai = CTX.get("decompose_requirement_with_ai", lambda *a, **k: "")
    run_freeform = CTX.get("run_freeform", lambda *a, **k: "")
    # NEW: dense-need decomposition hook
    decompose_need_into_requirements = CTX.get("decompose_need_into_requirements", lambda *a, **k: "")

    # Prefer engine-level ambiguity checker if available; otherwise fall back to CTX helper.
    def _ambig(text: str) -> list:
        try:
            if hasattr(rule_engine, "check_ambiguity"):
                out = rule_engine.check_ambiguity(text)
                return out or []
        except Exception:
            pass
        try:
            return safe_call_ambiguity(text, rule_engine) or []
        except TypeError:
            # some CTX implementations expect only the text
            return safe_call_ambiguity(text, None) or []
        except Exception:
            return []


    # ---------- State ----------
    if "need_ui" not in st.session_state:
        st.session_state.need_ui = {}
    S = st.session_state.need_ui
    S.setdefault("req_type", "Functional")
    S.setdefault("priority", "Should")
    S.setdefault("lifecycle", "Operations")
    S.setdefault("stakeholder", "")
    S.setdefault("need_text", "")
    S.setdefault("rationale", "")
    S.setdefault("ai_questions", [])
    S.setdefault("requirements", [])
    S.setdefault("child_counts", {})  # parent_id -> next int
    S.setdefault("need_id", "NEED-001")  # Need ID for validation/traceability

    # ---------- Header ----------
    pname = st.session_state.selected_project[1] if st.session_state.selected_project else None
    st.header("✍️ Need → Requirement Assistant" + (f" — Project: {pname}" if pname else ""))

    # ---------- Top controls ----------
    c_top = st.columns(5)
    with c_top[0]:
        S["req_type"] = st.selectbox("Requirement Type",
                                     ["Functional", "Performance", "Constraint", "Interface"],
                                     index=["Functional", "Performance", "Constraint", "Interface"].index(S["req_type"]))
    with c_top[1]:
        S["priority"] = st.selectbox("Priority", ["Must", "Should", "Could", "Won't (now)"],
                                     index=["Must", "Should", "Could", "Won't (now)"].index(S["priority"]))
    with c_top[2]:
        S["lifecycle"] = st.selectbox("Life-cycle",
                                      ["Concept", "Development", "P/I/V&V", "Operations", "Maintenance", "Disposal"],
                                      index=["Concept", "Development", "P/I/V&V", "Operations", "Maintenance", "Disposal"].index(S["lifecycle"]))
    with c_top[3]:
        S["stakeholder"] = st.text_input("Stakeholder / Role", value=S["stakeholder"],
                                         placeholder="e.g., Flight operator, Acquirer")
    with c_top[4]:
        S["need_id"] = st.text_input("Need ID", value=S["need_id"], help="Used for Validation link & traceability (e.g., NEED-001)")

    # ---------- Need & Rationale ----------
    st.subheader("🧩 Stakeholder Need")
    S["need_text"] = st.text_area(
        "Describe the need (no 'shall')",
        value=S["need_text"],
        height=110,
        placeholder="e.g., Enable autonomous, fault-tolerant operations meeting defined performance and safety constraints across modes and environments.",
    )

    if _NEED_SHALL_RX.search(S["need_text"] or ""):
        st.info("Heads-up: Your need text contains “shall/must/will”. I’ll treat it as an objective (not a requirement) during generation.")

    st.subheader("🎯 Rationale")
    S["rationale"] = st.text_area(
        "Why this matters",
        value=S["rationale"],
        height=80,
        placeholder="e.g., Performance and safety directly affect mission success, risk posture, and regulatory compliance.",
    )

    # ---------- Helpers ----------
    def _qc(text: str):
        try:
            amb = _ambig(text)
        except Exception:
            amb = []
        pas = check_passive_voice(text)
        inc = check_incompleteness(text)
        sing = check_singularity(text)
        return amb, pas, inc, sing


    def _badge_row(text: str) -> str:
        amb, pas, inc, sing = _qc(text)
        def mark(ok, label): return ("✅ " if ok else "⚠️ ") + label
        return f"{mark(not amb,'Unambiguous')}  {mark(not pas,'Active Voice')}  {mark(not inc,'Complete')}  {mark(not sing,'Singular')}"

    def _next_child_id(parent_id: str) -> str:
        S["child_counts"].setdefault(parent_id, 1)
        idx = S["child_counts"][parent_id]
        S["child_counts"][parent_id] = idx + 1
        return f"{parent_id}.{idx}"

    def _append_children_ids(base_parent: str, children_texts: list[str]) -> list[dict]:
        rows = []
        for txt in children_texts:
            rows.append({
                "ID": _next_child_id(base_parent),
                "ParentID": base_parent,
                "Text": txt,
                "Role": "Child",
                "Verification": "Test",
                "VerificationLevel": "Subsystem",
                "VerificationEvidence": "",
                "ValidationNeedID": S.get("need_id", "NEED-001"),
                "TestCaseIDs": "",
                "AllocatedTo": "",
                "Criticality": "Medium",
                "Status": "Draft"
            })
        return rows

    def _ai_rewrite_strict(text: str) -> str:
        if not st.session_state.get("api_key"):
            return text
        prompt = f"Rewrite as ONE singular, unambiguous, verifiable requirement using 'shall', ≤ 22 words. Return only the sentence.\n\n\"\"\"{text.strip()}\"\"\""
        out = _llm_retry(lambda p: run_freeform(st.session_state.api_key, p), prompt)
        return (out.splitlines()[0].strip() if out else text)

    # ---------- Generate (one click) ----------
    st.subheader("❓ Gaps & Clarifying Questions")
    cols_q = st.columns([1.4, 2.6])
    with cols_q[0]:
        if st.button("🔎 Generate Questions & Requirements"):
            need_clean = _normalize_need(S["need_text"])
            if not need_clean.strip():
                st.error("Enter the stakeholder need first.")
            elif not st.session_state.get("api_key"):
                st.error("Missing API key. Configure your AI provider.")
            else:
                with st.spinner("Thinking like a systems engineer…"):
                    # AI-only questions (aim 10, accept 6–12)
                    S["ai_questions"] = _ai_questions(need_clean, S["req_type"], run_freeform)

                    # AI-only requirements (1 parent + 8–12 children; repair loops)
                    reqs = _ai_generate_requirements(
                        need_clean, S["rationale"], run_freeform, S.get("need_id", "NEED-001")
                    )

                    # --- Dense-need fallback if counts are weak ---
                    parent_try = next((r for r in reqs if r["Role"] == "Parent"), None) if reqs else None
                    children_try = [r for r in (reqs or []) if r["Role"] == "Child"]
                    if (not parent_try) or (len(children_try) < 8):
                        decomp_raw = _llm_retry(
                            lambda _: decompose_need_into_requirements(st.session_state.api_key, need_clean),
                            "DENSE_DECOMP"
                        )
                        # Parse decomposition lines into child requirement texts
                        child_texts: List[str] = []
                        for ln in (decomp_raw or "").splitlines():
                            ln = ln.strip()
                            if not ln:
                                continue
                            # strip leading bullets/numbers/REQ-ids
                            ln = re.sub(r'^[\-\*\d]+\.\s*', '', ln)
                            ln = re.sub(r'^REQ-\d{3,5}[.\s:-]\s*', '', ln, flags=re.I).strip()
                            # keep only normative sentences
                            if " shall " in f" {ln.lower()} ":
                                if len(ln.split()) <= 26:
                                    child_texts.append(ln if ln.endswith(".") else (ln + "."))
                        if len(child_texts) >= 8:
                            parent_ai = _ai_parent_from_children(need_clean, child_texts, run_freeform)
                            if parent_ai:
                                # Build items WITHOUT IDs; structuring below will assign IDs
                                parent_ai["ID"] = ""
                                parent_ai["ParentID"] = ""
                                parent_ai["Role"] = "Parent"
                                parent_ai["VerificationLevel"] = parent_ai.get("VerificationLevel") or "System"
                                parent_ai["ValidationNeedID"] = S.get("need_id", "NEED-001")
                                child_items = [{
                                    "ID": "",
                                    "ParentID": "",
                                    "Text": txt,
                                    "Role": "Child",
                                    "Verification": "Test",
                                    "VerificationLevel": "Subsystem",
                                    "VerificationEvidence": "",
                                    "ValidationNeedID": S.get("need_id", "NEED-001"),
                                    "TestCaseIDs": "",
                                    "AllocatedTo": "",
                                    "Criticality": "Medium",
                                    "Status": "Draft",
                                } for txt in child_texts[:16]]
                                reqs = [parent_ai] + child_items
                                st.info("Used dense-need decomposition fallback for broader coverage.")

                    # Assign IDs & structure if we have anything substantial
                    if reqs:
                        parent = next((r for r in reqs if r["Role"] == "Parent"), None)
                        children = [r for r in reqs if r["Role"] == "Child"]

                        if parent:
                            parent_id = "REQ-001"
                            parent["ID"] = parent_id
                            parent["ParentID"] = ""
                            parent["VerificationLevel"] = parent.get("VerificationLevel") or "System"
                            S["child_counts"][parent_id] = 1
                            out_items = [parent]
                            for ch in children:
                                ch["ID"] = _next_child_id(parent_id)
                                ch["ParentID"] = parent_id
                                out_items.append(ch)
                            S["requirements"] = out_items
                            st.success("Questions and requirements generated.")
                        else:
                            # Show children standalone (no broken selector)
                            S["requirements"] = [{
                                **ch,
                                "ID": f"REQ-{i+1:03d}",
                                "ParentID": "",
                                "Role": ch.get("Role","Child")
                            } for i, ch in enumerate(reqs) if ch["Role"] == "Child"]
                            st.warning("No parent returned by AI. Showing child lines; regenerate for a proper hierarchy.")
                    else:
                        st.warning("The AI returned no usable requirement lines. Try refining the need/rationale and generate again.")

                    _rerun()
    with cols_q[1]:
        if not S.get("ai_questions"):
            st.caption("No questions yet. Click **Generate Questions & Requirements**.")
        else:
            for i, q in enumerate(S["ai_questions"], start=1):
                st.markdown(f"{i}. {q}")

    # ---------- Quick Add (Parent / Child) ----------
    st.subheader("🧱 Requirements")
    quick_cols = st.columns([0.20, 0.40, 0.40])
    with quick_cols[0]:
        if st.button("➕ Add Parent", key="add_parent_btn"):
            new_id = "REQ-001" if not S["requirements"] else f"REQ-{len([r for r in S['requirements'] if r['Role']=='Parent'])+1:03d}"
            S["child_counts"].setdefault(new_id, 1)
            S["requirements"].append({
                "ID": new_id,
                "ParentID": "",
                "Text": "The System shall TBD.",
                "Role": "Parent",
                "Verification": "Test",
                "VerificationLevel": "System",
                "VerificationEvidence": "",
                "ValidationNeedID": S.get("need_id", "NEED-001"),
                "TestCaseIDs": "",
                "AllocatedTo": "",
                "Criticality": "Medium",
                "Status": "Draft",
            })
            _rerun()
    with quick_cols[1]:
        parent_choices = [r["ID"] for r in S["requirements"] if r["Role"] == "Parent"]
        if parent_choices:
            sel_parent = st.selectbox("Parent for new child", parent_choices, key="add_child_parent")
        else:
            sel_parent = None
            st.info("Add a Parent first to enable child insertion.")
    with quick_cols[2]:
        if st.button("➕ Add Child", key="add_child_btn", disabled=not parent_choices):
            pid = sel_parent
            child_id = _next_child_id(pid)
            new_child = {
                "ID": child_id,
                "ParentID": pid,
                "Text": "The System shall TBD.",
                "Role": "Child",
                "Verification": "Test",
                "VerificationLevel": "Subsystem",
                "VerificationEvidence": "",
                "ValidationNeedID": S.get("need_id", "NEED-001"),
                "TestCaseIDs": "",
                "AllocatedTo": "",
                "Criticality": "Medium",
                "Status": "Draft",
            }
            insert_at = next((i for i, r in enumerate(S["requirements"]) if r["ID"] == pid), None)
            if insert_at is None:
                S["requirements"].append(new_child)
            else:
                S["requirements"].insert(insert_at + 1, new_child)
            _rerun()

    # ---------- Render cards ----------
    reqs = list(S.get("requirements", []))
    if not reqs:
        st.caption("No requirements yet. Use **Generate Questions & Requirements** or **Add Parent/Child**.")
    else:
        for idx, req in enumerate(reqs):
            rid, role = req["ID"], req["Role"]
            text = req.get("Text", "")
            # ensure defaults
            req.setdefault("Verification", req.get("Verification", "Test"))
            req.setdefault("VerificationLevel", req.get("VerificationLevel", "Subsystem"))
            req.setdefault("VerificationEvidence", req.get("VerificationEvidence", ""))
            req.setdefault("ValidationNeedID", req.get("ValidationNeedID", S.get("need_id", "NEED-001")))
            req.setdefault("TestCaseIDs", req.get("TestCaseIDs", ""))
            req.setdefault("AllocatedTo", req.get("AllocatedTo", ""))
            req.setdefault("Criticality", req.get("Criticality", "Medium"))
            req.setdefault("Status", req.get("Status", "Draft"))

            border = "1px solid #94a3b8" if role == "Parent" else "1px solid #e2e8f0"
            st.markdown(f"<div style='border:{border};border-radius:10px;padding:12px;margin-bottom:10px;'>", unsafe_allow_html=True)

            # Header + ID + Text
            title = "Parent" if role == "Parent" else ("Child" if role == "Child" else "Requirement")
            st.markdown(f"**{title}**")
            top = st.columns([0.20, 0.80])
            with top[0]:
                new_id = st.text_input("ID", value=rid, key=f"id_{rid}")
                if new_id and new_id != rid:
                    prefix_old = rid + "."
                    prefix_new = new_id + "."
                    for j, r2 in enumerate(S["requirements"]):
                        if r2["ID"] == rid:
                            S["requirements"][j]["ID"] = new_id
                            if r2["Role"] == "Parent":
                                if rid in S["child_counts"] and new_id not in S["child_counts"]:
                                    S["child_counts"][new_id] = S["child_counts"].pop(rid)
                        elif r2.get("ParentID") == rid:
                            S["requirements"][j]["ParentID"] = new_id
                        if r2["ID"].startswith(prefix_old):
                            S["requirements"][j]["ID"] = prefix_new + r2["ID"][len(prefix_old):]
                    _rerun()
            with top[1]:
                new_text = st.text_input("Requirement", value=text, key=f"text_{rid}")
                if new_text != text:
                    S["requirements"][idx]["Text"] = new_text

            # Tools row (Rewrite always; Decompose only for non-singular)
            tools = st.columns([0.18, 0.18, 0.18, 0.46])
            with tools[0]:
                if st.button("🪄 Rewrite", key=f"rw_{rid}"):
                    S["requirements"][idx]["Text"] = _ai_rewrite_strict(S["requirements"][idx]["Text"])
                    _rerun()

            _, _, _, sing_issues = _qc(S["requirements"][idx]["Text"])
            show_decompose = bool(sing_issues)

            with tools[1]:
                if show_decompose:
                    if st.button("🧩 Decompose", key=f"dc_{rid}"):
                        base_parent = S["requirements"][idx]["ID"]
                        if st.session_state.get("api_key"):
                            raw = _llm_retry(lambda _: decompose_requirement_with_ai(st.session_state.api_key, S["requirements"][idx]["Text"]), "DECOMPOSE")
                            kids_txt = [re.sub(r'^[\-\*\u2022]?\s*', '', ln.strip()) for ln in (raw or "").splitlines() if re.search(r'\w', ln)]
                            kids_txt = [k for k in kids_txt if len(k.split()) > 3]
                        else:
                            kids_txt = []
                        if kids_txt:
                            if S["requirements"][idx]["Role"] == "Standalone":
                                S["requirements"][idx]["Role"] = "Parent"
                                S["child_counts"][base_parent] = 1
                            S["child_counts"].setdefault(base_parent, 1)
                            children = _append_children_ids(base_parent, kids_txt)
                            S["requirements"][idx+1:idx+1] = children
                            st.success(f"Decomposed into {len(children)} child requirement(s).")
                            _rerun()
                        else:
                            st.info("No decomposable actions detected.")
                else:
                    st.write("")

            with tools[2]:
                if st.button("🗑️ Delete", key=f"del_{rid}"):
                    pref = rid + "."
                    S["requirements"] = [r for r in S["requirements"] if not (r["ID"] == rid or r["ID"].startswith(pref))]
                    _rerun()
            with tools[3]:
                st.caption("")

            # Structured edit (dropdowns / with custom)
            with st.expander("Structured edit (dropdowns / with custom)"):
                def _sel_or_custom(label, options, ksel, kcust, initial=""):
                    opts = [o for o in options if o != "function"] + (["function"] if "function" in options else [])
                    preset = initial if initial in opts else (opts[0] if opts else "")
                    sel = st.selectbox(
                        label,
                        opts + ["Custom…"],
                        index=(opts + ["Custom…"]).index(preset) if preset in opts else len(opts),
                        key=ksel
                    )
                    if sel == "Custom…":
                        return st.text_input(
                            f"{label} (custom)",
                            value=initial if (initial and initial not in opts) else "",
                            key=kcust
                        )
                    return sel

                txt_now = S["requirements"][idx]["Text"]
                parsed = _parse_req_text(txt_now)

                actor_guess = parsed["actor"]
                action_guess = parsed["action"]
                object_guess = parsed["object"]
                trigger_guess = parsed["trigger"]
                conditions_guess = parsed["conditions"]
                perf_guess = parsed["perf"]

                c1, c2 = st.columns(2)
                with c1:
                    actor = _sel_or_custom("Actor / System",
                                           ["System", "Thermal Control Subsystem", "Power Subsystem", "Payload", "Spacecraft", "UAV"],
                                           f"{rid}_actor_sel", f"{rid}_actor_custom", actor_guess)
                    modal = st.selectbox("Modal Verb", ["shall", "will", "must"], index=0, key=f"{rid}_modal")
                    action = _sel_or_custom("Action / Verb",
                                            ["maintain", "regulate", "limit", "detect", "log", "achieve", "provide", "enforce", "control", "acquire"],
                                            f"{rid}_action_sel", f"{rid}_action_custom", action_guess)
                    obj = _sel_or_custom("Object",
                                         ["payload optics temperature", "battery temperatures", "avionics temperatures", "temperatures", "C2 link", "endurance", "geo-fencing", "latency", "function"],
                                         f"{rid}_object_sel", f"{rid}_object_custom", object_guess)
                with c2:
                    trigger = _sel_or_custom("Trigger / Event (optional)",
                                             ["during all mission phases", "during eclipse", "during active imaging", "when commanded", "during flight operations", ""],
                                             f"{rid}_trigger_sel", f"{rid}_trigger_custom", trigger_guess)
                    conditions = _sel_or_custom("Operating Conditions / State (optional)",
                                                ["in eclipse and full sun", "in nominal mode", "in safe mode", "in nominal conditions", ""],
                                                f"{rid}_cond_sel", f"{rid}_cond_custom", conditions_guess)
                    perf = st.text_input("Performance / Constraint (optional, measurable)",
                                         value=perf_guess,
                                         placeholder="e.g., within ±2 °C; ≥ 15 km; ≤ 200 ms; ≥ 99.9% availability",
                                         key=f"{rid}_perf")

                def _norm_trig(t: str) -> str:
                    t = t.strip()
                    if not t:
                        return ""
                    return t if re.match(r'^(when|if|while|during)\b', t, flags=re.I) else f"during {t}"

                trig_part = (_norm_trig(trigger) + ", ") if trigger.strip() else ""
                perf_final = perf.strip() if perf.strip() else perf_guess
                tail_perf = f" {perf_final}" if perf_final else ""
                tail_cond = f" {conditions.strip()}" if conditions.strip() else ""
                rebuilt = f"{trig_part}{actor} {modal} {action} {obj}{tail_perf}{tail_cond}".strip()
                if not rebuilt.endswith("."):
                    rebuilt += "."
                rebuilt = re.sub(r"\s{2,}", " ", rebuilt)
                if st.button("Apply structured edit", key=f"apply_{rid}"):
                    S["requirements"][idx]["Text"] = rebuilt
                    _rerun()

            # 🔻 V&V & traceability
            with st.expander("Verification & Traceability"):
                row_vv1 = st.columns([0.26, 0.26, 0.24, 0.24])
                with row_vv1[0]:
                    ver_options = ["Test", "Analysis", "Inspection", "Demo"]
                    cur = S["requirements"][idx].get("Verification", "Test")
                    sel = st.selectbox("Verification Method", ver_options, index=ver_options.index(cur) if cur in ver_options else 0, key=f"{rid}_verif")
                    if sel != cur:
                        S["requirements"][idx]["Verification"] = sel
                with row_vv1[1]:
                    lvl_opts = ["Unit", "Subsystem", "System", "Mission"]
                    cur = S["requirements"][idx].get("VerificationLevel", "Subsystem")
                    sel = st.selectbox("Verification Level", lvl_opts, index=lvl_opts.index(cur) if cur in lvl_opts else 1, key=f"{rid}_verlvl")
                    if sel != cur:
                        S["requirements"][idx]["VerificationLevel"] = sel
                with row_vv1[2]:
                    cur = S["requirements"][idx].get("ValidationNeedID", S.get("need_id", "NEED-001"))
                    val = st.text_input("Validation Need ID", value=cur, key=f"{rid}_valneed")
                    if val != cur:
                        S["requirements"][idx]["ValidationNeedID"] = val
                with row_vv1[3]:
                    cur = S["requirements"][idx].get("AllocatedTo", "")
                    val = st.text_input(
                        "Allocated To",
                        value=cur,
                        key=f"{rid}_alloc",
                        placeholder="e.g., Propulsion Subsystem / Thermal Subsystem / Flight Software / API Service"
                    )
                    if val != cur:
                        S["requirements"][idx]["AllocatedTo"] = val

                row_vv2 = st.columns([0.50, 0.25, 0.25])
                with row_vv2[0]:
                    cur = S["requirements"][idx].get("VerificationEvidence", "")
                    val = st.text_input("Verification Evidence (link/ID)", value=cur, key=f"{rid}_verevid")
                    if val != cur:
                        S["requirements"][idx]["VerificationEvidence"] = val
                with row_vv2[1]:
                    cur = S["requirements"][idx].get("TestCaseIDs", "")
                    val = st.text_input("Test Case ID(s)", value=cur, key=f"{rid}_tcids", placeholder="e.g., HIL-BURN-07; TVAC-OPT-02")
                    if val != cur:
                        S["requirements"][idx]["TestCaseIDs"] = val
                with row_vv2[2]:
                    crit_options = ["High", "Medium", "Low"]
                    cur_crit = S["requirements"][idx].get("Criticality", "Medium")
                    sel_crit = st.selectbox("Criticality", crit_options, index=crit_options.index(cur_crit) if cur_crit in crit_options else 1, key=f"{rid}_crit")
                    if sel_crit != cur_crit:
                        S["requirements"][idx]["Criticality"] = sel_crit

                status_row = st.columns([1.0])
                with status_row[0]:
                    status_options = ["Draft", "Reviewed", "Approved"]
                    cur_status = S["requirements"][idx].get("Status", "Draft")
                    sel_status = st.selectbox("Status", status_options, index=status_options.index(cur_status) if cur_status in status_options else 0, key=f"{rid}_status")
                    if sel_status != cur_status:
                        S["requirements"][idx]["Status"] = sel_status

            # Quality badges
            st.markdown(_badge_row(S["requirements"][idx]["Text"]))
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Export ----------
    st.subheader("⬇️ Export Requirements (CSV)")
    if not S.get("requirements"):
        st.info("No requirements to export yet.")
    else:
        rows = []
        for r in S["requirements"]:
            rows.append({
                "Need ID": S.get("need_id", "NEED-001"),
                "Validation Need ID": r.get("ValidationNeedID", S.get("need_id", "NEED-001")),
                "ID": r["ID"],
                "ParentID": r["ParentID"],
                "Requirement Text": r["Text"],
                "Type": S.get("req_type", "Functional"),
                "Role": r["Role"],
                "Priority": S.get("priority", "Should"),
                "Lifecycle": S.get("lifecycle", "Operations"),
                "Stakeholder": S.get("stakeholder", ""),
                "Source": "Need",
                "Verification": r.get("Verification", ""),
                "Verification Level": r.get("VerificationLevel", ""),
                "Verification Evidence": r.get("VerificationEvidence", ""),
                "Test Case IDs": r.get("TestCaseIDs", ""),
                "Allocated To": r.get("AllocatedTo", ""),
                "Criticality": r.get("Criticality", ""),
                "Status": r.get("Status", ""),
                "Acceptance Criteria": "",
                "Rationale": S.get("rationale", "")
            })
        df_pro = pd.DataFrame(rows, columns=[
            "Need ID", "Validation Need ID", "ID", "ParentID", "Requirement Text", "Type", "Role",
            "Priority", "Lifecycle", "Stakeholder", "Source",
            "Verification", "Verification Level", "Verification Evidence",
            "Test Case IDs", "Allocated To", "Criticality", "Status",
            "Acceptance Criteria", "Rationale"
        ])
        st.download_button(
            "Download CSV",
            data=df_pro.to_csv(index=False).encode("utf-8"),
            file_name="Requirements_Export.csv",
            mime="text/csv",
            key="pro_export_csv"
        )
