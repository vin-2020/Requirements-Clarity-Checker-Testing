# llm/ai_suggestions.py
"""
Lightweight helpers for ReqCheck's AI features (Gemini via google.generativeai).

IMPORTANT:
- Streamlit caching (@st.cache_data) is preserved.
- Adds run_freeform(...) for prompts that need multi-line / JSON output.
"""

import streamlit as st
import google.generativeai as genai
import json
import re
from typing import List, Tuple

# ----------------------------- Core helpers -----------------------------

@st.cache_data
def run_freeform(api_key: str, prompt: str) -> str:
    """
    Generic freeform call: sends your prompt AS-IS and returns raw model text.
    Use this for prompts that expect multi-line lists, JSON, or JSON Lines.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception:
        # IMPORTANT: return empty string, never an error blob (callers handle retries/repairs)
        return ""


@st.cache_data
def get_ai_suggestion(api_key, requirement_text):
    """
    Ask Gemini to rewrite a requirement for clarity and testability.
    (Single-sentence polish helper; keep for the 🪄 Rewrite button.)
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        You are a lead Systems Engineer acting as a mentor. Your task is to review and rewrite a single requirement statement to make it exemplary.

        Follow these critical INCOSE-based principles for your rewrite:
        1.  **Verifiable:** The requirement must be testable. Replace subjective words (like "easy", "fast", "efficient") with specific, measurable criteria (like "within 500ms", "with 99.9% accuracy").
        2.  **Unambiguous:** The requirement must have only one possible interpretation. Use clear, direct language.
        3.  **Singular:** The requirement MUST state only a single capability. DO NOT use words like "and" or "or" to combine multiple requirements.
        4.  **Active Voice:** The requirement must be in the active voice (e.g., "The system shall...").
        5.  **Concise:** Remove unnecessary words like "be able to" or "be capable of".

        CRITICAL INSTRUCTION: Your final output must be ONLY the rewritten requirement sentence and nothing else. Do not add preambles like "Here is the rewritten requirement:".

        Original Requirement: "{requirement_text}"
        
        Rewritten Requirement:
        """
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"An error occurred with the AI service: {e}"


@st.cache_data
def generate_requirement_from_need(api_key, need_text):
    """
    Convert an informal stakeholder need into a structured requirement
    or ask a clarifying question if the need is too vague.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        You are a Systems Engineer creating a formal requirement from a stakeholder's informal need.
        Convert the following need into a structured requirement with the format:
        "[Condition], the [System/Actor] shall [Action] [Object] [Performance Metric]."

        If the need is too vague to create a full requirement, identify the missing pieces (like a measurable number or a clear action) and ask a clarifying question.

        Stakeholder Need: "{need_text}"

        Structured Requirement or Clarifying Question:
        """
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"An error occurred with the AI service: {e}"


@st.cache_data
def get_chatbot_response(api_key, chat_history):
    """
    Get a conversational reply from Gemini based on the entire chat history.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(chat_history)
        return response.text.strip()

    except Exception as e:
        return f"An error occurred with the AI service: {e}"

# --- NEW: AI Requirement Extractor (full JSON-based, robust) ---
@st.cache_data
def extract_requirements_with_ai(
    api_key: str,
    document_text: str,
    max_chunk_chars: int = 12000
) -> List[Tuple[str, str]]:
    """
    Use the LLM to extract ONLY requirement statements from raw text.
    Returns: list of (req_id, req_text)
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        return []

    paras = re.split(r"\n\s*\n", document_text or "")
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chunk_chars:
            buf += (p + "\n\n")
        else:
            if buf.strip():
                chunks.append(buf)
            buf = p + "\n\n"
    if buf.strip():
        chunks.append(buf)

    out: List[Tuple[str, str]] = []
    running_index = 1

    def _parse_or_fallback(resp_text: str) -> List[Tuple[str, str]]:
        json_text = None
        m = re.search(r"\{.*\}\s*$", resp_text or "", flags=re.S)
        if m:
            json_text = m.group(0)

        pairs: List[Tuple[str, str]] = []
        if json_text:
            try:
                data = json.loads(json_text)
                items = data.get("requirements", []) if isinstance(data, dict) else []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    rid = (item.get("id") or "").strip()
                    rtx = (item.get("text") or "").strip()
                    if rtx:
                        pairs.append((rid, rtx))
                if pairs:
                    return pairs
            except Exception:
                pass

        heur_pairs: List[Tuple[str, str]] = []
        bullets = re.findall(r"^\s*(?:-|\*|\d+[\.\)])\s*(.+)$", resp_text or "", flags=re.M)
        for b in bullets:
            t = (b or "").strip()
            if t:
                heur_pairs.append(("", t))

        norm_pat = re.compile(
            r"""(?ix)
            ^
            (?:
                (?P<id>[A-Z][A-Z0-9-]*-\d+|[A-Z]{2,}\d+|\d+[\.\)])\s+
            )?
            (?P<text>.*?\b(shall|must|will|should)\b.*)
            $
            """
        )
        for line in (resp_text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            m2 = norm_pat.match(line)
            if m2:
                rid = (m2.group("id") or "").strip()
                txt = (m2.group("text") or "").strip()
                if txt:
                    heur_pairs.append((rid, txt))

        return heur_pairs

    for ch in (chunks if chunks else [""]):
        prompt = f"""
You are an expert Systems Engineer and requirements analyst.

Extract ONLY formal requirement statements from the TEXT below and output STRICT JSON with this exact schema (no extra commentary, no markdown, no prefixes/suffixes):

{{
  "requirements": [
    {{"id": "optional-id-or-empty", "text": "the requirement text (original phrasing, trimmed)"}}
  ]
}}

Extraction rules:
- Include normative, testable statements: contain "shall", "must", "will", or "should", or measurable constraints.
- Accept both formats:
  • Table/ID-based: e.g., "SYS-001 The system shall ..." or "SAT-REQ-12 The payload shall ..."
  • Narrative/numbered/bulleted: e.g., "1. The drone shall ..." or "- The controller will ..."
- Keep the original sentence wording except trimming bullets/numbering. Do not rewrite.
- If an explicit identifier exists (e.g., "SYS-001", "1."), put it in "id"; otherwise, use "" (empty string).
- Return VALID JSON ONLY. Do not add any text before or after the JSON.

TEXT:
\"\"\"{ch}\"\"\""""
        try:
            resp = model.generate_content(prompt)
            text_out = (resp.text or "").strip()
        except Exception:
            text_out = ""

        pairs = _parse_or_fallback(text_out)
        for rid, rtx in pairs:
            if not rtx:
                continue
            final_id = rid.strip() if rid.strip() else f"R-{running_index:03d}"
            out.append((final_id, rtx.strip()))
            running_index += 1

    if not out and (document_text or "").strip():
        norm = re.compile(
            r'(?im)^(?:(?P<id>[A-Z][A-Z0-9-]*-\d+|\d+[.)])\s+)?(?P<txt>.*?\b(shall|must|will|should)\b.*)$'
        )
        idx = 1
        for line in (document_text or "").splitlines():
            m = norm.match(line.strip())
            if m:
                rid = (m.group("id") or f"R-{idx:03d}").strip()
                txt = (m.group("txt") or "").strip()
                if txt:
                    out.append((rid, txt))
                    idx += 1

    seen = set()
    unique: List[Tuple[str, str]] = []
    for rid, txt in out:
        key = (txt or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((rid, txt))

    return unique

@st.cache_data
def decompose_requirement_with_ai(api_key, requirement_text):
    """
    Uses the Gemini LLM to decompose a complex requirement into multiple singular requirements.
    Returns a plain list of lines, each: 'The system shall ...'.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
Split the compound requirement below into 3–8 singular, verifiable requirements.

RULES
- Return ONLY the final lines, one per line.
- Each line MUST be: "The system shall ...".
- ≤ 22 words per line; active voice; include measurable criteria where meaningful.
- No bullets, numbering, IDs, or extra text.

INPUT
\"\"\"{(requirement_text or '').strip()}\"\"\""""
        resp = model.generate_content(prompt)
        raw = (getattr(resp, "text", "") or "").strip()
        lines = [re.sub(r"^[\-\*\d\)\.]+\s*", "", ln.strip()) for ln in raw.splitlines() if ln.strip()]
        out = []
        for ln in lines:
            if " shall " not in f" {ln.lower()} ":
                continue
            out.append(ln.rstrip(".") + ".")
        return "\n".join(out[:8]) if out else "No decomposition produced."
    except Exception as e:
        return f"An error occurred with the AI service: {e}"

# ---------------------- Dense-Need → Requirement Set ----------------------
@st.cache_data
def decompose_need_into_requirements(api_key: str, need_text: str) -> str:
    """
    Systematically decomposes a dense stakeholder need into a numbered list of
    singular, verifiable 'The system shall ...' requirements with REQ-xxx IDs.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        return f"An error occurred with the AI service: {e}"

    def _call_model(prompt: str) -> str:
        try:
            resp = model.generate_content(prompt)
            return (resp.text or "").strip()
        except Exception as e:
            return f"__ERR__ {e}"

    base_prompt = f"""
You are a senior systems engineer following INCOSE practices.

TASK
1) ANALYZE the Stakeholder Need and identify every distinct capability, constraint, resource limit, failure/fault tolerance, timing window, environmental/operational condition, interface/safety/compliance.
2) DECOMPOSE into a set of singular, verifiable system requirements.

STRICT OUTPUT FORMAT
- Return ONLY the final list of requirements, one per line.
- Each line MUST start with a unique ID like "REQ-001." followed by a single sentence in active voice: "The system shall ...".
- Each requirement must be singular, testable/verifiable, and measurable (numbers/units or crisp thresholds when meaningful).
- ≤ 26 words per sentence. Avoid lists inside a requirement.
- Do NOT include analysis notes, headings, bullets, or commentary. FINAL LIST ONLY.

STAKEHOLDER NEED
"""
    base_prompt += f'\"\"\"{(need_text or "").strip()}\"\"\"'

    raw = _call_model(base_prompt)

    def _normalize_lines(text: str) -> list[str]:
        if not text or text.startswith("__ERR__"):
            return []
        lines = [re.sub(r"^[\-\*\d\)\.]+\s*", "", ln.strip()) for ln in text.splitlines() if ln.strip()]
        out = []
        for ln in lines:
            if " shall " not in f" {ln.lower()} ":
                continue
            m = re.match(r"^(REQ-\d{3,4})[.\s:-]\s*(.*)$", ln, flags=re.I)
            if m:
                rid = m.group(1).upper()
                body = m.group(2).strip()
                ln = f"{rid}. {body}"
            out.append(ln.rstrip(".") + ".")
        seen = set()
        uniq = []
        for ln in out:
            k = re.sub(r"\s+", " ", ln.lower())
            if k not in seen:
                seen.add(k)
                uniq.append(ln)
        return uniq

    lines = _normalize_lines(raw)

    if len(lines) < 8:
        repair_prompt = f"""
Your previous answer did not cover the full scope or enough requirements.

REPAIR:
- Rewrite into 8–16 distinct, singular, verifiable requirements.
- EXACTLY the format: "REQ-001. The system shall ...".
- One per line. No commentary, no bullets, no headings.
- Keep ≤ 26 words per sentence; include measurable criteria when meaningful.

STAKEHOLDER NEED
\"\"\"{(need_text or '').strip()}\"\"\"\n
YOUR PRIOR ANSWER
\"\"\"{raw}\"\"\""""
        raw2 = _call_model(repair_prompt)
        lines2 = _normalize_lines(raw2)
        if len(lines2) >= len(lines):
            lines = lines2

    if not lines:
        return "An error occurred with the AI service: empty response from model"

    out = []
    for i, ln in enumerate(lines[:20], 1):
        body = re.sub(r"^REQ-\d{3,4}[.\s:-]\s*", "", ln, flags=re.I).strip()
        out.append(f"REQ-{i:03d}. {body}")
    return "\n".join(out)

# ------------------------------ Req Tutor helpers ------------------------------

def _extract_json_or_none(text: str):
    """Try to parse JSON; tolerate code fences and trailing prose."""
    if not text:
        return None
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    candidate = m.group(0) if m else text.strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None

def _kv_lines_to_dict(text: str) -> dict:
    """Fallback parser for 'Key: value' lines."""
    out = {}
    for line in (text or "").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().strip("-").strip()
            if k:
                out[k] = v
    return out

_NEED_AUTOFILL_FIELDS = {
    "Functional": ["Actor","Action","Object","Trigger","Conditions","Performance","ModalVerb"],
    "Performance": ["Function","Metric","Threshold","Unit","Conditions","Measurement","VerificationMethod"],
    "Constraint": ["Subject","ConstraintText","DriverOrStandard","Rationale"],
    "Interface": ["System","ExternalSystem","InterfaceStandard","Direction","DataItems","Performance","Conditions"],
}

@st.cache_data
def analyze_need_autofill(api_key: str, need_text: str, req_type: str) -> dict:
    """
    Return a dict with the fields required by the chosen requirement type.
    Uses Gemini directly with a strict JSON prompt + light post-processing.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        fields = _NEED_AUTOFILL_FIELDS.get(req_type or "Functional", _NEED_AUTOFILL_FIELDS["Functional"])
        return {k: "" for k in fields}

    req_type = (req_type or "Functional").strip()
    fields = _NEED_AUTOFILL_FIELDS.get(req_type, _NEED_AUTOFILL_FIELDS["Functional"])

    if req_type == "Functional":
        fewshot = """Example (Functional, JSON only):
{"Actor":"UAV","Action":"present","Object":"low-battery alert to operator","Trigger":"battery state-of-charge < 20%","Conditions":"in all flight modes","Performance":"within 1 s","ModalVerb":"shall"}"""
    elif req_type == "Performance":
        fewshot = """Example (Performance, JSON only):
{"Function":"position estimator","Metric":"RMSE","Threshold":"1.5","Unit":"m","Conditions":"steady hover","Measurement":"flight log analysis","VerificationMethod":"Analysis"}"""
    elif req_type == "Constraint":
        fewshot = """Example (Constraint, JSON only):
{"Subject":"avionics enclosure","ConstraintText":"IP65 ingress protection","DriverOrStandard":"IEC 60529","Rationale":"dust and water resistance for field ops"}"""
    else:
        fewshot = """Example (Interface, JSON only):
{"System":"flight computer","ExternalSystem":"ground control station","InterfaceStandard":"MAVLink v2","Direction":"Bi-directional","DataItems":"heartbeat, position, battery_status","Performance":"latency ≤ 150 ms","Conditions":"nominal flight modes"}"""

    prompt = f"""
You are assisting a systems engineer. Given this stakeholder need, produce a FIRST-DRAFT for a {req_type} requirement.

NEED:
\"\"\"{(need_text or '').strip()}\"\"\"\n
Return ONLY a VALID JSON object with EXACTLY these keys (no extra text, no code fences):
{json.dumps(fields)}

Guidance:
- Map scenario context into fields (e.g., contested airspace, stealth/avoiding detection, mission completion/return).
- Keep Object as the thing acted on (do NOT include metrics or percentages in Object).
- Put numbers/thresholds/units in Performance (or Threshold/Unit for Performance type).
- Prefer an EARS style (use Trigger/Conditions when implied).
- Avoid vague phrases: "all specified", "as soon as possible", "as needed", "etc.", "including but not limited to".
- ModalVerb should usually be "shall".
- VerificationMethod (if present): one of "Test","Analysis","Inspection","Demonstration".

{fewshot}
"""
    try:
        resp = model.generate_content(prompt)
        raw = (getattr(resp, "text", "") or "").strip()
    except Exception:
        raw = ""

    data = _extract_json_or_none(raw)
    if data is None or not isinstance(data, dict):
        data = _kv_lines_to_dict(raw)

    out = {k: (data.get(k, "") if isinstance(data, dict) else "") for k in fields}

    def _ban_vague_phrases(txt: str) -> str:
        banned = ["all specified", "as needed", "as soon as possible", "etc.", "including but not limited to"]
        t = (txt or "")
        for b in banned:
            t = t.replace(b, "").strip()
        return t

    def _strip_perf_from_object(obj: str, perf: str) -> tuple[str, str]:
        o = (obj or "").strip()
        p = (perf or "").strip()
        if not o:
            return o, p
        patterns = [
            r'\bwith (a )?probability of\s*[0-9]*\.?[0-9]+%?',
            r'\b(minimum|maximum|at least|no more than)\s*[0-9]*\.?[0-9]+%?\b',
            r'\b\d+(\.\d+)?\s*(ms|s|sec|m|km|hz|khz|mhz|kbps|mbps|gbps|fps)\b',
            r'\b\d+(\.\d+)?\s*%(\b|$)'
        ]
        extracted = []
        for rx in patterns:
            m = re.search(rx, o, flags=re.IGNORECASE)
            if m:
                extracted.append(m.group(0).strip())
                o = (o[:m.start()] + o[m.end():]).strip().strip(',. ')
        if extracted:
            extra = " ".join(extracted)
            if p:
                if extra not in p:
                    p = f"{p}; {extra}"
            else:
                p = extra
        return o, p

    need_lower = (need_text or "").lower()
    def _maybe_push_context_to_conditions(conds: str) -> str:
        bits = []
        if "contested airspace" in need_lower and "contested airspace" not in (conds or "").lower():
            bits.append("in contested airspace")
        if "avoiding detection" in need_lower and "avoid" not in (conds or "").lower():
            bits.append("while minimizing detectability by adversary sensors")
        if "return" in need_lower and "return" not in (conds or "").lower():
            bits.append("and return safely to base")
        if bits:
            return (f"{conds} " + " ".join(bits)).strip() if conds else " ".join(bits)
        return conds

    if "ModalVerb" in out:
        mv = (out["ModalVerb"] or "shall").lower()
        out["ModalVerb"] = mv if mv in ("shall","will","must") else "shall"
    if "VerificationMethod" in out:
        vm = (out["VerificationMethod"] or "").title()
        out["VerificationMethod"] = vm if vm in ("Test","Analysis","Inspection","Demonstration") else "Test"
    if "Direction" in out:
        dr = (out["Direction"] or "")
        out["Direction"] = dr if dr in ("In","Out","Bi-directional") else "Bi-directional"

    if req_type == "Functional":
        out["Object"], out["Performance"] = _strip_perf_from_object(out.get("Object",""), out.get("Performance",""))
        out["Object"] = _ban_vague_phrases(out.get("Object",""))
        out["Conditions"] = _maybe_push_context_to_conditions(out.get("Conditions",""))
    elif req_type == "Performance":
        out["Function"] = _ban_vague_phrases(out.get("Function",""))
    elif req_type == "Interface":
        out["DataItems"] = _ban_vague_phrases(out.get("DataItems",""))
    elif req_type == "Constraint":
        out["ConstraintText"] = _ban_vague_phrases(out.get("ConstraintText",""))

    return out

def review_requirement_with_ai(api_key: str, requirement_text: str, preferred_verification: str | None = None) -> dict:
    """
    Return {"review": str, "acceptance": [str, ...]}.
    Uses freeform call to preserve the JSON response.
    """
    pv = preferred_verification if preferred_verification in ("Test","Analysis","Inspection","Demonstration") else ""
    hint = f' "preferredVerification": "{pv}",' if pv else ""

    prompt = f"""
Act as a systems engineering reviewer. Evaluate the requirement and propose precise, testable acceptance criteria.

Return ONLY this JSON (no extra text, no code fences):
{{
  "review": "short critique focusing on ambiguity, measurability, singularity, feasibility",
  {hint}
  "acceptance": [
    "bullet 1 with threshold(s), setup/conditions, verification method",
    "bullet 2 ..."
  ]
}}

Requirement:
\"\"\"{(requirement_text or '').strip()}\"\"\""""
    raw = run_freeform(api_key, prompt)
    data = _extract_json_or_none(raw)
    if not isinstance(data, dict):
        return {"review": raw.strip(), "acceptance": []}

    review = str(data.get("review", "")).strip()
    acceptance = data.get("acceptance", [])
    if not isinstance(acceptance, list):
        acceptance = [str(acceptance)]
    acceptance = [str(x).strip() for x in acceptance if str(x).strip()]
    return {"review": review, "acceptance": acceptance}
