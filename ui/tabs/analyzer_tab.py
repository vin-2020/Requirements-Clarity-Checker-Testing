# ui/tabs/analyzer_tab.py
import os
import re
import docx
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st


def render(st, db, rule_engine, CTX):
    """
    Document Analyzer tab.
    Expects:
      - db: your db module (already imported/reloaded in app.py)
      - rule_engine: RuleEngine instance (or stub)
      - CTX: dict of helpers injected from app.py
    """
    # ---- CTX helpers ---------------------------------------------------------
    HAS_AI_PARSER = CTX.get("HAS_AI_PARSER", False)
    get_ai_suggestion = CTX["get_ai_suggestion"]
    decompose_requirement_with_ai = CTX["decompose_requirement_with_ai"]
    extract_requirements_with_ai = CTX.get("extract_requirements_with_ai")

    _read_docx_text_and_rows = CTX["_read_docx_text_and_rows"]
    _read_docx_text_and_rows_from_path = CTX["_read_docx_text_and_rows_from_path"]
    _extract_requirements_from_table_rows = CTX["_extract_requirements_from_table_rows"]
    extract_requirements_from_string = CTX["extract_requirements_from_string"]
    extract_requirements_from_file = CTX["extract_requirements_from_file"]

    format_requirement_with_highlights = CTX["format_requirement_with_highlights"]
    safe_call_ambiguity = CTX["safe_call_ambiguity"]
    check_passive_voice = CTX["check_passive_voice"]
    check_incompleteness = CTX["check_incompleteness"]
    check_singularity = CTX["check_singularity"]
    safe_clarity_score = CTX["safe_clarity_score"]

    _save_uploaded_file_for_doc = CTX["_save_uploaded_file_for_doc"]



    # ---- Harden get_ai_suggestion: truncate + retry + HARD timeout ----
    import time, concurrent.futures

    _orig_get_ai_suggestion = CTX["get_ai_suggestion"]  # keep original

    def _safe_get_ai_suggestion(
        api_key: str,
        requirement_text: str | None = None,
        prompt: str | None = None,
        *,
        max_chars: int = 6000,
        retries: int = 2,
        backoff_base: float = 0.6,
        timeout_s: int = 20,
        **kwargs,
    ):
        if not api_key:
            return ""
        content = (prompt if prompt is not None else requirement_text) or ""
        safe_prompt = content[:max_chars]

        def _call_once():
            try:
                return _orig_get_ai_suggestion(api_key, safe_prompt, **kwargs)
            except TypeError:
                return _orig_get_ai_suggestion(api_key, safe_prompt)

        last_err = None
        for attempt in range(retries + 1):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_call_once)
                    return (fut.result(timeout=timeout_s) or "").strip()
            except Exception as e:
                msg = str(e)
                last_err = e
                if any(code in msg for code in (" 500", " 502", " 503", " 504", " 429", "deadline", "timeout")):
                    time.sleep(backoff_base * (2 ** attempt))
                    continue
                return ""
        return ""

    # Install the shim globally
    CTX["get_ai_suggestion"] = _safe_get_ai_suggestion

    # ---- AI contradiction config & caller ----
    CTX.update({
        "USE_AI_CONTRA": True,
        # Model (tune if you have another deployed model id)
        "AI_CONTRA_MODEL": "gemini-1.5-pro",
        # Explore more pairs (helps catch cross-domain policy clashes)
        "AI_CONTRA_TOPK": 200,           # was 60
        # More time overall and per call (helps on larger sets)
        "AI_CONTRA_BUDGET_S": 70,        # was 35
        "AI_CONTRA_PER_CALL_S": 20,      # was 12
        # Allow more returned contradictions
        "AI_CONTRA_MAX_FINDINGS": 200,   # new: runner should respect if implemented
        # Give the AI a domain-agnostic “what to look for” brief
        "AI_CONTRA_SYSTEM_PROMPT": (
            "You analyze plain-English requirements and output concrete, testable CONTRADICTIONS only. "
            "Catch policy or mode conflicts (e.g., 'always do X' vs 'never do X' or 'operator approval required' "
            "vs 'no approval for time-critical'), priority/trade-off inversions (accuracy vs latency), "
            "security vs speed (apply any authenticated command vs operator review), timelines that disagree "
            "(deploy within 24h vs wait 90 days), and scope/exception clashes "
            "(no changes after launch vs use latest post-launch). "
            "Be strict about mutual exclusivity; ignore mere differences in non-overlapping scopes. "
            "Return concise reason text and do not invent facts."
        ),
    })
    CTX["api_key"] = st.session_state.get("api_key")

    # Fail fast if no API key (so UI shows clear instruction and we don't call AI)
    if not CTX.get("api_key"):
        st.info("AI key missing — paste your Google AI Studio API key in the sidebar to enable AI features.")

    # --- Getting started (indented) -------------------------------------------
    with st.expander("Getting started (2–3 minutes)", expanded=False):
        st.markdown(
            "<div style='margin-left:16px'>"
            "• Paste 3–10 requirements or upload a document.<br/>"
            "• Click Analyze to see clarity issues (ambiguity, passive voice, etc.).<br/>"
            "• Use AI actions to rewrite or decompose, then run Contradiction Scan."
            "</div>",
            unsafe_allow_html=True,
        )

    # --- UI mode & onboarding (Beginner UI removed) ---------------------------
    beginner = False
    # Quick utility to reset noisy inline AI results
    def _reset_ai_caches():
        keys = [k for k in list(st.session_state.keys()) if k.startswith("rewritten_cache_") or k.startswith("decomp_cache_")]
        for k in keys:
            try:
                del st.session_state[k]
            except Exception:
                pass

    # --- Quick sanity helpers (AI connectivity) ---
    def _has_api_key() -> bool:
        key = (st.session_state.get("api_key") or "").strip()
        return bool(key)

    def _ai_smoke_check(label_key: str = "ai_smoke_doc"):
        if st.button("🔧 AI connectivity check", key=label_key):
            if not _has_api_key():
                st.error("No API key in session_state.api_key.")
                return
            sample = 'Return ONLY this JSON array: [{"ok": true}]'
            try:
                out = CTX["call_ai_contra"](sample, timeout_s=10) or ""
                st.code(out or "<empty>", language="json")
                if not out.strip():
                    st.error("AI call returned empty. Check API key/network or model quota.")
            except Exception as e:
                st.error(f"AI call failed: {e}")

    def _call_ai_contra(prompt: str, *, timeout_s: int | None = None) -> str:
        """
        Always inline the system brief into the prompt so we are not relying on kwargs
        that might be stripped by _safe_get_ai_suggestion's TypeError path.
        """
        if not CTX.get("api_key"):
            return ""
        sys_brief = CTX.get("AI_CONTRA_SYSTEM_PROMPT", "").strip()
        fused = (sys_brief + "\n\n" + prompt) if sys_brief else prompt
        return _safe_get_ai_suggestion(
            CTX["api_key"],
            prompt=fused,
            timeout_s=timeout_s or CTX.get("AI_CONTRA_PER_CALL_S", 20),
        ) or ""

    CTX["call_ai_contra"] = _call_ai_contra



    # ---- Normalizers & merge/dedupe so counts are correct --------------------
    # Accept IDs like R-001, S-12, SYS-001, FLT-003, etc. (1–8 letters)
    _ID_RE = re.compile(r"^[A-Z]{1,8}-\d{1,5}\b")

    def _normalize_req_text(t: str) -> str:
        """
        Strong normalizer for deduping cross sources:
        - lowercase
        - collapse whitespace and trim space before punctuation
        - remove leading '"text":' (or text:) labels
        - normalize curly quotes/dashes
        - strip surrounding quotes if whole sentence is quoted
        - drop trailing terminal punctuation for near-duplicates
        """
        t = (t or "").strip()
        # Normalize curly quotes/dashes to straight
        trans = {
            ord("“"): '"', ord("”"): '"', ord("‘"): "'", ord("’"): "'",
            ord("\u2013"): "-",  # en dash
            ord("\u2014"): "-",  # em dash
        }
        t = t.translate(trans)

        # Remove leading label like: "text": or text:
        t = re.sub(r'^\s*"?text"?\s*:\s*', "", t, flags=re.IGNORECASE)

        # If the whole sentence is wrapped in matching quotes, strip them
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            t = t[1:-1].strip()

        # Lowercase
        t = t.lower()

        # Collapse whitespace
        t = re.sub(r"\s+", " ", t)

        # Remove space before punctuation
        t = re.sub(r"[ \t]+([,.;:])", r"\1", t)

        # Drop trailing terminal punctuation to align near-duplicates
        t = t.rstrip(" .;:")

        return t

    def _merge_unique_reqs(*lists):
        """
        Merge lists of (id, text) with de-duplication.
        Priority: first occurrence wins.
        Deduping by:
          - exact ID (when present), and
          - normalized text (plus a stronger alphanumeric-only signature)
        """
        seen_ids = set()
        seen_texts = set()        # normalized text
        seen_texts_alnum = set()  # alphanumeric signature
        out = []
        auto_idx = 1

        def _norm_alnum(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", s).strip()

        def _next_auto_id():
            nonlocal auto_idx
            rid = f"R-{auto_idx:03d}"
            auto_idx += 1
            return rid

        for lst in lists:
            for rid, rtxt in (lst or []):
                rid = (rid or "").strip()
                rtxt = (rtxt or "").strip()  # fixed: use rtxt, not rtx
                if not rtxt:
                    continue

                has_id = bool(_ID_RE.match(rid))
                norm = _normalize_req_text(rtxt)
                norm_alnum = _norm_alnum(norm)

                # Dedup by text signature (even if different ID)
                if norm in seen_texts or norm_alnum in seen_texts_alnum:
                    continue

                if has_id:
                    if rid in seen_ids:
                        # already saw this ID; keep first occurrence
                        continue
                    seen_ids.add(rid)

                # Remember signatures
                seen_texts.add(norm)
                seen_texts_alnum.add(norm_alnum)

                # If no valid ID, assign synthetic
                if not has_id:
                    rid = _next_auto_id()

                out.append((rid, rtxt))
        return out

    # ---- Header --------------------------------------------------------------
    pname = st.session_state.selected_project[1] if st.session_state.selected_project else None
    if st.session_state.selected_project is None:
        st.header("Analyze Documents & Text")
        st.warning("You can analyze documents without a project, but results won’t be saved.")
    else:
        project_name = st.session_state.selected_project[1]
        st.header(f"Analyze & Add Documents to: {project_name}")

    # ===== Quick Paste Analyzer ------------------------------------------------
    st.subheader("🔍 Quick Paste Analyzer — single or small set")
    quick_text = st.text_area(
        "Paste one or more requirements (one per line). You may prefix with an ID like `REQ-001:`",
        height=160,
        key="quick_paste_area"
    )
    st.caption("Example:\nREQ-001: The system shall report position within 500 ms.\nREQ-001.1.")

    # session keys
    if "quick_results" not in st.session_state:
        st.session_state.quick_results = []   # list of dicts with analysis
    if "quick_issue_counts" not in st.session_state:
        st.session_state.quick_issue_counts = {"Ambiguity": 0, "Passive Voice": 0, "Incompleteness": 0, "Singularity": 0}
    if "quick_analyzed" not in st.session_state:
        st.session_state.quick_analyzed = False
    if "quick_text_snapshot" not in st.session_state:
        st.session_state.quick_text_snapshot = ""

    def _parse_quick_lines(raw: str):
        rows = []
        idx = 1
        for ln in (raw or "").splitlines():
            t = ln.strip()
            if not t:
                continue
            if ":" in t:
                left, right = t.split(":", 1)
                rid = left.strip()
                rtx = right.strip()
                if not rtx:
                    continue
            else:
                rid = f"R-{idx:03d}"
                rtx = t
            rows.append((rid, rtx))
            idx += 1
        # dedupe quick paste too
        return _merge_unique_reqs(rows)

    # --- AI prompts (rewrite + decompose) ------------------------------------
    def _ai_rewrite_prompt(req_text: str) -> str:
        # preserves all numbers/units/ranges; no inventions
        return f"""
You are a senior systems engineer. Rewrite the requirement below into EXACTLY ONE clear, verifiable sentence in ACTIVE voice using the verb "shall".
CRITICAL RULES:
- Preserve ALL original numeric values, ranges, thresholds, probabilities, units, symbols, and enumerations EXACTLY (e.g., 99.9%, -25°C to +55°C, 0–100% (non-condensing), 120 km/h). Do not invent values. Do not change units.
- Keep the original intent and scope. If details are not specified, do NOT add any new conditions or numbers.
- Remove vagueness (e.g., "robust", "approximately", "user-friendly") only if you can restate without introducing new numbers.
- Make it singular (one action) if possible; otherwise keep the main action clear.
- OUTPUT: the single rewritten sentence ONLY (no lists, no commentary).

Requirement:
\"\"\"{(req_text or '').strip()}\"\"\""""

    def _ai_decompose_prompt(parent_id: str, cleaned_sentence: str) -> str:
        # minimal children 2–4; preserve numbers/units; strict IDing
        return f"""
You are decomposing a requirement that contains multiple distinct actions.
Produce the MINIMUM number of child requirements (2–4) needed to make each child a SINGLE, testable "shall" statement.
RULES:
- Preserve ALL original numeric values, units, symbols, ranges EXACTLY. Do NOT invent numbers or tighten/relax thresholds.
- Each child must be independent and verifiable.
- Use child IDs in the format {parent_id}.1, {parent_id}.2, ... (no other text).
- OUTPUT FORMAT: each child on its own line as:
{parent_id}.n: <child shall sentence>

Parent requirement:
\"\"\"{cleaned_sentence.strip()}\"\"\""""

    # ---- Local AI helpers ----------------------------------------------------
    def _ai_rewrite_clarity(api_key: str, req_text: str) -> str:
        """
        Smart rewrite:
          - If rule engine says the requirement lacks measurable criteria or has an alert w/o trigger,
            return a deterministic rewrite with explicit TBD placeholders (no guessing).
          - Otherwise, call the AI with a stricter prompt.
        """
        text = (req_text or "").strip()

        # 1) Fix non-binding modal to "shall" up front (no invention)
        text_fixed_modal = re.sub(r"\b(will|would|can|could|may|should)\b", "shall", text, flags=re.IGNORECASE)

        # 2) Ask the rule engine what’s wrong
        amb = safe_call_ambiguity(text_fixed_modal, rule_engine)

        def _has(label: str) -> bool:
            return any(label.lower() in tok.lower() for tok in amb)

        # 3) Deterministic rewrites with explicit TBDs (no invented numbers)
        # 3a) Alerts/warnings but no trigger/condition -> add WHEN + time-to-alert TBD
        if _has("Alert without trigger"):
            return "The system shall annunciate a warning to the crew within TBD s when <TRIGGER TBD>."

        # 3b) Weak verbs and no measurable criterion -> add measurable placeholders
        if _has("No measurable criterion"):
            if re.search(r"\b(enable|support|provide|allow)\b", text_fixed_modal, re.IGNORECASE):
                return (
                    "The system shall provide a configuration interface to set <PARAMETERS TBD>; "
                    "changes shall take effect within TBD s and be recorded per <LOGGING POLICY TBD>."
                )
            # generic “no measurable” fallback
            return "The system shall perform the specified function with <PERFORMANCE METRIC TBD> under <CONDITIONS TBD>."

        # 4) If the only issue was non-binding modal, keep user wording but with “shall”
        if _has("Non-binding modal"):
            if text_fixed_modal.lower().startswith("shall "):
                return "The system shall " + text_fixed_modal[6:]
            return text_fixed_modal if "shall" in text_fixed_modal.lower() else f"The system shall {text_fixed_modal}"

        # 5) Otherwise, use AI — with a stricter prompt that prefers TBD placeholders over vagueness
        out = get_ai_suggestion(api_key, _ai_rewrite_prompt(text_fixed_modal)) or ""
        for ln in out.splitlines():
            ln = ln.strip()
            if ln:
                return ln
        return out.strip() or text_fixed_modal

    def _ai_decompose_children(api_key: str, parent_id: str, cleaned_sentence: str) -> str:
        return decompose_requirement_with_ai(api_key, _ai_decompose_prompt(parent_id, cleaned_sentence)) or ""

    # --- Batch rewrite helper -------------------------------------------------
    def _ai_batch_rewrite(api_key: str, items):
        """
        items: list[dict] with keys: id, text, ambiguous, passive, incomplete, singularity
        Stores results in st.session_state[f"rewritten_cache_{id}"]
        """
        total = len(items)
        if total == 0:
            return 0
        prog = st.progress(0.0)
        done = 0
        for r in items:
            try:
                suggestion = _ai_rewrite_clarity(api_key, r["text"])
                st.session_state[f"rewritten_cache_{r['id']}"] = suggestion.strip()
                done += 1
            except Exception:
                pass
            prog.progress(done / total)
        return done

    # --- Batch rewrite + conditional decompose (for non-singular only) -------
    def _ai_batch_rewrite_and_decompose(api_key: str, items):
        """
        For each flagged item:
          - rewrite to clear, singular text
          - if the original had singularity issues, also decompose (use rewritten text when available)
        Stores:
          - st.session_state['rewritten_cache_{id}']
          - st.session_state['decomp_cache_{id}']  (append if multiple decompositions)
        Returns (rewrote_count, decomposed_count)
        """
        total = len(items)
        if total == 0:
            return 0, 0
        prog = st.progress(0.0)
        done = 0
        rewrote = 0
        decomped = 0
        for r in items:
            # 1) rewrite
            rewritten = ""
            try:
                rewritten = _ai_rewrite_clarity(api_key, r["text"]) or ""
                st.session_state[f"rewritten_cache_{r['id']}"] = rewritten.strip()
                if rewritten.strip():
                    rewrote += 1
            except Exception:
                pass

            # 2) decompose only if singularity issue present
            if r.get("singularity"):
                base = (rewritten.strip() or r["text"]).strip()
                try:
                    dec = _ai_decompose_children(api_key, r["id"], base) or ""
                    if dec.strip():
                        key = f"decomp_cache_{r['id']}"
                        existing = st.session_state.get(key, "").strip()
                        combined = (existing + "\n" + dec.strip()).strip() if existing and dec.strip() not in existing else (dec.strip() or existing)
                        st.session_state[key] = combined
                        decomped += 1
                except Exception:
                    pass

            done += 1
            prog.progress(done / total)
        return rewrote, decomped

    # --- Helper: show only failing categories ---------------------------------
    def _error_badges(r):
        labels = []
        if r.get("ambiguous"):
            labels.append("⚠️ Ambiguity")
        if r.get("passive"):
            labels.append("⚠️ Passive Voice")
        if r.get("incomplete"):
            labels.append("⚠️ Incomplete")
        if r.get("singularity"):
            labels.append("⚠️ Not Singular")
        if not labels:
            return ""
        chips = " ".join(
            f"<span style='background:#FFF3CD;color:#856404;padding:4px 10px;border-radius:999px;display:inline-block;margin:0 6px 6px 0;font-size:0.85rem'>{t}</span>"
            for t in labels
        )
        return f"<div>{chips}</div>"

    # --- Helper: turn decomposition markdown/text into child rows --------------
    def _extract_child_rows_from_decomp(parent_id: str, decomp_text: str):
        """
        Accepts the AI decomposition text and returns [(child_id, child_text), ...].
        Accepts lines like:
          PARENT.1: text
          - PARENT.2: text
          * PARENT.3: text
          • PARENT.4: text
        """
        if not decomp_text:
            return []
        rows = []
        pid = parent_id.split(":")[0].strip()
        patt = re.compile(rf"^\s*[-*•]?\s*({re.escape(pid)}\.\d+)\s*:\s*(.+)$")
        for raw in decomp_text.splitlines():
            ln = raw.strip()
            if not ln:
                continue
            m = patt.match(ln)
            if m:
                rows.append((m.group(1).strip(), m.group(2).strip()))
        return rows

    def _post_filter_ambiguity(modal_text: str, amb_list):
        """
        Suppress 'Non-binding modal' when the requirement already contains 'shall'.
        Also harden tokenization to avoid matching words like 'Marshall'.
        """
        t = (modal_text or "").lower()
        # If there's a real 'shall' token, drop the non-binding modal warning
        has_shall = re.search(r"\bshall\b", t) is not None

        cleaned = []
        for item in (amb_list or []):
            s = str(item)
            if has_shall and re.search(r"non[- ]binding modal", s, re.I):
                # skip this specific warning
                continue
            cleaned.append(s)
        return cleaned

    # === AI JSON sanitizer + global+pairwise contradiction scan ===================
    import json as _json, re as _re

    def _heuristic_contradictions(req_rows):
        """
        Fast, deterministic catches for classic clashes:
        - Timing: 'within X s' vs 'may take up to Y s' on same action/domain
        - Operator confirmation policy: 'without operator' vs 'require operator' (except emergency)
        - Redundancy vs cost/limit: 'single-fault tolerant / full redundancy' vs 'prohibit duplication/limit mass/cost'
        - Priority inversions: 'always use A' vs 'always prioritize B' for same scope
        """
        out = []
        def _seconds(text):
            m = re.search(r"within\s+(\d+(?:\.\d+)?)\s*s\b", text, re.I)
            return float(m.group(1)) if m else None
        def _may_take_up_to_seconds(text):
            m = re.search(r"may\s+take\s+up\s+to\s+(\d+(?:\.\d+)?)\s*s\b", text, re.I)
            return float(m.group(1)) if m else None
        def _mentions_uplink(text):
            return bool(re.search(r"\buplink\b|\buplink commands?\b|\baccept(ed)?\b|\bexecute(d)?\b", text, re.I))
        def _operator_required(text):
            return bool(re.search(r"(require|shall require).*(operator|acknowledg(e|ment))", text, re.I))
        def _operator_not_required(text):
            return bool(re.search(r"without\s+operator|no\s+operator", text, re.I))
        def _emergency_exception(text):
            return bool(re.search(r"except.*(emergency|catastrophic)", text, re.I))
        def _redundancy_required(text):
            return bool(re.search(r"(single[- ]fault tolerant|full hardware redundancy|triple[- ]redundant)", text, re.I))
        def _no_duplication(text):
            return bool(re.search(r"(prohibit|limit).*(duplication|duplicate|full duplication)", text, re.I))
        def _priority_a(text, a, scope=None):
            pat = rf"(always\s+use|use.*primary|prioritize).*{re.escape(a)}"
            ok = re.search(pat, text, re.I)
            if ok and scope:
                ok = ok and (re.search(scope, text, re.I) is not None)
            return bool(ok)
        N = len(req_rows)
        for i in range(N):
            ai, adoc, at = req_rows[i]["id"], req_rows[i]["doc"], req_rows[i]["text"]
            for j in range(i+1, N):
                bi, bdoc, bt = req_rows[j]["id"], req_rows[j]["doc"], req_rows[j]["text"]
                a_w = _seconds(at); b_w = _seconds(bt)
                a_m = _may_take_up_to_seconds(at); b_m = _may_take_up_to_seconds(bt)
                if _mentions_uplink(at) and _mentions_uplink(bt):
                    if (a_w is not None and b_m is not None and a_w < b_m) or (b_w is not None and a_m is not None and b_w < a_m):
                        out.append({
                            "kind": "timing_conflict",
                            "reason": "Command path requires ≤ {:.0f}s but another step may take up to {:.0f}s.".format(
                                a_w if a_w is not None else b_w, b_m if b_m is not None else a_m
                            ),
                            "a_id": ai, "a_doc": adoc, "a_text": at,
                            "b_id": bi, "b_doc": bdoc, "b_text": bt,
                            "scope": "Uplink/command processing"
                        })
                        continue
                if (_operator_not_required(at) and _operator_required(bt) and not _emergency_exception(bt)) or \
                   (_operator_not_required(bt) and _operator_required(at) and not _emergency_exception(at)):
                    out.append({
                        "kind": "policy_conflict",
                        "reason": "One clause removes operator confirmation while another requires it.",
                        "a_id": ai, "a_doc": adoc, "a_text": at,
                        "b_id": bi, "b_doc": bdoc, "b_text": bt,
                        "scope": ""
                    })
                    continue
                if (_redundancy_required(at) and _no_duplication(bt)) or (_redundancy_required(bt) and _no_duplication(at)):
                    out.append({
                        "kind": "resource_conflict",
                        "reason": "Redundancy is required but another clause prohibits duplication/limits that prevent it.",
                        "a_id": ai, "a_doc": adoc, "a_text": at,
                        "b_id": bi, "b_doc": bdoc, "b_text": bt,
                        "scope": "Propulsion/avionics redundancy"
                    })
                    continue
                if (_priority_a(at, "inertial navigation", r"orbit insertion") and _priority_a(bt, "star tracker", r"orbit insertion")) or \
                   (_priority_a(bt, "inertial navigation", r"orbit insertion") and _priority_a(at, "star tracker", r"orbit insertion")):
                    out.append({
                        "kind": "policy_conflict",
                        "reason": "Conflicting primary/prioritized navigation sources during orbit insertion.",
                        "a_id": ai, "a_doc": adoc, "a_text": at,
                        "b_id": bi, "b_doc": bdoc, "b_text": bt,
                        "scope": "Orbit insertion"
                    })
                    continue
        return out

    def _json_array_from_any(s: str):
        if not s:
            return []
        s = s.strip()

        # Strip code fences if present ```json ... ```
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
            s = re.sub(r"\s*```$", "", s)

        # strict parse first
        try:
            obj = _json.loads(s)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                # common keys the model might use
                for k in ("findings", "decision", "contradictions", "results"):
                    v = obj.get(k)
                    if isinstance(v, list):
                        return v
                # or first top-level array value
                for v in obj.values():
                    if isinstance(v, list):
                        return v
        except Exception:
            pass

        # extract first [...] and strip trailing commas
        start, end = s.find("["), s.rfind("]")
        if start != -1 and end > start:
            payload = re.sub(r",\s*([\]\}])", r"\1", s[start:end+1])
            try:
                return _json.loads(payload)
            except Exception:
                return []
        return []

    def _force_ai_contra_scan(req_rows, CTX, *, timeout_s: int = 60, temperature: float = 0.1):
        """
        req_rows: list of {"id": "...", "text": "...", "doc": "..."}.
        Returns list[dict] with keys: kind, reason, a_id, a_doc, a_text, b_id, b_doc, b_text, scope
        Strategy:
          1) Single global scan over all requirements.
          2) If empty, pairwise scan in chunks over capped pairs.
        """
        if not CTX.get("call_ai_contra"):
            return []

        # 0) Heuristic pre-pass (fast, deterministic)
        heur = _heuristic_contradictions(req_rows)

        # ---------- (1) Global scan ----------
        sys_brief = CTX.get("AI_CONTRA_SYSTEM_PROMPT", "")
        global_prompt = (
            (sys_brief + "\n\n") +
            "You are checking a set of natural-language requirements for contradictions ONLY.\n"
            "Return STRICT JSON (no markdown, no commentary): a JSON array of objects with EXACT keys:\n"
            "[{\n"
            '  "kind":"policy_conflict|timing_conflict|accuracy_vs_latency|safety_vs_speed|scope_conflict|resource_conflict|other",\n'
            '  "reason":"concise explanation",\n'
            '  "a_id":"R-###","a_doc":"DocName","a_text":"full text",\n'
            '  "b_id":"R-###","b_doc":"DocName","b_text":"full text",\n'
            '  "scope": "optional scope string or empty"\n'
            "}]\n"
            "Flag only truly incompatible pairs (mutually exclusive or practically impossible to satisfy together).\n"
            "Do not return unclear/conditional items. JSON array only.\n\n"
            "Requirements:\n"
            + "\n".join(f"- [{r['id']}] ({r.get('doc','')}): {r['text']}" for r in req_rows)
        )
        raw_global = CTX["call_ai_contra"](
            global_prompt,
            timeout_s=timeout_s,
        )
        findings = _json_array_from_any(raw_global)

        # ---------- (2) Pairwise fallback in chunks ----------
        if not findings:
            # Limit to TOPK to avoid prompt explosion
            topk = int(CTX.get("AI_CONTRA_TOPK", 200))
            req_subset = req_rows[:topk]

            pairs = []
            for i in range(len(req_subset)):
                for j in range(i + 1, len(req_subset)):
                    a, b = req_subset[i], req_subset[j]
                    pairs.append(
                        f"[{a['id']}] ({a.get('doc','QuickPaste')}): {a['text']} || "
                        f"[{b['id']}] ({b.get('doc','QuickPaste')}): {b['text']}"
                    )

            chunk_size = 100
            all_found = []
            for cstart in range(0, len(pairs), chunk_size):
                chunk = pairs[cstart:cstart + chunk_size]
                if not chunk:
                    continue
                pair_prompt = (
                    "You will receive PAIRS of requirements separated by ' || '.\n"
                    "For each pair, decide ONLY whether they are contradictory (mutually exclusive in practice).\n"
                    "Return STRICT JSON array; one object PER contradictory pair using EXACT keys as below. Omit non-contradictory pairs.\n"
                    "[{\n"
                    '  "kind":"policy_conflict|timing_conflict|accuracy_vs_latency|safety_vs_speed|scope_conflict|resource_conflict|other",\n'
                    '  "reason":"concise explanation",\n'
                    '  "a_id":"R-###","a_doc":"DocName","a_text":"full text",\n'
                    '  "b_id":"R-###","b_doc":"DocName","b_text":"full text",\n'
                    '  "scope": ""\n'
                    "}]\n"
                    "Return [] if none in this chunk. No prose.\n\n"
                    "Pairs:\n" + "\n".join(chunk)
                )
                raw_pairs = CTX["call_ai_contra"](
                    pair_prompt,
                    timeout_s=max(timeout_s, CTX.get("AI_CONTRA_BUDGET_S", 70)),
                )
                all_found.extend(_json_array_from_any(raw_pairs))
                try:
                    import streamlit as st
                    st.session_state.setdefault("_ai_contra_raw_pairs", "")
                    st.session_state["_ai_contra_raw_pairs"] += f"\n\n// chunk {cstart//chunk_size}\n{raw_pairs}"
                except Exception:
                    pass

            findings = all_found
        else:
            try:
                import streamlit as st
                st.session_state["_ai_contra_raw_global"] = raw_global
            except Exception:
                pass

        # Merge heuristics + AI (dedupe by (a_id,b_id,reason))
        all_findings = []
        seen = set()
        for f in (heur or []):
            k = (f["a_id"], f["b_id"], f["reason"])
            if k not in seen:
                seen.add(k); all_findings.append(f)
        for f in (findings or []):
            f_std = {
                "kind": str(f.get("kind","other")),
                "reason": str(f.get("reason","")).strip(),
                "a_id": str(f.get("a_id","")),
                "a_doc": str(f.get("a_doc","QuickPaste")),
                "a_text": str(f.get("a_text","")),
                "b_id": str(f.get("b_id","")),
                "b_doc": str(f.get("b_doc","QuickPaste")),
                "b_text": str(f.get("b_text","")),
                "scope": str(f.get("scope","")).strip(),
            }
            k = (f_std["a_id"], f_std["b_id"], f_std["reason"])
            if k not in seen:
                seen.add(k); all_findings.append(f_std)
        return all_findings

    # --- Deterministic contradiction scan (numeric/policy) -----------------------
    def _deterministic_contra_scan(req_rows):
        """
        req_rows: [{"id","text","doc"}]
        Returns: list[{kind, reason, a_id,a_doc,a_text, b_id,b_doc,b_text, scope}]
        Catches: policy conflicts, timeline conflicts, accuracy-vs-latency, resource caps, etc.
        """
        import re as _re2
        out = []
        def _has(t, *needles):
            t = (t or "").lower()
            return all(n.lower() in t for n in needles)
        def _num_ms(s):
            m = _re2.search(r'(\d+(?:\.\d+)?)\s*ms\b', s or "", _re2.I)
            return float(m.group(1)) if m else None
        def _num_s(s):
            m = _re2.search(r'(\d+(?:\.\d+)?)\s*s\b', s or "", _re2.I)
            return float(m.group(1)) if m else None
        def _num_meters(s):
            m = _re2.search(r'≤?\s*(\d+(?:\.\d+)?)\s*m\b', s or "", _re2.I)
            return float(m.group(1)) if m else None
        def _mass_fraction(s):
            m = _re2.search(r'(\d+(?:\.\d+)?)\s*%', s or "")
            return float(m.group(1)) if m else None
        themes = {
            "timing": [], "approval": [], "calibration": [], "commands": [],
            "navigation": [], "propulsion": [], "updates": [], "resource": [],
        }
        for r in req_rows or []:
            t = (r.get("text") or "").lower()
            if any(k in t for k in ["ms", "deadline", "latency", "within "]): themes["timing"].append(r)
            if any(k in t for k in ["operator", "confirmation", "acknowled", "approval", "human"]): themes["approval"].append(r)
            if "calibration" in t or "coeff" in t: themes["calibration"].append(r)
            if "command" in t: themes["commands"].append(r)
            if "navigation" in t or "position" in t or "accuracy" in t or "latency" in t: themes["navigation"].append(r)
            if "propulsion" in t or "burn" in t or "thruster" in t: themes["propulsion"].append(r)
            if "update" in t or "patch" in t or "regression" in t: themes["updates"].append(r)
            if any(k in t for k in ["mass", "fraction", "limit", "cap", "redundan"]): themes["resource"].append(r)
        def _add(kind, reason, a, b, scope=""):
            out.append({
                "kind": kind, "reason": reason,
                "a_id": a["id"], "a_doc": a.get("doc",""), "a_text": a["text"],
                "b_id": b["id"], "b_doc": b.get("doc",""), "b_text": b["text"],
                "scope": scope
            })
        # 1) Timing: deadline vs scheduler timeslice
        for a in themes["timing"]:
            for b in themes["timing"]:
                if a["id"] >= b["id"]:
                    continue
                a_ms = _num_ms(a["text"])
                b_slice = None
                if _re2.search(r'(slice|timeslice|time[- ]?slice).*(?:min|minimum)\s*(\d+(?:\.\d+)?)\s*ms', b["text"], _re2.I):
                    b_slice = float(_re2.search(r'(?:min|minimum)\s*(\d+(?:\.\d+)?)\s*ms', b["text"], _re2.I).group(1))
                if a_ms is not None and b_slice is not None and b_slice > a_ms:
                    _add("timing_conflict",
                         f"Control deadline {a_ms} ms cannot be met with scheduler minimum slice {b_slice} ms.",
                         a, b, "control-loop vs scheduler")
        # 2) Approval vs no-approval
        for a in themes["approval"]:
            for b in themes["approval"]:
                if a["id"] >= b["id"]:
                    continue
                no_op = _has(a["text"], "without") and (_has(a["text"], "operator") or _has(a["text"], "human"))
                needs_op = _has(b["text"], "require") and (_has(b["text"], "operator") or _has(b["text"], "approval") or _has(b["text"], "acknowled"))
                if no_op and needs_op:
                    _add("policy_conflict",
                         "One requires operator approval; the other forbids it for time-critical actions.",
                         a, b, "operator-approval")
        # 3) Calibration policy clash
        for a in themes["calibration"]:
            for b in themes["calibration"]:
                if a["id"] >= b["id"]:
                    continue
                always_latest = _has(a["text"], "always") and _has(a["text"], "latest", "calibration")
                forbid_change = _has(b["text"], "no", "change") and _has(b["text"], "regression", "test")
                if always_latest and forbid_change:
                    _add("policy_conflict",
                         "Policy requires always using latest coefficients but also forbids changes without a full regression campaign.",
                         a, b, "sensor calibration")
        # 4) Commands: immediate apply vs queue for approval
        for a in themes["commands"]:
            for b in themes["commands"]:
                if a["id"] >= b["id"]:
                    continue
                accept_apply = _has(a["text"], "accept") and _has(a["text"], "apply") and _has(a["text"], "authenticated")
                queue_only = _has(b["text"], "queue") and (_has(b["text"], "operator") or _has(b["text"], "approval"))
                if accept_apply and queue_only:
                    _add("safety_vs_speed",
                         "One mandates immediate application of authenticated commands; the other requires queueing for operator approval.",
                         a, b, "command handling")
        # 5) Navigation accuracy vs latency preference
        for a in themes["navigation"]:
            for b in themes["navigation"]:
                if a["id"] >= b["id"]:
                    continue
                tight = _has(a["text"], "≤ 0.05 m") or (_num_meters(a["text"]) is not None and _num_meters(a["text"]) <= 0.05)
                loose_latency = _has(b["text"], "lowest-latency") and ("1 m" in b["text"] or (_num_meters(b["text"]) and _num_meters(b["text"]) >= 1.0))
                if tight and loose_latency:
                    _add("accuracy_vs_latency",
                         "One fixes accuracy at ≤0.05 m while the other accepts up to 1 m to reduce latency for the same phase.",
                         a, b, "orbit insertion")
        # 6) Propulsion sizing vs mass fraction cap
        for a in themes["propulsion"]:
            for b in themes["resource"]:
                if a["id"] >= b["id"]:
                    continue
                needs_three = _has(a["text"], "three", "contingency") or _has(a["text"], "three unused contingency")
                cap_pct = _mass_fraction(b["text"])
                if needs_three and cap_pct is not None and cap_pct <= 15.0:
                    _add("resource_conflict",
                         "Sizing to guarantee three unused contingency burns may violate the ≤15% propellant mass fraction cap.",
                         a, b, "propellant margin")
        # 7) Update timeline: 24h vs 90 days
        for a in themes["updates"]:
            for b in themes["updates"]:
                if a["id"] >= b["id"]:
                    continue
                fast = (_has(a["text"], "within") and "24" in a["text"] and "hour" in a["text"].lower()) or _num_s(a["text"]) in (86400.0,)
                slow = _has(b["text"], "only") and ("90" in b["text"] and "day" in b["text"].lower())
                if fast and slow:
                    _add("timing_conflict",
                         "Security patch policy requires deployment within 24h, but reliability policy requires a 90-day soak.",
                         a, b, "software update policy")
        return out

    # ---- Analyze (Quick Paste) -----------------------------------------------
    if st.button("Analyze Pasted Lines", key="quick_analyze_btn"):
        pairs = _parse_quick_lines(quick_text)
        if not pairs:
            st.warning("No non-empty lines found.")
            st.session_state.quick_results = []
            st.session_state.quick_issue_counts = {"Ambiguity": 0, "Passive Voice": 0, "Incompleteness": 0, "Singularity": 0}
            st.session_state.quick_analyzed = False
            st.session_state.quick_text_snapshot = ""
        else:
            issue_counts = {"Ambiguity": 0, "Passive Voice": 0, "Incompleteness": 0, "Singularity": 0}
            quick_results = []
            for rid, rtx in pairs:
                amb_raw = safe_call_ambiguity(rtx, rule_engine)
                amb = _post_filter_ambiguity(rtx, amb_raw)

                pas = check_passive_voice(rtx)
                inc = check_incompleteness(rtx)
                try:
                    sing = check_singularity(rtx)
                except Exception:
                    sing = []
                if amb:
                    issue_counts["Ambiguity"] += 1
                if pas:
                    issue_counts["Passive Voice"] += 1
                if inc:
                    issue_counts["Incompleteness"] += 1
                if sing:
                    issue_counts["Singularity"] += 1
                quick_results.append({
                    "id": rid, "text": rtx,
                    "ambiguous": amb, "passive": pas, "incomplete": inc, "singularity": sing
                })
            st.session_state.quick_results = quick_results
            st.session_state.quick_issue_counts = issue_counts
            st.session_state.quick_analyzed = True
            st.session_state.quick_text_snapshot = quick_text
    # Remove the AI smoke test button
    # if st.button("AI Smoke Test: get_ai_suggestion", key="ai_smoke"):
    #     try:
    #         key = st.session_state.get("api_key")
    #         if not key:
    #             st.error("No API key in session_state.api_key.")
    #         else:
    #             sample = "Reply with JSON only: {\"ok\": true}"
    #             out = CTX["get_ai_suggestion"](key, prompt=sample, timeout_s=10) or ""
    #             st.code(out or "<empty>", language="json")
    #     except Exception as e:
    #         st.error(f"AI call blew up: {e}")


    # ---- Render quick results -------------------------------------------------
    if st.session_state.quick_analyzed and st.session_state.quick_results:
        quick_results = st.session_state.quick_results
        issue_counts = st.session_state.quick_issue_counts

        total = len(quick_results)
        flagged = sum(1 for r in quick_results if r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"])
        st.markdown(f"**Analyzed:** {total} • **Flagged:** {flagged}")
        cqa = st.columns(4)
        cqa[0].metric("Ambiguity", issue_counts["Ambiguity"])
        cqa[1].metric("Passive", issue_counts["Passive Voice"])
        cqa[2].metric("Incomplete", issue_counts["Incompleteness"])
        cqa[3].metric("Multiple actions", issue_counts["Singularity"])

        # --- bulk buttons (Quick Paste) ---
        with st.expander("Bulk AI actions (optional)", expanded=not beginner):
            st.caption("Power tools and troubleshooting.")
            cols_bulk = st.columns([1.3, 1.9])
            with cols_bulk[0]:
                if st.button("🧹 Reset AI rewrite/decompose caches", on_click=_reset_ai_caches, key="btn_reset_ai_caches"):
                    st.success("AI caches reset.")
            with cols_bulk[1]:
                _ai_smoke_check(label_key="ai_smoke_global")

        # Define lists before rendering expanders
        flagged_list = [r for r in quick_results if r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"]]
        clear_list = [r for r in quick_results if not (r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"])]

        with st.expander(f"Flagged ({len(flagged_list)})", expanded=True):
            if not flagged_list:
                st.caption("None 🎉")
            for r in flagged_list:
                with st.container():
                    st.markdown(
                        format_requirement_with_highlights(r["id"], r["text"], r),
                        unsafe_allow_html=True,
                    )
                    badges_html = _error_badges(r)
                    if badges_html:
                        st.markdown(badges_html, unsafe_allow_html=True)

                    # AI actions with strict gating
                    has_amb = bool(r.get("ambiguous"))
                    has_pas = bool(r.get("passive"))
                    has_inc = bool(r.get("incomplete"))
                    has_sing = bool(r.get("singularity"))
                    only_sing = has_sing and not (has_amb or has_pas or has_inc)

                    if st.session_state.api_key:
                        if only_sing:
                            cols = st.columns(1)
                            with cols[0]:
                                if st.button(f"🧩 Decompose [{r['id']}]", key=f"quick_dec_only_{r['id']}"):
                                    try:
                                        base = st.session_state.get(f"rewritten_cache_{r['id']}", "").strip() or r["text"]
                                        d = _ai_decompose_children(st.session_state.api_key, r["id"], base)
                                        if d.strip():
                                            key = f"decomp_cache_{r['id']}"
                                            existing = st.session_state.get(key, "").strip()
                                            st.session_state[key] = ((existing + "\n" + d.strip()).strip()
                                                                     if existing and d.strip() not in existing else (d.strip() or existing))
                                        st.info("Decomposition:")
                                        st.markdown(st.session_state.get(f"decomp_cache_{r['id']}", d))
                                    except Exception as e:
                                        st.warning(f"AI decomposition failed: {e}")
                        elif has_sing:
                            cols = st.columns(3)
                            with cols[0]:
                                if st.button(f"⚒️ Fix Clarity [{r['id']}]", key=f"quick_fix_{r['id']}"):
                                    try:
                                        suggestion = _ai_rewrite_clarity(st.session_state.api_key, r['text'])
                                        st.session_state[f"rewritten_cache_{r['id']}"] = (suggestion or "").strip()
                                        st.info("Rewritten:")
                                        st.markdown(f"> {suggestion}")
                                    except Exception as e:
                                        st.warning(f"AI rewrite failed: {e}")
                            with cols[1]:
                                if st.button(f"🧩 Decompose [{r['id']}]", key=f"quick_dec_{r['id']}"):
                                    try:
                                        base = st.session_state.get(f"rewritten_cache_{r['id']}", "").strip() or r["text"]
                                        d = _ai_decompose_children(st.session_state.api_key, r["id"], base)
                                        if d.strip():
                                            key = f"decomp_cache_{r['id']}"
                                            existing = st.session_state.get(key, "").strip()
                                            st.session_state[key] = ((existing + "\n" + d.strip()).strip()
                                                                     if existing and d.strip() not in existing else (d.strip() or existing))
                                        st.info("Decomposition:")
                                        st.markdown(st.session_state.get(f"decomp_cache_{r['id']}", d))
                                    except Exception as e:
                                        st.warning(f"AI decomposition failed: {e}")
                            with cols[2]:
                                if st.button(f"Auto: Fix → Decompose [{r['id']}]", key=f"quick_pipe_{r['id']}"):
                                    try:
                                        cleaned = st.session_state.get(f"rewritten_cache_{r['id']}", "").strip() or _ai_rewrite_clarity(st.session_state.api_key, r["text"])
                                        st.session_state[f"rewritten_cache_{r['id']}"] = cleaned
                                        d = _ai_decompose_children(st.session_state.api_key, r["id"], cleaned)
                                        if d.strip():
                                            key = f"decomp_cache_{r['id']}"
                                            existing = st.session_state.get(key, "").strip()
                                            st.session_state[key] = ((existing + "\n" + d.strip()).strip()
                                                                     if existing and d.strip() not in existing else (d.strip() or existing))
                                        st.success("Rewritten requirement:")
                                        st.markdown(f"> {cleaned}")
                                        if st.session_state.get(f"decomp_cache_{r['id']}", ""):
                                            st.info("Decomposition:")
                                            st.markdown(st.session_state[f"decomp_cache_{r['id']}"])
                                    except Exception as e:
                                        st.warning(f"AI pipeline failed: {e}")
                    else:
                        cols = st.columns(1)
                        with cols[0]:
                            if st.button(f"⚒️ Fix Clarity [{r['id']}]", key=f"quick_fix_only_{r['id']}"):
                                try:
                                    suggestion = _ai_rewrite_clarity(st.session_state.api_key, r['text'])
                                    st.session_state[f"rewritten_cache_{r['id']}"] = (suggestion or "").strip()
                                    st.info("Rewritten:")
                                    st.markdown(f"> {suggestion}")
                                except Exception as e:
                                    st.warning(f"AI rewrite failed: {e}")

                # Show cached results inline (inside flagged container)
                cached_rw = st.session_state.get(f"rewritten_cache_{r['id']}", "")
                cached_dc = st.session_state.get(f"decomp_cache_{r['id']}", "")
                if cached_rw:
                    st.caption("AI Rewrite (cached):")
                    st.code(cached_rw)
                if cached_dc:
                    st.caption("AI Decomposition (cached):")
                    st.markdown(cached_dc)
        with st.expander(f"Clear ({len(clear_list)})", expanded=False):
            for r in clear_list:
                st.markdown(
                    f'<div style="background-color:#D4EDDA;color:#155724;padding:10px;'
                    f'border-radius:5px;margin-bottom:10px;">✅ <strong>{r["id"]}</strong> {r["text"]}</div>',
                    unsafe_allow_html=True,
                )

    # --------------- NEW (Quick Paste): Contradiction detection ---------------
    st.subheader("AI Contradiction Scan (Quick Paste)")
    ai_temp_qp = 0.1
    if not beginner:
        with st.expander("Advanced (tuning & debug)", expanded=False):
            sensitivity_qp = st.select_slider(
                "Contradiction sensitivity",
                options=["Conservative", "Balanced", "Aggressive"],
                value="Balanced",
                key="contra_sensitivity_qp",
                help="Conservative finds fewer conflicts; Aggressive may raise more potential contradictions.",
            )
            ai_temp_qp = {"Conservative": 0.0, "Balanced": 0.1, "Aggressive": 0.3}[sensitivity_qp]
            st.caption("Tip: Use 'Show raw model output' if results look empty or odd.")
    if st.button("🔎 Run AI Contradiction Scan ", key="ai_contra_qp_btn"):
        if not CTX.get("api_key"):
            st.info("AI key not set — deterministic contradictions will still be shown.")
        if not (st.session_state.quick_analyzed and st.session_state.quick_results):
            st.warning("Paste text and click **Analyze Pasted Lines** first.")
        else:
            req_rows = [{"id": r["id"], "text": r["text"], "doc": "QuickPaste"}
                        for r in st.session_state.get("quick_results", [])]
            # 1) Deterministic pass
            det = _deterministic_contra_scan(req_rows)
            # 2) Optional AI pass
            ai = []
            if CTX.get("api_key"):
                with st.spinner("Scanning for contradictions with AI..."):
                    ai = _force_ai_contra_scan(
                        req_rows, CTX,
                        timeout_s=max(CTX.get("AI_CONTRA_BUDGET_S", 70), CTX.get("AI_CONTRA_PER_CALL_S", 20)),
                        temperature=ai_temp_qp,
                    )
            else:
                st.info("AI key not set — showing deterministic contradictions only.")
            # 3) Merge & de-dup
            def _sig(f):
                return (min(f["a_id"], f["b_id"]), max(f["a_id"], f["b_id"]), f.get("kind",""), (f.get("reason","") or "")[:160])
            seen = set(); merged = []
            for src in (det or []) + (ai or []):
                s = _sig(src)
                if s in seen:
                    continue
                seen.add(s); merged.append(src)
            # 4) Render
            if merged:
                st.caption(f"Found {len(merged)} contradiction(s):")
                for f in merged[:300]:
                    st.markdown(
                        f"🚨 **{f['kind'].capitalize()}** — {f['reason']}"
                        f"\n\n• **{f['a_id']}** ({f['a_doc']}): {f['a_text']}"
                        f"\n\n• **{f['b_id']}** ({f['b_doc']}): {f['b_text']}"
                        + (f"\n\n_Scope_: `{f['scope']}`" if f.get('scope') else "")
                    )
            else:
                st.info("No contradictions detected by the deterministic + AI scan.")
            # Optional debug: show raw AI outputs (helps when nothing is found)
            if (not beginner) and st.checkbox("Show raw model output (debug)", key="show_ai_contra_raw_qp"):
                raw_g = st.session_state.get("_ai_contra_raw_global", "")
                raw_p = st.session_state.get("_ai_contra_raw_pairs", "")
                if raw_g:
                    st.caption("Raw (global scan):")
                    st.code(raw_g, language="json")
                if raw_p:
                    st.caption("Raw (pairwise scan):")
                    st.code(raw_p, language="json")

    # ===================== Full Document Analyzer =============================
    st.subheader("📁 Upload Documents — analyze one or more files")
    use_ai_parser = st.toggle(
        "Use Advanced AI Parser (requires API key)",
        key="use_ai_parser_toggle",  # unique key to avoid toggle collisions
        help="Tries a more capable parser (if available); otherwise falls back to the standard extractor."
    )
    if use_ai_parser and not (HAS_AI_PARSER and st.session_state.api_key):
        st.info("AI Parser not available (missing function or API key). Falling back to Standard Parser.")

    project_id = st.session_state.selected_project[0] if st.session_state.selected_project else None

    stored_to_analyze = None
    if project_id is not None and hasattr(db, "get_documents_for_project"):
        try:
            _rows = db.get_documents_for_project(project_id)
            stored_docs, labels = [], []
            for (doc_id, file_name, version, uploaded_at, clarity_score) in _rows:
                conv_path = os.path.join("data", "projects", str(project_id), "documents",
                                         f"{doc_id}_{CTX['_sanitize_filename'](file_name)}")
                if os.path.exists(conv_path):
                    stored_docs.append((doc_id, file_name, version, conv_path))
                    labels.append(f"{file_name} (v{version})")
            if stored_docs:
                sel = st.selectbox("Re-analyze a saved document:", ["— Select —"] + labels, key="rean_select")
                if sel != "— Select —":
                    if st.button("Analyze Selected", key="rean_btn"):
                        idx = labels.index(sel)
                        _doc_id, _fn, _ver, _path = stored_docs[idx]
                        stored_to_analyze = (_fn, _path)
        except Exception:
            pass

    uploaded_files = st.file_uploader(
        "Upload one or more requirements documents (.txt or .docx)",
        type=['txt', 'docx'],
        accept_multiple_files=True,
        key=f"uploader_unified_{project_id or 'none'}",
    )

    example_files = {"Choose an example...": None, "Drone System SRS (Complex Example)": "DRONE_SRS_v1.0.docx"}
    selected_example = st.selectbox(
        "Or, select an example to analyze:",
        options=list(example_files.keys()),
        key="example_unified",
    )

    docs_to_process = []
    if uploaded_files:
        for up in uploaded_files:
            docs_to_process.append(("upload", up.name, up))
    if selected_example != "Choose an example...":
        example_path = example_files[selected_example]
        try:
            if example_path.endswith(".docx"):
                d = docx.Document(example_path)
                example_text = "\n".join([p.text for p in d.paragraphs if p.text.strip()])
            else:
                with open(example_path, "r", encoding="utf-8") as f:
                    example_text = f.read()
            docs_to_process.append(("example", selected_example, example_text))
        except FileNotFoundError:
            st.error(f"Example file not found: {example_path}. Place it in the project folder.")
    if stored_to_analyze:
        _fn, _path = stored_to_analyze
        docs_to_process.append(("stored", _fn, _path))

    # ---------- STRICT requirement gate (shared) ----------
    _REQ_MODAL_RE = re.compile(r"\b(shall|must|will|should)\b", re.I)

    # Known section/heading terms
    _COMMON_HEADING_TERMS = {
        "introduction", "overview", "system overview", "scope", "requirements",
        "appendix", "glossary", "references", "purpose", "background", "summary",
        "assumptions", "constraints", "abbreviations", "definitions", "table of contents"
    }

    # Strip leading requirement ID like "R-041:" or "REQ_12 :"
    _LEADING_ID_RE = re.compile(r"^\s*[A-Z]{1,8}-?\d{1,5}\s*:?\s*", re.I)

    # Catch code-like lines and obvious non-requirements (balanced & 3.13-safe)
    _CODE_LINE_RE = re.compile(
        r'(?:^\s*(?:#|//))'                      # comment at start of line
        r'|(?:^\s*(?:from\s+\w+|import\s+\w+))'  # imports
        r'|(?:^\s*(?:class|def)\s+\w+\s*\()'     # class/def signatures
        r'|(?:^\s*@\w+)'                         # decorators
        r'|re\.compile\('                        # explicit re.compile(
    )

    # --- Subject start: broader natural-language subjects (detectors/nouns) ---
    _SUBJECT_START_RE = re.compile(
        r"^(the|this|a|an|all|any|each|every|"
        r"system|software|application|device|module|service|"
        r"user|operator|controller|interface|api|component|"
        r"satellite|spacecraft|payload|tt&c|ttc|link|uplink|downlink|"
        r"eps|battery|detector|components?|deployables?|c&dh|cdh|fsw|"
        r"telemetry|command(?:\s+links?)?|project|ground(?:\s+segment)?|mission|data(?:\s+products?)?"
        r")\b",
        re.I
    )

    def _base_text_without_id(txt: str) -> str:
        s = (txt or "").strip()
        return _LEADING_ID_RE.sub("", s)  # remove any leading ID/colon

    def _looks_like_heading(txt: str) -> bool:
        s = _base_text_without_id(txt or "")
        s = s.strip()
        if not s:
            return True

        # NEW: known section/heading terms
        if s.lower() in _COMMON_HEADING_TERMS:
            return True

        # If it already has a binding modal, it's not a heading.
        if _REQ_MODAL_RE.search(s):
            return False

        s_upper = s.upper()

        # Very short lines without binding modals/verbs -> likely headings
        if len(s.split()) <= 4:
            return True

        # Numbered headings like "1. Introduction", "2.3.4 Requirements"
        if s_upper and s_upper[0].isdigit():
            return True

        if len(s) <= 80 and s == s_upper and " SHALL " not in f" {s_upper} ":
            return True

        if len(s.split()) <= 8 and s.endswith((':', ';')):
            return True

        return False

    def _is_requirement_strict(s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return False
        if _CODE_LINE_RE.search(t):
            return False
        if _looks_like_heading(t):
            return False
        if not _REQ_MODAL_RE.search(t):   # must contain shall/must/will/should
            return False

        # Remove any leading ID like "R-041: " before subject check
        core = _base_text_without_id(t)

        # 1) Preferred: recognizable natural-language subject
        if _SUBJECT_START_RE.search(core):
            return True

        # 2) Fallback: if it has a binding modal and looks like a sentence (≥ 6 tokens),
        # treat it as a requirement (prevents false negatives).
        if len(core.split()) >= 6:
            return True

        return False

    # === Analyzer-local AI helpers (shared) ===================================
    def _ai_rewrite_clarity_doc(api_key: str, req_text: str) -> str:
        out = get_ai_suggestion(st.session_state.api_key, _ai_rewrite_prompt(req_text)) or ""
        for ln in out.splitlines():
            ln = ln.strip()
            if ln:
                return ln
        return out.strip()

    def _ai_decompose_clean_doc(api_key: str, parent_id: str, req_text: str) -> str:
        return decompose_requirement_with_ai(api_key, _ai_decompose_prompt(parent_id, req_text)) or ""
    # ==========================================================================

    # Collect rows for cross-document contradiction scanning
    all_doc_req_rows = []   # list[{"id","text","doc"}] across all analyzed docs

    if docs_to_process:
        with st.spinner("Processing and analyzing documents..."):
            for doc_idx, (src_type, display_name, payload) in enumerate(docs_to_process):
                # --- Extract requirements (AI + standard + table) and merge/dedupe ---
                std_reqs, table_reqs, ai_reqs = [], [], []

                if src_type == "upload":
                    if payload.name.endswith(".txt"):
                        raw = payload.getvalue().decode("utf-8", errors="ignore")
                        std_reqs = extract_requirements_from_string(raw) or []
                        if use_ai_parser and HAS_AI_PARSER and st.session_state.api_key:
                            ai_reqs = extract_requirements_with_ai(st.session_state.api_key, raw) or []
                    elif payload.name.endswith(".docx"):
                        flat_text, table_rows = _read_docx_text_and_rows(payload)
                        table_reqs = _extract_requirements_from_table_rows(table_rows) or []
                        std_reqs = extract_requirements_from_string(flat_text) or []
                        if use_ai_parser and HAS_AI_PARSER and st.session_state.api_key:
                            ai_reqs = extract_requirements_with_ai(st.session_state.api_key, flat_text) or []
                elif src_type == "stored":
                    path = payload
                    if path.endswith(".txt"):
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            raw = f.read()
                        std_reqs = extract_requirements_from_string(raw) or []
                        if use_ai_parser and HAS_AI_PARSER and st.session_state.api_key:
                            ai_reqs = extract_requirements_with_ai(st.session_state.api_key, raw) or []
                    elif path.endswith(".docx"):
                        flat_text, table_rows = _read_docx_text_and_rows_from_path(path)
                        table_reqs = _extract_requirements_from_table_rows(table_rows) or []
                        std_reqs = extract_requirements_from_string(flat_text) or []
                        if use_ai_parser and HAS_AI_PARSER and st.session_state.api_key:
                            ai_reqs = extract_requirements_with_ai(st.session_state.api_key, flat_text) or []
                else:  # example
                    std_reqs = extract_requirements_from_string(payload) or []
                    if use_ai_parser and HAS_AI_PARSER and st.session_state.api_key:
                        ai_reqs = extract_requirements_with_ai(st.session_state.api_key, payload) or []

                # Merge + dedupe
                reqs = _merge_unique_reqs(table_reqs, std_reqs, ai_reqs)
                # Filter headings BEFORE analysis
                reqs = [(rid, rtx) for (rid, rtx) in reqs if not _looks_like_heading(rtx)]
                if not reqs:
                    st.warning(f"⚠️ No recognizable requirements in **{display_name}** after filtering headings.")
                    continue

                # --- Analyze requirements --------------------------------------------
                results = []
                issue_counts = {"Ambiguity": 0, "Passive Voice": 0, "Incompleteness": 0, "Singularity": 0}

                for rid, rtext in reqs:
                    # Hard gate: if not a requirement, record and skip analysis
                    if not _is_requirement_strict(rtext):
                        results.append({
                            "id": rid,
                            "text": rtext,
                            "ambiguous": [],
                            "passive": [],
                            "incomplete": False,
                            "singularity": [],
                            "non_requirement": True,
                        })
                        continue

                    ambiguous = _post_filter_ambiguity(rtext, safe_call_ambiguity(rtext, rule_engine))
                    passive = check_passive_voice(rtext)
                    incomplete = check_incompleteness(rtext)
                    try:
                        singular = check_singularity(rtext)
                    except Exception:
                        singular = []

                    if ambiguous:
                        issue_counts["Ambiguity"] += 1
                    if passive:
                        issue_counts["Passive Voice"] += 1
                    if incomplete:
                        issue_counts["Incompleteness"] += 1
                    if singular:
                        issue_counts["Singularity"] += 1

                    results.append({
                        "id": rid,
                        "text": rtext,
                        "ambiguous": ambiguous,
                        "passive": passive,
                        "incomplete": incomplete,
                        "singularity": singular,
                    })

                # Totals should use only true requirements
                analyzed_only = [r for r in results if not r.get("non_requirement")]
                total_reqs = len(analyzed_only)
                flagged_total = sum(
                    1 for r in analyzed_only
                    if r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"]
                )
                clarity_score = int(((total_reqs - flagged_total) / total_reqs) * 100) if total_reqs else 100

                # Prepare rows for per-document and cross-document AI contradiction scan
                doc_req_rows = [{"id": r["id"], "text": r["text"], "doc": display_name} for r in analyzed_only]
                all_doc_req_rows.extend(doc_req_rows)

                # --- Save to DB if helpers exist and a project is selected ------------
                if (st.session_state.selected_project is not None) and (src_type in ("upload", "example")):
                    project_id = st.session_state.selected_project[0]
                    try:
                        if hasattr(db, "add_document") and hasattr(db, "add_requirements") and hasattr(db, "get_documents_for_project"):
                            existing = []
                            try:
                                existing = [d for d in db.get_documents_for_project(project_id) if d[1] == display_name]
                            except Exception:
                                existing = []
                            next_version = (max([d[2] for d in existing], default=0) + 1)
                            doc_id = db.add_document(project_id, display_name, next_version, clarity_score)
                            db.add_requirements(doc_id, reqs)

                            if src_type == "upload":
                                try:
                                    file_path = _save_uploaded_file_for_doc(project_id, doc_id, display_name, payload)
                                    if hasattr(db, "add_document_file"):
                                        try:
                                            db.add_document_file(doc_id, file_path)
                                        except Exception:
                                            pass
                                    elif hasattr(db, "set_document_file_path"):
                                        try:
                                            db.set_document_file_path(doc_id, file_path)
                                        except Exception:
                                            pass
                                except Exception as _e:
                                    st.warning(f"Saved analysis, but file persistence failed for '{display_name}': {_e}")

                        elif hasattr(db, "add_document_to_project") and hasattr(db, "add_requirements_to_document"):
                            doc_id = db.add_document_to_project(project_id, display_name, clarity_score)
                            db.add_requirements_to_document(doc_id, reqs)

                            if src_type == "upload":
                                try:
                                    file_path = _save_uploaded_file_for_doc(project_id, doc_id, display_name, payload)
                                    if hasattr(db, "add_document_file"):
                                        try:
                                            db.add_document_file(doc_id, file_path)
                                        except Exception:
                                            pass
                                    elif hasattr(db, "set_document_file_path"):
                                        try:
                                            db.set_document_file_path(doc_id, file_path)
                                        except Exception:
                                            pass
                                except Exception as _e:
                                    st.warning(f"Saved analysis, but file persistence failed for '{display_name}': {_e}")
                        else:
                            st.info("Analysis done — DB helpers not found, so nothing was saved.")
                    except Exception as e:
                        st.warning(f"Saved analysis for **{display_name}**, but DB write failed: {e}")

                # --- Per-document results UI -----------------------------------------
                with st.expander(f"📄 {display_name} — Clarity {clarity_score}/100 • {total_reqs} requirements"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Requirements", total_reqs)
                    c2.metric("Flagged", flagged_total)
                    c3.metric("Clarity Score", f"{clarity_score} / 100")
                    st.progress(clarity_score)

                    # Bulk actions for this document
                    if st.session_state.api_key:
                        cols_bulk_doc = st.columns([1.3, 1.9])
                        with cols_bulk_doc[0]:
                            if st.button("✨ Rewrite all flagged (AI) — this document", key=f"doc_bulk_rew_{doc_idx}"):
                                with st.spinner("AI rewriting all flagged requirements in this document..."):
                                    flagged_for_rewrite = [
                                        {"id": r["id"], "text": r["text"], "ambiguous": r["ambiguous"],
                                         "passive": r["passive"], "incomplete": r["incomplete"],
                                         "singularity": r["singularity"]}
                                        for r in analyzed_only
                                        if r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"]
                                    ]
                                    rew_count = _ai_batch_rewrite(st.session_state.api_key, flagged_for_rewrite)
                                st.success(f"Rewrote {rew_count} requirement(s).")
                        with cols_bulk_doc[1]:
                            if st.button("⚡ Rewrite & Decompose all flagged (AI) — this document", key=f"doc_bulk_rewdec_{doc_idx}"):
                                with st.spinner("AI rewriting and decomposing flagged requirements in this document..."):
                                    flagged_for_action = [
                                        {"id": r["id"], "text": r["text"], "ambiguous": r["ambiguous"],
                                         "passive": r["passive"], "incomplete": r["incomplete"],
                                         "singularity": r["singularity"]}
                                        for r in analyzed_only
                                        if r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"]
                                    ]
                                    rew_count, dec_count = _ai_batch_rewrite_and_decompose(st.session_state.api_key, flagged_for_action)
                                st.success(f"Rewrote {rewrote} and decomposed {dec_count} requirement(s).")
                    else:
                        st.caption("ℹ️ Enter your Google AI API key to enable bulk AI rewrites/decomposition.")
 
                    # --- AI Contradiction Scan (this document) ---
                    # st.subheader("AI Contradiction Scan — This Document")
                    # if st.button("🔎 Run AI Contradiction Scan (this document)", key=f"ai_contra_doc_btn_{doc_idx}"):
                    #     if not CTX.get("api_key"):
                    #         st.error("AI key missing — paste your Google AI Studio API key in the sidebar.")
                    #     else:
                    #         req_rows_doc = [{"id": r["id"], "text": r["text"], "doc": display_name} for r in analyzed_only]
                    #         if not req_rows_doc:
                    #             st.info("No requirements found to scan.")
                    #         else:
                    #             with st.spinner("Scanning this document for contradictions with AI..."):
                    #                 findings_doc = _force_ai_contra_scan(
                    #                     req_rows_doc, CTX,
                    #                     timeout_s=max(CTX.get("AI_CONTRA_BUDGET_S", 70), CTX.get("AI_CONTRA_PER_CALL_S", 20))
                    #                 )
                    #             if findings_doc:
                    #                 st.caption(f"Found {len(findings_doc)} contradiction(s):")
                    #                 for f in findings_doc[:300]:
                    #                     st.markdown(
                    #                         f"🚨 **{f['kind'].capitalize()}** — {f['reason']}"
                    #                         f"\n\n• **{f['a_id']}** ({f['a_doc']}): {f['a_text']}"
                    #                         f"\n\n• **{f['b_id']}** ({f['b_doc']}): {f['b_text']}"
                    #                         + (f"\n\n_Scope_: `{f['scope']}`" if f.get('scope') else "")
                    #                     )
                    #             else:
                    #                 st.info("No contradictions detected by AI for this document.")
                    # --- AI Contradiction Scan (this document) — instrumented ---
                    doc_req_rows = [{"id": r["id"], "text": r["text"], "doc": display_name}
                                    for r in analyzed_only]
                    subkey = re.sub(r"[^A-Za-z0-9]+", "_", f"{display_name}_{doc_idx}")
                    st.subheader("AI Contradiction Scan — This Document")
                    ai_temp_doc = 0.1
                    dbg = False
                    if not beginner:
                        with st.expander("Advanced (tuning & debug)", expanded=False):
                            _ai_smoke_check(label_key=f"ai_smoke_doc_{subkey}")
                            sensitivity_doc = st.select_slider(
                                "Contradiction sensitivity",
                                options=["Conservative", "Balanced", "Aggressive"],
                                value="Balanced",
                                key=f"contra_sensitivity_{subkey}",
                                help="Adjust model strictness for contradiction detection.",
                            )
                            ai_temp_doc = {"Conservative": 0.0, "Balanced": 0.1, "Aggressive": 0.3}[sensitivity_doc]
                            dbg = st.toggle("Show debug (prompts & raw model output)", key=f"ai_contra_dbg_{subkey}", value=False)
                    if st.button("🔎 Run AI Contradiction Scan", key=f"ai_contra_doc_btn_{subkey}"):
                        if len(doc_req_rows) < 2:
                            st.info("Need at least two requirements in this document to check contradictions.")
                        else:
                            # 1) Deterministic pass
                            det_doc = _deterministic_contra_scan(doc_req_rows)
                            # 2) Optional AI pass
                            ai_doc = []
                            if _has_api_key():
                                # Optional prompt preview (debug)
                                sys_brief = CTX.get("AI_CONTRA_SYSTEM_PROMPT", "").strip()
                                header = (
                                    "You are checking a set of natural-language requirements for contradictions ONLY.\n"
                                    "Return STRICT JSON (no markdown, no commentary): a JSON array of objects with EXACT keys:\n"
                                    "[{\n"
                                    '  "kind":"policy_conflict|timing_conflict|accuracy_vs_latency|safety_vs_speed|scope_conflict|resource_conflict|other",\n'
                                    '  "reason":"concise explanation",\n'
                                    '  "a_id":"R-###","a_doc":"DocName","a_text":"full text",\n'
                                    '  "b_id":"R-###","b_doc":"DocName","b_text":"full text",\n'
                                    '  "scope": "optional scope string or empty"\n'
                                    "}]\n"
                                    "Flag only truly incompatible pairs. JSON array only.\n\n"
                                    "Requirements:\n"
                                )
                                listing = "\n".join(f"- [{r['id']}] ({r.get('doc','Doc')}): {r['text']}" for r in doc_req_rows)
                                fused_prompt = (sys_brief + "\n\n" + header + listing) if sys_brief else (header + listing)
                                if dbg:
                                    st.caption(f"Prompt size: {len(fused_prompt)} chars • items: {len(doc_req_rows)}")
                                    st.text_area("Prompt (read-only)", fused_prompt, height=180, key=f"ai_contra_prompt_{subkey}")
                                try:
                                    with st.spinner("Scanning for contradictions with AI..."):
                                        ai_doc = _force_ai_contra_scan(
                                            doc_req_rows, CTX,
                                            timeout_s=max(CTX.get("AI_CONTRA_BUDGET_S", 70), CTX.get("AI_CONTRA_PER_CALL_S", 20)),
                                            temperature=ai_temp_doc,
                                        )
                                except Exception as e:
                                    st.error(f"AI contradiction scan crashed: {e}")
                                    ai_doc = []
                            else:
                                st.info("AI key not set — showing deterministic contradictions only.")
                            # 3) Merge & render
                            def _sigd(f):
                                return (min(f["a_id"], f["b_id"]), max(f["a_id"], f["b_id"]), f.get("kind",""), (f.get("reason","") or "")[:160])
                            seen = set(); merged_doc = []
                            for src in (det_doc or []) + (ai_doc or []):
                                s = _sigd(src)
                                if s in seen:
                                    continue
                                seen.add(s); merged_doc.append(src)
                            if merged_doc:
                                st.caption(f"Found {len(merged_doc)} contradiction(s):")
                                for f in merged_doc[:300]:
                                    st.markdown(
                                        f"🚨 **{f['kind'].capitalize()}** — {f['reason']}"
                                        f"\n\n• **{f['a_id']}** ({f['a_doc']}): {f['a_text']}"
                                        f"\n\n• **{f['b_id']}** ({f['b_doc']}): {f['b_text']}"
                                        + (f"\n\n_Scope_: `{f['scope']}`" if f.get('scope') else "")
                                    )
                            else:
                                st.info("No contradictions detected by the deterministic + AI scan.")
                                if dbg:
                                    st.caption("Tip: Enable debug above to inspect prompts and raw model output.")

                    st.subheader("Issues by Type")
                    st.bar_chart(issue_counts)

                    # Debug: show all accepted requirement lines
                    with st.expander("Debug: show all accepted requirement lines"):
                        for rid, rtext in [(r["id"], r["text"]) for r in analyzed_only]:
                            st.markdown(f"- **{rid}**: {rtext}")

                    st.subheader("Detailed Analysis")
                    for r_idx, r in enumerate(analyzed_only):
                        is_flagged = r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"]
                        if is_flagged:
                            with st.container():
                                st.markdown(
                                    format_requirement_with_highlights(r["id"], r["text"], r),
                                    unsafe_allow_html=True,
                                )
                                badges_html = _error_badges(r)
                                if badges_html:
                                    st.markdown(badges_html, unsafe_allow_html=True)
                                if r["ambiguous"]:
                                    st.caption(f"ⓘ **Ambiguity:** {', '.join(r['ambiguous'])}")
                                if r["passive"]:
                                    st.caption(f"ⓘ **Passive Voice:** {', '.join(r['passive'])}")
                                if r["incomplete"]:
                                    st.caption("ⓘ **Incompleteness** detected.")
                                if r["singularity"]:
                                    st.caption(f"ⓘ **Singularity:** {', '.join(r['singularity'])}")

                                # === AI actions (Document view) — mirror Quick-Paste ===
                                has_amb = bool(r.get("ambiguous"))
                                has_pas = bool(r.get("passive"))
                                has_inc = bool(r.get("incomplete"))
                                has_sing = bool(r.get("singularity"))
                                only_sing = has_sing and not (has_amb or has_pas or has_inc)
                                if st.session_state.api_key:
                                    if only_sing:
                                        cols = st.columns(1)
                                        with cols[0]:
                                            if st.button(f"🧩 Decompose [{r['id']}]", key=f"doc_dec_only_{doc_idx}_{r_idx}_{r['id']}"):
                                                try:
                                                    base = st.session_state.get(f"rewritten_cache_{r['id']}", "").strip() or r["text"]
                                                    d = _ai_decompose_children(st.session_state.api_key, r["id"], base)
                                                    if d.strip():
                                                        key = f"decomp_cache_{r['id']}"
                                                        existing = st.session_state.get(key, "").strip()
                                                        st.session_state[key] = ((existing + "\n" + d.strip()).strip()
                                                                                 if existing and d.strip() not in existing else (d.strip() or existing))
                                                    st.info("Decomposition:")
                                                    st.markdown(st.session_state.get(f"decomp_cache_{r['id']}", d))
                                                except Exception as e:
                                                    st.warning(f"AI decomposition failed: {e}")
                                    elif has_sing:
                                        cols = st.columns(3)
                                        with cols[0]:
                                            if st.button(f"⚒️ Fix Clarity [{r['id']}]", key=f"doc_fix_{doc_idx}_{r_idx}_{r['id']}"):
                                                try:
                                                    suggestion = _ai_rewrite_clarity(st.session_state.api_key, r['text'])
                                                    st.session_state[f"rewritten_cache_{r['id']}"] = (suggestion or "").strip()
                                                    st.info("Rewritten:")
                                                    st.markdown(f"> {suggestion}")
                                                except Exception as e:
                                                    st.warning(f"AI rewrite failed: {e}")
                                        with cols[1]:
                                            if st.button(f"🧩 Decompose [{r['id']}]", key=f"doc_dec_{doc_idx}_{r_idx}_{r['id']}"):
                                                try:
                                                    base = st.session_state.get(f"rewritten_cache_{r['id']}", "").strip() or r["text"]
                                                    d = _ai_decompose_children(st.session_state.api_key, r["id"], base)
                                                    if d.strip():
                                                        key = f"decomp_cache_{r['id']}"
                                                        existing = st.session_state.get(key, "").strip()
                                                        st.session_state[key] = ((existing + "\n" + d.strip()).strip()
                                                                                 if existing and d.strip() not in existing else (d.strip() or existing))
                                                    st.info("Decomposition:")
                                                    st.markdown(st.session_state.get(f"decomp_cache_{r['id']}", d))
                                                except Exception as e:
                                                    st.warning(f"AI decomposition failed: {e}")
                                        with cols[2]:
                                            if st.button(f"Auto: Fix → Decompose [{r['id']}]", key=f"doc_pipe_{doc_idx}_{r_idx}_{r['id']}"):
                                                try:
                                                    cleaned = st.session_state.get(f"rewritten_cache_{r['id']}", "").strip() or _ai_rewrite_clarity(st.session_state.api_key, r["text"])
                                                    st.session_state[f"rewritten_cache_{r['id']}"] = cleaned
                                                    d = _ai_decompose_children(st.session_state.api_key, r["id"], cleaned)
                                                    if d.strip():
                                                        key = f"decomp_cache_{r['id']}"
                                                        existing = st.session_state.get(key, "").strip()
                                                        st.session_state[key] = ((existing + "\n" + d.strip()).strip()
                                                                                 if existing and d.strip() not in existing else (d.strip() or existing))
                                                    st.success("Rewritten requirement:")
                                                    st.markdown(f"> {cleaned}")
                                                    if st.session_state.get(f"decomp_cache_{r['id']}", ""):
                                                        st.info("Decomposition:")
                                                        st.markdown(st.session_state[f"decomp_cache_{r['id']}"])
                                                except Exception as e:
                                                    st.warning(f"AI pipeline failed: {e}")
                                    else:
                                        cols = st.columns(1)
                                        with cols[0]:
                                            if st.button(f"⚒️ Fix Clarity [{r['id']}]", key=f"doc_fix_only_{doc_idx}_{r_idx}_{r['id']}"):
                                                try:
                                                    suggestion = _ai_rewrite_clarity(st.session_state.api_key, r['text'])
                                                    st.session_state[f"rewritten_cache_{r['id']}"] = (suggestion or "").strip()
                                                    st.info("Rewritten:")
                                                    st.markdown(f"> {suggestion}")
                                                except Exception as e:
                                                    st.warning(f"AI rewrite failed: {e}")

                                # Show cached results inline (inside flagged container)
                                cached_rw = st.session_state.get(f"rewritten_cache_{r['id']}", "")
                                cached_dc = st.session_state.get(f"decomp_cache_{r['id']}", "")
                                if cached_rw:
                                    st.caption("AI Rewrite (cached):")
                                    st.code(cached_rw)
                                if cached_dc:
                                    st.caption("AI Decomposition (cached):")
                                    st.markdown(cached_dc)
                        else:
                            st.markdown(
                                f'<div style="background-color:#D4EDDA;color:#155724;padding:10px;'
                                f'border-radius:5px;margin-bottom:10px;">✅ <strong>{r["id"]}</strong> {r["text"]}</div>',
                                unsafe_allow_html=True,
                            )
                    # NEW: Not requirements (skipped)
                    nonreq_list_doc = [r for r in results if r.get("non_requirement")]
                    if nonreq_list_doc:
                        st.subheader("🚫 Not requirements (skipped)")
                        for r in nonreq_list_doc:
                            st.markdown(
                                f'<div style="background:#F8D7DA;color:#721C24;padding:10px;border-radius:5px;'
                                f'margin-bottom:10px;">'
                                f'🚫 <strong>{r["id"]}</strong> {r["text"]} — <em>not a proper requirement</em></div>',
                                unsafe_allow_html=True,
                            )

    # -------------------- Cross-document contradiction scan ---------------------
    if all_doc_req_rows:
        st.subheader("AI Contradiction Scan — Across All Analyzed Documents")
        if st.button("🔎 Run Cross-Document Scan", key="ai_contra_all_docs"):
            if not CTX.get("api_key"):
                st.error("AI key missing — paste your Google AI Studio API key in the sidebar.")
            else:
                with st.spinner("Scanning across documents for contradictions..."):
                    findings_all = _force_ai_contra_scan(
                        all_doc_req_rows, CTX,
                        timeout_s=max(CTX.get("AI_CONTRA_BUDGET_S", 70), CTX.get("AI_CONTRA_PER_CALL_S", 20))
                    )
                if findings_all:
                    st.caption(f"Found {len(findings_all)} contradiction(s) across documents:")
                    for f in findings_all[:300]:
                        st.markdown(
                            f"🚨 **{f['kind'].capitalize()}** — {f['reason']}"
                            f"\n\n• **{f['a_id']}** ({f['a_doc']}): {f['a_text']}"
                            f"\n\n• **{f['b_id']}** ({f['b_doc']}): {f['b_text']}"
                            + (f"\n\n_Scope_: `{f['scope']}`" if f.get('scope') else "")
                        )
                else:
                    st.info("No contradictions detected across documents.")

    # -------------- Optional: Quick-Paste + Documents combined scan -------------
    combined_rows = []
    if st.session_state.get("quick_analyzed") and st.session_state.get("quick_results"):
        combined_rows.extend([{"id": r["id"], "text": r["text"], "doc": "QuickPaste"}
                              for r in st.session_state["quick_results"]])
    combined_rows.extend(all_doc_req_rows)

    if combined_rows:
        st.subheader("AI Contradiction Scan — Combined (Quick Paste + Documents)")
        if st.button("🔎 Run Combined Scan", key="ai_contra_combined"):
            if not CTX.get("api_key"):
                st.error("AI key missing — paste your Google AI Studio API key in the sidebar.")
            else:
                with st.spinner("Scanning combined inputs for contradictions..."):
                    findings_combo = _force_ai_contra_scan(
                        combined_rows, CTX,
                        timeout_s=max(CTX.get("AI_CONTRA_BUDGET_S", 70), CTX.get("AI_CONTRA_PER_CALL_S", 20))
                    )
                if findings_combo:
                    st.caption(f"Found {len(findings_combo)} contradiction(s) in combined inputs:")
                    for f in findings_combo[:300]:
                        st.markdown(
                            f"🚨 **{f['kind'].capitalize()}** — {f['reason']}"
                            f"\n\n• **{f['a_id']}** ({f['a_doc']}): {f['a_text']}"
                            f"\n\n• **{f['b_id']}** ({f['b_doc']}): {f['b_text']}"
                            + (f"\n\n_Scope_: `{f['scope']}`" if f.get('scope') else "")
                        )
                else:
                    st.info("No contradictions detected in the combined set.")

    # --- Project library (document versions) ----------------------------------
    st.divider()
    st.header("Documents in this Project")
    try:
        project_id = st.session_state.selected_project[0]
    except Exception:
        project_id = None

    if project_id is not None:
        try:
            _rows = db.get_documents_for_project(project_id)
            if not _rows:
                st.info("No documents found for this project.")
            else:
                for (doc_id, file_name, version, uploaded_at, clarity_score) in _rows:
                    conv_path = os.path.join("data", "projects", str(project_id), "documents",
                                             f"{doc_id}_{CTX['_sanitize_filename'](file_name)}")
                    if os.path.exists(conv_path):
                        rel_path = f"data/projects/{project_id}/documents/{doc_id}_{CTX['_sanitize_filename'](file_name)}"
                        st.markdown(
                            f"- [{file_name} (v{version})]({st.file_uploader.get_file_url(rel_path)}) — Clarity: {clarity_score}/100",
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning(f"Document file not found: {conv_path}")
        except Exception as e:
            st.error(f"Failed to load documents: {e}")
    else:
        st.info("Select a project to view its documents.")



