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

    # ---- Hardening for flaky AI calls (retry, truncate, swallow 500s) --------
    import time
    _orig_get_ai_suggestion = get_ai_suggestion  # keep original

    def get_ai_suggestion(api_key: str, prompt: str, *, max_chars: int = 6000, retries: int = 2, backoff_base: float = 0.6):
        """
        Wrapper around CTX['get_ai_suggestion'] that:
          - truncates overly long prompts (helps avoid server 500s)
          - retries on transient 5xx/429
          - never raises (returns "" on failure)
        """
        if not api_key:
            return ""
        safe_prompt = (prompt or "")[:max_chars]
        last_err = None
        for attempt in range(retries + 1):
            try:
                out = _orig_get_ai_suggestion(api_key, safe_prompt)
                return (out or "").strip()
            except Exception as e:
                msg = str(e)
                last_err = e
                if any(code in msg for code in (" 500", " 502", " 503", " 504", " 429")):
                    time.sleep(backoff_base * (2 ** attempt))
                    continue
                return ""
        return ""

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
                rtxt = (rtxt or "").strip()
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

    # ---- AI prompts (rewrite + decompose) ------------------------------------
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
            # common weak “enable/support/provide/allow/configuration” pattern
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
                amb = safe_call_ambiguity(rtx, rule_engine)
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
        if st.session_state.api_key:
            cols_bulk = st.columns([1.3, 1.9])
            with cols_bulk[0]:
                if st.button("✨ Rewrite all flagged (AI)", key="quick_rewrite_all"):
                    with st.spinner("AI rewriting all flagged requirements..."):
                        flagged_list_for_rewrite = [
                            r for r in quick_results
                            if r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"]
                        ]
                        rew_count = _ai_batch_rewrite(st.session_state.api_key, flagged_list_for_rewrite)
                    st.success(f"Rewrote {rew_count} requirement(s).")
            with cols_bulk[1]:
                if st.button("⚡ Rewrite & Decompose all flagged (AI)", key="quick_rewrite_decomp_all"):
                    with st.spinner("AI rewriting and decomposing flagged requirements..."):
                        flagged_list_for_action = [
                            r for r in quick_results
                            if r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"]
                        ]
                        rew_count, dec_count = _ai_batch_rewrite_and_decompose(
                            st.session_state.api_key, flagged_list_for_action
                        )
                    st.success(f"Rewrote {rew_count} and decomposed {dec_count} requirement(s).")
        else:
            st.caption("ℹ️ Enter your Google AI API key to enable bulk AI rewrites/decomposition.")

        flagged_list = [r for r in quick_results if r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"]]
        clear_list = [r for r in quick_results if not (r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"])]

        st.subheader("Flagged")
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
                        # ONLY Not Singular -> show only Decompose
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
                        # Not Singular + other issues -> Fix, Decompose, Auto
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
                        # No Not-Singular, but has other issues -> show only Fix Clarity
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

                # show cached results inline
                cached_rw = st.session_state.get(f"rewritten_cache_{r['id']}", "")
                cached_dc = st.session_state.get(f"decomp_cache_{r['id']}", "")
                if cached_rw:
                    st.caption("AI Rewrite (cached):")
                    st.code(cached_rw)
                if cached_dc:
                    st.caption("AI Decomposition (cached):")
                    st.markdown(cached_dc)

        st.subheader("Clear")
        for r in clear_list:
            st.markdown(
                f'<div style="background-color:#D4EDDA;color:#155724;padding:10px;'
                f'border-radius:5px;margin-bottom:10px;">✅ <strong>{r["id"]}</strong> {r["text"]}</div>',
                unsafe_allow_html=True,
            )

        # --- Single CSV export: corrected + decomposed (children as rows) -----
        expanded_rows = []
        for r in quick_results:
            issues = []
            if r["ambiguous"]:
                issues.append(f"Ambiguity: {', '.join(r['ambiguous'])}")
            if r["passive"]:
                issues.append(f"Passive Voice: {', '.join(r['passive'])}")
            if r["incomplete"]:
                issues.append("Incompleteness")
            if r["singularity"]:
                issues.append(f"Singularity: {', '.join(r['singularity'])}")
            ai_rew = st.session_state.get(f"rewritten_cache_{r['id']}", "")
            ai_dec = st.session_state.get(f"decomp_cache_{r['id']}", "")

            expanded_rows.append({
                "Requirement ID": r["id"],
                "Requirement Text": r["text"],
                "Status": "Clear" if not issues else "Flagged",
                "Issues Found": "; ".join(issues),
                "AI Rewrite": ai_rew,
                "AI Decomposition": ai_dec,
                "Is Decomposed Child": "No",
                "Parent ID": "",
            })

            try:
                for child_id, child_text in _extract_child_rows_from_decomp(r["id"], ai_dec):
                    expanded_rows.append({
                        "Requirement ID": child_id,
                        "Requirement Text": child_text,
                        "Status": "Decomposed Child",
                        "Issues Found": "",
                        "AI Rewrite": "",
                        "AI Decomposition": "",
                        "Is Decomposed Child": "Yes",
                        "Parent ID": r["id"],
                    })
            except Exception:
                pass

        df_quick_expanded = pd.DataFrame(expanded_rows)
        st.download_button(
            "Download Analysis (CSV)",
            data=df_quick_expanded.to_csv(index=False).encode("utf-8-sig"),
            file_name="ReqCheck_Analysis.csv",
            mime="text/csv",
            key="dl_quick_csv_expanded_single"
        )

    # ===================== Full Document Analyzer =============================
    st.subheader("📁 Upload Documents — analyze one or more files")
    use_ai_parser = st.toggle("Use Advanced AI Parser (requires API key)")
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

    if docs_to_process:
        with st.spinner("Processing and analyzing documents..."):
            saved_count = 0
            for src_type, display_name, payload in docs_to_process:
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

                #           Merge + dedupe => correct counts, no spurious inflation
                reqs = _merge_unique_reqs(table_reqs, std_reqs, ai_reqs)
                total_reqs = len(reqs)
                if total_reqs == 0:
                    st.warning(f"⚠️ No recognizable requirements in **{display_name}**.")
                    continue

                # --- Analyze requirements --------------------------------------------
                results = []
                issue_counts = {"Ambiguity": 0, "Passive Voice": 0, "Incompleteness": 0, "Singularity": 0}

                for rid, rtext in reqs:
                    ambiguous = safe_call_ambiguity(rtext, rule_engine)
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

                flagged_total = sum(
                    1 for r in results
                    if r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"]
                )
                clarity_score = int(((total_reqs - flagged_total) / total_reqs) * 100) if total_reqs else 100

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
                    flagged_total = sum(
                        1 for r in results
                        if r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"]
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Requirements", total_reqs)
                    c2.metric("Flagged", flagged_total)
                    c3.metric("Clarity Score", f"{clarity_score} / 100")
                    st.progress(clarity_score)

                    # bulk buttons (Per-document)
                    flagged_for_doc = [
                        r for r in results
                        if r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"]
                    ]
                    if st.session_state.api_key:
                        cols_bulk_doc = st.columns([1.7, 2.1])
                        with cols_bulk_doc[0]:
                            if st.button(f"✨ Rewrite all flagged in '{display_name}'", key=f"doc_rewrite_all_{display_name}"):
                                with st.spinner(f"AI rewriting flagged requirements in '{display_name}'..."):
                                    rew_count = _ai_batch_rewrite(st.session_state.api_key, flagged_for_doc)
                                st.success(f"Rewrote {rew_count} requirement(s).")
                        with cols_bulk_doc[1]:
                            if st.button(f"⚡ Rewrite & Decompose all flagged in '{display_name}'", key=f"doc_rewrite_decomp_all_{display_name}"):
                                with st.spinner(f"AI rewriting and decomposing flagged requirements in '{display_name}'..."):
                                    rew_count, dec_count = _ai_batch_rewrite_and_decompose(
                                        st.session_state.api_key, flagged_for_doc
                                    )
                                st.success(f"Rewrote {rew_count} and decomposed {dec_count} requirement(s).")
                    else:
                        st.caption("ℹ️ Enter your Google AI API key to enable bulk AI rewrites/decomposition for this document.")

                    # --- Single CSV export per document (parent+children) --------------
                    expanded_rows_doc = []
                    for r in results:
                        issues = []
                        if r["ambiguous"]:
                            issues.append(f"Ambiguity: {', '.join(r['ambiguous'])}")
                        if r["passive"]:
                            issues.append(f"Passive Voice: {', '.join(r['passive'])}")
                        if r["incomplete"]:
                            issues.append("Incompleteness")
                        if r["singularity"]:
                            issues.append(f"Singularity: {', '.join(r['singularity'])}")

                        ai_rew = st.session_state.get(f"rewritten_cache_{r['id']}", "")
                        ai_dec = st.session_state.get(f"decomp_cache_{r['id']}", "")

                        expanded_rows_doc.append({
                            "Document": display_name,
                            "Requirement ID": r["id"],
                            "Requirement Text": r["text"],
                            "Status": "Clear" if not issues else "Flagged",
                            "Issues Found": "; ".join(issues),
                            "AI Rewrite": ai_rew,
                            "AI Decomposition": ai_dec,
                            "Is Decomposed Child": "No",
                            "Parent ID": "",
                        })

                        try:
                            for child_id, child_text in _extract_child_rows_from_decomp(r["id"], ai_dec):
                                expanded_rows_doc.append({
                                    "Document": display_name,
                                    "Requirement ID": child_id,
                                    "Requirement Text": child_text,
                                    "Status": "Decomposed Child",
                                    "Issues Found": "",
                                    "AI Rewrite": "",
                                    "AI Decomposition": "",
                                    "Is Decomposed Child": "Yes",
                                    "Parent ID": r["id"],
                                })
                        except Exception:
                            pass

                    df_doc_expanded = pd.DataFrame(expanded_rows_doc)
                    st.download_button(
                        label=f"Download '{display_name}' Analysis (CSV)",
                        data=df_doc_expanded.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"{os.path.splitext(display_name)[0]}_ReqCheck_Report.csv",
                        mime="text/csv",
                        key=f"dl_csv_expanded_{display_name}",
                    )

                    st.subheader("Issues by Type")
                    st.bar_chart(issue_counts)

                    # Word cloud of ambiguous terms
                    all_ambiguous_words = []
                    for r in results:
                        if r["ambiguous"]:
                            all_ambiguous_words.extend(r["ambiguous"])
                    with st.expander("Common Weak Words (Word Cloud)"):
                        if all_ambiguous_words:
                            text_for_cloud = " ".join(all_ambiguous_words)
                            wordcloud = WordCloud(
                                width=800, height=300, background_color="white", collocations=False
                            ).generate(text_for_cloud)
                            fig, ax = plt.subplots()
                            ax.imshow(wordcloud, interpolation="bilinear")
                            ax.axis("off")
                            st.pyplot(fig)
                        else:
                            st.write("No ambiguous words found.")

                    st.subheader("Detailed Analysis")
                    for r in results:
                        is_flagged = r["ambiguous"] or r["passive"] or r["incomplete"] or r["singularity"]
                        if is_flagged:
                            with st.container(border=True):
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

                                # Action buttons with strict gating
                                has_amb = bool(r.get("ambiguous"))
                                has_pas = bool(r.get("passive"))
                                has_inc = bool(r.get("incomplete"))
                                has_sing = bool(r.get("singularity"))
                                only_sing = has_sing and not (has_amb or has_pas or has_inc)

                                if st.session_state.api_key:
                                    if only_sing:
                                        # ONLY Not Singular -> Decompose only
                                        cols = st.columns(1)
                                        with cols[0]:
                                            if st.button(f"🧩 Decompose [{r['id']}]", key=f"decomp_only_{r['id']}"):
                                                with st.spinner("Decomposing into single-action children..."):
                                                    base = st.session_state.get(f"rewritten_cache_{r['id']}", "").strip() or _ai_rewrite_clarity_doc(st.session_state.api_key, r["text"])
                                                    decomp = _ai_decompose_clean_doc(st.session_state.api_key, r["id"], base)
                                                    if decomp.strip():
                                                        key = f"decomp_cache_{r['id']}"
                                                        existing = st.session_state.get(key, "").strip()
                                                        st.session_state[key] = ((existing + "\n" + decomp.strip()).strip()
                                                                                 if existing and decomp.strip() not in existing else (decomp.strip() or existing))
                                                st.info("Decomposition:")
                                                st.markdown(st.session_state.get(f"decomp_cache_{r['id']}", decomp))
                                    elif has_sing:
                                        # Not Singular + other issues -> Fix, Decompose, Auto
                                        cols = st.columns(3)
                                        with cols[0]:
                                            if st.button(f"⚒️ Fix Clarity (Rewrite) [{r['id']}]", key=f"fix_{r['id']}"):
                                                with st.spinner("Rewriting to remove ambiguity/passive/incompleteness..."):
                                                    cleaned = _ai_rewrite_clarity_doc(st.session_state.api_key, r["text"])
                                                    st.session_state[f"rewritten_cache_{r['id']}"] = cleaned
                                                if cleaned:
                                                    st.success("Rewritten draft:")
                                                    st.markdown(f"> {cleaned}")
                                        with cols[1]:
                                            if st.button(f"🧩 Decompose [{r['id']}]", key=f"decomp_{r['id']}"):
                                                with st.spinner("Decomposing into single-action children..."):
                                                    base = st.session_state.get(f"rewritten_cache_{r['id']}", "").strip() or _ai_rewrite_clarity_doc(st.session_state.api_key, r["text"])
                                                    decomp = _ai_decompose_clean_doc(st.session_state.api_key, r["id"], base)
                                                    if decomp.strip():
                                                        key = f"decomp_cache_{r['id']}"
                                                        existing = st.session_state.get(key, "").strip()
                                                        st.session_state[key] = ((existing + "\n" + decomp.strip()).strip()
                                                                                 if existing and decomp.strip() not in existing else (decomp.strip() or existing))
                                                st.info("Decomposition:")
                                                st.markdown(st.session_state.get(f"decomp_cache_{r['id']}", decomp))
                                        with cols[2]:
                                            if st.button(f"Auto: Fix → Decompose [{r['id']}]", key=f"pipeline_{r['id']}"):
                                                with st.spinner("Rewriting, then decomposing..."):
                                                    cleaned = st.session_state.get(f"rewritten_cache_{r['id']}", "").strip() or _ai_rewrite_clarity_doc(st.session_state.api_key, r["text"])
                                                    st.session_state[f"rewritten_cache_{r['id']}"] = cleaned
                                                    decomp = _ai_decompose_clean_doc(st.session_state.api_key, r["id"], cleaned)
                                                    if decomp.strip():
                                                        key = f"decomp_cache_{r['id']}"
                                                        existing = st.session_state.get(key, "").strip()
                                                        st.session_state[key] = ((existing + "\n" + decomp.strip()).strip()
                                                                                 if existing and decomp.strip() not in existing else (decomp.strip() or existing))
                                                if cleaned:
                                                    st.success("Rewritten requirement:")
                                                    st.markdown(f"> {cleaned}")
                                                if st.session_state.get(f"decomp_cache_{r['id']}", ""):
                                                    st.info("Decomposition:")
                                                    st.markdown(st.session_state[f"decomp_cache_{r['id']}"])
                                    else:
                                        # No Not-Singular, but has other issues -> Fix only
                                        cols = st.columns(1)
                                        with cols[0]:
                                            if st.button(f"⚒️ Fix Clarity (Rewrite) [{r['id']}]", key=f"fix_only_{r['id']}"):
                                                with st.spinner("Rewriting to remove ambiguity/passive/incompleteness..."):
                                                    cleaned = _ai_rewrite_clarity_doc(st.session_state.api_key, r["text"])
                                                    st.session_state[f"rewritten_cache_{r['id']}"] = cleaned
                                                if cleaned:
                                                    st.success("Rewritten draft:")
                                                    st.markdown(f"> {cleaned}")

                                # show cached (if any)
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

            st.success("Analysis complete.")

    # --- Project library (document versions) ----------------------------------
    st.divider()
    st.header("Documents in this Project")

    if st.session_state.selected_project is None:
        st.info("Select a project to view its saved documents.")
    else:
        pid = st.session_state.selected_project[0]
        try:
            if hasattr(db, "get_documents_for_project"):
                rows = db.get_documents_for_project(pid)  # (doc_id, file_name, version, uploaded_at, clarity_score)
                if not rows:
                    st.info("No documents have been added to this project yet.")
                else:
                    # One row per version
                    doc_data = []
                    for (doc_id, file_name, version, uploaded_at, clarity_score) in rows:
                        doc_data.append({
                            "File Name": file_name,
                            "Version": version,
                            "Uploaded On": uploaded_at.replace("T", " ")[:19] if isinstance(uploaded_at, str) else uploaded_at,
                            "Clarity Score": f"{clarity_score} / 100" if clarity_score is not None else "—",
                        })
                    df_docs = pd.DataFrame(doc_data).sort_values(["File Name", "Version"], ascending=[True, False])
                    st.dataframe(df_docs, use_container_width=True)

                    # Export summary
                    proj_csv = df_docs.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="Download Project Documents Summary (CSV)",
                        data=proj_csv,
                        file_name=f"Project_{pid}_Documents_Summary.csv",
                        mime="text/csv",
                        key="dl_csv_project_docs",
                    )
            else:
                st.info("get_documents_for_project() not found in db.database.")
        except Exception as e:
            st.error(f"Failed to load documents for this project: {e}")
