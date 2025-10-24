# ui/tabs/chat_tab.py
from __future__ import annotations
from typing import List, Dict, Any
import json
from datetime import datetime
import re
import streamlit as st

# ---- Config ----
HISTORY_TURNS = 10          # last N message pairs to keep context lean
MAX_CTX_LINES = 200         # limit project-context bleed
SYS_PROMPT = (
    "You are ReqCheck AI — a professional Systems Engineering assistant following INCOSE/ISO 29148. "
    "Your answers must be short, structured, and readable. "
    "When giving requirements, use a compact layout with bold field labels (ID, Statement, Rationale, Verification). "
    "Never exceed 6 lines. No long explanations unless asked."
)
COACH_JSON_INSTRUCTIONS = """
Return ONLY valid JSON with keys:
- "reply": short structured markdown string (under 6 lines)
- "follow_up": one concise question
- "quick_replies": 2–4 short strings (next suggestions)
Keep answers short and readable.
"""

def render(st, db, rule_engine, CTX):
    get_chatbot_response = CTX["get_chatbot_response"]

    # ---- Header ----
    pname = st.session_state.selected_project[1] if st.session_state.get("selected_project") else None
    st.header("💬 AI Chat" + (f" — Project: {pname}" if pname else ""))

    # ---- State ----
    st.session_state.setdefault("messages", [])           # [{role:"user"|"assistant", content:str}]
    st.session_state.setdefault("attach_ctx", True)
    st.session_state.setdefault("last_bot_json", {})      # last parsed JSON (for quick replies)

    # ---- Controls ----
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_bot_json = {}
            st.rerun()
    with c2:
        st.session_state.attach_ctx = st.toggle(
            "Attach project context",
            value=st.session_state.attach_ctx,
            help="Include recent requirements from the selected project."
        )

    # ---- Helpers ----
    def _safe_project_context() -> str:
        if not st.session_state.get("selected_project"):
            return ""
        try:
            pid = st.session_state.selected_project[0]
            lines: List[str] = []
            if hasattr(db, "get_requirements_for_project"):
                for r in db.get_requirements_for_project(pid)[:MAX_CTX_LINES]:
                    rid, text = (r[0], r[1]) if len(r) >= 2 else (str(r[0]), str(r[0]))
                    lines.append(f"{rid}: {text}")
            elif hasattr(db, "get_documents_for_project") and hasattr(db, "get_requirements_for_document"):
                docs = db.get_documents_for_project(pid)
                for (doc_id, *_rest) in docs[:3]:
                    try:
                        for rr in db.get_requirements_for_document(doc_id)[:80]:
                            rid, text = (rr[0], rr[1]) if len(rr) >= 2 else (str(rr[0]), str(rr[0]))
                            lines.append(f"{rid}: {text}")
                    except Exception:
                        continue
            return "PROJECT CONTEXT:\n" + "\n".join(lines[:MAX_CTX_LINES]) if lines else ""
        except Exception:
            return ""

    def _to_history(messages: List[Dict[str, str]], sys_preface: str, proj_ctx: str) -> List[Dict[str, Any]]:
        hist: List[Dict[str, Any]] = []
        preface = sys_preface + ("\n\n" + proj_ctx if proj_ctx else "") + "\n\n" + COACH_JSON_INSTRUCTIONS
        hist.append({"role": "user", "parts": [preface]})
        trimmed = messages[-(HISTORY_TURNS * 2):]
        for m in trimmed:
            hist.append({"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]})
        return hist

    # --- sanitize helpers -----------------------------------------------------
    CODE_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*([\s\S]+?)\s*```\s*$")

    def _strip_code_fences(text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` fences if present."""
        m = CODE_FENCE_RE.match(text.strip())
        return m.group(1) if m else text

    def _parse_bot_json(text: str) -> Dict[str, Any]:
        """Parse model output into {reply, follow_up, quick_replies}, forgivingly."""
        s = _strip_code_fences(text or "").strip()
        # If a JSON object seems embedded, try to isolate it
        if "{" in s and "}" in s:
            try:
                s = s[s.index("{"): s.rindex("}") + 1]
            except Exception:
                pass
        try:
            data = json.loads(s)
            reply = str(data.get("reply", "")).strip()
            follow = str(data.get("follow_up", "")).strip()
            qrs = data.get("quick_replies", [])
            if not isinstance(qrs, list):
                qrs = []
            qrs = [str(x).strip() for x in qrs if str(x).strip()]
            return {"reply": reply or "(No reply)", "follow_up": follow, "quick_replies": qrs[:4]}
        except Exception:
            # Fallback: treat whole thing as plain text
            return {"reply": s or "(No reply)", "follow_up": "", "quick_replies": []}

    # --- compacting helper ----------------------------------------------------
    def _short_format_requirement(text: str) -> str:
        """
        Compress long requirement-style text into <= 5 lines focusing on:
        ID, Statement, Rationale, Verification. Truncates long free text.
        """
        field_map = {
            "ID": "🆔 **ID:**",
            "Statement": "📝 **Statement:**",
            "Rationale": "💡 **Rationale:**",
            "Verification": "✅ **Verification:**",
            "Verification Method": "✅ **Verification:**",
        }
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        formatted: List[str] = []
        for line in lines:
            matched = False
            for key, label in field_map.items():
                if re.match(fr"^{key}\s*[:\-]", line, re.IGNORECASE):
                    value = re.sub(fr"^{key}\s*[:\-]\s*", "", line, flags=re.IGNORECASE)
                    formatted.append(f"{label} {value}")
                    matched = True
                    break
            if not matched:
                # keep brief free text (truncate if too long)
                if len(line) > 150:
                    line = line[:147] + "..."
                formatted.append(line)
        # Only keep first 5 lines to stay compact
        return "\n".join(formatted[:5])

    def _call_ai(api_key: str, history: List[Dict[str, Any]]) -> str:
        try:
            return get_chatbot_response(api_key, history)
        except TypeError:
            return get_chatbot_response(api_key, history)

    # ---- Transcript (render) ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if msg["role"] == "assistant":
                # If assistant message contains JSON-like content, parse then compact-render
                if content.strip().startswith("{") and '"reply"' in content:
                    data = _parse_bot_json(content)
                    st.markdown(_short_format_requirement(data["reply"]))
                    if data.get("follow_up"):
                        st.markdown(f"**Quick question:** {data['follow_up']}")
                else:
                    st.markdown(_short_format_requirement(content))
            else:
                st.markdown(content)

    # ---- Quick replies (buttons under the chat) ----
    def _inject_user_message(txt: str):
        st.session_state.messages.append({"role": "user", "content": txt})
        st.rerun()

    if st.session_state.last_bot_json.get("quick_replies"):
        with st.container():
            bcols = st.columns(len(st.session_state.last_bot_json["quick_replies"]))
            for i, qr in enumerate(st.session_state.last_bot_json["quick_replies"]):
                with bcols[i]:
                    if st.button(qr, key=f"qr_{len(st.session_state.messages)}_{i}"):
                        _inject_user_message(qr)

    # ---- Chat input (unique key!) ----
    user_input = st.chat_input("Type your message…", key="chat_input_main")

    if user_input:
        if not st.session_state.get("api_key"):
            st.warning("⚠️ Please enter your Google AI API Key at the top.")
            return

        # 1) Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2) History
        proj_ctx = _safe_project_context() if st.session_state.attach_ctx else ""
        history = _to_history(st.session_state.messages, SYS_PROMPT, proj_ctx)

        # 3) Call AI
        with st.spinner("🤖 Thinking…"):
            try:
                raw = _call_ai(st.session_state.api_key, history)
            except Exception as e:
                raw = f'{{"reply":"AI error: {e}","follow_up":"","quick_replies":[]}}'

        # 4) Parse & store
        data = _parse_bot_json(raw)
        st.session_state.last_bot_json = data

        # 5) Main reply (compact formatting)
        short_reply = _short_format_requirement(data["reply"])
        st.session_state.messages.append({"role": "assistant", "content": short_reply})
        with st.chat_message("assistant"):
            st.markdown(short_reply)

        # 6) Follow-up question (separate bubble)
        if data["follow_up"]:
            st.session_state.messages.append({"role": "assistant", "content": f"**Quick question:** {data['follow_up']}"})
            with st.chat_message("assistant"):
                st.markdown(f"**Quick question:** {data['follow_up']}")

        # 7) Quick reply chips
        if data["quick_replies"]:
            chip_cols = st.columns(len(data["quick_replies"]))
            for i, qr in enumerate(data["quick_replies"]):
                with chip_cols[i]:
                    if st.button(qr, key=f"chip_{len(st.session_state.messages)}_{i}"):
                        _inject_user_message(qr)
