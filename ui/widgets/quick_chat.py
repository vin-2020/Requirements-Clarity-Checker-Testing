# ui/widgets/quick_chat.py
# Inline Quick Chat using Streamlit's chat UI, with sticky composer + quick replies.

from typing import Dict, Any, List
import re
import streamlit as st
import json
import urllib.parse

# --- Chip styles (pill buttons + wrap) ----------------------------------------
st.markdown("""
<style>
.qc-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 4px;
}
.qc-chip-btn button {
    border-radius: 999px !important;  /* pill shape */
    background: rgba(255,255,255,0.08) !important;
    color: #f2f3f5 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: 1.1 !important;
    padding: 6px 14px !important;
    height: auto !important;
    min-height: 0 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    white-space: nowrap !important;
}
.qc-chip-btn button:hover {
    background: linear-gradient(135deg,#8B5CF6,#22D3EE) !important;
    color: white !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# --- Chip styles (anchor pills) ----------------------------------------------
st.markdown("""
<style>
.qc-chips { 
  display:flex; flex-wrap: wrap; gap: 8px; 
  margin: 4px 0 6px 0;
}
.qc-chip {
  display:inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  color: #f2f3f5 !important;
  border: 1px solid rgba(255,255,255,0.15);
  text-decoration: none !important;
  white-space: nowrap;
  font-size: 13px; font-weight: 600; line-height: 1.15;
}
.qc-chip:hover {
  background: linear-gradient(135deg,#8B5CF6,#22D3EE);
  color: white !important; border-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# --- Small rounded clear button styles ---------------------------------------
st.markdown("""
<style>
.qc-clear-btn button {
    background: rgba(255,255,255,0.08);
    color: #f2f3f5;
    font-size: 12px;
    border-radius: 999px;
    padding: 4px 12px;
    border: 1px solid rgba(255,255,255,0.15);
    width: auto !important;
    min-width: 80px;
}
.qc-clear-btn button:hover {
    background: linear-gradient(135deg,#8B5CF6,#22D3EE);
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

def _suggest_quick_replies(user_msg: str, assistant_msg: str | None = None) -> List[str]:
    """Lightweight, local heuristics for smart reply chips."""
    txt = (user_msg or "").lower()

    # Identity / about
    if any(k in txt for k in ["who are you", "what are you", "creator", "made you", "built you"]):
        return [
            "What can you do for requirements?",
            "How do you check ambiguity?",
            "Summarize how ReqCheck works",
        ]

    # Requirements help / rewriting
    if any(k in txt for k in ["rewrite", "improve", "clarity", "refactor", "polish"]):
        return [
            "Rewrite this requirement to be unambiguous",
            "Show verification method options (T/I/A/D)",
            "Point out passive voice and fix it",
        ]

    # Analysis / diagnosis
    if any(k in txt for k in ["analyze", "check", "flags", "issues", "quality"]):
        return [
            "List top 3 issues and fixes",
            "Give an INCOSE-style rewrite",
            "Suggest acceptance criteria",
        ]

    # MBSE / systems prompts
    if any(k in txt for k in ["mbse", "sysml", "ibd", "bdd", "state", "requirements flow"]):
        return [
            "How to trace to SysML blocks?",
            "Map to verification cases",
            "Create a concise requirement set",
        ]

    # Default options
    return [
        "Give 3 action items",
        "Explain with a short example",
        "Rewrite more concisely",
    ]


# --- AI quick-replies helper (LLM-backed) -------------------------------------
def _ai_quick_replies(st, CTX: Dict[str, Any], last_user: str, last_assistant: str | None) -> List[str]:
    """Ask the LLM for 3–4 short, button-ready follow-ups tailored to the last exchange."""
    api_key = st.session_state.get("api_key", "")
    if not api_key or not last_user:
        return []

    cache_key = ("qc_suggest", len(st.session_state.quick_chat_hist))
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    run_freeform = CTX.get("run_freeform")
    get_chatbot_response = CTX.get("get_chatbot_response")

    # Tight, deterministic prompt that returns a JSON array of short strings.
    sys_prompt = (
        "You are ReqCheck AI (INCOSE/ISO 29148). Generate 3–4 SHORT follow-up buttons "
        "that would help continue the conversation usefully. Each item must be <= 60 chars, "
        "actionable, and specific to the last user message and your last answer. "
        "Return ONLY a JSON array of strings. No prose."
    )
    user_prompt = json.dumps({
        "last_user": last_user.strip(),
        "last_assistant": (last_assistant or "").strip()
    }, ensure_ascii=False)

    try:
        if callable(run_freeform):
            raw = run_freeform(api_key, f"{sys_prompt}\nINPUT:\n{user_prompt}")
        else:
            history = [
                {"role": "user", "parts": [sys_prompt]},
                {"role": "user", "parts": [f"INPUT:\n{user_prompt}"]},
            ]
            raw = get_chatbot_response(api_key, history)

        raw = (raw or "").strip()
        try:
            arr = json.loads(raw)
            if not isinstance(arr, list):
                raise ValueError("not a list")
        except Exception:
            lines = re.findall(r'^\s*[-*\d.)]\s*(.+)$', raw, flags=re.M)
            arr = [l.strip() for l in lines if l.strip()]
            if not arr:
                arr = [raw[:60]] if raw else []

        cleaned: List[str] = []
        seen = set()
        for s in arr:
            s = re.sub(r'[\s]+', ' ', str(s)).strip()
            if not s or s.lower() in seen:
                continue
            seen.add(s.lower())
            if len(s) > 60:
                s = s[:57] + "..."
            cleaned.append(s)
            if len(cleaned) >= 4:
                break

        st.session_state[cache_key] = cleaned
        return cleaned
    except Exception:
        return []

def render_inline_quick_chat(st, CTX: Dict[str, Any]) -> None:
    get_chatbot_response = CTX.get("get_chatbot_response")
    if get_chatbot_response is None:
        st.info("Chat backend not available.")
        return

    # --- State ---------------------------------------------------------------
    if "quick_chat_hist" not in st.session_state:
        st.session_state.quick_chat_hist = []  # [{role, content}]

    # --- Helpers -------------------------------------------------------------
    def _assistant_reply(user_text: str):
        """Append user text, call model, append assistant text."""
        st.session_state.quick_chat_hist.append({"role": "user", "content": user_text})

        # Identity prompt to keep branding consistent
        coach = (
            "You are ReqCheck AI — an intelligent systems engineering assistant "
            "created by Vinodh Kumar Rajkumar. "
            "You specialize in INCOSE/ISO 29148 requirements analysis, "
            "clarity checking, and systems thinking. "
            "Respond in short, clear, professional sentences (max 3). "
            "If asked who you are, state you are part of the ReqCheck project, "
            "built by Vinodh to help engineers improve requirement quality."
        )

        history = [{"role": "user", "parts": [coach]}]
        for msg in st.session_state.quick_chat_hist:
            history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [msg["content"]],
            })

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…"):
                    ans = get_chatbot_response(st.session_state.api_key, history)
            except Exception as e:
                ans = f"(Quick Chat error: {e})"
            st.markdown(ans)
        st.session_state.quick_chat_hist.append({"role": "assistant", "content": ans})

    def _render_quick_replies(last_user: str, last_assistant: str | None):
        chips = _suggest_quick_replies(last_user, last_assistant)
        if not chips:
            return
        st.write("")  # small spacer
        st.caption("Quick replies")
        cols = st.columns(min(4, len(chips)))
        for i, text in enumerate(chips):
            if cols[i].button(text, key=f"qc_chip_{len(st.session_state.quick_chat_hist)}_{i}"):
                # Send the chip text as user's next message
                _assistant_reply(text)
                st.rerun()

    # --- Handle chip clicks via query params (before rendering history) -------
    qp = st.query_params
    if "qc_chip" in qp and qp["qc_chip"]:
        chip_text = qp["qc_chip"]
        st.query_params.clear()
        if st.session_state.get("api_key"):
            _assistant_reply(chip_text)
        else:
            with st.chat_message("assistant"):
                st.warning("Enter your Google AI API key at the top to use Quick Chat.")
        st.rerun()

    # --- History -------------------------------------------------------------
    for m in st.session_state.quick_chat_hist[-20:]:
        role = "user" if m["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(m["content"])

    # ------- AI-generated quick replies under the last assistant turn -------
    if st.session_state.quick_chat_hist and st.session_state.quick_chat_hist[-1]["role"] == "assistant":
        last_user = ""
        for prev in reversed(st.session_state.quick_chat_hist):
            if prev["role"] == "user":
                last_user = prev["content"]
                break
        last_assistant = st.session_state.quick_chat_hist[-1]["content"]
        chips = _ai_quick_replies(st, CTX, last_user, last_assistant)
        if chips:
            st.caption("Quick replies")
            # Render as pill links with query param ?qc_chip=...
            html_parts = ['<div class="qc-chips">']
            for text in chips:
                href = "?qc_chip=" + urllib.parse.quote_plus(text)
                html_parts.append(f'<a class="qc-chip" href="{href}">{text}</a>')
            html_parts.append("</div>")
            st.markdown("".join(html_parts), unsafe_allow_html=True)

    # ====================== NEW: chat_input composer ======================
    prompt = st.chat_input("Type your message… (Press Enter to send, Shift+Enter for newline)")
    if prompt is not None:
        msg = prompt.strip()
        if msg:
            if not st.session_state.get("api_key"):
                with st.chat_message("assistant"):
                    st.warning("Enter your Google AI API key at the top to use Quick Chat.")
            else:
                _assistant_reply(msg)
                st.rerun()  # ensures the field is blank and the reply appears

    # --- Clear (compact rounded button) --------------------------------------
    st.write("")  # spacer
    st.markdown("<div class='qc-clear-btn'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear chat", key="qc_clear_btn_small"):
        st.session_state.quick_chat_hist = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
