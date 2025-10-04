# ui/widgets/quick_chat.py
# Inline Quick Chat using Streamlit's chat UI.

from typing import Dict, Any

def render_inline_quick_chat(st, CTX: Dict[str, Any]) -> None:
    get_chatbot_response = CTX.get("get_chatbot_response")
    if get_chatbot_response is None:
        st.info("Chat backend not available.")
        return

    # state
    if "quick_chat_hist" not in st.session_state:
        st.session_state.quick_chat_hist = []  # [{role, content}]

    

    # history
    for m in st.session_state.quick_chat_hist[-20:]:
        role = "user" if m["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(m["content"])

    # chat input
    prompt = st.chat_input("Type your message…")
    if prompt:
        if not st.session_state.get("api_key"):
            with st.chat_message("assistant"):
                st.warning("Enter your Google AI API key at the top to use Quick Chat.")
            return

        # append user message
        st.session_state.quick_chat_hist.append({"role": "user", "content": prompt})

        # concise coach context (no project context here)
        coach = (
               "You are an expert Systems Engineering coach (INCOSE/ISO 29148). "
    "Answer in **short, crisp sentences** (max 3 sentences). "
    "Focus only on the key issue and 1–2 next steps. "
    "Do NOT expand into long explanations unless explicitly asked."
        )

        history = [{"role": "user", "parts": [coach]}]
        for msg in st.session_state.quick_chat_hist:
            history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [msg["content"]],
            })

        # assistant reply
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…"):
                    ans = get_chatbot_response(st.session_state.api_key, history)
            except Exception as e:
                ans = f"(Quick Chat error: {e})"
            st.markdown(ans)
        st.session_state.quick_chat_hist.append({"role": "assistant", "content": ans})

    # clear button BELOW the chat box
    st.write("")  # small spacer
    if st.button("Clear chat", key="qc_clear_btn", use_container_width=True):
        st.session_state.quick_chat_hist = []
        st.rerun()
