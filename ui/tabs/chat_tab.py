# ui/tabs/chat_tab.py
import re

def render(st, db, rule_engine, CTX):
    get_chatbot_response = CTX["get_chatbot_response"]
    get_ai_suggestion = CTX["get_ai_suggestion"]
    decompose_requirement_with_ai = CTX["decompose_requirement_with_ai"]

    pname = st.session_state.selected_project[1] if st.session_state.selected_project else None
    st.header("Chat with an AI Systems Engineering Coach" + (f" — Project: {pname}" if pname else ""))

    # --- state
    if "messages" not in st.session_state:
        st.session_state.messages = []   # [{role: user|assistant, content: str}]
    if "chat_temp" not in st.session_state:
        st.session_state.chat_temp = 0.2
    if "attach_ctx" not in st.session_state:
        st.session_state.attach_ctx = True

    # --- controls
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("Clear chat", key="chat_clear"):
            st.session_state.messages = []
            st.rerun()
    with c2:
        st.session_state.chat_temp = st.slider(
            "Creativity", 0.0, 0.9, st.session_state.chat_temp, 0.1,
            help="Model temperature (lower = more deterministic)"
        )
    with c3:
        st.session_state.attach_ctx = st.toggle(
            "Attach project context", value=st.session_state.attach_ctx,
            help="Adds recent project requirements to ground the conversation"
        )

   

    # --- render transcript
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- helpers
    SE_SYSTEM_PROMPT = (
        "You are an expert Systems Engineering COACH. Follow INCOSE/ISO 29148. "
        "Your primary role is critique + guidance: identify ambiguity, passive voice, incompleteness, and non-singularity. "
        "Prefer EARS patterns; require measurable thresholds; propose verification guidance (V&V). "
        "ALWAYS ask clarifying questions when needed and provide alternatives/patterns. "
        "Only provide a full one-sentence requirement when explicitly asked (e.g., '/rewrite'). "
        "When giving a rewrite, label it clearly as 'Example (for consideration)' and keep it to ONE sentence."
    )

    def _safe_project_context() -> str:
        if not st.session_state.selected_project:
            return ""
        try:
            pid = st.session_state.selected_project[0]
            lines = []
            if hasattr(db, "get_requirements_for_project"):
                rows = db.get_requirements_for_project(pid)
                for r in rows[:200]:
                    rid, rtx = (r[0], r[1]) if len(r) >= 2 else (str(r[0]), str(r[0]))
                    lines.append(f"{rid}: {rtx}")
            elif hasattr(db, "get_documents_for_project") and hasattr(db, "get_requirements_for_document"):
                docs = db.get_documents_for_project(pid)
                docs = sorted(docs, key=lambda x: x[2], reverse=True)[:3]
                for (doc_id, file_name, version, uploaded_at, clarity_score) in docs:
                    try:
                        reqs = db.get_requirements_for_document(doc_id)
                        for rr in reqs[:80]:
                            rid, rtx = (rr[0], rr[1]) if len(rr) >= 2 else (str(rr[0]), str(rr[0]))
                            lines.append(f"{rid}: {rtx}")
                    except Exception:
                        continue
            if not lines:
                return ""
            return "PROJECT CONTEXT:\n" + "\n".join(lines[:200])
        except Exception:
            return ""

    def _to_gemini_history(messages: list[dict], system_prompt: str = "", project_ctx: str = "") -> list[dict]:
        hist = []
        preface_parts = []
        if system_prompt.strip():
            preface_parts.append(system_prompt.strip())
        if project_ctx.strip():
            preface_parts.append(project_ctx.strip())
        if preface_parts:
            hist.append({"role": "user", "parts": ["\n\n".join(preface_parts)]})
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not content:
                continue
            if role == "user":
                hist.append({"role": "user", "parts": [content]})
            else:
                hist.append({"role": "model", "parts": [content]})
        return hist

    def _analyze(text: str) -> str:
        amb = CTX["safe_call_ambiguity"](text, rule_engine)
        pas = CTX["check_passive_voice"](text)
        inc = CTX["check_incompleteness"](text)
        try:
            sing = CTX["check_singularity"](text)
        except Exception:
            sing = []
        issues = []
        if amb: issues.append(f"Ambiguity: {', '.join(amb)}")
        if pas: issues.append(f"Passive: {', '.join(pas)}")
        if inc: issues.append("Incompleteness")
        if sing: issues.append(f"Multiple actions: {', '.join(sing)}")
        tips = []
        if amb: tips.append("Replace weak/vague terms with measurable thresholds.")
        if pas: tips.append("Switch to active voice with an explicit actor.")
        if inc: tips.append("Ensure a complete, testable statement with conditions and outcome.")
        if sing: tips.append("Split multiple actions into separate shall-statements.")
        out = "Analysis:\n- " + ("\n- ".join(issues) if issues else "No major issues found.")
        if tips: out += "\n\nCoach Tips:\n- " + "\n- ".join(tips)
        return out

    def _rewrite_example(text: str) -> str:
        prompt = f"""
You are a senior systems engineering COACH.
Rewrite the requirement as ONE sentence using 'shall', active voice, unambiguous and testable.
Keep intent; do not add scope. Return ONLY the sentence.

Requirement:
\"\"\"{(text or '').strip()}\"\"\""""
        try:
            suggestion = get_ai_suggestion(st.session_state.api_key, prompt) or ""
        except Exception as e:
            return f"(AI rewrite unavailable: {e})"
        for ln in suggestion.splitlines():
            ln = ln.strip()
            if ln:
                return ln
        return suggestion.strip()

    def _ac_suggest(text: str, method: str) -> str:
        prompt_ac = f"""Return ONLY 3–6 bullets. Each bullet must include: setup/conditions + numeric threshold + verification method.
Requirement:
\"\"\"{text.strip()}\"\"\" 
Verification Method: {method or 'Test'}"""
        try:
            return get_ai_suggestion(st.session_state.api_key, prompt_ac)
        except Exception as e:
            return f"(AI AC unavailable: {e})"

    def _decompose(text: str) -> str:
        try:
            return decompose_requirement_with_ai(st.session_state.api_key, text)
        except Exception as e:
            return f"(AI decomposition unavailable: {e})"

    def _handle_slash(prompt: str):
        p = prompt.strip()
        if p == "/help":
            return (
                "Commands:\n"
                "- `/analyze <req>` → quality checks + coach tips\n"
                "- `/rewrite <req>` → show a **sample** one-sentence rewrite (on demand)\n"
                "- `/ac <req> method=Test|Analysis|Inspection|Demonstration` → suggest AC bullets\n"
                "- `/decompose <req>` → split into singular requirements"
            )
        if p.startswith("/analyze "):
            return _analyze(p[len("/analyze "):])
        if p.startswith("/rewrite "):
            t = p[len("/rewrite "):]
            ex = _rewrite_example(t)
            return f"**Example rewrite (for consideration):**\n> {ex}"
        if p.startswith("/ac "):
            body = p[len("/ac "):]
            method = None
            if " method=" in body:
                body, method = body.split(" method=", 1)
            return _ac_suggest(body, (method or "Test").strip())
        if p.startswith("/decompose "):
            return _decompose(p[len("/decompose "):])
        return None

    # --- chat input
    prompt = st.chat_input("Ask about requirements… (or /help)")
    if prompt:
        if not st.session_state.api_key:
            st.warning("Please enter your Google AI API Key at the top of the page to use the chatbot.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = _handle_slash(prompt)
            if response is None:
                project_ctx = _safe_project_context() if st.session_state.attach_ctx else ""
                api_history = _to_gemini_history(st.session_state.messages, SE_SYSTEM_PROMPT, project_ctx)
                with st.spinner("AI is thinking..."):
                    response = get_chatbot_response(st.session_state.api_key, api_history)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

            if st.session_state.api_key and prompt and not prompt.startswith("/rewrite "):
                if st.button("Show example rewrite for my last message", key=f"coach_rewrite_btn_{len(st.session_state.messages)}"):
                    example = _rewrite_example(prompt)
                    st.info("Example rewrite (for consideration):")
                    st.markdown(f"> {example}")
