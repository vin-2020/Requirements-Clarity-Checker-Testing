# ui/tabs/home_tab.py
from __future__ import annotations
from streamlit.components.v1 import html

def render(st, db, rule_engine, CTX):
    # ---------- Styles (clean white/black + higher contrast) ----------
    st.markdown("""
    <style>
      .rc-wrap { max-width: 1200px; margin: 0 auto; }
      .rc-wrap, .rc-wrap * { color: #0b1220; }
      .rc-muted { color:#1f2937; }

      /* Section headings */
      .rc-h2 { font-size: 24px; font-weight: 800; margin: 4px 0 10px; color:#0b1220; }

      /* Soft info blocks */
      .story, .contact{
        background:#ffffff; border:1px solid #e5e7eb; border-radius:12px;
        padding:14px 16px; box-shadow:0 2px 6px rgba(15,23,42,.04);
      }

      /* Cards grid for tools */
      .rc-grid { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:14px; }
      @media (max-width: 1000px){ .rc-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
      @media (max-width: 640px){ .rc-grid { grid-template-columns: 1fr; } }

      .rc-card {
        background:#ffffff; border:1px solid #e5e7eb; border-radius:12px;
        padding:16px; box-shadow:0 2px 6px rgba(15,23,42,.04);
        display:flex; flex-direction:column; gap:10px;
      }
      .rc-card h4 { margin:0; font-size:18px; color:#0b1220; }
      .rc-card p  { margin:0; color:#111; line-height:1.5; }

      /* Primary button */
      .rc-btn {
        align-self:flex-start;
        cursor:pointer; border:none; font-weight:700; font-size:14px;
        padding:10px 14px; border-radius:10px; color:#fff;
        background: linear-gradient(135deg, var(--rc-primary, #4F46E5), #22D3EE);
        transition: filter .18s ease, transform .18s ease;
      }
      .rc-btn:hover { filter:brightness(1.05); transform: translateY(-1px); }

      /* Divider */
      .rc-divider { height:1px; background:#e5e7eb; margin:18px 0; }

      /* Tiny foot links row */
      .rc-links { display:flex; gap:12px; flex-wrap:wrap; justify-content:center; }
      .rc-links a { color: #2563eb !important; text-decoration:none; }
      .rc-links a:hover { text-decoration:underline; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="rc-wrap">', unsafe_allow_html=True)

    # ---------- What is ReqCheck? ----------
    st.markdown('<div class="rc-h2">What is ReqCheck?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="story rc-muted">'
        'ReqCheck is your <b>assistant coach</b> for requirements. It helps you turn fuzzy stakeholder needs into '
        '<b>clear, testable requirements</b> (INCOSE/ISO&nbsp;29148 style). It highlights weak wording, passive voice, '
        'incomplete fragments, and multi-action statements, then offers <b>AI assists</b> to rewrite or decompose.'
        '</div>',
        unsafe_allow_html=True
    )

    # ---------- Tools overview (each with a button that jumps to its tab) ----------
    st.markdown('<div class="rc-h2">Explore the tools</div>', unsafe_allow_html=True)
    st.markdown('<div class="rc-grid">', unsafe_allow_html=True)

    # Analyzer
    st.markdown("""
      <div class="rc-card">
        <h4>📄 Document Analyzer</h4>
        <p>Upload a <code>.docx</code> or paste text. Get instant highlights for ambiguity (e.g., <i>may, should</i>), passive voice,
        fragments, and non-singularity. View explanations and a quick clarity score.</p>
        <button class="rc-btn" id="rc-open-analyze">Go to Analyzer</button>
      </div>
    """, unsafe_allow_html=True)

    # Need→Requirement
    st.markdown("""
      <div class="rc-card">
        <h4>💡 Need → Requirement Helper</h4>
        <p>Start with a stakeholder need and turn it into a well-formed, testable requirement.
        Use AI to suggest actors, triggers, metrics, thresholds, and acceptance criteria.</p>
        <button class="rc-btn" id="rc-open-need">Go to Need→Req</button>
      </div>
    """, unsafe_allow_html=True)

    # Chatbot
    st.markdown("""
      <div class="rc-card">
        <h4>💬 Requirements Chatbot</h4>
        <p>Ask quick questions about wording, structure, and good practice.
        Get short, actionable guidance aligned with INCOSE/ISO&nbsp;29148.</p>
        <button class="rc-btn" id="rc-open-chat">Go to Chatbot</button>
      </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # /.rc-grid

    st.markdown('<div class="rc-divider"></div>', unsafe_allow_html=True)

    # ---------- Contact ----------
    st.markdown('<div class="rc-h2">📩 Contact</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="contact rc-muted">'
        'Have feedback, suggestions, or questions? Reach out at <b>reqcheck.dev@gmail.com</b>.'
        '</div>',
        unsafe_allow_html=True
    )

   

    st.markdown('</div>', unsafe_allow_html=True)  # /.rc-wrap

    # ---------- Wire the buttons to real tabs (indices: Home=0, Analyzer=1, Need→Req=2, Chatbot=3) ----------
    html("""
    <script>
      (function(){
        const P = window.parent || window;
        const doc = P.document;
        function tabs(){ return Array.from(doc.querySelectorAll('button[role="tab"]')); }

        const go = (idx) => {
          const t = tabs();
          if (t && t[idx] && typeof t[idx].click === 'function') t[idx].click();
        };

        document.getElementById('rc-open-analyze')?.addEventListener('click', () => go(1), {passive:true});
        document.getElementById('rc-open-need')?.addEventListener('click', () => go(2), {passive:true});
        document.getElementById('rc-open-chat')?.addEventListener('click', () => go(3), {passive:true});
      })();
    </script>
    """, height=0)
