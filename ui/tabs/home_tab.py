# ui/tabs/home_tab.py
from __future__ import annotations

LOGO_URL = "https://raw.githubusercontent.com/vin-2020/Requirements-Clarity-Checker/main/ReqCheck_Logo.png"

def render(st, db, rule_engine, CTX):
    # ---------- Styles ----------
    st.markdown("""
    <style>
      /* Hero (logo + title) */
      .rc-hero{
        display:flex; align-items:center; gap:16px;
        padding:16px 20px; border-radius:14px;
        background:#eef1f5;
        margin-bottom: 12px;
      }
      .rc-hero img{
        height:130px; width:auto; border-radius:14px;
        box-shadow:0 2px 6px rgba(0,0,0,.05);
      }
      .rc-hero .rc-title{ margin:0; font-size:44px; line-height:1.0; }
      .rc-hero .rc-sub  { margin:6px 0 0 0; font-size:18px; color:#6b7280; }

      /* Small cards / features */
      .feature{ color:#374151; margin-top:4px; }

      /* Story callout */
      .story{
        background:#f8fafc; border:1px solid #e5e7eb; border-radius:10px;
        padding:14px 16px;
      }

      @media (max-width: 700px){
        .rc-hero{gap:12px}
        .rc-hero img{height:76px}
        .rc-hero .rc-title{font-size:34px}
        .rc-hero .rc-sub {font-size:16px}
      }
    </style>
    """, unsafe_allow_html=True)

   

    # ---------- What is ReqCheck? ----------
    st.markdown("### What is ReqCheck?")
    st.markdown(
        "ReqCheck is an **assistant coach** for systems engineers. It helps you convert fuzzy stakeholder needs "
        "into **clear, testable requirements** that follow good practices (INCOSE/ISO 29148 style). It analyzes text, "
        "highlights weak language, suggests rewrites (optionally with AI), and helps **decompose** big needs into "
        "atomic, verifiable items."
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("#### 💡 Need Tutor")
        st.markdown("<div class='feature'>Guide needs → structured requirement with fields & preview.</div>", unsafe_allow_html=True)
    with f2:
        st.markdown("#### 🤖 AI Assist")
        st.markdown("<div class='feature'>Autofill & review (actors, triggers, metrics, thresholds).</div>", unsafe_allow_html=True)
    with f3:
        st.markdown("#### 🚩 Quality Checks")
        st.markdown("<div class='feature'>Flags ambiguity, passive voice, incompleteness, non-singularity.</div>", unsafe_allow_html=True)
    with f4:
        st.markdown("#### 🧩 Decomposition")
        st.markdown("<div class='feature'>Break high-level needs into atomic, testable children.</div>", unsafe_allow_html=True)

    st.divider()

    # ---------- Contact ----------
    st.subheader("📩 Contact")
    st.write(
        "Have feedback, suggestions, or questions? "
        "Reach out at **reqcheck.dev@gmail.com**"
    )
