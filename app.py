import streamlit as st
from utils.claims import extract_claims
from utils.verifier import verify_all_claims
from utils.scorer import score_results

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Hallucination Detector", layout="centered")

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0a;
    color: #e0e0e0;
  }

  #MainMenu, footer, header { visibility: hidden; }

  .block-container { max-width: 760px; padding-top: 3rem; }

  textarea {
    background-color: #111111 !important;
    color: #e0e0e0 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
  }
  textarea:focus { border-color: #555 !important; box-shadow: none !important; }

  label { color: #7c6fcd !important; font-size: 0.78rem !important;
          letter-spacing: 0.1em; text-transform: uppercase; }

  div.stButton > button {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    color: #a78bfa;
    border: 1px solid #4c1d95;
    border-radius: 6px;
    padding: 0.5rem 2rem;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    transition: all 0.2s;
    width: 100%;
  }
  div.stButton > button:hover {
    border-color: #a78bfa;
    color: #ffffff;
    box-shadow: 0 0 12px rgba(167, 139, 250, 0.3);
    background: linear-gradient(135deg, #1e1b4b, #1a1a2e);
  }

  .score-card {
    background: #111111;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
  }
  .score-number { font-size: 4rem; font-weight: 600; letter-spacing: -2px; line-height: 1; }
  .verdict-label {
    font-size: 0.85rem; font-weight: 500; letter-spacing: 0.15em;
    text-transform: uppercase; margin-top: 0.5rem;
  }
  .score-meta { font-size: 0.75rem; color: #555; margin-top: 0.75rem; }

  .bar-track {
    height: 4px; background: #1e1e1e; border-radius: 2px;
    margin: 1.25rem auto 0; width: 60%;
  }
  .bar-fill { height: 100%; border-radius: 2px; }

  .claim-card {
    background: #111111; border: 1px solid #1e1e1e;
    border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;
  }
  .claim-text { font-size: 0.88rem; color: #cccccc; margin-bottom: 0.5rem; line-height: 1.5; }
  .explanation-text { font-size: 0.78rem; color: #666; line-height: 1.5; }

  .badge {
    display: inline-block; font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 0.2rem 0.6rem; border-radius: 4px; margin-bottom: 0.4rem;
  }
  .badge-supported   { background: #0d2b0d; color: #4caf50; border: 1px solid #1a4a1a; }
  .badge-weak        { background: #2b2000; color: #ff9800; border: 1px solid #4a3800; }
  .badge-unsupported { background: #2b0d0d; color: #f44336; border: 1px solid #4a1a1a; }

  .example-box {
    background: #0d0d1a; border: 1px solid #2d2b6b; border-radius: 8px;
    padding: 1rem 1.25rem; margin-bottom: 2rem;
  }

  hr { border-color: #1e1e1e; }

  details summary {
    font-size: 0.72rem; color: #444; cursor: pointer;
    letter-spacing: 0.05em; text-transform: uppercase;
  }
  details summary:hover { color: #888; }
  details p {
    font-size: 0.75rem; color: #444; line-height: 1.6;
    margin-top: 0.5rem; border-left: 2px solid #222; padding-left: 0.75rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-bottom: 2rem;">
  <p style="font-size:0.7rem; letter-spacing:0.3em; text-transform:uppercase;
            color:#6d5fc7; margin-bottom:0.4rem;">AI Fact Checker</p>
  <h1 style="font-size:2rem; font-weight:600; letter-spacing:-1px; margin:0; color:#a78bfa;">
    AI Hallucination Detector
  </h1>
  <p style="font-size:0.78rem; color:#6d5fc7; margin:0.2rem 0 0; letter-spacing:0.15em;
            text-transform:uppercase;">
    CS + AI Knowledge Base
  </p>
  <p style="font-size:0.82rem; color:#555; margin-top:0.75rem; max-width:520px;
            margin-left:auto; margin-right:auto; line-height:1.6;">
    Enter a question and paste an AI-generated answer below. The app will extract
    factual claims from the answer and check each one against a curated CS and AI
    knowledge base.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Example box ───────────────────────────────────────────────────────────────
with st.expander("See an example"):
    st.markdown("""
**Question:** What is a transformer in machine learning?

**AI Answer:** A transformer is a deep learning model that uses self-attention mechanisms to process sequences. It was introduced in 2017 and is the architecture behind models like GPT and BERT. Transformers rely on recurrent layers to handle long-range dependencies.

*The last sentence is false — transformers replaced recurrent layers, not used them. The detector should catch it.*
""")


# ── Input fields ──────────────────────────────────────────────────────────────
question = st.text_area(
    label="Question",
    placeholder="e.g. What is a transformer in machine learning?",
    height=80,
)

ai_answer = st.text_area(
    label="AI Answer",
    placeholder="Paste the AI-generated answer you want to fact-check here...",
    height=180,
)

analyze_clicked = st.button("Analyze")


# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_clicked:
    if not ai_answer.strip():
        st.warning("Please paste an AI-generated answer before analyzing.")
    else:
        with st.spinner("Extracting claims from the answer..."):
            claims = extract_claims(ai_answer)

        if not claims:
            st.info("No checkable factual claims found in the answer.")
        else:
            st.markdown(f"""
            <p style="font-size:0.72rem; color:#444; letter-spacing:0.1em;
                      text-transform:uppercase; margin-bottom:0.5rem;">
              {len(claims)} claim(s) extracted
            </p>
            """, unsafe_allow_html=True)

            with st.spinner(f"Verifying {len(claims)} claim(s) against knowledge base..."):
                results = verify_all_claims(claims)

            summary = score_results(results)

            # ── Score card ────────────────────────────────────────────────────
            score = summary["score"]
            verdict = summary["verdict"]
            total = summary["total"]
            counts = summary["counts"]

            score_color = "#4caf50" if score >= 0.75 else "#ff9800" if score >= 0.4 else "#f44336"
            bar_pct = int(score * 100)

            st.markdown(f"""
            <div class="score-card">
              <div class="score-number" style="color:{score_color};">{score:.2f}</div>
              <div class="verdict-label" style="color:{score_color};">{verdict}</div>
              <div class="bar-track">
                <div class="bar-fill" style="width:{bar_pct}%; background:{score_color};"></div>
              </div>
              <div class="score-meta">
                {counts.get('Supported', 0)} supported &nbsp;·&nbsp;
                {counts.get('Weakly Supported', 0)} weakly supported &nbsp;·&nbsp;
                {counts.get('Unsupported', 0)} unsupported &nbsp;·&nbsp;
                {total} total claims
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Claims breakdown ──────────────────────────────────────────────
            st.markdown("""
            <p style="font-size:0.7rem; letter-spacing:0.2em; text-transform:uppercase;
                      color:#444; margin-bottom:1rem;">Claims Breakdown</p>
            """, unsafe_allow_html=True)

            badge_class = {
                "Supported": "badge-supported",
                "Weakly Supported": "badge-weak",
                "Unsupported": "badge-unsupported",
            }

            for r in results:
                v = r["verdict"]
                cls = badge_class.get(v, "badge-unsupported")
                evidence_html = "".join(f"<p>{e}</p>" for e in r.get("evidence", []))

                correct_html = ""
                if r.get("correct"):
                    correct_html = f"""
                  <div style="margin-top:0.6rem; padding:0.6rem 0.75rem;
                              background:#0d1a0d; border-left:2px solid #2a5a2a; border-radius:4px;">
                    <span style="font-size:0.65rem; font-weight:600; letter-spacing:0.1em;
                                 text-transform:uppercase; color:#4caf50;">Correct</span>
                    <div style="font-size:0.78rem; color:#aaa; margin-top:0.25rem;
                                line-height:1.5;">{r['correct']}</div>
                  </div>"""

                st.markdown(f"""
                <div class="claim-card">
                  <span class="badge {cls}">{v}</span>
                  <div class="claim-text">{r['claim']}</div>
                  <div class="explanation-text">{r['explanation']}</div>
                  {correct_html}
                  <details style="margin-top:0.6rem;">
                    <summary>View evidence</summary>
                    {evidence_html}
                  </details>
                </div>
                """, unsafe_allow_html=True)
