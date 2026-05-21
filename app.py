"""
app.py — Phishing URL Detector
MSc Applied Data Science & Analytics — TU Dublin
Model: XGBoost Exp 27 (HP Tuning R2) — 94.73% accuracy, 98.42% AUC
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.feature_extractor import extract_features, get_feature_explanations
from utils.validators import validate_url, clean_url, heuristic_flags

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.result-phishing {
    background: #7f0000;
    border-radius: 12px; padding: 24px 28px;
    color:  #ffffff; text-align: center;
}
.result-legit {
    background: #1a5c1a;
    border-radius: 12px; padding: 24px 28px;
    color: white; text-align: center;
}
.result-uncertain {
    background: #7a5000;
    border-radius: 12px; padding: 24px 28px;
    color: white; text-align: center;
}
.feat-row {
    padding: 8px 0;
    border-bottom: 1px solid rgba(128,128,128,0.15);
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    art = os.path.join(ROOT, "model", "artifacts")
    try:
        with open(os.path.join(art, "xgb_model.pkl"),     "rb") as f: model    = pickle.load(f)
        with open(os.path.join(art, "tld_freq.pkl"),      "rb") as f: tld_freq = pickle.load(f)
        with open(os.path.join(art, "feature_names.pkl"), "rb") as f: feat_names = pickle.load(f)
        with open(os.path.join(art, "metrics.pkl"),       "rb") as f: metrics  = pickle.load(f)
        return model, tld_freq, feat_names, metrics
    except FileNotFoundError as e:
        return None, None, None, None


# ── Predict ──────────────────────────────────────────────────────────────────────
def predict(url, model, tld_freq, feat_names):
    # Extract features — tld_freq passed in so encoding matches training
    feat_dict  = extract_features(url, tld_freq)

    # Build DataFrame in exact column order the model was trained on
    row        = pd.DataFrame([feat_dict])[feat_names]

    # Model predict
    proba      = model.predict_proba(row)[0]   # [prob_class0=phishing, prob_class1=legit]
    phish_prob = float(proba[0])
    legit_prob = float(proba[1])

    # Label with uncertainty band
    if phish_prob >= 0.55:
        label = "Phishing"
        conf  = round(phish_prob * 100, 1)
    elif phish_prob <= 0.35:
        label = "Legitimate"
        conf  = round(legit_prob * 100, 1)
    else:
        label = "Uncertain"
        conf  = round(max(phish_prob, legit_prob) * 100, 1)

    return {
        "label":      label,
        "confidence": conf,
        "phish_pct":  round(phish_prob * 100, 1),
        "legit_pct":  round(legit_prob  * 100, 1),
        "features":   feat_dict,
    }


# ── Sample URLs ──────────────────────────────────────────────────────────────────
PHISHING_SAMPLES = [
    "http://paypal-secure-login.tk/update/account/verify",
    "http://192.168.1.1/login.php?user=verify&account=confirm",
    "http://microsofft-login.xyz/verify?id=abc123&token=xyz",
    "http://bankofamerica-secure.cf/webscr?cmd=login",
    "http://www.google.com.login.secure.account-verify.evil.com/signin",
    "http://secure-paypal-login.tk/account/update/confirm",
    "http://apple-id-verify.cf/login?user=admin&secure=true",
]

LEGIT_SAMPLES = [
    "https://www.google.com",
    "https://www.linkedin.com/in/profile",
    "https://www.bbc.com/news",
    "https://www.wikipedia.org",
    "https://docs.python.org/3/library/urllib.parse.html",
]


# ── App ──────────────────────────────────────────────────────────────────────────
def main():
    model, tld_freq, feat_names, metrics = load_artifacts()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🛡️ Phishing URL Detector")
        st.markdown("---")
        st.markdown("""
**About this app**

Detects phishing URLs in real time using URL structure alone —
no page loading, no external lookups.

**Project:** MSc Applied Data Science & Analytics  
**Institution:** TU Dublin  
**Model:** XGBoost (Exp 27 — HP Tuning R2)
        """)

        if metrics:
            st.markdown("---")
            st.markdown("**Model performance on test set**")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy",  f"{metrics['accuracy']*100:.1f}%")
            c2.metric("AUC-ROC",   f"{metrics['auc']*100:.1f}%")
            c1.metric("Recall",    f"{metrics['recall']*100:.1f}%")
            c2.metric("Precision", f"{metrics['precision']*100:.1f}%")
            st.caption(f"Tested on {metrics['test_size']:,} URLs")

        st.markdown("---")
        st.markdown("""
**How it works**
1. URL is parsed into components
2. 19 structural and lexical features extracted
3. XGBoost predicts phishing probability
4. Key risk factors displayed
        """)
        st.markdown("---")
        st.caption(
            "Educational project only. Not a production security tool. "
            "Always verify suspicious URLs through official channels."
        )

    # Header
    st.markdown("# 🛡️ Phishing URL Detector")
    st.markdown(
        "Enter any URL to check if it shows phishing indicators. "
        "Analysis is instant — the model reads URL structure only, no website is loaded."
    )
    st.markdown("---")

    # Model missing warning
    if model is None:
        st.error(
            "Model artifacts not found in model/artifacts/. "
            "Please ensure xgb_model.pkl, tld_freq.pkl, "
            "feature_names.pkl and metrics.pkl are present."
        )
        st.stop()

    # Input
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com/path?query=value",
            label_visibility="collapsed",
            key="url_input",
        )
    with col_btn:
        go = st.button("Analyse", use_container_width=True, type="primary")

    # Sample URL pickers
    with st.expander("Try a sample URL"):
        t1, t2 = st.tabs(["Phishing examples", "Legitimate examples"])
        with t1:
            for s in PHISHING_SAMPLES:
                if st.button(s, key="ph_" + s):
                    st.session_state.url_input = s
                    st.rerun()
        with t2:
            for s in LEGIT_SAMPLES:
                if st.button(s, key="le_" + s):
                    st.session_state.url_input = s
                    st.rerun()

    # Analysis
    if go and url_input:
        url = clean_url(url_input)

        valid, msg = validate_url(url)
        if not valid:
            st.warning(msg)
            st.stop()

        with st.spinner("Analysing..."):
            try:
                result = predict(url, model, tld_freq, feat_names)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.markdown("---")

        # Result card
        label = result["label"]
        if label == "Phishing":
            css   = "result-phishing"
            icon  = "PHISHING DETECTED"
            advice = "Do not visit this URL. Strong phishing indicators found."
        elif label == "Legitimate":
            css   = "result-legit"
            icon  = "LIKELY LEGITIMATE"
            advice = "No strong phishing indicators found. Always stay cautious."
        else:
            css   = "result-uncertain"
            icon  = "UNCERTAIN — proceed with caution"
            advice = "Model is not confident. Treat this URL carefully."

        st.markdown(f"""
        <div class="{css}">
            <h2 style="margin:0">{icon}</h2>
            <p style="margin:8px 0 0; opacity:0.9">{advice}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Phishing probability", f"{result['phish_pct']}%")
        c2.metric("Legitimate probability", f"{result['legit_pct']}%")
        c3.metric("Confidence", f"{result['confidence']}%")

        p1, p2, p3 = st.columns(3)
        p1.progress(result["phish_pct"] / 100)
        p2.progress(result["legit_pct"] / 100)
        p3.progress(result["confidence"] / 100)

        st.markdown("---")

        # Heuristic flags
        flags = heuristic_flags(url)
        if flags:
            st.markdown("### Risk indicators found")
            for f in flags:
                st.markdown(f"- {f}")
        else:
            st.markdown("### No obvious risk indicators in URL structure")

        st.markdown("---")

        # Feature breakdown
        col_feat, col_vals = st.columns([3, 2])
        explanations = get_feature_explanations()

        with col_feat:
            st.markdown("### Feature breakdown")
            st.caption(
                "The 19 features the model used. Ranked by XGBoost feature importance "
                "(from report Table 14). Red = known high-risk signal."
            )

            # Importance order from your report Table 14
            importance_order = [
                "NoOfSubDomain", "SuspiciousKeywordFlag", "NoOfHyphensInDomain",
                "TLD_freq", "URLEntropy", "HasAtSymbol", "DomainEntropy",
                "IsFreeTLD", "NoOfOtherSpecialCharsInURL", "NoOfDegitsInURL",
                "TLDLength", "BrandSimilarityScore", "DomainLength",
                "NoOfEqualsInURL", "LetterRatioInURL", "HasObfuscation",
                "URLLength", "NoOfObfuscatedChar", "NoOfQMarkInURL",
            ]

            high_risk_feats = {
                "SuspiciousKeywordFlag", "IsFreeTLD", "HasAtSymbol",
                "HasObfuscation", "NoOfHyphensInDomain"
            }

            feats = result["features"]
            for fname in importance_order:
                if fname not in feats:
                    continue
                val   = feats[fname]
                expl  = explanations.get(fname, "")
                color = "#ff6666" if fname in high_risk_feats else "inherit"
                flag  = " ⚠️" if (fname in high_risk_feats and val not in (0, 0.0)) else ""

                if isinstance(val, float):
                    disp = f"{val:.4f}"
                else:
                    disp = str(val)

                st.markdown(
                    f'<div class="feat-row">'
                    f'<span style="color:{color};font-weight:500">{fname}{flag}</span>'
                    f' = <code>{disp}</code><br>'
                    f'<small style="color:#888">{expl}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with col_vals:
            st.markdown("### All feature values")
            feat_df = pd.DataFrame([
                {
                    "Feature": k,
                    "Value": round(v, 4) if isinstance(v, float) else v
                }
                for k, v in feats.items()
            ])
            st.dataframe(feat_df, use_container_width=True, height=500)

        st.markdown("---")

        # URL breakdown
        from urllib.parse import urlparse as up
        try:
            p = up(url if url.startswith("http") else "http://" + url)
            st.markdown("### URL breakdown")
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Scheme",   p.scheme or "—")
            bc2.metric("Domain",   p.netloc or "—")
            bc3.metric("Path",     (p.path[:25] + "…") if len(p.path) > 25 else (p.path or "—"))
            bc4.metric("Query",    (p.query[:25] + "…") if len(p.query) > 25 else (p.query or "—"))
        except Exception:
            pass

    elif go and not url_input:
        st.warning("Please enter a URL first.")

    # Batch analysis
    st.markdown("---")
    with st.expander("Batch analysis — upload a CSV"):
        st.markdown("Upload a CSV with a column named **`url`** to analyse multiple URLs at once.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            try:
                batch = pd.read_csv(uploaded)
                if "url" not in batch.columns:
                    st.error("CSV must have a column named 'url'")
                else:
                    batch = batch.head(100)
                    st.info(f"Analysing {len(batch)} URLs...")
                    results = []
                    bar = st.progress(0)
                    for i, row in batch.iterrows():
                        u = str(row["url"])
                        ok, _ = validate_url(u)
                        if ok:
                            try:
                                r = predict(u, model, tld_freq, feat_names)
                                results.append({
                                    "URL":        u,
                                    "Prediction": r["label"],
                                    "Phishing_%": r["phish_pct"],
                                    "Legit_%":    r["legit_pct"],
                                    "Confidence": r["confidence"],
                                })
                            except Exception:
                                results.append({"URL": u, "Prediction": "Error",
                                    "Phishing_%": None, "Legit_%": None, "Confidence": None})
                        else:
                            results.append({"URL": u, "Prediction": "Invalid URL",
                                "Phishing_%": None, "Legit_%": None, "Confidence": None})
                        bar.progress((i + 1) / len(batch))

                    out = pd.DataFrame(results)
                    st.dataframe(out, use_container_width=True)
                    st.download_button(
                        "Download results CSV",
                        data=out.to_csv(index=False).encode(),
                        file_name="phishing_results.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
