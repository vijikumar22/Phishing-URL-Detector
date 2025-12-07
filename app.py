import os
import datetime as dt

import joblib
import pandas as pd
import streamlit as st
import tldextract

from src.feature_extraction import extract_features_from_url, build_feature_dataframe

# Paths
MODEL_PATH = os.path.join("models", "url_model.pkl")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python -m src.train_model' first."
        )
    model = joblib.load(MODEL_PATH)
    return model


def predict_single_url(model, url: str):
    features_dict = extract_features_from_url(url)
    features_df = pd.DataFrame([features_dict])
    proba = model.predict_proba(features_df)[0]
    phishing_proba = float(proba[1])
    legit_proba = float(proba[0])
    return phishing_proba, legit_proba, features_dict


def get_risk_flags(features: dict):
    """
    Simple human-understandable explanation based on URL features.
    """
    flags = []

    if features["has_ip"]:
        flags.append("Uses IP address instead of domain (very suspicious).")

    if features["url_length"] > 80:
        flags.append(f"Very long URL ({features['url_length']} characters).")

    if features["count_dots"] > 4:
        flags.append(f"Too many dots in URL ({features['count_dots']}), may indicate nested subdomains.")

    if features["count_hyphens"] > 3:
        flags.append(f"Many hyphens ({features['count_hyphens']}), often used to spoof brands.")

    if features["suspicious_words"] >= 2:
        flags.append("Contains multiple suspicious words (login, verify, account, bank, etc.).")
    elif features["suspicious_words"] == 1:
        flags.append("Contains a suspicious word in the URL text.")

    if not features["has_https"]:
        flags.append("Does not use HTTPS (starts with http://).")

    if features["num_parameters"] > 3:
        flags.append(f"Has many query parameters ({features['num_parameters']}), may be trying to capture data.")

    if features["num_subdirectories"] > 4:
        flags.append(f"Deep path structure ({features['num_subdirectories']} subdirectories).")

    return flags


def extract_domain(url: str) -> str:
    extracted = tldextract.extract(url)
    if extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    return extracted.domain or url


def init_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []


def main():
    st.set_page_config(
        page_title="Phishing URL Detector",
        page_icon="🛡️",
        layout="wide"
    )

    init_session_state()
    model = load_model()

    st.title("🛡️ Phishing URL Detector (Advanced)")
    st.write(
        "This tool uses a machine learning model on URL-based features to predict whether a URL is "
        "**Phishing** or **Legitimate**, and explains *why* it looks suspicious."
    )

    # Global settings panel
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider(
            "Phishing detection threshold",
            min_value=0.30,
            max_value=0.90,
            value=0.50,
            step=0.01,
            help=(
                "If the model's phishing probability is above this value, "
                "the URL will be classified as phishing."
            )
        )
        st.caption(
            f"Current threshold: **{threshold:.2f}**. "
            "Increase to be more strict (fewer false positives, more false negatives)."
        )

    tab1, tab2, tab3 = st.tabs(
        ["🔗 Single URL Check", "📁 Bulk URL Check (CSV)", "📊 History & Analytics"]
    )

    # ========== TAB 1: SINGLE URL ==========
    with tab1:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            url_input = st.text_input(
                "Enter URL to analyse:",
                placeholder="e.g. http://secure-login-paypal.com/account/verify"
            )
            check_btn = st.button("Check URL", type="primary")

        with col_right:
            st.markdown("### ℹ️ Tips")
            st.markdown(
                "- Paste full URL including `http://` or `https://`\n"
                "- The model only analyses the URL text (no page content)\n"
                "- Use the sidebar to adjust how strict detection is"
            )

        if check_btn:
            if not url_input.strip():
                st.warning("Please enter a URL.")
            else:
                url = url_input.strip()
                phishing_proba, legit_proba, features = predict_single_url(model, url)

                # Final decision using threshold
                pred_label = "Phishing" if phishing_proba >= threshold else "Legitimate"

                # --- Result card ---
                st.subheader("Prediction")

                col1, col2 = st.columns([2, 1])

                with col1:
                    if pred_label == "Phishing":
                        st.error(
                            f"⚠️ This URL is predicted as **Phishing** "
                            f"(phishing probability: {phishing_proba:.2f})"
                        )
                    else:
                        st.success(
                            f"✅ This URL is predicted as **Legitimate** "
                            f"(phishing probability: {phishing_proba:.2f})"
                        )

                with col2:
                    st.markdown("**Confidence**")
                    st.progress(phishing_proba)
                    st.write(f"Phishing: **{phishing_proba:.2f}**")
                    st.write(f"Legitimate: **{legit_proba:.2f}**")

                # Save to history
                st.session_state["history"].append(
                    {
                        "time": dt.datetime.now().isoformat(timespec="seconds"),
                        "url": url,
                        "domain": extract_domain(url),
                        "phishing_proba": phishing_proba,
                        "legit_proba": legit_proba,
                        "threshold_used": threshold,
                        "predicted_label": pred_label,
                    }
                )

                # --- Risk explanation ---
                st.subheader("Why does it look suspicious?")
                flags = get_risk_flags(features)
                if flags:
                    for f in flags:
                        st.warning("• " + f)
                else:
                    st.info("No strong red flags detected from the URL structure, but always verify manually.")

                # --- Raw features ---
                with st.expander("🔍 View extracted features"):
                    st.json(features)

    # ========== TAB 2: BULK CSV ==========
    with tab2:
        st.write(
            "Upload a CSV file with a column named **`url`**. "
            "The app will return predictions and basic analytics."
        )
        uploaded_file = st.file_uploader("Upload CSV with 'url' column", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if "url" not in df.columns:
                st.error("CSV must contain a 'url' column.")
            else:
                st.write("Sample of uploaded data:")
                st.dataframe(df.head())

                if st.button("Run Bulk Prediction"):
                    X = build_feature_dataframe(df)
                    probas = model.predict_proba(X)
                    df["phishing_proba"] = probas[:, 1]
                    df["legit_proba"] = probas[:, 0]
                    df["prediction"] = (df["phishing_proba"] >= threshold).astype(int)
                    df["prediction_label"] = df["prediction"].map(
                        {0: "Legitimate", 1: "Phishing"}
                    )
                    df["domain"] = df["url"].apply(extract_domain)

                    st.subheader("Results")
                    st.dataframe(df)

                    # Simple analytics
                    st.subheader("Bulk Analytics")
                    col_a, col_b, col_c = st.columns(3)

                    total = len(df)
                    phish_count = int((df["prediction"] == 1).sum())
                    legit_count = int((df["prediction"] == 0).sum())

                    with col_a:
                        st.metric("Total URLs", total)
                    with col_b:
                        st.metric("Predicted Phishing", phish_count)
                    with col_c:
                        st.metric("Predicted Legitimate", legit_count)

                    # Bar chart phishing vs legitimate
                    summary = pd.DataFrame(
                        {
                            "label": ["Phishing", "Legitimate"],
                            "count": [phish_count, legit_count],
                        }
                    ).set_index("label")
                    st.bar_chart(summary)

                    # Top risky domains
                    st.markdown("#### 🔝 Top domains with phishing predictions")
                    risky = df[df["prediction"] == 1]
                    if not risky.empty:
                        domain_counts = (
                            risky["domain"].value_counts()
                            .reset_index()
                            .rename(columns={"index": "domain", "domain": "count"})
                        )
                        st.dataframe(domain_counts.head(10))
                    else:
                        st.info("No phishing URLs detected in this batch with the current threshold.")

                    # Download results
                    csv_download = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download results as CSV",
                        data=csv_download,
                        file_name="phishing_predictions_bulk.csv",
                        mime="text/csv",
                    )

    # ========== TAB 3: HISTORY & ANALYTICS ==========
    with tab3:
        st.write(
            "This tab shows all URLs you have checked in this session, with probabilities and decisions."
        )

        history = st.session_state["history"]

        if not history:
            st.info("No history yet. Run a single URL check first.")
        else:
            hist_df = pd.DataFrame(history)
            st.dataframe(hist_df)

            col1, col2, col3 = st.columns(3)
            total_h = len(hist_df)
            phish_h = int((hist_df["predicted_label"] == "Phishing").sum())
            legit_h = int((hist_df["predicted_label"] == "Legitimate").sum())

            with col1:
                st.metric("Total URLs checked", total_h)
            with col2:
                st.metric("Phishing predictions", phish_h)
            with col3:
                st.metric("Legitimate predictions", legit_h)

            # Simple chart
            hist_summary = (
                hist_df["predicted_label"].value_counts().rename_axis("label").to_frame("count")
            )
            st.bar_chart(hist_summary)

            # Download history
            csv_hist = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download history as CSV",
                data=csv_hist,
                file_name="phishing_detection_history.csv",
                mime="text/csv",
            )

            if st.button("Clear history"):
                st.session_state["history"] = []
                st.success("History cleared. Run more checks to see new entries.")


if __name__ == "__main__":
    main()
