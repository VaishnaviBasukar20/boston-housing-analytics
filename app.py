import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, precision_recall_curve
)
from sklearn.model_selection import train_test_split

# =========================
# PAGE CONFIG (no CSS hacks)
# =========================
st.set_page_config(
    page_title="Boston Housing — FinTech Analytics",
    layout="wide",
    page_icon="💹",
)

# ==============
# DATA LOADING
# ==============
def load_boston_df():
    """Loads boston_housing.csv if present; otherwise tries sklearn fallback.
       If neither is available, shows a helpful error."""
    try:
        df_ = pd.read_csv("boston_housing.csv")
        return df_.dropna()
    except Exception:
        try:
            # NOTE: load_boston is deprecated in latest sklearn.
            from sklearn.datasets import load_boston  # type: ignore
            data = load_boston()
            df_ = pd.DataFrame(data.data, columns=data.feature_names)
            df_["MEDV"] = data.target
            return df_.dropna()
        except Exception:
            st.error(
                "Could not find 'boston_housing.csv' and sklearn fallback isn't available. "
                "Place 'boston_housing.csv' next to app.py and rerun."
            )
            st.stop()

df = load_boston_df()

# Identify target column (price-like)
possible_targets = ["MEDV","medv","PRICE","price","SalePrice","target","y"]
target_col = next((c for c in df.columns if c.lower() in [x.lower() for x in possible_targets]), df.columns[-1])

# Create binary target for Naive Bayes classification
median_val = df[target_col].median()
df["HighPrice"] = (df[target_col] > median_val).astype(int)

# Select numeric features only
X_all = df.select_dtypes(include=[float, int]).drop(columns=[target_col, "HighPrice"], errors="ignore").copy()
y_all = df["HighPrice"].copy()

# =================
# SIDEBAR CONTROLS
# =================
st.sidebar.header("⚙️ Controls")
use_features = st.sidebar.multiselect(
    "Select features to include",
    options=X_all.columns.tolist(),
    default=X_all.columns.tolist()
)
if len(use_features) < 2:
    st.sidebar.warning("Select at least 2 features for a meaningful model.")
    use_features = X_all.columns.tolist()

test_size = st.sidebar.slider("Test size (%)", 10, 40, 25, step=5) / 100.0
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))

X = X_all[use_features].copy()
y = y_all.copy()

# Scale and split
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Base metrics (threshold=0.5)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
prec_curve, rec_curve, thr_pr = precision_recall_curve(y_test, y_proba)

# =========
# HEADER
# =========
st.markdown(
    "<h1 style='font-weight:800; letter-spacing:0.2px;'>💹 Boston Housing — FinTech Analytics (Naive Bayes)</h1>",
    unsafe_allow_html=True,
)
st.caption(
    f"Target: **{target_col}** | Median threshold: **{median_val:.2f}** | "
    f"Features used: **{len(use_features)} / {X_all.shape[1]}** | Test size: **{int(test_size*100)}%**"
)

# =========
# NAV TABS
# =========
tab_overview, tab_corr, tab_model, tab_threshold, tab_explorer, tab_shap, tab_sim, tab_data = st.tabs([
    "🏠 Overview",
    "📈 Correlations",
    "📊 Model Performance",
    "🎯 Threshold Tuning",
    "🔎 Predictions Explorer",
    "🧠 Explainability (SHAP)",
    "🧪 What-If Simulator",
    "📄 Data"
])

# --------------
# OVERVIEW TAB
# --------------
# --------------
# OVERVIEW TAB
# --------------
with tab_overview:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{acc:.3f}")
    k2.metric("Precision", f"{prec:.3f}")
    k3.metric("Recall", f"{rec:.3f}")
    k4.metric("F1 Score", f"{f1:.3f}")

    st.subheader("Price Distribution (with Density)")

    mean_val = df[target_col].mean()
    median_val = df[target_col].median()

    fig_hist = go.Figure()

    # Histogram with soft transparency & outline
    fig_hist.add_trace(go.Histogram(
        x=df[target_col],
        nbinsx=30,
        name="Price Distribution",
        marker=dict(color="#00E5A8", line=dict(color="#00FFC6", width=1)),
        opacity=0.55,
    ))

    # Smooth KDE curve
    import scipy.stats as st_kde
    kde_x = np.linspace(df[target_col].min(), df[target_col].max(), 200)
    kde_y = st_kde.gaussian_kde(df[target_col])(kde_x)
    kde_y = kde_y * (len(df[target_col]) * (df[target_col].max() - df[target_col].min()) / 30)

    fig_hist.add_trace(go.Scatter(
        x=kde_x, y=kde_y,
        mode="lines",
        line=dict(color="#06B6D4", width=3),
        name="Density Curve (KDE)"
    ))

    # Mean & Median reference lines
    fig_hist.add_vline(x=mean_val, line_width=2, line_dash="dash", line_color="#00E5A8", annotation_text="Mean", annotation_position="top left")
    fig_hist.add_vline(x=median_val, line_width=2, line_dash="dot", line_color="#FFD166", annotation_text="Median", annotation_position="top left")

    fig_hist.update_layout(
        xaxis_title="House Price (MEDV)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=450,
        legend_title_text="",
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Class Balance (High vs Low)")
    fig_pie = px.pie(df, names="HighPrice", hole=0.35,
                     color_discrete_sequence=["#06B6D4", "#00E5A8"])
    st.plotly_chart(fig_pie, use_container_width=True)

# ----------------
# CORRELATIONS TAB
# ----------------
with tab_corr:
    st.subheader("Correlation Heatmap — Sorted & Annotated")

    corr = df[use_features + [target_col, "HighPrice"]].corr()

    # Sort columns by correlation with target (HighPrice)
    corr = corr.reindex(corr["HighPrice"].abs().sort_values(ascending=False).index, axis=0)
    corr = corr.reindex(corr["HighPrice"].abs().sort_values(ascending=False).index, axis=1)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            reversescale=True,
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation"),
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>ρ = %{z:.3f}<extra></extra>"
        )
    )

    fig.update_layout(
        height=800,
        xaxis=dict(tickangle=45),
        margin=dict(t=50, b=50, l=80, r=80),
    )

    # Add numeric annotations
    for i, row in enumerate(corr.values):
        for j, val in enumerate(row):
            fig.add_annotation(
                x=corr.columns[j],
                y=corr.index[i],
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(color="white" if abs(val) > 0.5 else "lightgray", size=11)
            )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # Automatic Insights
    # --------------------
    st.subheader("📌 Key Relationship Insights")

    strong_pos = corr[target_col].sort_values(ascending=False).iloc[1:4]
    strong_neg = corr[target_col].sort_values().iloc[:3]

    st.write("**Top Positive Correlations with Price**:")
    st.write(", ".join([f"**{idx}** (+{val:.2f})" for idx, val in strong_pos.items()]))

    st.write("**Top Negative Correlations with Price**:")
    st.write(", ".join([f"**{idx}** ({val:.2f})" for idx, val in strong_neg.items()]))

# --------------------
# MODEL PERFORMANCE TAB
# --------------------
with tab_model:
    st.subheader("ROC & Precision–Recall")
    c1, c2 = st.columns(2)
    with c1:
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"ROC (AUC={roc_auc:.3f})",
                                     line=dict(width=3, color="#00E5A8")))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dash", color="#7C8594"),
                                     name="Random"))
        roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(roc_fig, use_container_width=True)
    with c2:
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=rec_curve, y=prec_curve, mode="lines",
                                    name="Precision–Recall", line=dict(width=3, color="#06B6D4")))
        pr_fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(pr_fig, use_container_width=True)

    st.subheader("Decile Lift (ranking by predicted probability)")
    deck = pd.DataFrame({"proba": y_proba, "actual": y_test.values}).sort_values("proba", ascending=False)
    deck["decile"] = pd.qcut(deck["proba"], 10, labels=False, duplicates="drop")
    decile = deck.groupby("decile")["actual"].mean().rename("response_rate").reset_index()
    decile["decile"] = 10 - decile["decile"]  # top = 10
    decile = decile.sort_values("decile")
    base_rate = deck["actual"].mean()
    decile["lift"] = decile["response_rate"] / max(base_rate, 1e-9)

    lift_fig = go.Figure()
    lift_fig.add_trace(go.Bar(x=decile["decile"], y=decile["lift"],
                              marker_color="#00E5A8", name="Lift"))
    lift_fig.update_layout(xaxis_title="Decile (10=Top)", yaxis_title="Lift vs Base")
    st.plotly_chart(lift_fig, use_container_width=True)

# -------------------
# THRESHOLD TUNING TAB
# -------------------
with tab_threshold:
    st.subheader("Decision Threshold Tuning")
    thr = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
    y_thr = (y_proba >= thr).astype(int)

    acc_t = accuracy_score(y_test, y_thr)
    prec_t = precision_score(y_test, y_thr, zero_division=0)
    rec_t = recall_score(y_test, y_thr, zero_division=0)
    f1_t = f1_score(y_test, y_thr, zero_division=0)
    cm_t = confusion_matrix(y_test, y_thr)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc_t:.3f}")
    m2.metric("Precision", f"{prec_t:.3f}")
    m3.metric("Recall", f"{rec_t:.3f}")
    m4.metric("F1", f"{f1_t:.3f}")

    cm_fig = go.Figure(data=go.Heatmap(
        z=cm_t, x=["Pred 0","Pred 1"], y=["Actual 0","Actual 1"],
        colorscale="Reds", colorbar=dict(title="Count")
    ))
    cm_fig.update_layout(title=f"Confusion Matrix @ threshold={thr:.2f}")
    st.plotly_chart(cm_fig, use_container_width=True)

# ----------------------
# PREDICTIONS EXPLORER TAB
# ----------------------
with tab_explorer:
    st.subheader("Explore Prediction Probability vs Feature")
    feat = st.selectbox("Feature", use_features)
    ex_df = X_test.copy()
    ex_df["Actual"] = y_test.values
    ex_df["Proba(High)"] = y_proba
    chart = px.scatter(
        ex_df, x=feat, y="Proba(High)", color=ex_df["Actual"].astype(str),
        color_discrete_sequence=["#06B6D4","#00E5A8"],
        opacity=0.9, render_mode="webgl"
    )
    st.plotly_chart(chart, use_container_width=True)

# -----------------------
# EXPLAINABILITY (SHAP) TAB
# -----------------------
with tab_shap:
    st.subheader("Explainability (SHAP)")

    try:
        import shap
        import matplotlib.pyplot as plt

        # Use small background sample for performance
        bg = X_train.sample(min(200, len(X_train)), random_state=0)

        # Use probability of class=1 for SHAP explanation
        def proba_fn(z):
            z_df = pd.DataFrame(z, columns=X_train.columns)
            return model.predict_proba(z_df)[:, 1]

        explainer = shap.Explainer(proba_fn, bg.values)
        sample = X_test.sample(min(200, len(X_test)), random_state=1)
        sv = explainer(sample.values)

        # Global Feature Importance
        st.write("### 🔥 Global Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.bar(sv, show=False)
        st.pyplot(fig)
        plt.clf()

        # Beeswarm
        st.write("### 🐝 Beeswarm Impact Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.beeswarm(sv, max_display=20, show=False)
        st.pyplot(fig)
        plt.clf()

        # Single Example Waterfall
        st.write("### 🌊 Single Prediction Explanation")
        idx = st.slider("Pick sample index:", 0, len(sample)-1, 0)
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(sv[idx], max_display=20, show=False)
        st.pyplot(fig)
        plt.clf()

    except Exception as e:
        st.warning(
            "SHAP is installed incorrectly or missing.\n"
            "Install it using:\n\n`pip install shap`\n\n"
            f"Error details: {e}"
        )

# --------------------
# WHAT-IF SIMULATOR TAB
# --------------------
with tab_sim:
    st.subheader("What-If Scenario Simulator")
    st.caption("Move sliders to set feature values; see probability & decision at your threshold.")
    thr_sim = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, key="thr_sim")

    # Build dynamic sliders using original (unscaled) ranges for interpretability
    cols = st.columns(3)
    user_vals = {}
    for i, feat in enumerate(use_features):
        c = cols[i % 3]
        vmin = float(df[feat].min())
        vmax = float(df[feat].max())
        vdef = float(df[feat].mean())
        user_vals[feat] = c.slider(feat, vmin, vmax, vdef)

    sim_df_raw = pd.DataFrame([user_vals])
    # Scale with the same scaler (fit on train+test above)
    sim_df_scaled = pd.DataFrame(scaler.transform(sim_df_raw[X.columns]), columns=X.columns)
    p = float(model.predict_proba(sim_df_scaled)[:, 1][0])
    y_hat = int(p >= thr_sim)

    g, k = st.columns([2,1])
    with g:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p * 100.0,
            number={'suffix': "%"},
            title={'text': "Predicted Probability: High Price"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)
    with k:
        st.metric("Decision", "HIGH (1)" if y_hat==1 else "LOW (0)", help=f"Threshold = {thr_sim:.2f}")

# -----------
# DATA TAB
# -----------
with tab_data:
    st.subheader("Dataset")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download processed CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="boston_housing_processed.csv",
        mime="text/csv"
    )

st.caption("Built with Streamlit • Plotly • scikit-learn | Model: Gaussian Naive Bayes | Theme: FinTech Terminal")
