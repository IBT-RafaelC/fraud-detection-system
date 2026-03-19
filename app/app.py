import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
 
# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark financial theme */
    .stApp { background-color: #0a0e1a; }
    .main { background-color: #0a0e1a; }
 
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #12192c, #1a2540);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .kpi-value { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .kpi-label { font-size: 0.85rem; color: #8899aa; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-fraud .kpi-value { color: #ff4d6d; }
    .kpi-normal .kpi-value { color: #00d4aa; }
    .kpi-total .kpi-value { color: #4da6ff; }
    .kpi-rate .kpi-value { color: #ffd700; }
 
    /* Alert banner */
    .fraud-alert {
        background: linear-gradient(135deg, #3d0015, #5c0020);
        border: 1px solid #ff4d6d;
        border-radius: 8px;
        padding: 12px 20px;
        color: #ff4d6d;
        font-weight: 600;
        margin: 10px 0;
    }
 
    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 24px 0 12px 0;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 8px;
    }
 
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1321;
        border-right: 1px solid #1e3a5f;
    }
 
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
 
# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Works both locally and on Streamlit Cloud
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "src" / "fraud_model.pkl"
 
    if not model_path.exists():
        st.error(f"❌ Model not found at: `{model_path}`")
        st.info("Make sure `src/fraud_model.pkl` is in your repository.")
        st.stop()
 
    package = joblib.load(model_path)
    return package["model"], package["features"]
 
model, expected_features = load_model()
 
# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Fraud Detection")
    st.markdown("---")
    st.markdown("### 📂 Upload Transactions")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload transaction data with the same features used during training."
    )
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Fraud Probability Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Adjust sensitivity. Lower = more fraud alerts."
    )
    st.markdown("---")
    st.markdown(
        "<div style='color:#8899aa; font-size:0.75rem;'>Model: Random Forest<br>Dataset: Credit Card Fraud</div>",
        unsafe_allow_html=True
    )
 
# ─── Main Content ──────────────────────────────────────────────────────────────
st.markdown("# 💳 AI Fraud Detection Dashboard")
st.markdown("<p style='color:#8899aa;'>Real-time financial fraud analysis powered by Machine Learning</p>", unsafe_allow_html=True)
 
if uploaded_file is None:
    # Landing state
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class='kpi-card'>
            <p class='kpi-value kpi-total' style='color:#4da6ff'>⚡</p>
            <p class='kpi-label'>Real-time Detection</p>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class='kpi-card'>
            <p class='kpi-value' style='color:#00d4aa'>🎯</p>
            <p class='kpi-label'>High Accuracy Model</p>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class='kpi-card'>
            <p class='kpi-value' style='color:#ffd700'>📊</p>
            <p class='kpi-label'>Interactive Analytics</p>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Upload a CSV file from the sidebar to begin fraud analysis.")
    st.stop()
 
# ─── Process Data ──────────────────────────────────────────────────────────────
try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Could not read file: {e}")
    st.stop()
 
# Prepare features for model
model_data = data.copy()
for col in ['Class', 'Fraud Prediction', 'Fraud Probability']:
    if col in model_data.columns:
        model_data = model_data.drop(col, axis=1)
 
missing_cols = [c for c in expected_features if c not in model_data.columns]
if missing_cols:
    st.error(f"❌ Missing columns: {missing_cols}")
    st.stop()
 
model_data = model_data[expected_features]
 
# Predict with custom threshold
probabilities = model.predict_proba(model_data)[:, 1]
predictions = (probabilities >= threshold).astype(int)
 
data['Fraud Prediction'] = predictions
data['Fraud Probability'] = probabilities
data['Risk Level'] = pd.cut(
    probabilities,
    bins=[0, 0.3, 0.6, 1.0],
    labels=['🟢 Low', '🟡 Medium', '🔴 High']
)
 
# ─── KPIs ──────────────────────────────────────────────────────────────────────
total = len(data)
frauds = int(data['Fraud Prediction'].sum())
normal = total - frauds
fraud_rate = (frauds / total * 100) if total > 0 else 0
avg_fraud_amount = data[data['Fraud Prediction'] == 1]['Amount'].mean() if frauds > 0 else 0
 
if frauds > 0:
    st.markdown(f"""
    <div class='fraud-alert'>
        🚨 ALERT — {frauds} fraudulent transaction{'s' if frauds > 1 else ''} detected 
        ({fraud_rate:.2f}% of total)
    </div>""", unsafe_allow_html=True)
 
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class='kpi-card kpi-total'>
        <p class='kpi-value' style='color:#4da6ff'>{total:,}</p>
        <p class='kpi-label'>Total Transactions</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class='kpi-card kpi-fraud'>
        <p class='kpi-value' style='color:#ff4d6d'>{frauds:,}</p>
        <p class='kpi-label'>Fraud Detected 🚨</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class='kpi-card kpi-normal'>
        <p class='kpi-value' style='color:#00d4aa'>{normal:,}</p>
        <p class='kpi-label'>Normal Transactions ✅</p>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class='kpi-card kpi-rate'>
        <p class='kpi-value' style='color:#ffd700'>{fraud_rate:.2f}%</p>
        <p class='kpi-label'>Fraud Rate</p>
    </div>""", unsafe_allow_html=True)
 
# ─── Charts ────────────────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>Analytics</p>", unsafe_allow_html=True)
 
col4, col5, col6 = st.columns(3)
 
with col4:
    fig_pie = px.pie(
        values=[normal, frauds],
        names=['Normal', 'Fraud'],
        color_discrete_sequence=['#00d4aa', '#ff4d6d'],
        hole=0.5,
        title="Transaction Distribution"
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#8899aa',
        title_font_color='#ffffff',
        showlegend=True
    )
    st.plotly_chart(fig_pie, use_container_width=True)
 
with col5:
    fig_hist = px.histogram(
        data, x='Amount', color='Fraud Prediction',
        color_discrete_map={0: '#00d4aa', 1: '#ff4d6d'},
        nbins=50,
        title="Amount by Transaction Type",
        labels={'Fraud Prediction': 'Type', 'Amount': 'Transaction Amount ($)'}
    )
    fig_hist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#8899aa',
        title_font_color='#ffffff',
        barmode='overlay'
    )
    fig_hist.update_traces(opacity=0.75)
    st.plotly_chart(fig_hist, use_container_width=True)
 
with col6:
    fig_prob = px.histogram(
        data, x='Fraud Probability',
        nbins=40,
        title="Fraud Probability Distribution",
        color_discrete_sequence=['#4da6ff']
    )
    fig_prob.add_vline(x=threshold, line_dash="dash", line_color="#ffd700",
                       annotation_text=f"Threshold ({threshold})", annotation_font_color="#ffd700")
    fig_prob.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#8899aa',
        title_font_color='#ffffff'
    )
    st.plotly_chart(fig_prob, use_container_width=True)
 
# ─── Tables ────────────────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>Transaction Data</p>", unsafe_allow_html=True)
 
tab1, tab2 = st.tabs(["🚨 Fraudulent Transactions", "📋 All Transactions"])
 
with tab1:
    fraud_df = data[data['Fraud Prediction'] == 1].copy()
    fraud_df = fraud_df.sort_values('Fraud Probability', ascending=False)
    if len(fraud_df) > 0:
        st.markdown(f"Showing **{len(fraud_df)}** flagged transactions sorted by risk")
        st.dataframe(
            fraud_df[['Amount', 'Fraud Probability', 'Risk Level'] +
                     [c for c in fraud_df.columns if c not in ['Amount', 'Fraud Probability', 'Risk Level', 'Fraud Prediction']]].head(100),
            use_container_width=True,
            hide_index=True
        )
        # Download button
        csv = fraud_df.to_csv(index=False)
        st.download_button("⬇️ Download Fraud Report", csv, "fraud_report.csv", "text/csv")
    else:
        st.success("✅ No fraudulent transactions detected with current threshold.")
 
with tab2:
    st.dataframe(data.head(200), use_container_width=True, hide_index=True)
