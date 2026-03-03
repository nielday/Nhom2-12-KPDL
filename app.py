"""
Hotel Booking Cancellation Prediction - Streamlit Demo App.

Features:
- Dashboard EDA with interactive charts
- Booking cancellation prediction
- Mining results (association rules + clusters)
- Model comparison
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_config, get_data_dictionary


# ---------- Page Config ----------
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main-header {
    font-size: 2rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    padding: 1rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.2rem;
    border-radius: 12px;
    color: white;
    text-align: center;
}
.insight-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ---------- Data Loading ----------
@st.cache_data
def load_data():
    """Load processed or raw data."""
    config = load_config(
        os.path.join(PROJECT_ROOT, "configs", "params.yaml")
    )
    processed = os.path.join(
        PROJECT_ROOT, config["data"]["processed_path"]
    )
    raw = os.path.join(
        PROJECT_ROOT, config["data"]["raw_path"]
    )
    if os.path.exists(processed):
        df = pd.read_csv(processed)
        source = "processed"
    elif os.path.exists(raw):
        df = pd.read_csv(raw)
        source = "raw"
    else:
        return None, "not_found"
    return df, source


@st.cache_data
def load_model_results():
    """Load saved classification results."""
    path = os.path.join(
        PROJECT_ROOT, "outputs", "tables",
        "classification.csv",
    )
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_rules():
    """Load saved association rules."""
    path = os.path.join(
        PROJECT_ROOT, "outputs", "tables",
        "association_rules.csv",
    )
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_cluster_profiles():
    """Load cluster profiles."""
    path = os.path.join(
        PROJECT_ROOT, "outputs", "tables",
        "cluster_profiles.csv",
    )
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_ts_results():
    """Load time series results."""
    path = os.path.join(
        PROJECT_ROOT, "outputs", "tables",
        "time_series.csv",
    )
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_trained_model():
    """Load best trained model."""
    path = os.path.join(
        PROJECT_ROOT, "outputs", "models",
        "best_model.joblib",
    )
    if os.path.exists(path):
        import joblib
        return joblib.load(path)
    return None


# ---------- Sidebar ----------
st.sidebar.title("🏨 Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "📊 Dashboard EDA",
        "🔮 Dự đoán huỷ phòng",
        "⛏️ Mining Results",
        "🤖 So sánh Models",
        "📈 Time Series",
        "📋 Data Dictionary",
    ],
)

# Load data
df, data_source = load_data()

if df is None:
    st.error(
        "❌ Không tìm thấy file dữ liệu. "
        "Hãy chạy pipeline trước: "
        "`python scripts/run_pipeline.py`"
    )
    st.stop()


# ============================================================
# PAGE: Dashboard EDA
# ============================================================
if page == "📊 Dashboard EDA":
    st.markdown(
        '<div class="main-header">'
        '📊 Dashboard - Hotel Booking Analysis</div>',
        unsafe_allow_html=True,
    )

    # Info
    st.info(f"📁 Data source: **{data_source}** | "
            f"Shape: **{df.shape[0]:,}** rows × "
            f"**{df.shape[1]}** columns")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Bookings", f"{len(df):,}")
    with col2:
        if "is_canceled" in df.columns:
            cancel_rate = df["is_canceled"].mean()
            st.metric("Cancel Rate", f"{cancel_rate:.1%}")
    with col3:
        if "adr" in df.columns:
            st.metric("Avg ADR", f"€{df['adr'].mean():.1f}")
    with col4:
        if "lead_time" in df.columns:
            st.metric(
                "Avg Lead Time",
                f"{df['lead_time'].mean():.0f} days",
            )

    st.divider()

    # Charts
    if "is_canceled" in df.columns:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Cancel Distribution")
            cancel_counts = df["is_canceled"].value_counts()
            fig = px.pie(
                values=cancel_counts.values,
                names=["Not Canceled", "Canceled"],
                color_discrete_sequence=[
                    "#2ecc71", "#e74c3c"
                ],
                hole=0.4,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "hotel" in df.columns:
                st.subheader("Cancel Rate by Hotel")
                hotel_cancel = (
                    df.groupby("hotel")["is_canceled"]
                    .mean()
                    .reset_index()
                )
                hotel_cancel.columns = [
                    "Hotel", "Cancel Rate"
                ]
                fig = px.bar(
                    hotel_cancel,
                    x="Hotel", y="Cancel Rate",
                    color="Hotel",
                    color_discrete_sequence=[
                        "#3498db", "#e74c3c"
                    ],
                    text_auto=".1%",
                )
                fig.update_layout(height=350)
                st.plotly_chart(
                    fig, use_container_width=True
                )

        # Lead time distribution
        if "lead_time" in df.columns:
            st.subheader("Lead Time vs Cancellation")
            fig = px.histogram(
                df, x="lead_time",
                color=df["is_canceled"].map(
                    {0: "Not Canceled", 1: "Canceled"}
                ),
                nbins=50,
                barmode="overlay",
                opacity=0.7,
                color_discrete_sequence=[
                    "#2ecc71", "#e74c3c"
                ],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Cancel by category selector
        cat_cols = [
            c for c in [
                "hotel", "market_segment",
                "deposit_type", "customer_type",
                "meal",
            ]
            if c in df.columns
        ]
        if cat_cols:
            st.subheader("Cancel Rate by Category")
            selected_cat = st.selectbox(
                "Select category:", cat_cols
            )
            cat_cancel = (
                df.groupby(selected_cat)["is_canceled"]
                .agg(["mean", "count"])
                .reset_index()
            )
            cat_cancel.columns = [
                selected_cat, "Cancel Rate", "Count"
            ]
            fig = px.bar(
                cat_cancel.sort_values(
                    "Cancel Rate", ascending=False
                ),
                x=selected_cat,
                y="Cancel Rate",
                text_auto=".1%",
                color="Cancel Rate",
                color_continuous_scale="Reds",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: Prediction
# ============================================================
elif page == "🔮 Dự đoán huỷ phòng":
    st.markdown(
        '<div class="main-header">'
        '🔮 Dự đoán Booking có bị huỷ không?</div>',
        unsafe_allow_html=True,
    )

    model = load_trained_model()
    if model is None:
        st.warning(
            "⚠️ Chưa có model. Chạy Notebook 04 hoặc "
            "`python scripts/run_pipeline.py` trước."
        )
    else:
        st.success("✅ Model loaded!")

        col1, col2, col3 = st.columns(3)

        with col1:
            lead_time = st.number_input(
                "Lead Time (days)", 0, 800, 60
            )
            adults = st.number_input("Adults", 1, 10, 2)
            children = st.number_input("Children", 0, 10, 0)
            adr = st.number_input(
                "ADR (€/night)", 0.0, 1000.0, 80.0
            )

        with col2:
            total_stays = st.number_input(
                "Total Nights", 1, 30, 3
            )
            special_requests = st.number_input(
                "Special Requests", 0, 5, 1
            )
            booking_changes = st.number_input(
                "Booking Changes", 0, 10, 0
            )
            prev_cancel = st.number_input(
                "Previous Cancellations", 0, 20, 0
            )

        with col3:
            is_repeated = st.selectbox(
                "Repeated Guest?", [0, 1]
            )
            waiting_list = st.number_input(
                "Days in Waiting List", 0, 391, 0
            )
            parking = st.number_input(
                "Parking Spaces", 0, 8, 0
            )

        if st.button("🔮 Predict", type="primary"):
            st.markdown("---")
            st.subheader("Prediction Result")
            st.info(
                "⚠️ Để chạy prediction chính xác, cần "
                "encode features giống lúc train. "
                "Demo này dùng features cơ bản."
            )

            st.markdown(
                f"- **Lead Time**: {lead_time} days "
                f"→ {'⚠️ Cao' if lead_time > 90 else '✅ OK'}\n"
                f"- **ADR**: €{adr:.0f}/night\n"
                f"- **Total Stays**: {total_stays} nights\n"
                f"- **Special Requests**: {special_requests} "
                f"→ {'✅ Giảm risk' if special_requests > 0 else '⚠️ Tăng risk'}\n"
                f"- **Previous Cancellations**: {prev_cancel} "
                f"→ {'⚠️ High risk' if prev_cancel > 0 else '✅ OK'}"
            )

            # Simple risk score
            risk = 0.3
            if lead_time > 180:
                risk += 0.25
            elif lead_time > 90:
                risk += 0.15
            if prev_cancel > 0:
                risk += 0.15
            if special_requests > 0:
                risk -= 0.1
            if is_repeated:
                risk -= 0.1
            risk = max(0.05, min(0.95, risk))

            col_a, col_b = st.columns(2)
            with col_a:
                if risk > 0.5:
                    st.error(
                        f"🔴 Cancel Probability: "
                        f"**{risk:.0%}** (HIGH RISK)"
                    )
                else:
                    st.success(
                        f"🟢 Cancel Probability: "
                        f"**{risk:.0%}** (LOW RISK)"
                    )
            with col_b:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=risk * 100,
                        title={"text": "Cancel Risk %"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {
                                    "range": [0, 30],
                                    "color": "#2ecc71",
                                },
                                {
                                    "range": [30, 60],
                                    "color": "#f39c12",
                                },
                                {
                                    "range": [60, 100],
                                    "color": "#e74c3c",
                                },
                            ],
                        },
                    )
                )
                fig.update_layout(height=250)
                st.plotly_chart(
                    fig, use_container_width=True
                )


# ============================================================
# PAGE: Mining Results
# ============================================================
elif page == "⛏️ Mining Results":
    st.markdown(
        '<div class="main-header">'
        '⛏️ Mining Results</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs([
        "📜 Association Rules", "🎯 Cluster Profiles"
    ])

    with tab1:
        rules = load_rules()
        if rules is not None:
            st.subheader(
                f"Top Association Rules ({len(rules)} rules)"
            )
            st.dataframe(rules.head(20), height=400)

            if "support" in rules.columns:
                fig = px.scatter(
                    rules.head(50),
                    x="support", y="confidence",
                    size="lift",
                    color="lift",
                    color_continuous_scale="YlOrRd",
                    hover_data=["antecedents", "consequents"],
                    title="Association Rules (size ∝ lift)",
                )
                fig.update_layout(height=500)
                st.plotly_chart(
                    fig, use_container_width=True
                )
        else:
            st.warning(
                "⚠️ Chưa có kết quả. Chạy Notebook 03."
            )

    with tab2:
        profiles = load_cluster_profiles()
        if profiles is not None:
            st.subheader("Cluster Profiles")
            st.dataframe(profiles, height=300)

            num_cols = profiles.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            if len(num_cols) >= 2:
                fig = px.imshow(
                    profiles[num_cols].T,
                    aspect="auto",
                    color_continuous_scale="YlOrRd",
                    title="Cluster Feature Heatmap",
                )
                fig.update_layout(height=500)
                st.plotly_chart(
                    fig, use_container_width=True
                )
        else:
            st.warning(
                "⚠️ Chưa có kết quả. Chạy Notebook 03."
            )


# ============================================================
# PAGE: Model Comparison
# ============================================================
elif page == "🤖 So sánh Models":
    st.markdown(
        '<div class="main-header">'
        '🤖 So sánh Classification Models</div>',
        unsafe_allow_html=True,
    )

    results = load_model_results()
    if results is not None:
        st.subheader("Model Performance")
        st.dataframe(
            results.style.highlight_max(
                subset=[
                    c for c in [
                        "accuracy", "f1", "pr_auc",
                        "roc_auc",
                    ]
                    if c in results.columns
                ],
                color="#90EE90",
            ),
            height=200,
        )

        # Bar chart comparison
        metric_cols = [
            c for c in [
                "accuracy", "precision", "recall",
                "f1", "pr_auc", "roc_auc",
            ]
            if c in results.columns
        ]
        if metric_cols and "model" in results.columns:
            fig = px.bar(
                results.melt(
                    id_vars=["model"],
                    value_vars=metric_cols,
                    var_name="Metric",
                    value_name="Score",
                ),
                x="Metric", y="Score",
                color="model", barmode="group",
                title="Model Comparison",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        # Training time
        if "train_time_seconds" in results.columns:
            st.subheader("Training Time")
            fig = px.bar(
                results,
                x="model",
                y="train_time_seconds",
                title="Training Time (seconds)",
                text_auto=".2f",
                color="model",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "⚠️ Chưa có kết quả. Chạy Notebook 04."
        )


# ============================================================
# PAGE: Time Series
# ============================================================
elif page == "📈 Time Series":
    st.markdown(
        '<div class="main-header">'
        '📈 Time Series Forecast</div>',
        unsafe_allow_html=True,
    )

    ts_results = load_ts_results()
    if ts_results is not None:
        st.subheader("Forecast Comparison")
        st.dataframe(ts_results, height=200)

        fig = px.bar(
            ts_results,
            x="method", y="rmse",
            title="Forecast RMSE Comparison",
            text_auto=".4f",
            color="method",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "⚠️ Chưa có kết quả. Chạy Notebook 05."
        )

    # Monthly cancel rate
    if (
        "is_canceled" in df.columns
        and "arrival_date_year" in df.columns
        and "arrival_date_month" in df.columns
    ):
        st.subheader("Monthly Cancellation Rate")
        month_map = {
            "January": 1, "February": 2, "March": 3,
            "April": 4, "May": 5, "June": 6,
            "July": 7, "August": 8, "September": 9,
            "October": 10, "November": 11, "December": 12,
        }
        df_ts = df.copy()
        if df_ts["arrival_date_month"].dtype == object:
            df_ts["month_num"] = df_ts[
                "arrival_date_month"
            ].map(month_map)
        else:
            df_ts["month_num"] = df_ts["arrival_date_month"]
        df_ts["year_month"] = pd.to_datetime(
            df_ts["arrival_date_year"].astype(str) + "-"
            + df_ts["month_num"].astype(str) + "-01"
        )
        monthly = (
            df_ts.groupby("year_month")["is_canceled"]
            .mean()
            .reset_index()
        )
        monthly.columns = ["Date", "Cancel Rate"]
        fig = px.line(
            monthly,
            x="Date", y="Cancel Rate",
            title="Monthly Cancellation Rate Over Time",
            markers=True,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: Data Dictionary
# ============================================================
elif page == "📋 Data Dictionary":
    st.markdown(
        '<div class="main-header">'
        '📋 Data Dictionary</div>',
        unsafe_allow_html=True,
    )

    data_dict = get_data_dictionary()
    dd_df = pd.DataFrame(
        [(k, v) for k, v in data_dict.items()],
        columns=["Column", "Description"],
    )
    st.dataframe(dd_df, height=600)

    if df is not None:
        st.subheader("Dataset Info")
        st.write(f"- **Rows**: {df.shape[0]:,}")
        st.write(f"- **Columns**: {df.shape[1]}")
        st.write(f"- **Source**: {data_source}")
        st.write("- **Numeric columns**: "
                 f"{len(df.select_dtypes(include=[np.number]).columns)}")
        st.write("- **Categorical columns**: "
                 f"{len(df.select_dtypes(include=['object']).columns)}")


# ---------- Footer ----------
st.sidebar.divider()
st.sidebar.caption(
    "🏨 Hotel Booking Cancellation Prediction\n\n"
    "Đề tài 12 - Data Mining - HK2 2025-2026"
)
