# =============================================================================
# pages/3_analytics.py — Full detection analytics dashboard
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import csv
import json
import pandas as pd
import streamlit as st

from config import GLOBAL_CSS
from core.logger import (
    fetch_all_detections, fetch_sessions, fetch_class_summary,
    fetch_detections_over_time, clear_all_data,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Analytics | RT-DETR Sentinel Pro",
                   page_icon="📊", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.markdown('<p class="gradient-text">📊 Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-heading">Detection history, class distribution, confidence analysis</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
detections = fetch_all_detections(limit=50_000)
sessions   = fetch_sessions()
class_sum  = fetch_class_summary()
time_series = fetch_detections_over_time()

df_det  = pd.DataFrame(detections)  if detections  else pd.DataFrame()
df_sess = pd.DataFrame(sessions)    if sessions    else pd.DataFrame()
df_cls  = pd.DataFrame(class_sum)   if class_sum   else pd.DataFrame()
df_ts   = pd.DataFrame(time_series) if time_series else pd.DataFrame()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="medium")
kpi1.metric("Total Detections",    len(df_det))
kpi2.metric("Total Sessions",      len(df_sess))
kpi3.metric("Unique Classes",      df_det["class_name"].nunique() if not df_det.empty else 0)
kpi4.metric("Avg Confidence",
            f"{df_det['confidence'].mean()*100:.1f}%" if not df_det.empty else "—")

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
if df_det.empty:
    st.info(
        "No detections logged yet. Run the **Live Detection** or **File Detection** pages first.",
        icon="📭",
    )
else:
    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        with st.container(border=True):
            st.subheader("🏷️ Class Distribution (Top 20)")
            if not df_cls.empty:
                top20 = df_cls.head(20).set_index("class_name")["count"]
                st.bar_chart(top20, height=320)

    with chart_col2:
        with st.container(border=True):
            st.subheader("🎯 Confidence Histogram")
            if "confidence" in df_det.columns:
                # bucket into 0.05-wide bins
                bins = pd.cut(df_det["confidence"], bins=20)
                hist = df_det.groupby(bins, observed=True).size()
                hist.index = hist.index.astype(str)
                st.bar_chart(hist, height=320)

    st.markdown("")

    with st.container(border=True):
        st.subheader("📈 Detections Over Time")
        if not df_ts.empty:
            df_ts_plot = df_ts.set_index("bucket")["count"]
            st.line_chart(df_ts_plot, height=260)
        else:
            st.info("Not enough data yet.", icon="⏳")

    st.markdown("")

    # Per-class average confidence table
    row3_l, row3_r = st.columns(2, gap="large")

    with row3_l:
        with st.container(border=True):
            st.subheader("📋 Class Summary Table")
            if not df_cls.empty:
                df_cls_display = df_cls.copy()
                df_cls_display["avg_conf"] = df_cls_display["avg_conf"].map("{:.2%}".format)
                df_cls_display.columns = ["Class", "Count", "Avg Confidence"]
                st.dataframe(df_cls_display, use_container_width=True, hide_index=True)

    with row3_r:
        with st.container(border=True):
            st.subheader("🗂️ Session History")
            if not df_sess.empty:
                cols_to_show = [c for c in
                                ["session_id","source","model_name","started_at",
                                 "total_frames","total_detections"]
                                if c in df_sess.columns]
                st.dataframe(df_sess[cols_to_show], use_container_width=True, hide_index=True)
            else:
                st.info("No sessions recorded yet.", icon="🗂️")

    # ---------------------------------------------------------------------------
    # Raw detections table + search
    # ---------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("🔍 Raw Detections Table", expanded=False):
        search_cls = st.text_input("Filter by class name", placeholder="person, car, …")
        df_show = df_det.copy()
        if search_cls.strip():
            df_show = df_show[df_show["class_name"].str.contains(search_cls.strip(), case=False)]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Export & management
# ---------------------------------------------------------------------------
st.markdown("---")
exp_col1, exp_col2, exp_col3 = st.columns(3, gap="medium")

with exp_col1:
    if not df_det.empty:
        csv_buf = io.StringIO()
        df_det.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Export All Detections (CSV)",
            csv_buf.getvalue().encode(),
            "all_detections.csv", "text/csv",
            use_container_width=True,
        )
    else:
        st.button("⬇️ Export All Detections (CSV)", disabled=True, use_container_width=True)

with exp_col2:
    if not df_det.empty:
        st.download_button(
            "⬇️ Export All Detections (JSON)",
            json.dumps(detections, indent=2).encode(),
            "all_detections.json", "application/json",
            use_container_width=True,
        )
    else:
        st.button("⬇️ Export All Detections (JSON)", disabled=True, use_container_width=True)

with exp_col3:
    with st.popover("🗑️ Clear All Data", use_container_width=True):
        st.warning("This will permanently delete **all** detection logs and session records.")
        if st.button("⚠️ Yes, delete everything", type="primary", use_container_width=True):
            clear_all_data()
            st.toast("All data cleared.", icon="🗑️")
            st.rerun()
