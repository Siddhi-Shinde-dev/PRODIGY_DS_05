import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RTA Severity Analyser", page_icon="🚦", layout="wide")
 
# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {
    background: #1a1d27; border: 1px solid #2a2d3e;
    border-radius: 12px; padding: 20px 24px; text-align: center; margin-bottom: 6px;
}
.metric-card .label { font-size: 12px; color: #8b8fa8; text-transform: uppercase; letter-spacing:.07em; margin-bottom:6px; }
.metric-card .value { font-size: 30px; font-weight: 700; color: #ffffff; }
.metric-card .sub   { font-size: 12px; color: #555870; margin-top: 4px; }
.section-title { font-size: 17px; font-weight: 700; color: #e2e4f0; margin-bottom: 3px; }
.section-sub   { font-size: 13px; color: #6b6f87; margin-bottom: 18px; }
hr { border-color: #2a2d3e; margin: 28px 0; }
</style>
""", unsafe_allow_html=True)
 
# ── Palette ───────────────────────────────────────────────────────────────────
SEV_COLOR = {"Slight Injury": "#22c55e", "Serious Injury": "#f59e0b", "Fatal injury": "#ef4444"}
BG, TEXT, GRID = "#1a1d27", "#e2e4f0", "#2a2d3e"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": GRID, "grid.linewidth": 0.5,
})
 
# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    drop_cols = ["Service_year_of_vehicle","Defect_of_vehicle","Work_of_casuality","Fitness_of_casuality","Time"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].fillna(df[c].mode()[0])
    df["Accident_severity"] = df["Accident_severity"].str.strip()
    return df
 
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🚦 Road Traffic Accident Severity Analyser")
st.markdown('<p style="color:#6b6f87;margin-top:-10px;">Explore patterns across 12,316 accident records</p>', unsafe_allow_html=True)
st.markdown("---")
 
uploaded = st.file_uploader("Upload `RTA_Dataset.csv` to get started", type="csv")
if not uploaded:
    st.info("⬆️ Please upload the `RTA_Dataset.csv` file to continue.")
    st.stop()
try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Could not load data: {e}"); st.stop()
 
# ── KPIs ──────────────────────────────────────────────────────────────────────
total   = len(df)
fatal   = int((df["Accident_severity"] == "Fatal injury").sum())
serious = int((df["Accident_severity"] == "Serious Injury").sum())
slight  = int((df["Accident_severity"] == "Slight Injury").sum())
 
c1, c2, c3, c4 = st.columns(4)
for col, lbl, val, sub in [
    (c1, "Total Records",  f"{total:,}",   "after cleaning"),
    (c2, "Slight Injury",  f"{slight:,}",  f"{slight/total*100:.1f}% of accidents"),
    (c3, "Serious Injury", f"{serious:,}", f"{serious/total*100:.1f}% of accidents"),
    (c4, "Fatal Injury",   f"{fatal:,}",   f"{fatal/total*100:.1f}% of accidents"),
]:
    col.markdown(f"""<div class="metric-card">
        <div class="label">{lbl}</div><div class="value">{val}</div><div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)
 
st.markdown("---")
 
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🌦️ Weather & Light", "🛣️ Road Factors", "🚗 Driver Profile", "📄 Raw Data"])
 
# ════ TAB 1 — Overview ════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Accident Severity Distribution</div>', unsafe_allow_html=True)
    sev_counts = df["Accident_severity"].value_counts()
    col_l, col_r = st.columns([1.3, 1])
 
    with col_l:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(sev_counts.index, sev_counts.values,
                      color=[SEV_COLOR.get(s, "#888") for s in sev_counts.index], width=0.5, zorder=3)
        ax.yaxis.grid(True, zorder=0); ax.set_ylabel("Count")
        ax.set_title("Severity Classes", fontweight="bold", pad=10)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+50, f"{h:,}", ha="center", fontsize=10, color=TEXT)
        fig.tight_layout(); st.pyplot(fig); plt.close()
 
    with col_r:
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
        wedges, texts, autos = ax2.pie(
            sev_counts.values, labels=sev_counts.index, autopct="%1.1f%%",
            colors=[SEV_COLOR.get(s, "#888") for s in sev_counts.index],
            startangle=140, wedgeprops=dict(width=0.55))
        for t in texts:  t.set_color(TEXT)
        for t in autos:  t.set_color("#111"); t.set_fontsize(10)
        ax2.set_title("Share", fontweight="bold", pad=10)
        fig2.tight_layout(); st.pyplot(fig2); plt.close()
 
    st.markdown("---")
    st.markdown('<div class="section-title">Accidents by Day of Week</div>', unsafe_allow_html=True)
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_df = df.groupby(["Day_of_week","Accident_severity"]).size().unstack(fill_value=0)
    day_df = day_df.reindex([d for d in day_order if d in day_df.index])
    fig3, ax3 = plt.subplots(figsize=(9, 3.8))
    bottom = np.zeros(len(day_df))
    for sev in ["Slight Injury","Serious Injury","Fatal injury"]:
        if sev in day_df.columns:
            ax3.bar(day_df.index, day_df[sev], bottom=bottom, label=sev,
                    color=SEV_COLOR[sev], width=0.6, zorder=3)
            bottom += day_df[sev].values
    ax3.yaxis.grid(True, zorder=0); ax3.set_ylabel("Accidents")
    ax3.set_title("Severity by Day of Week", fontweight="bold", pad=10)
    patches = [mpatches.Patch(color=SEV_COLOR[s], label=s) for s in SEV_COLOR]
    ax3.legend(handles=patches, loc="upper right", framealpha=0.2)
    fig3.tight_layout(); st.pyplot(fig3); plt.close()
 
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Vehicles Involved</div>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        vc = df["Number_of_vehicles_involved"].value_counts().sort_index()
        ax4.bar(vc.index.astype(str), vc.values, color="#3b82f6", width=0.5, zorder=3)
        ax4.yaxis.grid(True, zorder=0); ax4.set_xlabel("Vehicles"); ax4.set_ylabel("Count")
        ax4.set_title("No. of Vehicles Involved", fontweight="bold", pad=8)
        fig4.tight_layout(); st.pyplot(fig4); plt.close()
 
    with col_b:
        st.markdown('<div class="section-title">Casualties per Accident</div>', unsafe_allow_html=True)
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        cc = df["Number_of_casualties"].value_counts().sort_index()
        ax5.bar(cc.index.astype(str), cc.values, color="#a855f7", width=0.5, zorder=3)
        ax5.yaxis.grid(True, zorder=0); ax5.set_xlabel("Casualties"); ax5.set_ylabel("Count")
        ax5.set_title("No. of Casualties", fontweight="bold", pad=8)
        fig5.tight_layout(); st.pyplot(fig5); plt.close()
 
# ════ TAB 2 — Weather & Light ═════════════════════════════════════════════════
with tab2:
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown('<div class="section-title">Weather Conditions</div>', unsafe_allow_html=True)
        wc = df["Weather_conditions"].value_counts()
        fig6, ax6 = plt.subplots(figsize=(5.5, 4))
        ax6.barh(wc.index[::-1], wc.values[::-1], color="#3b82f6", height=0.6, zorder=3)
        ax6.xaxis.grid(True, zorder=0); ax6.set_title("Accidents by Weather", fontweight="bold", pad=8)
        for i, v in enumerate(wc.values[::-1]):
            ax6.text(v+10, i, str(v), va="center", fontsize=9)
        fig6.tight_layout(); st.pyplot(fig6); plt.close()
 
    with col_r2:
        st.markdown('<div class="section-title">Light Conditions</div>', unsafe_allow_html=True)
        lc = df["Light_conditions"].value_counts()
        fig7, ax7 = plt.subplots(figsize=(5.5, 4))
        wedges, texts, autos = ax7.pie(
            lc.values, labels=lc.index, autopct="%1.1f%%",
            colors=["#f59e0b","#1e40af","#6b7280","#0ea5e9"],
            startangle=140, wedgeprops=dict(width=0.6))
        for t in texts:  t.set_color(TEXT); t.set_fontsize(9)
        for t in autos:  t.set_color("#111"); t.set_fontsize(9)
        ax7.set_title("Light Conditions", fontweight="bold", pad=8)
        fig7.tight_layout(); st.pyplot(fig7); plt.close()
 
    st.markdown("---")
    st.markdown('<div class="section-title">Severity vs Weather Conditions</div>', unsafe_allow_html=True)
    weather_sev = df.groupby(["Weather_conditions","Accident_severity"]).size().unstack(fill_value=0)
    fig8, ax8 = plt.subplots(figsize=(11, 4))
    bottom8 = np.zeros(len(weather_sev))
    for sev in ["Slight Injury","Serious Injury","Fatal injury"]:
        if sev in weather_sev.columns:
            ax8.bar(weather_sev.index, weather_sev[sev], bottom=bottom8,
                    label=sev, color=SEV_COLOR[sev], width=0.6, zorder=3)
            bottom8 += weather_sev[sev].values
    ax8.yaxis.grid(True, zorder=0); ax8.set_ylabel("Count")
    ax8.tick_params(axis="x", rotation=30)
    ax8.set_title("Severity by Weather", fontweight="bold", pad=10)
    patches = [mpatches.Patch(color=SEV_COLOR[s], label=s) for s in SEV_COLOR]
    ax8.legend(handles=patches, loc="upper right", framealpha=0.2)
    fig8.tight_layout(); st.pyplot(fig8); plt.close()
 
# ════ TAB 3 — Road Factors ════════════════════════════════════════════════════
with tab3:
    col_l3, col_r3 = st.columns(2)
    with col_l3:
        st.markdown('<div class="section-title">Road Surface Type</div>', unsafe_allow_html=True)
        rs = df["Road_surface_type"].value_counts()
        fig9, ax9 = plt.subplots(figsize=(5.5, 3.5))
        ax9.barh(rs.index[::-1], rs.values[::-1], color="#22c55e", height=0.55, zorder=3)
        ax9.xaxis.grid(True, zorder=0); ax9.set_title("Road Surface", fontweight="bold", pad=8)
        fig9.tight_layout(); st.pyplot(fig9); plt.close()
 
    with col_r3:
        st.markdown('<div class="section-title">Types of Junction</div>', unsafe_allow_html=True)
        tj = df["Types_of_Junction"].value_counts()
        fig10, ax10 = plt.subplots(figsize=(5.5, 3.5))
        ax10.barh(tj.index[::-1], tj.values[::-1], color="#f59e0b", height=0.55, zorder=3)
        ax10.xaxis.grid(True, zorder=0); ax10.set_title("Junction Type", fontweight="bold", pad=8)
        fig10.tight_layout(); st.pyplot(fig10); plt.close()
 
    st.markdown("---")
    st.markdown('<div class="section-title">Top 10 Causes of Accidents</div>', unsafe_allow_html=True)
    causes = df["Cause_of_accident"].value_counts().head(10)
    fig11, ax11 = plt.subplots(figsize=(10, 4))
    ax11.barh(causes.index[::-1], causes.values[::-1], color="#ef4444", height=0.6, zorder=3)
    ax11.xaxis.grid(True, zorder=0); ax11.set_title("Top 10 Accident Causes", fontweight="bold", pad=10)
    for i, v in enumerate(causes.values[::-1]):
        ax11.text(v+10, i, str(v), va="center", fontsize=9)
    fig11.tight_layout(); st.pyplot(fig11); plt.close()
 
# ════ TAB 4 — Driver Profile ══════════════════════════════════════════════════
with tab4:
    col_l4, col_r4 = st.columns(2)
    with col_l4:
        st.markdown('<div class="section-title">Driver Age Band</div>', unsafe_allow_html=True)
        age_order = ["Under 18","18-30","31-50","Over 51","Unknown"]
        age_sev = df.groupby(["Age_band_of_driver","Accident_severity"]).size().unstack(fill_value=0)
        age_sev = age_sev.reindex([a for a in age_order if a in age_sev.index])
        fig12, ax12 = plt.subplots(figsize=(5.5, 4))
        bottom12 = np.zeros(len(age_sev))
        for sev in ["Slight Injury","Serious Injury","Fatal injury"]:
            if sev in age_sev.columns:
                ax12.bar(age_sev.index, age_sev[sev], bottom=bottom12,
                         color=SEV_COLOR[sev], width=0.55, zorder=3)
                bottom12 += age_sev[sev].values
        ax12.yaxis.grid(True, zorder=0); ax12.set_ylabel("Count")
        ax12.set_title("Age Band vs Severity", fontweight="bold", pad=8)
        patches = [mpatches.Patch(color=SEV_COLOR[s], label=s) for s in SEV_COLOR]
        ax12.legend(handles=patches, fontsize=8, framealpha=0.2)
        fig12.tight_layout(); st.pyplot(fig12); plt.close()
 
    with col_r4:
        st.markdown('<div class="section-title">Driving Experience</div>', unsafe_allow_html=True)
        exp_order = ["Below 1yr","1-2yr","2-5yr","5-10yr","Above 10yr","No Licence","unknown"]
        exp_sev = df.groupby(["Driving_experience","Accident_severity"]).size().unstack(fill_value=0)
        exp_sev = exp_sev.reindex([e for e in exp_order if e in exp_sev.index])
        fig13, ax13 = plt.subplots(figsize=(5.5, 4))
        bottom13 = np.zeros(len(exp_sev))
        for sev in ["Slight Injury","Serious Injury","Fatal injury"]:
            if sev in exp_sev.columns:
                ax13.bar(exp_sev.index, exp_sev[sev], bottom=bottom13,
                         color=SEV_COLOR[sev], width=0.55, zorder=3)
                bottom13 += exp_sev[sev].values
        ax13.yaxis.grid(True, zorder=0); ax13.set_ylabel("Count")
        ax13.set_title("Driving Experience vs Severity", fontweight="bold", pad=8)
        ax13.tick_params(axis="x", rotation=30)
        patches = [mpatches.Patch(color=SEV_COLOR[s], label=s) for s in SEV_COLOR]
        ax13.legend(handles=patches, fontsize=8, framealpha=0.2)
        fig13.tight_layout(); st.pyplot(fig13); plt.close()
 
    st.markdown("---")
    st.markdown('<div class="section-title">Correlation Heatmap (Numerical Features)</div>', unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig14, ax14 = plt.subplots(figsize=(6, 3.5))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, ax=ax14, cbar_kws={"shrink": 0.8})
    ax14.set_title("Numerical Feature Correlation", fontweight="bold", pad=10)
    fig14.tight_layout(); st.pyplot(fig14); plt.close()
 
# ════ TAB 5 — Raw Data ════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Dataset Explorer</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    f_sev     = f1.selectbox("Severity",    ["All"] + sorted(df["Accident_severity"].unique()))
    f_weather = f2.selectbox("Weather",     ["All"] + sorted(df["Weather_conditions"].dropna().unique()))
    f_day     = f3.selectbox("Day of Week", ["All"] + ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
 
    view = df.copy()
    if f_sev     != "All": view = view[view["Accident_severity"]  == f_sev]
    if f_weather != "All": view = view[view["Weather_conditions"] == f_weather]
    if f_day     != "All": view = view[view["Day_of_week"]        == f_day]
 
    st.markdown(f"**{len(view):,} rows** matching filters")
    show_cols = ["Day_of_week","Age_band_of_driver","Driving_experience","Weather_conditions",
                 "Light_conditions","Cause_of_accident","Number_of_vehicles_involved",
                 "Number_of_casualties","Accident_severity"]
    st.dataframe(view[[c for c in show_cols if c in view.columns]].reset_index(drop=True),
                 width='stretch', height=430)
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p style="text-align:center;color:#555870;font-size:12px;">Built by Siddhi Shinde · PRODIGY_DS_05 · Road Traffic Accident Severity Analysis</p>', unsafe_allow_html=True)