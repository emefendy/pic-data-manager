"""Real-data PIC dashboard — Tidy3D simulations + ScreeningDatasetV2 lab data.

Two clearly separated domains, each with the analytical features (overview,
records, parameter/process analysis, anomaly insights, predictive model) applied
to REAL data only — no synthetic/seeded data.

Run:  streamlit run app_real.py
The screening store is built on first use from the raw dataset
(~/Downloads/ScreeningDatasetV2_candidate, or set SCREENING_DATASET).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import dashboard_data as dd  # noqa: E402

st.set_page_config(page_title="QuEra · Real PIC Data", layout="wide",
                   initial_sidebar_state="collapsed")

# ── Theme (matches app.py) ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root{--bg:#fff;--surface:#f8f7ff;--border:#e2dff5;--purple:#7c3aed;--violet:#a855f7;
--pink:#ec4899;--orange:#f97316;--teal:#06b6d4;--text:#1a1040;--muted:#7168a8;}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;
background-image:radial-gradient(circle,#d4cff0 1.2px,transparent 1.2px)!important;
background-size:22px 22px!important;color:var(--text)!important;font-family:'Inter',sans-serif!important;}
[data-testid="stHeader"]{background:rgba(255,255,255,.85)!important;backdrop-filter:blur(8px)!important;}
[data-testid="block-container"]{padding-top:2rem!important;} footer{visibility:hidden;}
[data-testid="stTabs"] button{color:var(--muted)!important;font-weight:500!important;font-size:.85rem!important;
border-bottom:2px solid transparent!important;background:transparent!important;padding:.5rem 1.1rem!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--purple)!important;border-bottom:2px solid var(--purple)!important;}
[data-baseweb="tab-highlight"]{display:none!important;}
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:rgba(255,255,255,.85)!important;backdrop-filter:blur(6px)!important;
border-radius:10px!important;padding:4px 12px!important;gap:8px!important;border:1px solid var(--border)!important;}
[data-testid="metric-container"]{background:rgba(255,255,255,.88)!important;border:1px solid var(--border)!important;
border-radius:12px!important;padding:1.2rem 1.4rem!important;box-shadow:0 2px 12px rgba(124,58,237,.07)!important;}
[data-testid="metric-container"] label{color:var(--muted)!important;font-size:.78rem!important;text-transform:uppercase;letter-spacing:.08em;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--text)!important;font-size:1.7rem!important;font-weight:700!important;}
[data-testid="stAlert"]{background:rgba(255,255,255,.88)!important;border:1px solid var(--border)!important;
border-left:3px solid var(--teal)!important;border-radius:10px!important;color:var(--text)!important;}
.pic-table-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:10px;background:#fff;margin-bottom:.5rem;}
.pic-table{width:100%;border-collapse:collapse;font-size:.82rem;color:#1a1040;background:#fff;}
.pic-table thead th{background:#f3f0ff;font-weight:600;padding:.55rem .9rem;text-align:left;border-bottom:2px solid #e2dff5;white-space:nowrap;}
.pic-table tbody td{padding:.45rem .9rem;white-space:nowrap;} .pic-table tbody tr{border-bottom:1px solid #f0eef9;}
.pic-table tbody tr:hover{background:#faf9ff;}
hr{border-color:var(--border)!important;}
::-webkit-scrollbar{width:6px;height:6px;}::-webkit-scrollbar-thumb{background:#d4cff0;border-radius:3px;}
.src-pill{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-size:.72rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.85)",
    font=dict(family="Inter", color="#1a1040"),
    title_font=dict(family="Inter", size=15, color="#1a1040"),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2dff5", borderwidth=1),
    margin=dict(t=50, l=10, r=10, b=10),
    xaxis=dict(gridcolor="#e2dff5", linecolor="#e2dff5"),
    yaxis=dict(gridcolor="#e2dff5", linecolor="#e2dff5"),
)
PALETTE = ["#7c3aed", "#06b6d4", "#ec4899", "#f97316", "#10b981", "#a855f7"]

try:  # OLS trendlines need statsmodels; degrade gracefully if absent
    import statsmodels.api as _sm  # noqa: F401
    TRENDLINE = "ols"
except Exception:
    TRENDLINE = None


def style_fig(fig):
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def section(title, subtitle=""):
    st.markdown(f"""<div style="margin:.4rem 0 1.1rem;">
      <h2 style="font-size:1.4rem;font-weight:700;margin:0;
      background:linear-gradient(90deg,#a855f7,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{title}</h2>
      {"<p style='color:#9591c4;font-size:.88rem;margin:.2rem 0 0;'>"+subtitle+"</p>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)


def render_table(df, max_rows=500):
    html = df.head(max_rows).to_html(index=False, classes="pic-table", border=0)
    st.markdown(f'<div class="pic-table-wrap">{html}</div>', unsafe_allow_html=True)
    if len(df) > max_rows:
        st.caption(f"Showing {max_rows} of {len(df)} rows")


def corr_bar(corr: pd.Series, title: str):
    if corr.empty:
        st.info("Not enough variation to compute correlations.")
        return
    c = corr.iloc[::-1]
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h",
                           marker_color=["#ec4899" if v < 0 else "#7c3aed" for v in c.values]))
    fig.update_layout(title=title, xaxis_title="Pearson r", height=max(240, 40*len(c)))
    st.plotly_chart(style_fig(fig), use_container_width=True)


def model_panel(X, y, target_label):
    res = dd.fit_predictive_model(X, y)
    if not res["ok"]:
        st.info(f"Predictive model skipped — {res['reason']}.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Cross-val R²", f"{res['r2']:.3f}")
    c2.metric("MAE", f"{res['mae']:.4f}")
    c3.metric("Samples", res["n"])
    a, b = st.columns(2)
    with a:
        lo, hi = float(min(res["y_true"].min(), res["y_pred"].min())), float(max(res["y_true"].max(), res["y_pred"].max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["y_true"], y=res["y_pred"], mode="markers",
                                 marker=dict(color="#7c3aed", size=8, opacity=0.7), name="designs"))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color="#ec4899", dash="dash"), name="ideal"))
        fig.update_layout(title=f"Predicted vs actual {target_label}",
                          xaxis_title=f"actual {target_label}", yaxis_title="predicted")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with b:
        imp = res["importances"].head(8).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#06b6d4"))
        fig.update_layout(title="Feature importance", xaxis_title="importance",
                          height=max(240, 40*len(imp)))
        st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption("Model: 5-fold cross-validated RandomForest. Honest small-sample estimate; "
               "high R² reflects that geometry strongly determines the simulated/measured metric.")


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:.2rem;"><span style="font-size:1.05rem;font-weight:600;color:#9591c4;letter-spacing:.04em;">QuEra · PIC Data Manager</span></div>
<h1 style="font-size:2.5rem;font-weight:700;margin:.2rem 0 .1rem;
background:linear-gradient(90deg,#f97316 0%,#ec4899 40%,#a855f7 100%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.15;">Real PIC Data Platform</h1>
<p style="color:#9591c4;font-size:.95rem;margin:0 0 1.4rem;">Real Tidy3D simulations · Real ScreeningDatasetV2 lab data · no synthetic data</p>
""", unsafe_allow_html=True)

domain = st.radio("Dataset", ["🔷 Tidy3D PIC Simulations", "🔬 ScreeningDatasetV2 — Lab Data"],
                  horizontal=True, label_visibility="collapsed")
st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)


# ============================================================================ #
# TIDY3D DOMAIN
# ============================================================================ #
@st.cache_data(show_spinner=False)
def _tidy3d_designs():
    return dd.load_tidy3d_designs()


def render_tidy3d():
    if not dd.DB_PATH.exists():
        st.error("pic_data.db not found in this folder.")
        return
    df = _tidy3d_designs()
    if df.empty:
        st.warning("No real Tidy3D designs found (designer in 'sweep'/'simulation_import').")
        return
    feats = dd.tidy3d_geometry_features(df)

    st.markdown('<span class="src-pill" style="background:#ede9fe;color:#7c3aed;">REAL FDTD · Tidy3D</span>',
                unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Real designs", len(df))
    k2.metric("Component types", df["device_type"].nunique())
    k3.metric("Mean insertion loss", f"{df['simulated_loss_db'].mean():.3f} dB")
    k4.metric("Best loss", f"{df['simulated_loss_db'].min():.3f} dB")

    t_ov, t_lib, t_spec, t_par, t_anom, t_pred = st.tabs(
        ["  Overview  ", "  Design Library  ", "  Spectra  ",
         "  Parameter Analysis  ", "  Anomaly Insights  ", "  Predictive Model  "])

    with t_ov:
        section("Tidy3D simulation overview", "Real FDTD sweeps of Y-junctions and Euler bends")
        c1, c2 = st.columns(2)
        with c1:
            counts = df["device_type"].value_counts().reset_index()
            counts.columns = ["device_type", "n"]
            fig = px.bar(counts, x="device_type", y="n", color="device_type",
                         color_discrete_sequence=PALETTE, title="Designs per component type")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            fig = px.box(df, x="device_type", y="simulated_loss_db", color="device_type",
                         color_discrete_sequence=PALETTE, title="Insertion loss by component type",
                         points="all")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        st.caption("Source: real Tidy3D FDTD runs in pic_data.db (designer = sweep / simulation_import). "
                   "Synthetic seed designs are excluded.")

    with t_lib:
        section("Design library", "Filterable table of real designs + geometry")
        types = st.multiselect("Component type", sorted(df["device_type"].unique()),
                               default=sorted(df["device_type"].unique()))
        view_cols = ["design_id", "device_type", "designer", "simulated_loss_db"] + feats
        sub = df[df["device_type"].isin(types)][[c for c in view_cols if c in df.columns]]
        render_table(sub.round(4))

    with t_spec:
        section("Transmission / insertion-loss spectra", "Read from data/spectra/*.json")
        did = st.selectbox("Design", df["design_id"].tolist())
        spec = dd.load_spectrum(did)
        if spec is None:
            st.info("No spectrum file for this design.")
        else:
            wl, curves = dd.spectrum_loss_curves(spec)
            if wl is not None and curves:
                fig = go.Figure()
                for i, (name, arr) in enumerate(curves.items()):
                    fig.add_trace(go.Scatter(x=wl, y=arr, mode="lines", name=name,
                                             line=dict(color=PALETTE[i % len(PALETTE)])))
                fig.update_layout(title=f"Insertion loss vs wavelength — {did}",
                                  xaxis_title="Wavelength (µm)", yaxis_title="Loss (dB)")
                st.plotly_chart(style_fig(fig), use_container_width=True)
            else:
                st.json({k: f"<{len(v)} pts>" if isinstance(v, list) else v for k, v in spec.items()})

    with t_par:
        section("Parameter analysis", "How geometry drives simulated insertion loss")
        corr_bar(dd.correlations(df, feats, "simulated_loss_db"),
                 "Geometry ↔ insertion-loss correlation (Pearson r)")
        xcol = st.selectbox("Geometry parameter", feats,
                            index=feats.index("geom_l_junction") if "geom_l_junction" in feats else 0)
        fig = px.scatter(df, x=xcol, y="simulated_loss_db", color="device_type",
                         color_discrete_sequence=PALETTE, trendline=TRENDLINE,
                         title=f"Insertion loss vs {xcol}")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with t_anom:
        section("Anomaly insights", "Designs with unusual geometry→loss behavior (IsolationForest)")
        X = dd.build_feature_matrix(df, feats + ["simulated_loss_db"], categorical=["device_type"])
        lab, score = dd.anomaly_flags(X, contamination=0.12)
        dfa = df.assign(anomaly=np.where(lab == 1, "anomaly", "normal"), anomaly_score=score.round(3))
        fig = px.scatter(dfa, x="geom_l_junction" if "geom_l_junction" in dfa else feats[0],
                         y="simulated_loss_db", color="anomaly",
                         color_discrete_map={"anomaly": "#ec4899", "normal": "#7c3aed"},
                         hover_data=["design_id", "device_type", "anomaly_score"],
                         title="Flagged designs")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.markdown(f"**{int(lab.sum())}** of {len(lab)} designs flagged as anomalous.")
        render_table(dfa[dfa.anomaly == "anomaly"][["design_id", "device_type",
                     "simulated_loss_db", "anomaly_score"] + feats].round(4))

    with t_pred:
        section("Predictive model", "Predict simulated insertion loss from geometry")
        X = dd.build_feature_matrix(df, feats, categorical=["device_type"])
        model_panel(X, df["simulated_loss_db"], "loss (dB)")


# ============================================================================ #
# SCREENING DOMAIN
# ============================================================================ #
@st.cache_data(show_spinner=False)
def _screening_frames(root_str, token):
    """Load all screening DataFrames once (picklable), keyed by a freshness token."""
    s = dd.open_screening_store(root=Path(root_str))
    try:
        out = {
            "features": s.feature_table(),
            "metrics": s.device_metrics_summary(),
            "logs": s.lab_logs(),
            "captures": s.captures(),
            "devices": s.devices(),
        }
        ms = s.measurements()
        for kind in ("static_sweep", "fra"):
            row = ms[ms["kind"] == kind]
            out[kind] = s.read_table(row.iloc[0]["measurement_id"]) if len(row) else pd.DataFrame()
    finally:
        s.close()
    return out


def render_screening():
    root = dd.SCREEN_ROOT
    built = (root / "screening.duckdb").exists()
    if not built:
        st.warning("The ScreeningDatasetV2 store hasn't been built yet.")
        ds = st.text_input("Path to ScreeningDatasetV2_candidate folder",
                           value=str(Path.home() / "Downloads" / "ScreeningDatasetV2_candidate"))
        if st.button("⚙️ Build store from dataset"):
            with st.spinner("Ingesting static sweep, FRA, 100 captures, lab logs…"):
                _, status = dd.ensure_screening_store(dataset=ds, root=root)
            if status.startswith("error"):
                st.error(f"Build failed: {status}")
            else:
                st.success("Store built. Reloading…"); st.rerun()
        return

    token = (root / "screening.duckdb").stat().st_mtime
    F = _screening_frames(str(root), token)
    feats = F["features"]
    metrics = {r["metric_name"]: r["value"] for _, r in F["metrics"].iterrows()}

    st.markdown('<span class="src-pill" style="background:#cffafe;color:#0e7490;">REAL LAB DATA · cascaded MZI</span>',
                unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Captures", len(feats))
    k2.metric("−3 dB BW", f"{metrics.get('fra_bandwidth_3db', float('nan')):.3f} MHz")
    k3.metric("V_null", f"{metrics.get('static_V_null', float('nan')):.2f} V")
    k4.metric("V_π (est)", f"{metrics.get('static_V_pi_est', float('nan')):.2f} V")
    k5.metric("Extinction", f"{metrics.get('static_extinction', float('nan')):.1f} dB")

    t_ov, t_cap, t_char, t_wave, t_par, t_anom, t_pred, t_logs = st.tabs(
        ["  Overview  ", "  Captures  ", "  Characterization  ", "  Waveform  ",
         "  Parameter Analysis  ", "  Anomaly Insights  ", "  Predictive Model  ", "  Lab Logs  "])

    with t_ov:
        section("Device characterization overview", "Cascaded Mach–Zehnder modulator — real bench data")
        render_table(F["metrics"])
        dev = F["devices"]
        if len(dev):
            d = dev.iloc[0]
            st.markdown(f"**Device:** {d['name']} · PD bandwidth ~{d['pd_bandwidth_hz']/1e6:.0f} MHz · "
                        f"{d['n_stages']} stages")
            known = d.get("known_facts") or []
            if isinstance(known, list) and known:
                st.markdown("**Known facts:** " + "; ".join(known[:4]))

    with t_cap:
        section("Capture records", "100 step-pulse captures — filter by acquisition params")
        c1, c2 = st.columns(2)
        reps = sorted(feats["rep_hz"].dropna().unique())
        vlbls = sorted(feats["vlbl"].dropna().unique())
        sel_rep = c1.multiselect("Rep rate (Hz)", reps, default=reps)
        sel_v = c2.multiselect("Drive variant", vlbls, default=vlbls)
        sub = feats[feats["rep_hz"].isin(sel_rep) & feats["vlbl"].isin(sel_v)]
        render_table(sub.round(3))

    with t_char:
        section("Static fringe & small-signal Bode", "From the stored Parquet tables")
        c1, c2 = st.columns(2)
        with c1:
            s = F["static_sweep"]
            if len(s):
                fig = go.Figure(go.Scatter(x=s["V_DC_volts"], y=s["PD_mean_mV"], mode="lines",
                                           line=dict(color="#7c3aed")))
                if "static_V_null" in metrics:
                    fig.add_vline(x=metrics["static_V_null"], line=dict(color="#ec4899", dash="dash"))
                fig.update_layout(title="Static fringe", xaxis_title="V_DC (V)", yaxis_title="PD (mV)")
                st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            fr = F["fra"]
            if len(fr):
                fig = go.Figure(go.Scatter(x=fr["freq_hz"], y=fr["H_mag_db"], mode="lines",
                                           line=dict(color="#06b6d4")))
                if metrics.get("fra_bandwidth_3db"):
                    fig.add_vline(x=metrics["fra_bandwidth_3db"]*1e6, line=dict(color="#ec4899", dash="dash"))
                fig.update_layout(title="Small-signal Bode |H|", xaxis_title="Freq (Hz)",
                                  yaxis_title="|H| (dB)", xaxis_type="log")
                st.plotly_chart(style_fig(fig), use_container_width=True)

    with t_wave:
        section("Capture waveform viewer", "Trace-averaged PD readout (read from Zarr)")
        tag = st.selectbox("Capture", feats["tag"].tolist())
        try:
            s = dd.open_screening_store(root=root)
            arrays, attrs = s.read_waveform(tag); s.close()
            t_us = np.asarray(arrays["time"]) * 1e6
            mean = np.asarray(arrays["ch1_mean"]) * 1e3
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_us, y=mean, mode="lines", name="PD (ch1 mean)",
                                     line=dict(color="#7c3aed")))
            if "ch1_std" in arrays:
                sd = np.asarray(arrays["ch1_std"]) * 1e3
                fig.add_trace(go.Scatter(x=np.concatenate([t_us, t_us[::-1]]),
                                         y=np.concatenate([mean+sd, (mean-sd)[::-1]]),
                                         fill="toself", fillcolor="rgba(124,58,237,0.15)",
                                         line=dict(color="rgba(0,0,0,0)"), name="±1σ"))
            fig.update_layout(title=f"Step capture {tag}", xaxis_title="Time (µs)", yaxis_title="PD (mV)")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        except Exception as e:
            st.info(f"Waveform not available ({e}). Rebuild the store with full traces if needed.")

    with t_par:
        section("Parameter analysis", "How drive parameters affect the step response")
        param_feats = ["rep_hz", "edge_ns", "V_pp"] if "V_pp" in feats else ["rep_hz", "edge_ns"]
        param_feats = [c for c in param_feats if c in feats.columns]
        target = st.selectbox("Response metric", [c for c in feats.columns if c.startswith("cap_")],
                              index=0)
        corr_bar(dd.correlations(feats, param_feats, target), f"Drive params ↔ {target}")
        xcol = st.selectbox("X axis", param_feats)
        fig = px.scatter(feats, x=xcol, y=target, color="vlbl",
                         color_discrete_sequence=PALETTE, title=f"{target} vs {xcol}",
                         log_x=(xcol == "rep_hz"))
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with t_anom:
        section("Anomaly insights", "Captures with unusual step features (IsolationForest)")
        fcols = [c for c in feats.columns if c.startswith("cap_")]
        X = dd.build_feature_matrix(feats, fcols)
        lab, score = dd.anomaly_flags(X, contamination=0.1)
        fa = feats.assign(anomaly=np.where(lab == 1, "anomaly", "normal"), anomaly_score=score.round(3))
        fig = px.scatter(fa, x="rep_hz", y="cap_swing_mV", color="anomaly",
                         color_discrete_map={"anomaly": "#ec4899", "normal": "#06b6d4"},
                         hover_data=["tag", "vlbl", "anomaly_score"], log_x=True,
                         title="Flagged captures")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.markdown(f"**{int(lab.sum())}** of {len(lab)} captures flagged. "
                    "Cross-reference the Lab Logs tab — the 2026-04-23 note reports a thermal-drift event.")
        render_table(fa[fa.anomaly == "anomaly"][["tag", "rep_hz", "vlbl", "cap_swing_mV",
                     "cap_rise_time_ns", "anomaly_score"]].round(3))

    with t_pred:
        section("Predictive model", "Predict PD swing from drive parameters")
        param_feats = [c for c in ["rep_hz", "edge_ns", "V_pp"] if c in feats.columns]
        X = dd.build_feature_matrix(feats, param_feats, categorical=["vlbl"])
        model_panel(X, feats["cap_swing_mV"], "swing (mV)")

    with t_logs:
        section("Operator lab logs", "The RAG seam — real bench notes")
        for _, r in F["logs"].iterrows():
            with st.expander(f"📋 {r['date']} — {r['title']}"):
                st.markdown(r["body"])
                if isinstance(r.get("referenced_tags"), list) and r["referenced_tags"]:
                    st.caption("References: " + ", ".join(r["referenced_tags"]))


# ── Route ───────────────────────────────────────────────────────────────────
if domain.startswith("🔷"):
    render_tidy3d()
else:
    render_screening()
