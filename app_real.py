
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import dashboard_data as dd

st.set_page_config(page_title="QuEra · Real PIC Data", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root{--bg:#fff;--surface:#f8f7ff;--border:#e2dff5;--purple:#7c3aed;--teal:#06b6d4;
--pink:#ec4899;--orange:#f97316;--text:#1a1040;--muted:#7168a8;}
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
.src-pill{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-size:.72rem;font-weight:600;margin-right:.3rem;}
.stat-row{display:flex;gap:12px;flex-wrap:wrap;margin:.3rem 0 .9rem;}
.stat-card{flex:1 1 0;min-width:140px;background:rgba(255,255,255,.88);border:1px solid var(--border);
border-radius:12px;padding:1rem 1.2rem;box-shadow:0 2px 12px rgba(124,58,237,.07);}
.stat-label{color:var(--muted);font-size:.74rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem;}
.stat-value{color:var(--text);font-size:1.25rem;font-weight:700;line-height:1.25;overflow-wrap:anywhere;}
.pic-table tbody tr.hl-row{background:#fef3c7!important;}
.pic-table tbody tr.hl-row:hover{background:#fde9a8!important;}
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

import importlib.util

TRENDLINE = "ols" if importlib.util.find_spec("statsmodels") else None

_FMT_COLORS = {"DuckDB": ("#ede9fe", "#7c3aed"), "Parquet": ("#cffafe", "#0e7490"),
               "Zarr": ("#fef3c7", "#b45309"), "SQLite": ("#e2e8f0", "#475569"),
               "JSON": ("#e2e8f0", "#475569"), "HDF5": ("#e2e8f0", "#475569")}


def badge(fmt):
    bg, fg = _FMT_COLORS.get(fmt, ("#eee", "#333"))
    return f'<span class="src-pill" style="background:{bg};color:{fg};">{fmt}</span>'


def style_fig(fig):
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def section(title, subtitle=""):
    st.markdown(f"""<div style="margin:.4rem 0 1.1rem;">
      <h2 style="font-size:1.4rem;font-weight:700;margin:0;
      background:linear-gradient(90deg,#a855f7,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{title}</h2>
      {"<p style='color:#9591c4;font-size:.88rem;margin:.2rem 0 0;'>"+subtitle+"</p>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)


def render_table(df, max_rows=500, highlight_mask=None):
    d = df.head(max_rows)
    if highlight_mask is None:
        html = d.to_html(index=False, classes="pic-table", border=0)
    else:
        hm = highlight_mask.reindex(d.index).fillna(False)
        head = "".join(f"<th>{c}</th>" for c in d.columns)
        rows = []
        for i, (_, r) in enumerate(d.iterrows()):
            cls = ' class="hl-row"' if bool(hm.iloc[i]) else ""
            cells = "".join(f"<td>{v}</td>" for v in r)
            rows.append(f"<tr{cls}>{cells}</tr>")
        html = (f'<table class="pic-table" border="0"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table>')
    st.markdown(f'<div class="pic-table-wrap">{html}</div>', unsafe_allow_html=True)
    if len(df) > max_rows:
        st.caption(f"Showing {max_rows} of {len(df)} rows")


def stat_cards(items):
    cells = "".join(
        f'<div class="stat-card"><div class="stat-label">{lab}</div>'
        f'<div class="stat-value">{val}</div></div>' for lab, val in items)
    st.markdown(f'<div class="stat-row">{cells}</div>', unsafe_allow_html=True)


def model_panel(X, y, target_label, ctx="", ids=None, z_thresh=2.0,
                flag_anomalies=False):
    res = dd.fit_predictive_model(X, y)
    if not res["ok"]:
        st.info(f"Predictive model skipped — {res['reason']}.")
        return
    suffix = f" — {ctx}" if ctx else ""

    resid = res["y_true"] - res["y_pred"]
    z = (resid - resid.mean()) / (resid.std() or 1.0)
    flags = (np.abs(z) >= z_thresh) if flag_anomalies else np.zeros(len(z), dtype=bool)

    cols = st.columns(4 if flag_anomalies else 3)
    cols[0].metric("Cross-val R²", f"{res['r2']:.3f}")
    cols[1].metric("MAE", f"{res['mae']:.4f}")
    cols[2].metric("Samples", res["n"])
    if flag_anomalies:
        cols[3].metric("Anomalies", int(flags.sum()))
    a, b = st.columns(2)
    with a:
        lo = float(min(res["y_true"].min(), res["y_pred"].min()))
        hi = float(max(res["y_true"].max(), res["y_pred"].max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["y_true"][~flags], y=res["y_pred"][~flags],
                                 mode="markers",
                                 marker=dict(color="#7c3aed", size=8, opacity=0.7),
                                 name="normal" if flag_anomalies else "samples"))
        if flags.any():
            fig.add_trace(go.Scatter(x=res["y_true"][flags], y=res["y_pred"][flags],
                                     mode="markers",
                                     marker=dict(color="#ec4899", size=10),
                                     name="anomaly (|z| ≥ 2)"))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color="#94a3b8", dash="dash"), name="ideal"))
        fig.update_layout(title=f"Predicted vs actual {target_label}{suffix}",
                          xaxis_title=f"actual {target_label}", yaxis_title="predicted")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with b:
        imp = res["importances"].head(8).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#06b6d4"))
        fig.update_layout(title=f"Feature importance{suffix}", xaxis_title="importance",
                          height=max(240, 40 * len(imp)))
        st.plotly_chart(style_fig(fig), use_container_width=True)
    note = ("Model: 5-fold cross-validated RandomForest (honest small-sample estimate).")
    if flag_anomalies:
        note += (" Anomalies = samples whose residual against their out-of-fold prediction "
                 "is ≥ 2σ — unusual *for their parameters*, not just unusual values.")
    st.caption(note)
    if flags.any() and ids is not None:
        ids_used = pd.Series(ids).reset_index(drop=True).iloc[res["row_index"]]
        tbl = pd.DataFrame({
            "id": ids_used.to_numpy()[flags],
            f"actual {target_label}": res["y_true"][flags],
            "predicted": res["y_pred"][flags],
            "residual": resid[flags], "z": z[flags],
        }).sort_values("z", key=lambda s: s.abs(), ascending=False)
        st.markdown("**Flagged samples**")
        render_table(tbl.round(4))


def shap_panel(sub, X, y, id_col, label):
    res = dd.shap_attributions(X, y)
    if not res["ok"]:
        st.caption(f"SHAP attribution unavailable — {res['reason']}.")
        return
    sv = res["shap"]; feats = res["features"]; base = res["base"]
    ids = sub.iloc[res["row_index"]][id_col].astype(str).tolist()
    st.markdown(f"**SHAP feature attribution** — how each parameter pushes the predicted "
                f"{label} above/below the baseline (signed, per sample).")
    a, b = st.columns(2)
    with a:
        mean_abs = np.abs(sv).mean(0)
        order = np.argsort(mean_abs)
        fig = go.Figure(go.Bar(x=mean_abs[order], y=[feats[i] for i in order],
                               orientation="h", marker_color="#7c3aed"))
        fig.update_layout(title="Global importance — mean |SHAP|", xaxis_title="mean |SHAP|",
                          height=max(240, 40 * len(feats)))
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with b:
        sel = st.selectbox(f"Per-sample attribution ({label})", ids, key=f"shap_{label}")
        i = ids.index(sel)
        contrib = sv[i]
        order = np.argsort(np.abs(contrib))
        fig = go.Figure(go.Bar(x=contrib[order], y=[feats[j] for j in order], orientation="h",
                               marker_color=["#ec4899" if v < 0 else "#10b981" for v in contrib[order]]))
        pred = base + float(contrib.sum())
        fig.update_layout(title=f"{sel}: baseline {base:.3f} → predicted {pred:.3f}",
                          xaxis_title="SHAP contribution (signed)", height=max(240, 40 * len(feats)))
        st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption("Green pushes the prediction up, pink pushes it down; baseline + contributions = "
               "the model's prediction for that sample. TreeSHAP (exact) on the RandomForest.")


def get_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        import os
        return os.environ.get("ANTHROPIC_API_KEY", "")


st.markdown("""
<div style="margin-bottom:.2rem;"><span style="font-size:1.05rem;font-weight:600;color:#9591c4;letter-spacing:.04em;">QuEra · PIC Data Manager</span></div>
<h1 style="font-size:2.5rem;font-weight:700;margin:.2rem 0 .1rem;
background:linear-gradient(90deg,#f97316 0%,#ec4899 40%,#a855f7 100%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.15;">PIC Data Platform</h1>
<p style="color:#9591c4;font-size:.95rem;margin:0 0 1.4rem;">Tidy3D Simulations · Real ScreeningDatasetV2 Lab Data</p>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="First-time setup: ingesting raw data into the stores…")
def _ensure_stores_once():
    _, t = dd.ensure_tidy3d_pipeline_store()
    _, r = dd.ensure_screening_store()
    return {"Tidy3D": t, "real lab data": r}

for _name, _status in _ensure_stores_once().items():
    if str(_status).startswith("error"):
        st.warning(f"The {_name} store could not be built automatically — {_status}")

DOMAINS = ["🔷 Tidy3D · Y-junction", "🔶 Tidy3D · Euler bend",
           "🔬 ScreeningDatasetV2 — Lab Data", "🗂 Tables & Queries",
           "🧪 Pipeline Tests"]
domain = st.radio("View", DOMAINS, horizontal=True, label_visibility="collapsed")
st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _tidy3d_designs():
    return dd.load_tidy3d_designs()


@st.cache_data(show_spinner=False)
def _component_record(device_type: str):
    return dd.load_component_record(device_type)


@st.cache_data(show_spinner=False)
def _opt_records(device_type: str):
    return dd.load_optimization_records(device_type)


@st.cache_data(show_spinner="Fitting the GP loss landscape…")
def _gp_surface(h_bend: float):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    df = _tidy3d_designs()
    yj = df[df["device_type"] == "y_junction"].dropna(
        subset=["geom_w7", "geom_h_bend", "geom_l_junction", "simulated_loss_db"])
    X = yj[["geom_w7", "geom_h_bend", "geom_l_junction"]].to_numpy(float)
    y = yj["simulated_loss_db"].to_numpy(float)
    kernel = (ConstantKernel(1.0)
              * Matern(length_scale=[0.15, 0.4, 0.4], nu=2.5,
                       length_scale_bounds=[(0.05, 1.0), (0.2, 2.0), (0.25, 2.0)])
              + WhiteKernel(1e-6, noise_level_bounds=(1e-12, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=3, random_state=0)
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    w7g = np.linspace(1.2, 1.6, 35)
    lg = np.linspace(1.5, 3.0, 45)
    W, L = np.meshgrid(w7g, lg)
    P = np.column_stack([W.ravel(), np.full(W.size, h_bend), L.ravel()])
    Z = gp.predict(P).reshape(W.shape)
    pts = yj[(yj["geom_h_bend"] - h_bend).abs() <= 0.25]
    return (w7g, lg, Z,
            pts["geom_w7"].to_numpy(), pts["geom_l_junction"].to_numpy(),
            pts["simulated_loss_db"].to_numpy(),
            pts["geom_h_bend"].to_numpy(), pts["design_id"].to_numpy(),
            pts["design_id"].astype(str).str.startswith("D_YJ_OPT").to_numpy(),
            len(yj))


def _pid_alive(pid):
    import os
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _live_optimizer_panel():
    import subprocess

    log_path = dd.ROOT / "data" / "tidy3d_processed" / "optimize_live.log"
    pid = st.session_state.get("opt_pid")
    running = pid is not None and _pid_alive(pid)

    with st.expander("🚀 Run a live closed-loop study (submits real Tidy3D jobs — "
                     "costs FlexCredits)", expanded=running):
        st.markdown(
            "Launches `scripts/optimize_yjunction.py`: **Bayesian optimization over "
            "all three swept parameters simultaneously** — a Gaussian process is "
            "first fitted to **every existing sweep design** (free seed data), then "
            "expected improvement proposes new (w7, h_bend, l_junction) "
            "combinations, one **real FDTD job per proposal** on your Flexcompute "
            "account. Every result is ingested with full provenance and the study "
            "appears below when done.")
        st.markdown(
            "**Varied (simultaneously):** w7 ∈ [1.2, 1.6] µm · h_bend ∈ [1.5, 2.5] "
            "µm · l_junction ∈ [1.5, 3.0] µm. **Held constant:** w1 = 0.5 µm, "
            "t = 0.22 µm, wavelength band 1.5–1.6 µm.")
        mx = st.number_input("Max new FDTD jobs", value=6, min_value=1, max_value=12,
                             step=1, key="opt_mx")
        sure = st.checkbox("I understand this submits real jobs to my Flexcompute "
                           "account and costs FlexCredits", key="opt_sure")
        if st.button("▶ Launch study", disabled=(not sure) or running, key="opt_go"):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(
                    [sys.executable, str(dd.ROOT / "scripts" / "optimize_yjunction.py"),
                     "--max-evals", str(int(mx))],
                    stdout=lf, stderr=subprocess.STDOUT, cwd=str(dd.ROOT))
            st.session_state["opt_pid"] = proc.pid
            st.rerun()

        if pid is not None:
            if running:
                st.info(f"Study running (pid {pid}) — each FDTD job takes a few "
                        "minutes. Refresh to follow the log.")
            else:
                st.success("Study finished — reload results to see the new study "
                           "in the dropdown below.")
            if log_path.exists():
                tail = log_path.read_text().splitlines()[-25:]
                st.code("\n".join(tail) or "(no output yet)")
            b1, b2 = st.columns(2)
            if b1.button("🔄 Refresh log", key="opt_refresh"):
                st.rerun()
            if not running and b2.button("📥 Reload results", key="opt_reload"):
                st.session_state.pop("opt_pid", None)
                st.cache_data.clear()
                st.rerun()


@st.cache_data(show_spinner=False)
def _spectra_long(device_type: str, param: str):
    alld = dd.load_tidy3d_designs()
    sub = alld[alld["device_type"] == device_type]
    sub = sub[~sub["design_id"].astype(str).str.startswith("D_YJ_OPT")]
    rows = []
    for _, r in sub.iterrows():
        spec = dd.load_spectrum(r["design_id"])
        if not spec:
            continue
        wl, curves = dd.spectrum_loss_curves(spec)
        if wl is None or not curves:
            continue
        arr = curves.get("insertion_loss_db", curves.get("loss_euler_db",
                                                         next(iter(curves.values()))))
        rows.append(pd.DataFrame({"wavelength_nm": wl * 1000.0, "loss_db": arr,
                                  "design_id": r["design_id"],
                                  "param": r.get(f"geom_{param}")}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _geom_loss_fig(df, xcol, label, hue_col=None, hue_name=None):
    vals = pd.to_numeric(df[xcol], errors="coerce")
    if vals.nunique() <= 6:
        d = df.assign(_g=vals).sort_values("_g")
        d["_g"] = d["_g"].astype(str)
        if hue_col is not None and hue_col != xcol:
            d["_h"] = df[hue_col].map(lambda v: f"{hue_name} = {v}")
            fig = px.box(d, x="_g", y="simulated_loss_db", color="_h", points="all",
                         color_discrete_sequence=PALETTE)
            fig.update_layout(legend_title_text="")
        else:
            fig = px.box(d, x="_g", y="simulated_loss_db", color="_g", points="all",
                         color_discrete_sequence=PALETTE)
            fig.update_layout(showlegend=False)
        fig.update_layout(title=f"Insertion loss by {label}", xaxis_title=label,
                          yaxis_title="mean insertion loss (dB)")
    else:
        fig = px.scatter(df, x=xcol, y="simulated_loss_db", trendline=TRENDLINE,
                         color_discrete_sequence=[PALETTE[1]])
        fig.update_layout(title=f"Insertion loss vs {label}", xaxis_title=label,
                          yaxis_title="mean insertion loss (dB)")
    return fig


def render_tidy3d(device_type: str, nice: str):
    alld = _tidy3d_designs()
    if alld.empty:
        st.error("No design manifest found. Run `python scripts/build_manifest.py` "
                 "(or it will auto-build from data/designs.json)."); return
    df = alld[alld["device_type"] == device_type].reset_index(drop=True)
    if df.empty:
        st.warning(f"No real {nice} designs found."); return
    feats = dd.tidy3d_geometry_features(df)

    st.markdown(badge("Tidy3D") + f'<span class="src-pill" style="background:#ede9fe;color:#7c3aed;">REAL FDTD</span>'
                f'<b style="color:#1a1040;">{nice}</b>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{nice} designs", len(df))
    k2.metric("Mean insertion loss", f"{df['simulated_loss_db'].mean():.3f} dB")
    k3.metric("Best loss", f"{df['simulated_loss_db'].min():.3f} dB")
    k4.metric("Geometry params", len(feats))

    labels = ["  Overview  ", "  Component Metadata  ", "  Design Library & Spectra  "]
    if device_type == "y_junction":
        labels.append("  Optimization  ")
    tabs = st.tabs(labels)
    t_ov, t_meta, t_lib = tabs[0], tabs[1], tabs[2]

    if device_type == "y_junction":
        with tabs[3]:
            section(f"{nice} — optimization",
                    "Gradient-free search over l_junction; every candidate is stored "
                    "as an OptimizationRecord linked to its SimulationRun")
            _live_optimizer_panel()

            st.markdown("#### Loss landscape — what the optimizer sees")
            st.caption("Axes: x = w7, y = l_junction, **z (height) = mean insertion "
                       "loss — the quantity being minimized**, so the optimum is the "
                       "deepest valley. The problem has a third parameter (h_bend) "
                       "that a 3-D plot can't show as an axis — pick its slice:")
            hb = st.radio("h_bend (µm) — 3rd parameter, sliced for this view",
                          [1.5, 2.0, 2.5], horizontal=True, key="opt_hb_radio")
            w7g, lg, Z, px_, py_, pz_, phb_, pid_, popt_, n_fit = _gp_surface(float(hb))
            fig3 = go.Figure()
            fig3.add_trace(go.Surface(
                x=w7g, y=lg, z=Z, colorscale="Plasma", opacity=0.92,
                colorbar=dict(title="loss (dB)", len=0.6),
                hovertemplate=("w7 = %{x:.3f} µm · l_junction = %{y:.3f} µm"
                               "<br>GP-predicted loss = %{z:.4f} dB"
                               "<extra>GP surface</extra>")))
            for mask, color, label in [(~popt_, "#06b6d4", "original sweep design"),
                                       (popt_, "#ec4899", "new BO-proposed design")]:
                if mask.any():
                    fig3.add_trace(go.Scatter3d(
                        x=px_[mask], y=py_[mask], z=pz_[mask], mode="markers",
                        marker=dict(size=4.5, color=color,
                                    line=dict(width=1, color="white")),
                        customdata=np.column_stack([pid_[mask], phb_[mask]]),
                        hovertemplate=("<b>%{customdata[0]}</b>"
                                       "<br>w7 = %{x:.3f} µm · l_junction = %{y:.3f} µm"
                                       " · h_bend = %{customdata[1]:.3f} µm"
                                       "<br>measured loss = %{z:.4f} dB"
                                       f"<extra>{label}</extra>"),
                        name=label))
            fig3.update_layout(
                title="GP loss landscape",
                scene=dict(xaxis_title="w7 (µm)", yaxis_title="l_junction (µm)",
                           zaxis_title="mean insertion loss (dB)"),
                height=540, margin=dict(t=40, l=0, r=0, b=0),
                showlegend=True, legend=dict(orientation="h", y=0.02))
            st.plotly_chart(style_fig(fig3), use_container_width=True,
                            key=f"opt_surface_{device_type}")
            st.caption("The 3-D view is hover-only (Plotly can't report clicks in 3-D "
                       "scenes) — use the 2-D plots below to click-highlight table rows.")
            st.markdown("<hr style='margin:.8rem 0;'>", unsafe_allow_html=True)

            recs = _opt_records(device_type)
            if recs.empty:
                st.info("No optimization studies yet. Launch the live loop above "
                        "(or run `python scripts/optimize_yjunction.py`) — when it "
                        "finishes, the progress plot and per-iteration table "
                        "populate here automatically.")
            else:
                runs_ = list(dict.fromkeys(recs["study_id"]))
                sid = st.selectbox(
                    "Loop run", runs_, index=len(runs_) - 1,
                    format_func=lambda s: f"Run {runs_.index(s) + 1} — {s}",
                    key=f"opt_study_{device_type}")
                d = recs[recs["study_id"] == sid].reset_index(drop=True)
                d["l_junction"] = d["parameters"].map(lambda p: p.get("l_junction"))
                extras = [c for c in ("w7", "h_bend")
                          if d["parameters"].map(lambda p: c in p).any()]
                for c in extras:
                    d[c] = d["parameters"].map(lambda p, c=c: p.get(c))
                p0 = d["parameters"].iloc[0]
                live = "seed_best_db" in p0
                best = float(d["best_so_far"].iloc[-1])
                best_row = d[d["is_best"]].iloc[-1] if d["is_best"].any() else d.iloc[0]

                if live:
                    seed_best = float(p0["seed_best_db"])
                    n_seeds = int(p0["n_seeds"])
                    st.caption(
                        f"**Bayesian optimization over (w7, h_bend, l_junction) "
                        f"simultaneously.** The GP surrogate was first fitted on the "
                        f"**{n_seeds} pre-existing sweep designs** — they are not "
                        "shown as iterations below, but their loss landscape told "
                        "the optimizer where the promising regions are before a "
                        "single new FlexCredit was spent. Only **new** FDTD "
                        "evaluations appear in the plot and table.")
                    stat_cards([
                        ("New evaluations", f"{len(d)}"),
                        ("Free GP seeds", f"{n_seeds}"),
                        ("Best pre-existing (seed)", f"{seed_best:.4f} dB"),
                        ("Best found", f"{best:.4f} dB"),
                        ("Improvement vs best seed", f"{seed_best - best:+.4f} dB"),
                    ])
                    hover = [f"w7={r['w7']} · h_bend={r['h_bend']} · "
                             f"l_junction={r['l_junction']}" for _, r in d.iterrows()]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=d["iteration"], y=d["objective_value"], mode="markers",
                        marker=dict(color="#ec4899", size=10,
                                    line=dict(width=1, color="white")),
                        text=hover, name="new BO evaluation"))
                    fig.add_trace(go.Scatter(
                        x=d["iteration"], y=d["best_so_far"], mode="lines",
                        line=dict(color="#7c3aed", width=2, shape="hv"),
                        name="best so far (incl. seeds)"))
                    fig.add_hline(y=seed_best, line=dict(color="#94a3b8", dash="dash"))
                    fig.update_layout(
                        title="Optimization progress — new evaluations only "
                              "(dashed = best pre-existing design)",
                        xaxis_title="new evaluation #",
                        yaxis_title="mean insertion loss (dB)")
                else:
                    baseline = float(d["objective_value"].iloc[0])
                    if extras:
                        ctx = " · ".join(f"{c} = {d[c].iloc[0]} µm" for c in extras)
                        st.caption(f"Held fixed across all candidates: **{ctx}** — so "
                                   "the objective differences are attributable to "
                                   "l_junction alone (apples to apples).")
                    stat_cards([
                        ("Candidates", f"{len(d)}"),
                        ("Baseline (iteration 0)", f"{baseline:.4f} dB"),
                        ("Best found", f"{best:.4f} dB"),
                        ("Improvement vs initial", f"−{baseline - best:.4f} dB"),
                        ("Best l_junction", f"{best_row['l_junction']} µm"),
                    ])
                    ds = d.sort_values("l_junction")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ds["l_junction"], y=ds["objective_value"], mode="lines",
                        line=dict(color="#d4cff0", width=1.5, dash="dot"),
                        showlegend=False))
                    fig.add_trace(go.Scatter(
                        x=d["l_junction"], y=d["objective_value"], mode="markers+text",
                        marker=dict(color=["#ec4899" if b else "#7c3aed"
                                           for b in d["is_best"]], size=10, opacity=0.9),
                        text=[str(i) for i in d["iteration"]],
                        textposition="top center",
                        textfont=dict(size=10, color="#7168a8"),
                        name="evaluated candidate"))
                    fig.add_hline(y=baseline, line=dict(color="#94a3b8", dash="dash"))
                    fig.update_layout(
                        title="Objective vs l_junction — the only varied parameter "
                              "(markers numbered by iteration; dashed = initial design)",
                        xaxis_title="l_junction (µm)",
                        yaxis_title="mean insertion loss (dB)")
                events = [(st.plotly_chart(
                    style_fig(fig), use_container_width=True,
                    key=f"opt_fig_{device_type}",
                    on_select="rerun", selection_mode="points"),
                    0 if live else 1)]
                sel_mask = pd.Series(False, index=d.index)
                for ev, pink_curve in events:
                    try:
                        pts = ev["selection"]["points"]
                    except Exception:
                        pts = []
                    for pt in pts:
                        if pt.get("curve_number") != pink_curve:
                            continue
                        idx = pt.get("point_index")
                        if idx is not None and 0 <= int(idx) < len(d):
                            sel_mask.iloc[int(idx)] = True
                tbl = d[["iteration", "l_junction"] + extras
                        + ["objective_value", "best_so_far", "is_best", "run_id"]].round(4)
                tbl = tbl.rename(columns={
                    "objective_value": "mean insertion loss (dB)",
                    "best_so_far": "best so far (dB)"})
                if sel_mask.any():
                    st.caption("Showing your **clicked point(s)** highlighted below — "
                               "click elsewhere or press Esc on the plot to clear.")
                render_table(tbl, highlight_mask=sel_mask)
                st.caption(
                    "Click any pink point in the plot above to highlight its row. "
                    "Each row records the full parameter set the GP proposed and "
                    "the loss the real FDTD job measured; is_best marks iterations "
                    "that set a new record.")

    with t_meta:
        section(f"{nice} — component metadata",
                "The minimal component record: name, function, geometry, "
                "material/platform assumptions, and expected performance metrics")
        rec = _component_record(device_type)
        if rec is None:
            st.info("No `components` record found in the Tidy3D DuckDB store.")
        else:
            st.markdown(badge("DuckDB")
                        + f"`components` table · **{rec['name']}**", unsafe_allow_html=True)
            stat_cards([
                ("Function", rec.get("function", "—")),
                ("Platform", rec.get("platform", "—")),
                ("Library", rec.get("library", "—")),
            ])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Geometry parameters**")
                g = rec.get("geometry", {})
                render_table(pd.DataFrame(
                    [(k, str(v)) for k, v in g.items()], columns=["parameter", "value"]))
                st.markdown("**Material assumptions**")
                m = rec.get("materials", {})
                render_table(pd.DataFrame(
                    [(k, str(v)) for k, v in m.items()], columns=["layer", "material"]))
            with c2:
                st.markdown("**Expected performance metrics** *(the spec set before "
                            "simulating — compare results against these)*")
                e = rec.get("expected_metrics", {})
                render_table(pd.DataFrame(
                    [(k, str(v)) for k, v in e.items()], columns=["metric", "expected"]))
                st.markdown("**Provenance**")
                st.markdown(f"- Source: [{rec.get('source_url', '—')}]({rec.get('source_url', '')})\n"
                            f"- Record created: {str(rec.get('created_at', '—'))[:19]}")
            if rec.get("notes"):
                st.caption(f"📝 {rec['notes']}")

    with t_ov:
        section(f"{nice} — overview", "Real Tidy3D FDTD sweep")
        ov = df[~df["design_id"].astype(str).str.startswith("D_YJ_OPT")]
        swept = [c for c in feats if ov[c].nunique(dropna=True) > 1]
        if not swept:
            st.info("No geometry parameter varies in this sweep."); st.stop()
        corr = dd.correlations(ov, swept, "simulated_loss_db")
        dom_col = corr.index[0] if len(corr) else swept[0]
        key_col = st.selectbox(
            "Geometry parameter (drives both plots)", swept,
            index=swept.index(dom_col) if dom_col in swept else 0,
            key=f"ov_par_{device_type}", format_func=dd.param_label)
        key = key_col.replace("geom_", "")
        label = dd.param_label(key)

        c1, c2 = st.columns(2)
        with c1:
            long = _spectra_long(device_type, key)
            if len(long):
                if long["param"].nunique() <= 6:
                    long = long.sort_values("param")
                    long["group"] = long["param"].map(lambda v: f"{key} = {v} µm")
                    fig = px.line(long, x="wavelength_nm", y="loss_db",
                                  line_group="design_id", color="group",
                                  color_discrete_sequence=PALETTE)
                else:
                    fig = go.Figure()
                    for did, g in long.groupby("design_id"):
                        fig.add_trace(go.Scatter(x=g["wavelength_nm"], y=g["loss_db"],
                                                 mode="lines", showlegend=False,
                                                 line=dict(color="#d8d3ee", width=1)))
                    means = long.groupby("design_id")["loss_db"].mean()
                    for did, color, tag in [(means.idxmin(), "#10b981", "best"),
                                            (means.idxmax(), "#ec4899", "worst")]:
                        g = long[long["design_id"] == did]
                        fig.add_trace(go.Scatter(x=g["wavelength_nm"], y=g["loss_db"],
                                                 mode="lines", name=f"{tag}: {did}",
                                                 line=dict(color=color, width=2.6)))
                fig.update_layout(title=f"All {len(ov)} original sweep spectra — "
                                        "loss vs wavelength",
                                  xaxis_title="wavelength (nm)",
                                  yaxis_title="insertion loss (dB)")
                st.plotly_chart(style_fig(fig), use_container_width=True,
                                key=f"ov_spectra_{device_type}")
        with c2:
            st.plotly_chart(style_fig(_geom_loss_fig(ov, key_col, label,
                                                     hue_col=dom_col if dom_col != key_col else None,
                                                     hue_name=dd.param_label(dom_col).split(" (")[0])),
                            use_container_width=True, key=f"ov_geom_{device_type}")

    with t_lib:
        section(f"{nice} — design library & spectra",
                "Spectrum viewer on top, every real FDTD run (variant) below")
        st.caption("Pick a design to view its insertion-loss spectrum (read from "
                   "data/tidy3d_raw/*.json) — the selected design is **highlighted** "
                   "in the table below.")
        did = st.selectbox("Design", df["design_id"].tolist(), key=f"spec_{device_type}")
        spec = dd.load_spectrum(did)
        if not spec:
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
                st.plotly_chart(style_fig(fig), use_container_width=True,
                                key=f"spec_fig_{device_type}")
        st.markdown("<hr style='margin:.8rem 0;'>", unsafe_allow_html=True)
        cols = ["design_id", "device_type", "simulated_loss_db"] + feats
        render_table(df[[c for c in cols if c in df.columns]].round(4),
                     highlight_mask=(df["design_id"] == did))


@st.cache_data(show_spinner=False)
def _screening_frames(root_str, token):
    s = dd.open_screening_store(root=Path(root_str))
    try:
        out = {"features": s.feature_table(), "metrics": s.device_metrics_summary(),
               "logs": s.lab_logs(), "captures": s.captures(), "devices": s.devices()}
        ms = s.measurements()
        for kind in ("static_sweep", "fra"):
            row = ms[ms["kind"] == kind]
            out[kind] = s.read_table(row.iloc[0]["measurement_id"]) if len(row) else pd.DataFrame()
    finally:
        s.close()
    return out


@st.cache_data(show_spinner="Fitting two-stage fringe model…")
def _two_stage_fit(v_bytes: bytes, p_bytes: bytes) -> dict:
    v = np.frombuffer(v_bytes, dtype=float)
    p = np.frombuffer(p_bytes, dtype=float)
    return dd.fit_two_stage_fringe(v, p)


def render_screening():
    root = dd.SCREEN_ROOT
    if not (root / "reallabdata.duckdb").exists():
        st.warning("The real-lab-data store hasn't been built yet.")
        ds = st.text_input("Path to the raw lab dataset folder",
                           value=str(dd.ROOT / "data" / "reallabdata_raw"))
        if st.button("⚙️ Build store from raw lab data"):
            with st.spinner("Ingesting static sweep, FRA, 100 captures, lab logs…"):
                _, status = dd.ensure_screening_store(dataset=ds, root=root)
            (st.error if status.startswith("error") else st.success)(status)
            if not status.startswith("error"):
                st.rerun()
        return

    token = (root / "reallabdata.duckdb").stat().st_mtime
    F = _screening_frames(str(root), token)
    feats = F["features"]
    metrics = {r["metric_name"]: r["value"] for _, r in F["metrics"].iterrows()}

    st.markdown(badge("DuckDB") + badge("Parquet") + badge("Zarr")
                + '<span class="src-pill" style="background:#cffafe;color:#0e7490;">REAL LAB DATA · cascaded MZI</span>',
                unsafe_allow_html=True)

    sw = F["static_sweep"]
    fit = (_two_stage_fit(sw["V_DC_volts"].to_numpy(dtype=float).tobytes(),
                          sw["PD_mean_mV"].to_numpy(dtype=float).tobytes())
           if len(sw) else {"ok": False, "reason": "no static sweep"})

    if fit["ok"]:
        s1, s2 = fit["stages"]
        vpi_card = ("V_π (stage 1 / 2)", f"{s1['v_pi']:.2f} / {s2['v_pi']:.2f} V")
        vnull_card = ("V_null (a stage-1 null)",
                      f"{metrics.get('static_V_null', float('nan')):.2f} V")
    else:
        vpi_card = ("V_π (composite est)",
                    f"{metrics.get('static_V_pi_est', float('nan')):.2f} V")
        vnull_card = ("V_null (composite)",
                      f"{metrics.get('static_V_null', float('nan')):.2f} V")
    stat_cards([
        ("Captures", f"{len(feats)}"),
        ("−3 dB BW", f"{metrics.get('fra_bandwidth_3db', float('nan')):.3f} MHz"),
        vnull_card, vpi_card,
        ("Extinction", f"{metrics.get('static_extinction', float('nan')):.1f} dB"),
    ])

    t_ov, t_cap, t_pred, t_logs = st.tabs(
        ["  Overview  ", "  Captures & Waveforms  ",
         "  Predictive Model  ", "  Lab Logs  "])

    with t_ov:
        _figures_group("Real lab data", dd.SCREEN_ROOT / "figures", [])
        section("Two-stage fringe decomposition",
                "P(V) = P0 + A·cos²(π(V−Vc1)/2Vπ1)·cos²(π(V−Vc2)/2Vπ2) — multi-start "
                "least-squares fit of the multiplicative two-stage model to the static sweep")
        if not fit["ok"]:
            st.info(f"Fit not available — {fit['reason']}.")
        else:
            s1, s2 = fit["stages"]
            stat_cards([
                ("Stage 1 V_π", f"{s1['v_pi']:.2f} V"),
                ("Stage 1 nulls", " / ".join(f"{n:+.2f} V" for n in s1["nulls"])),
                ("Stage 2 V_π", f"{s2['v_pi']:.2f} V"),
                ("Stage 2 nulls", " / ".join(f"{n:+.2f} V" for n in s2["nulls"])),
                ("Fit R²", f"{fit['r2']:.5f}"),
            ])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fit["v"], y=fit["p"], mode="markers",
                                     marker=dict(color="#7c3aed", size=5, opacity=0.55),
                                     name="data"))
            fig.add_trace(go.Scatter(x=fit["grid"], y=fit["fit"], mode="lines",
                                     line=dict(color="#ec4899", width=2, dash="dash"),
                                     name="product fit"))
            for i, (s, curve, col) in enumerate(zip(fit["stages"], fit["stage_curves"],
                                                    ["#10b981", "#06b6d4"]), start=1):
                fig.add_trace(go.Scatter(x=fit["grid"], y=curve, mode="lines",
                                         line=dict(color=col, width=1.5), opacity=0.7,
                                         name=f"stage {i} alone (V_π={s['v_pi']:.2f} V)"))
            fig.update_layout(title="Stage 1 × Stage 2 = the data — each stage is a "
                                    "simple cos²; their product makes the uneven fringe",
                              xaxis_title="V_DC (V)", yaxis_title="PD (mV)")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with t_cap:
        section("Captures & waveforms", "100 step-pulse captures — waveform viewer on top, "
                "records below")
        st.caption("Pick a capture to view its trace-averaged waveform (read from Zarr) — "
                   "the selected capture is **highlighted** in the table below.")
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
            st.info(f"Waveform not available ({e}).")

        st.markdown("<hr style='margin:.8rem 0;'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        reps = sorted(feats["rep_hz"].dropna().unique())
        vlbls = sorted(feats["vlbl"].dropna().unique())
        sel_rep = c1.multiselect("Rep rate (Hz)", reps, default=reps)
        sel_v = c2.multiselect("Drive variant", vlbls, default=vlbls)
        ftab = feats[feats["rep_hz"].isin(sel_rep) & feats["vlbl"].isin(sel_v)]
        render_table(ftab.round(3), highlight_mask=(ftab["tag"] == tag))
        if tag not in ftab["tag"].values:
            st.caption("⚠️ The selected capture is filtered out by the current filters.")

    with t_pred:
        section("Predictive model & anomalies",
                "Predict PD swing from drive parameters; captures that deviate ≥2σ "
                "from their drive-predicted swing are flagged")
        pf = [c for c in ["rep_hz", "edge_ns", "V_pp"] if c in feats.columns]
        X = dd.build_feature_matrix(feats, pf, categorical=["vlbl"])
        model_panel(X, feats["cap_swing_mV"], "swing (mV)", ctx="captures",
                    ids=feats["tag"], flag_anomalies=True)
        st.markdown("Flagged captures are worth cross-referencing against the Lab Logs — "
                    "the 2026-04-23 note reports a thermal-drift event.")
        with st.expander("How the anomaly flags are calculated"):
            st.markdown(
                "- A **RandomForest** model learns `swing = f(drive params)` "
                "(rep_hz, edge_ns, drive variant) with **5-fold cross-validation** "
                "(`cross_val_predict`), so each capture's prediction comes from a model "
                "that never saw that capture.\n"
                "- The scatter plots **actual** swing (x) vs **drive-predicted** swing (y); "
                "the dashed **ideal** line is the identity **y = x**, not a fit — distance "
                "from it = prediction error.\n"
                "- **Residual = actual − predicted**; residuals are z-scored and captures "
                "with **|z| ≥ 2** are flagged. This catches captures that are unusual "
                "*for their settings*, not just unusual values.")
        st.markdown("<hr style='margin:.8rem 0;'>", unsafe_allow_html=True)
        shap_panel(feats, X, feats["cap_swing_mV"], "tag", "swing (mV)")

    with t_logs:
        section("Operator lab logs", "The RAG seam — real bench notes")
        for _, r in F["logs"].iterrows():
            with st.expander(f"📋 {r['date']} — {r['title']}"):
                st.markdown(r["body"])
                if isinstance(r.get("referenced_tags"), list) and r["referenced_tags"]:
                    st.caption("References: " + ", ".join(r["referenced_tags"]))


def _duckdb_browser(label, db_path, key):
    st.markdown(badge("DuckDB") + f"**{label}** — `{dd.rel(db_path)}`", unsafe_allow_html=True)
    if not Path(db_path).exists():
        st.caption("Store not built yet — it builds automatically on first launch; "
                   "see the startup warning if it's missing."); return
    ov = dd.duckdb_overview(db_path)
    render_table(ov)
    if len(ov):
        t = st.selectbox(f"Preview a table ({label})", ov["table"].tolist(), key=key)
        render_table(dd.duckdb_preview(db_path, t, n=25))


def _nl_query(db_path, schema, placeholder, state_key):
    key = get_api_key()
    st.markdown("**…or ask in natural language**")
    if not key:
        st.info("Set `ANTHROPIC_API_KEY` (env var or `.streamlit/secrets.toml`) to enable "
                "natural-language queries.")
        return
    q = st.text_input("Ask a question", placeholder=placeholder, key=f"nl_{state_key}")
    if q:
        try:
            with st.spinner("Claude is writing the SQL…"):
                sql = dd.nl_to_duckdb_sql(q, key, schema=schema)
            res = dd.run_duckdb_sql(db_path, sql)
            st.session_state[state_key] = (f"NL query: _{q}_", sql, res)
            with st.spinner("Interpreting…"):
                st.caption(dd.interpret_result(q, sql, res.head(30).to_string(index=False), key))
        except Exception as e:
            st.error(f"Query failed: {e}")


def _run_and_store(state_key, caption, db_path, sql):
    try:
        st.session_state[state_key] = (caption, sql, dd.run_duckdb_sql(db_path, sql))
    except Exception as e:
        st.error(f"Query failed: {e}")


def _result_area(state_key):
    if state_key in st.session_state:
        cap, sql, res = st.session_state[state_key]
        st.markdown(f"**Result** — {cap}", unsafe_allow_html=True)
        st.markdown("*The DuckDB SQL that was executed:*")
        st.code(sql, language="sql")
        render_table(res)


def _query_tidy3d():
    ppdb = dd.PIPELINE_ROOT / "tidy3d.duckdb"
    if not ppdb.exists():
        st.info("The Tidy3D store is missing — it builds automatically at launch; "
                "see the startup warning for what went wrong.")
        return
    METRICS = ["insertion_loss_mean", "insertion_loss_min",
               "insertion_loss_at_center", "bandwidth_3db"]
    qcol, nlcol = st.columns(2)
    with qcol:
        st.markdown("**Quick queries** — each button runs the SQL shown below")
        comp = st.selectbox("Component", ["y_junction", "euler_bend"], key="q_comp")
        b1, b2 = st.columns(2)
        if b1.button("📥 All simulations for component", use_container_width=True):
            _run_and_store(
                "q_res_t3",
                f"All stored runs for **{comp}** — component identity, geometry, "
                "settings, timestamps, tool metadata, and analysis-ready outputs",
                ppdb,
f"""SELECT r.run_id, r.component_id,
       json_extract_string(r.geometry_snapshot, '$.design_id') AS design_id,
       r.geometry_snapshot,
       r.tool, r.tool_version, r.boundary_conditions, r.mesh_summary,
       r.wl_min_um, r.wl_max_um, r.n_wavelengths, r.created_at,
       COUNT(DISTINCT ar.artifact_id) AS artifacts,
       COUNT(DISTINCT res.result_id)  AS metrics
FROM simulation_runs r
LEFT JOIN raw_artifacts ar     ON ar.run_id = r.run_id
LEFT JOIN analysis_results res ON res.run_id = r.run_id
WHERE r.component_id = '{comp}'
GROUP BY ALL
ORDER BY design_id""")
        if b2.button("📈 Performance metrics", use_container_width=True):
            _run_and_store("q_res_t3", f"Metrics for **{comp}**", ppdb,
f"""SELECT component_id, metric_name, metric_value, units, run_id
FROM analysis_results
WHERE component_id = '{comp}'
ORDER BY metric_name, metric_value""")

        metric = st.selectbox("Metric", METRICS, key="q_metric")
        if st.button("⚖️ Compare variants by this metric", use_container_width=True):
            _run_and_store("q_res_t3", f"Variants of **{comp}** by **{metric}** (best first)", ppdb,
f"""SELECT sr.run_id, sr.geometry_snapshot, ar.metric_value, ar.units
FROM analysis_results ar
JOIN simulation_runs sr ON sr.run_id = ar.run_id
WHERE ar.component_id = '{comp}'
  AND ar.metric_name = '{metric}'
ORDER BY ar.metric_value""")

        params = dd.geometry_param_names() or ["l_junction"]
        geom = st.selectbox("Geometry parameter", params,
                            index=params.index("l_junction") if "l_junction" in params else 0,
                            key="q_geom")
        g1, g2 = st.columns(2)
        gmin = g1.number_input("min", value=0.0, step=0.1, key="q_gmin")
        gmax = g2.number_input("max", value=0.0, step=0.1, key="q_gmax")
        if st.button("🔎 Filter runs by geometry", use_container_width=True):
            conds = ["geom_value IS NOT NULL"]
            if gmin:
                conds.append(f"geom_value >= {gmin}")
            if gmax:
                conds.append(f"geom_value <= {gmax}")
            where = "\n  AND ".join(conds)
            _run_and_store("q_res_t3", f"Runs with **{geom}** in [{gmin}, {gmax}]", ppdb,
f"""SELECT run_id, component_id,
       CAST(json_extract(geometry_snapshot, '$.{geom}') AS DOUBLE) AS geom_value
FROM simulation_runs
WHERE {where}
ORDER BY geom_value""")
    with nlcol:
        _nl_query(ppdb, None,
                  "which y_junction run has the lowest mean insertion loss, and its l_junction?",
                  "q_res_t3")
    _result_area("q_res_t3")


def _query_lab():
    rldb = dd.SCREEN_ROOT / "reallabdata.duckdb"
    if not rldb.exists():
        st.info("The lab-data store is missing — it builds automatically at launch; "
                "see the startup warning for what went wrong.")
        return
    REPS = [30000.0, 100000.0, 300000.0, 600000.0, 1000000.0]
    VLBLS = ["sym1", "sym5", "sym10", "asymHi", "asymLo"]
    CAPTURE_SQL = """SELECT measurement_id, tag,
       CAST(json_extract(params, '$.rep_hz')  AS DOUBLE) AS rep_hz,
       CAST(json_extract(params, '$.edge_ns') AS DOUBLE) AS edge_ns,
       json_extract_string(params, '$.vlbl')             AS vlbl,
       CAST(json_extract(params, '$.V_pp')    AS DOUBLE) AS V_pp
FROM measurements
WHERE kind = 'capture'"""
    qcol, nlcol = st.columns(2)
    with qcol:
        st.markdown("**Quick queries** — each button runs the SQL shown below")
        b1, b2 = st.columns(2)
        if b1.button("📥 All captures", use_container_width=True):
            _run_and_store("q_res_lab", "All 100 step-pulse captures", rldb,
                           CAPTURE_SQL + "\nORDER BY rep_hz, edge_ns, vlbl")

        c1, c2 = st.columns(2)
        rep = c1.selectbox("Rep rate (Hz)", ["any"] + REPS, key="ql_rep")
        vlbl = c2.selectbox("Drive variant", ["any"] + VLBLS, key="ql_vlbl")
        if st.button("🔎 Filter captures by acquisition params", use_container_width=True):
            sql = CAPTURE_SQL
            if rep != "any":
                sql += f"\n  AND rep_hz = {rep}"
            if vlbl != "any":
                sql += f"\n  AND vlbl = '{vlbl}'"
            sql += "\nORDER BY rep_hz, edge_ns, vlbl"
            _run_and_store("q_res_lab", f"Captures with rep=**{rep}**, drive=**{vlbl}**", rldb, sql)
    with nlcol:
        _nl_query(rldb, dd.REALLAB_SCHEMA_HINT,
                  "which capture has the largest PD swing, and at what rep rate?",
                  "q_res_lab")
    _result_area("q_res_lab")


def _figures_group(label, figdir, csvs, key=""):
    st.markdown(f"#### {label} — `{dd.rel(figdir)}`")
    pngs = sorted(Path(figdir).glob("*.png"))
    if not pngs:
        st.caption("No figures yet — they're generated automatically during the "
                   "first-launch store build.")
    else:
        cols = st.columns(2)
        for i, p in enumerate(pngs):
            with cols[i % 2]:
                st.image(str(p), caption=p.stem.replace("_", " "),
                         use_container_width=True)
    for c in csvs:
        cp = Path(figdir) / c
        if cp.exists():
            with st.expander(f"📋 {c}"):
                render_table(pd.read_csv(cp).round(4))
    st.markdown("<hr style='margin:.8rem 0;'>", unsafe_allow_html=True)


def _pipeline_figures():
    figdir = dd.PIPELINE_ROOT / "figures"
    st.markdown(f"#### Tidy3D — `{dd.rel(figdir)}`")
    pngs = sorted(Path(figdir).glob("*.png"))[:2]
    if not pngs:
        st.caption("No figures yet — they're generated automatically during the "
                   "first-launch store build.")
        return
    cols = st.columns(2)
    for i, p in enumerate(pngs):
        with cols[i % 2]:
            st.image(str(p), caption=p.stem.replace("_", " "),
                     use_container_width=True)


def render_tables_queries():
    t1, t2, t3, t4, t5 = st.tabs(["  Query Tidy3D  ", "  Query lab data  ",
                                  "  Browse Tidy3D  ", "  Browse lab data  ",
                                  "  Pipeline figures  "])
    with t1:
        section("Query the Tidy3D simulations", "one-click query operations, or natural language")
        _query_tidy3d()
    with t2:
        section("Query the real lab data", "one-click query operations, or natural language")
        _query_lab()
    with t3:
        section("Browse the Tidy3D store", "every DuckDB table, previewable")
        _duckdb_browser("Tidy3D store", dd.PIPELINE_ROOT / "tidy3d.duckdb", "pp_browse")
    with t4:
        section("Browse the lab-data store", "every DuckDB table, previewable")
        _duckdb_browser("Real lab data store", dd.SCREEN_ROOT / "reallabdata.duckdb", "scr_browse")
    with t5:
        section("Pipeline-generated figures",
                "batch outputs written by the demos during the store build")
        _pipeline_figures()


@st.cache_data(show_spinner="Running pipeline validation checks…")
def _run_pipeline_checks() -> list[dict]:
    import hashlib

    import duckdb

    from reallabdata_pipeline import analysis as lab
    from tidy3d_pipeline import analysis as t3a
    from tidy3d_pipeline.components import COMPONENTS

    checks = []
    _grp = {"name": "", "blurb": ""}

    def group(name, blurb):
        _grp["name"], _grp["blurb"] = name, blurb

    def add(title, desc, code, expected, actual, passed):
        checks.append(dict(title=title, desc=desc, code=code,
                           expected=expected, actual=actual, passed=bool(passed),
                           group=_grp["name"], group_blurb=_grp["blurb"]))

    group("1. Calculation checks", "")

    val = t3a.mean_insertion_loss_db(np.ones(10))
    add("Unity transmission → 0 dB loss",
        "If a device transmits 100% of the light, insertion loss must be exactly "
        "0 dB. Any other answer means the dB conversion is wrong.",
        "from tidy3d_pipeline import analysis\n"
        "analysis.mean_insertion_loss_db(np.ones(10))   # T = 1.0 everywhere",
        "0.0 dB", f"{val:.6f} dB", np.isclose(val, 0.0, atol=1e-9))

    val = t3a.mean_insertion_loss_db(np.full(5, 0.5))
    add("Half transmission → 3.01 dB loss",
        "Transmitting half the power is the textbook 3 dB point "
        "(−10·log₁₀(0.5) = 3.0103 dB).",
        "analysis.mean_insertion_loss_db(np.full(5, 0.5))   # T = 0.5 everywhere",
        "3.0103 dB", f"{val:.4f} dB", np.isclose(val, 3.0103, atol=1e-3))

    bad = pd.DataFrame({"wavelength_um": [1.5, 1.55, 1.6], "transmission": [0.9, 2.0, 0.9]})
    warns = t3a.check_spectrum(bad)
    add("Nonphysical spectrum is flagged",
        "Transmission > 1 means more light out than in — physically impossible, "
        "usually a normalization bug. The sanity checker must catch it.",
        "bad = pd.DataFrame({'wavelength_um': [1.5, 1.55, 1.6],\n"
        "                    'transmission':  [0.9, 2.0, 0.9]})   # T=2 impossible\n"
        "analysis.check_spectrum(bad)",
        "≥1 warning mentioning 'exceeds'",
        f"{len(warns)} warning(s): {warns[:1]}", any("exceeds" in w for w in warns))

    v = np.linspace(-5, 5, 401)
    p = 0.05 * np.cos(np.pi * (v - (-1.5)) / (2 * 3.0)) ** 2 + 1e-4
    m = {k: t[0] for k, t in lab.static_fringe_metrics(v, p).items()}
    ok = (np.isclose(m["static_V_pi_est"], 3.0, atol=0.05)
          and np.isclose(m["static_V_null"], -4.5, atol=0.05)
          and m["static_visibility"] > 0.99)
    add("Synthetic fringe → V_π and V_null recovered",
        "A perfect cos² fringe is generated with V_π = 3.0 V and nulls at "
        "−4.5/+1.5 V. The static-fringe analyzer must recover those numbers.",
        "v = np.linspace(-5, 5, 401)\n"
        "p = 0.05 * np.cos(np.pi * (v + 1.5) / (2 * 3.0))**2 + 1e-4   # V_π = 3 V\n"
        "lab_analysis.static_fringe_metrics(v, p)",
        "V_π ≈ 3.0 V · V_null = −4.5 V · visibility ≈ 1",
        f"V_π = {m['static_V_pi_est']:.3f} V · V_null = {m['static_V_null']:.2f} V "
        f"· visibility = {m['static_visibility']:.4f}", ok)

    f = np.logspace(2, 7, 301)
    H = 1.0 / (1.0 + 1j * f / 1e5)
    m = {k: t[0] for k, t in lab.bode_metrics(f, 20 * np.log10(np.abs(H)),
                                              np.angle(H)).items()}
    ok = (np.isclose(m["fra_bandwidth_3db"], 0.1, rtol=0.02)
          and np.isclose(m["fra_phase_at_3db"], -45.0, atol=2.0))
    add("Single-pole filter → corner & phase recovered",
        "A one-pole low-pass H = 1/(1+jf/f_c) with f_c = 100 kHz has its −3 dB "
        "point at exactly f_c, where the phase is −45°. Textbook answer, no fitting.",
        "f = np.logspace(2, 7, 301)\n"
        "H = 1 / (1 + 1j * f / 1e5)                     # corner at 100 kHz\n"
        "lab_analysis.bode_metrics(f, 20*np.log10(abs(H)), np.angle(H))",
        "BW = 0.100 MHz · phase ≈ −45°",
        f"BW = {m['fra_bandwidth_3db']:.4f} MHz · phase = {m['fra_phase_at_3db']:.1f}°",
        ok)

    t = np.linspace(0, 1e-6, 2001)
    y = np.where(t < 0.4e-6, 0.01,
                 np.where(t > 0.5e-6, 0.05, 0.01 + 0.04 * (t - 0.4e-6) / 0.1e-6))
    m = {k: tt[0] for k, tt in lab.capture_step_features(t, y).items()}
    ok = (np.isclose(m["cap_swing_mV"], 40.0, atol=0.5)
          and np.isclose(m["cap_rise_time_ns"], 80.0, atol=2.0))
    add("Ideal step → swing and rise time recovered",
        "A clean 10→50 mV step with a 100 ns linear ramp: swing must be 40 mV and "
        "the 10–90% rise time must be 80% of the ramp = 80 ns.",
        "t = np.linspace(0, 1e-6, 2001)\n"
        "y = step(0.01 V → 0.05 V, linear ramp 100 ns)   # known answer\n"
        "lab_analysis.capture_step_features(t, y)",
        "swing = 40.0 mV · rise = 80 ns",
        f"swing = {m['cap_swing_mV']:.2f} mV · rise = {m['cap_rise_time_ns']:.1f} ns",
        ok)

    group("2. Metadata checks", "")

    df = _tidy3d_designs()
    detail, ok = [], True
    for dt, params in [("y_junction", ["w1", "w7", "l_junction", "t", "h_bend"]),
                       ("euler_bend", ["R_eff", "A", "w", "t"])]:
        sub = df[df["device_type"] == dt]
        swept = {p for p in params if f"geom_{p}" in sub.columns
                 and pd.to_numeric(sub[f"geom_{p}"], errors="coerce").dropna().nunique() > 1}
        declared = set(COMPONENTS[dt].geometry["swept_params"])
        ok &= (declared == swept)
        detail.append(f"{dt}: {sorted(declared)}")
    add("Component metadata matches the actual sweep",
        "The hand-written `swept_params` in the component records must equal the "
        "parameters that actually vary in the design manifest — this check exists "
        "because the original record claimed w1 was swept when it never was.",
        "declared = set(COMPONENTS[dt].geometry['swept_params'])\n"
        "actually_swept = {p for p in geometry_params if manifest[p].nunique() > 1}\n"
        "assert declared == actually_swept",
        "declared == actually swept, both components", " · ".join(detail), ok)

    group("3. End-to-end checks", "")

    spec = dd.load_spectrum("D_YJ_001")
    recomputed = t3a.mean_insertion_loss_db(np.asarray(spec["T_total"], float))
    con = duckdb.connect(str(dd.PIPELINE_ROOT / "tidy3d.duckdb"), read_only=True)
    stored = con.execute("""
        SELECT ar.metric_value FROM analysis_results ar
        JOIN simulation_runs r ON r.run_id = ar.run_id
        WHERE json_extract_string(r.geometry_snapshot, '$.design_id') = 'D_YJ_001'
          AND ar.metric_name = 'insertion_loss_mean'""").fetchone()[0]
    n_runs = con.execute("SELECT count(*) FROM simulation_runs").fetchone()[0]
    con.close()
    add("Real FDTD run: raw spectrum → stored metric (end-to-end)",
        "Not synthetic: design D_YJ_001's raw spectrum JSON (the actual solver "
        "output) is re-analyzed from scratch and compared against the metric the "
        "ingestion pipeline stored in `analysis_results`. Equality proves the real "
        "data went raw file → ingest → analysis → store without alteration.",
        "spec = json.load(open('data/tidy3d_raw/D_YJ_001.json'))     # raw solver output\n"
        "recomputed = analysis.mean_insertion_loss_db(spec['T_total'])\n"
        "stored = duckdb: analysis_results ⋈ simulation_runs (D_YJ_001)\n"
        "assert recomputed == stored",
        "recomputed loss == stored loss",
        f"recomputed {recomputed:.6f} dB == stored {stored:.6f} dB",
        np.isclose(recomputed, stored, atol=1e-9))

    d = np.load(dd.ROOT / "data/reallabdata_raw/fra_smallsignal.npz")
    m = {k: t[0] for k, t in lab.bode_metrics(d["freq_hz"], d["H_mag_db"],
                                              d["H_phase_rad"]).items()}
    con = duckdb.connect(str(dd.SCREEN_ROOT / "reallabdata.duckdb"), read_only=True)
    stored_bw = con.execute("SELECT metric_value FROM analysis_results "
                            "WHERE metric_name='fra_bandwidth_3db'").fetchone()[0]
    n_meas = con.execute("SELECT count(*) FROM measurements").fetchone()[0]
    n_arts = con.execute("SELECT count(*) FROM raw_artifacts").fetchone()[0]
    con.close()
    add("Real lab measurement: raw FRA .npz → stored metric (end-to-end)",
        "Same idea on the bench data: the raw FRA instrument file is re-analyzed "
        "and compared to the −3 dB bandwidth the ingestion stored. This is the "
        "0.319 MHz number on the lab-data overview, traced back to its source bytes.",
        "d = np.load('data/reallabdata_raw/fra_smallsignal.npz')     # raw instrument file\n"
        "recomputed = lab_analysis.bode_metrics(d['freq_hz'], d['H_mag_db'], d['H_phase_rad'])\n"
        "stored = duckdb: analysis_results['fra_bandwidth_3db']\n"
        "assert recomputed == stored",
        "recomputed bandwidth == stored bandwidth",
        f"recomputed {m['fra_bandwidth_3db']:.5f} MHz == stored {stored_bw:.5f} MHz",
        np.isclose(m["fra_bandwidth_3db"], stored_bw, atol=1e-9))

    group("4. Completeness checks", "")

    mf = pd.read_parquet(dd.ROOT / "data/tidy3d_raw/manifest.parquet")
    ok = (len(mf) == n_runs) and (n_meas == n_arts == 102)
    add("Ingestion completeness — nothing was dropped",
        "Every raw input must appear in the store exactly once: all manifest designs "
        "became simulation runs, and all 102 lab acquisitions (sweep + FRA + 100 "
        "captures) have a measurement row and a registered artifact.",
        "assert len(manifest) == count(simulation_runs)\n"
        "assert count(measurements) == count(raw_artifacts) == 102",
        f"{len(mf)} designs → {len(mf)} runs · 102 measurements ↔ 102 artifacts",
        f"{len(mf)} designs → {n_runs} runs · {n_meas} measurements ↔ {n_arts} artifacts",
        ok)

    group("5. Integrity checks", "")

    n_checked = n_ok = 0
    for db in [dd.PIPELINE_ROOT / "tidy3d.duckdb", dd.SCREEN_ROOT / "reallabdata.duckdb"]:
        if not db.exists():
            continue
        con = duckdb.connect(str(db), read_only=True)
        rows = con.execute("SELECT path, checksum_sha256 FROM raw_artifacts "
                           "WHERE checksum_sha256 IS NOT NULL").fetchall()
        con.close()
        for path, stored in rows:
            fp = dd.ROOT / path
            if fp.exists():
                n_checked += 1
                h = hashlib.sha256(fp.read_bytes()).hexdigest()
                n_ok += int(h == stored)
    add("Stored artifacts still match their SHA-256 checksums",
        "Every parquet artifact was checksummed at ingest. Re-hashing the files now "
        "proves the raw data hasn't been corrupted or edited since — the integrity "
        "guarantee the `raw_artifacts` catalog promises.",
        "for path, stored in raw_artifacts:           # all checksummed artifacts\n"
        "    assert sha256(open(path,'rb').read()).hexdigest() == stored",
        "all stored checksums match the files on disk",
        f"{n_ok}/{n_checked} artifacts verified", n_checked > 0 and n_ok == n_checked)

    return checks


def render_pipeline_tests():
    section("Pipeline tests",
            "Known-answer validation: synthetic inputs with hand-computable answers are "
            "run through the same ingestion/analysis functions the pipelines use — "
            "expected vs actual, live. The full suite lives in tests/ (pytest).")
    checks = _run_pipeline_checks()
    n_pass = sum(c["passed"] for c in checks)
    stat_cards([("Checks", f"{len(checks)}"), ("Passing", f"{n_pass}"),
                ("Failing", f"{len(checks) - n_pass}")])
    prev_group = None
    for i, c in enumerate(checks, 1):
        if c["group"] != prev_group:
            prev_group = c["group"]
            st.markdown(
                '<h2 style="font-size:1.4rem;font-weight:700;margin:.6rem 0 .9rem;'
                'background:linear-gradient(90deg,#a855f7,#06b6d4);'
                '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
                f'{c["group"]}</h2>', unsafe_allow_html=True)
        icon = "✅" if c["passed"] else "❌"
        st.markdown(f"#### {icon} {i}. {c['title']}")
        st.markdown(c["desc"])
        st.code(c["code"], language="python")
        e, a = st.columns(2)
        e.markdown(f"**Expected:** {c['expected']}")
        a.markdown(f"**Actual:** {c['actual']}")
        st.markdown("<hr style='margin:.8rem 0;'>", unsafe_allow_html=True)


if domain.startswith("🔷"):
    render_tidy3d("y_junction", "Y-junction")
elif domain.startswith("🔶"):
    render_tidy3d("euler_bend", "Euler bend")
elif domain.startswith("🔬"):
    render_screening()
elif domain.startswith("🧪"):
    render_pipeline_tests()
else:
    render_tables_queries()
