import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "pic_data.db"

st.set_page_config(
    page_title="QuEra · PIC Data Manager",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
  --bg:        #ffffff;
  --surface:   #f8f7ff;
  --card:      #ffffff;
  --border:    #e2dff5;
  --purple:    #7c3aed;
  --violet:    #a855f7;
  --pink:      #ec4899;
  --orange:    #f97316;
  --teal:      #06b6d4;
  --text:      #1a1040;
  --muted:     #7168a8;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  background-image: radial-gradient(circle, #d4cff0 1.2px, transparent 1.2px) !important;
  background-size: 22px 22px !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] {
  background: rgba(255,255,255,0.85) !important;
  backdrop-filter: blur(8px) !important;
}
[data-testid="stSidebar"] { background: var(--surface) !important; }
[data-testid="block-container"] { padding-top: 2rem !important; }

/* ── Hide Streamlit chrome ── */
footer { visibility: hidden; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
  color: var(--muted) !important;
  font-weight: 500 !important;
  font-size: 0.85rem !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
  padding: 0.5rem 1.1rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--purple) !important;
  border-bottom: 2px solid var(--purple) !important;
}
/* hide the red slider indicator */
[data-testid="stTabs"] [role="tablist"] ~ div [data-baseweb="tab-highlight"] {
  display: none !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.85) !important;
  backdrop-filter: blur(6px) !important;
  border-radius: 10px !important;
  padding: 4px 12px !important;
  gap: 8px !important;
  border: 1px solid var(--border) !important;
}

/* ── HTML Tables ── */
.pic-table-wrap {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: #ffffff;
  margin-bottom: 0.5rem;
}
.pic-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
  color: #1a1040;
  background: #ffffff;
}
.pic-table thead th {
  background: #f3f0ff;
  color: #1a1040;
  font-weight: 600;
  padding: 0.55rem 0.9rem;
  text-align: left;
  border-bottom: 2px solid #e2dff5;
  white-space: nowrap;
}
.pic-table tbody tr { border-bottom: 1px solid #f0eef9; }
.pic-table tbody tr:last-child { border-bottom: none; }
.pic-table tbody tr:hover { background: #faf9ff; }
.pic-table tbody td {
  padding: 0.45rem 0.9rem;
  color: #1a1040;
  white-space: nowrap;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
  background: rgba(255,255,255,0.88) !important;
  backdrop-filter: blur(6px) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 1.2rem 1.4rem !important;
  box-shadow: 0 2px 12px rgba(124,58,237,0.07) !important;
}
[data-testid="metric-container"] label {
  color: var(--muted) !important;
  font-size: 0.78rem !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-size: 2rem !important;
  font-weight: 700 !important;
}

/* ── Info/alert boxes ── */
[data-testid="stAlert"] {
  background: rgba(255,255,255,0.88) !important;
  border: 1px solid var(--border) !important;
  border-left: 3px solid var(--teal) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}

/* ── Code blocks ── */
[data-testid="stCode"] {
  background: rgba(248,247,255,0.95) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d4cff0; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plot theme ──────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,255,255,0.85)",
    font=dict(family="Inter", color="#1a1040"),
    title_font=dict(family="Inter", size=15, color="#1a1040"),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2dff5", borderwidth=1),
    margin=dict(t=50, l=10, r=10, b=10),
    xaxis=dict(gridcolor="#e2dff5", linecolor="#e2dff5", showgrid=True),
    yaxis=dict(gridcolor="#e2dff5", linecolor="#e2dff5", showgrid=True),
)
PALETTE = ["#7c3aed", "#a855f7", "#06b6d4", "#ec4899", "#f97316", "#10b981"]

def style_fig(fig):
    fig.update_layout(**PLOT_LAYOUT)
    return fig


# ── Schema description for NL queries ───────────────────────────────────────
_SCHEMA = """
SQLite database for Photonic Integrated Circuit (PIC) manufacturing data.

Tables:

joined_ai_results  ← USE THIS for most queries (all columns pre-joined)
  device_id, wafer_id, design_id, device_type, design_version,
  waveguide_width_nm, coupler_gap_nm, ring_radius_um, target_wavelength_nm,
  simulated_loss_db, simulated_bandwidth_ghz,
  fab_run, foundry, process_recipe,
  film_thickness_nm, etch_depth_nm, measured_width_nm, width_error_nm,
  sidewall_roughness_nm, defect_count,
  test_station, operator, timestamp, temperature_c, humidity_percent,
  wavelength_nm, insertion_loss_db, extinction_ratio_db, bandwidth_ghz,
  passed (1=pass, 0=fail), ai_flagged_anomaly (1=anomaly, 0=normal)

foundry_performance_summary  ← USE THIS for foundry-level comparisons
  foundry, avg_insertion_loss_db, avg_extinction_ratio_db, avg_bandwidth_ghz,
  avg_defect_count, avg_measured_width_nm, avg_sidewall_roughness_nm,
  yield_rate (0–100 percent), num_devices

process_performance_correlations  ← USE THIS for correlation data
  process_feature, performance_metric, correlation (Pearson r, –1 to 1)

designs
  design_id, device_type, design_version, waveguide_width_nm, coupler_gap_nm,
  ring_radius_um, target_wavelength_nm, simulated_loss_db, simulated_bandwidth_ghz

fabrication
  wafer_id, fab_run, foundry, design_id, process_recipe,
  film_thickness_nm, etch_depth_nm, measured_width_nm, width_error_nm,
  sidewall_roughness_nm, defect_count

tests
  test_id, device_id, wafer_id, design_id, test_station, operator, timestamp,
  temperature_c, humidity_percent, wavelength_nm,
  insertion_loss_db, extinction_ratio_db, bandwidth_ghz, passed
"""

def _get_api_key() -> str:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")

def _ask_generate_sql(question: str, api_key: str) -> str:
    """Call Claude Haiku to turn a natural-language question into SQL."""
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=f"""You are a SQLite expert for a Photonic Integrated Circuit (PIC) database.
Convert the user's question into a single, correct SQLite SELECT query.

Schema:
{_SCHEMA}

Rules:
- Return ONLY the raw SQL — no markdown fences, no explanation
- SELECT only (read-only)
- Prefer joined_ai_results for device-level questions
- Prefer foundry_performance_summary for foundry comparisons
- LIMIT 200 unless the user asks for all rows""",
        messages=[{"role": "user", "content": question}],
    )
    sql = msg.content[0].text.strip()
    # Strip any accidental markdown fences
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return sql

def _ask_interpret(question: str, sql: str, result_text: str, api_key: str) -> str:
    """Call Claude to interpret query results in plain English."""
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system="""You are a PIC (Photonic Integrated Circuit) data analyst.
Interpret the query results concisely. Highlight key numbers, patterns, and
engineering implications. Write for a technical audience (engineers/scientists).
Use plain prose — no bullet lists unless comparing 3+ items.""",
        messages=[{"role": "user", "content":
            f"Question: {question}\n\nSQL used:\n{sql}\n\nResults:\n{result_text}\n\n"
            "Give a concise interpretation with the key insight up front."}],
    )
    return msg.content[0].text

# ── DB helpers ──────────────────────────────────────────────────────────────
def _tmp_conn():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    shutil.copy2(DB_PATH, tmp_path)
    return sqlite3.connect(tmp_path), tmp_path

def load_table(table_name: str) -> pd.DataFrame:
    conn, tmp_path = _tmp_conn()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    tmp_path.unlink(missing_ok=True)
    return df

def render_table(df: pd.DataFrame, max_rows: int = 500):
    """Render a DataFrame as a fully visible styled HTML table."""
    html = df.head(max_rows).to_html(index=False, classes="pic-table", border=0)
    st.markdown(f'<div class="pic-table-wrap">{html}</div>', unsafe_allow_html=True)
    if len(df) > max_rows:
        st.caption(f"Showing {max_rows} of {len(df)} rows")


# ── Predictive model helpers ────────────────────────────────────────────────
_PRED_FEATURES = [
    "defect_count",
    "sidewall_roughness_nm",
    "width_error_nm",
    "film_thickness_nm",
    "etch_depth_nm",
    "measured_width_nm",
]

if _TORCH_AVAILABLE:
    class _InsertionLossNet(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 16), nn.ReLU(),
                nn.Linear(16, 8),          nn.ReLU(),
                nn.Linear(8, 1),
            )
        def forward(self, x):
            return self.net(x)

@st.cache_resource
def _load_pred_model():
    """Load saved InsertionLossNet weights + normalisation params. Returns (model, X_mean, X_std) or None."""
    if not _TORCH_AVAILABLE:
        return None
    model_path = ROOT / "models" / "insertion_loss_net.pt"
    norm_path  = ROOT / "models" / "norm_params.npz"
    if not model_path.exists() or not norm_path.exists():
        return None
    model = _InsertionLossNet(len(_PRED_FEATURES))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    norm   = np.load(norm_path)
    return model, norm["X_mean"].astype(np.float32), norm["X_std"].astype(np.float32)


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:0.2rem;">
  <span style="font-size:1.05rem; font-weight:600; color:#9591c4; letter-spacing:0.04em;">
    QuEra · PIC Data Manager
  </span>
</div>
<h1 style="
  font-size:2.6rem; font-weight:700; margin:0.2rem 0 0.1rem;
  background: linear-gradient(90deg, #f97316 0%, #ec4899 40%, #a855f7 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.15;">
  PIC Data Platform
</h1>
<p style="color:#9591c4; font-size:0.95rem; margin:0 0 1.8rem;">
  Design · Fabrication · Test · AI Insights
</p>
""", unsafe_allow_html=True)

# ── DB check ────────────────────────────────────────────────────────────────
if not DB_PATH.exists():
    st.error("Database not found. Run the setup scripts first.")
    st.code("""python scripts/seed_data.py
python scripts/ingest_jsons.py
python scripts/run_ai.py
python scripts/analyze_performance.py""")
    st.stop()

# ── Load data ───────────────────────────────────────────────────────────────
designs        = load_table("designs")
fabrication    = load_table("fabrication")
tests          = load_table("tests")
results        = load_table("joined_ai_results")
foundry_summary = load_table("foundry_performance_summary")
correlations   = load_table("process_performance_correlations")

try:
    shap_values = load_table("shap_values")
    shap_ready  = True
except Exception:
    shap_values = None
    shap_ready  = False

total_devices  = len(results)
overall_yield  = results["passed"].mean() * 100
anomaly_count  = int(results["ai_flagged_anomaly"].sum())

# ── KPI bar ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Designs",         len(designs))
k2.metric("Wafers",          len(fabrication))
k3.metric("Devices Tested",  total_devices)
k4.metric("Overall Yield",   f"{overall_yield:.1f}%")
k5.metric("AI Anomalies",    anomaly_count)

st.markdown("<hr style='margin:1.2rem 0;'>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab_sim, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "  Overview  ",
    "  Designs  ",
    "  Simulations  ",
    "  Fabrication  ",
    "  Test Data  ",
    "  AI Insights  ",
    "  Process Analysis  ",
    "  Predictive Model  ",
    "  Ask  ",
])


# ── Helper: section header ──────────────────────────────────────────────────
def section(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="margin:0.4rem 0 1.2rem;">
      <h2 style="
        font-size:1.45rem; font-weight:700; margin:0;
        background:linear-gradient(90deg,#a855f7,#06b6d4);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        {title}
      </h2>
      {"<p style='color:#9591c4;font-size:0.88rem;margin:0.2rem 0 0;'>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    section("System Overview", "End-to-end lifecycle of every PIC device")

    # yield by foundry bar
    fig_yield = px.bar(
        foundry_summary, x="foundry", y="yield_rate",
        color="foundry", color_discrete_sequence=PALETTE,
        labels={"yield_rate": "Yield %", "foundry": "Foundry"},
        title="Yield Rate by Foundry",
    )
    style_fig(fig_yield)

    # pass/fail pie
    pass_counts = results["passed"].value_counts().reset_index()
    pass_counts.columns = ["passed", "count"]
    pass_counts["label"] = pass_counts["passed"].map({1: "Pass", 0: "Fail"})
    fig_pie = px.pie(
        pass_counts, names="label", values="count",
        color_discrete_sequence=["#7c3aed", "#ec4899"],
        title="Pass / Fail Distribution",
        hole=0.55,
    )
    style_fig(fig_pie)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig_yield, use_container_width=True)
    with col_b:
        st.plotly_chart(fig_pie, use_container_width=True)

    section("Joined Lifecycle Dataset")
    render_table(results)


# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    section("PIC Design Library", "Layout parameters and simulation outputs")

    # ── Device type filter ───────────────────────────────────────────────
    all_types  = sorted(designs["device_type"].dropna().unique().tolist())
    sel_types  = st.multiselect("Filter by device type", all_types, default=all_types, key="des_filter")
    des_view   = designs[designs["device_type"].isin(sel_types)] if sel_types else designs

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = px.scatter(
            des_view,
            x="waveguide_width_nm", y="simulated_loss_db",
            color="device_type", color_discrete_sequence=PALETTE,
            hover_data=["design_id", "design_version"],
            title="Simulated Loss vs Waveguide Width",
            labels={"waveguide_width_nm": "Waveguide Width (nm)", "simulated_loss_db": "Sim. Loss (dB)"},
        )
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            des_view, x="device_type",
            color="device_type", color_discrete_sequence=PALETTE,
            title="Device Type Distribution",
            labels={"device_type": "Device Type", "count": "Count"},
        )
        style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    render_table(des_view)


# ═══════════════════════════════════════════════════════════════════════════
with tab_sim:
    section("Simulation Results", "FDTD results by device type — Y-Junction vs Euler Bend")

    sim_designs = designs[designs["device_type"].isin(["y_junction", "euler_bend"])].copy()

    if sim_designs.empty:
        st.info("No simulation records yet. Run `sweep_yjunction.py` and/or `sweep_eulerbend.py` first.")
    else:
        # ── KPIs per device type ─────────────────────────────────────────
        yj  = sim_designs[sim_designs["device_type"] == "y_junction"]
        eb  = sim_designs[sim_designs["device_type"] == "euler_bend"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Y-Junction Designs",      len(yj))
        k2.metric("Euler Bend Designs",       len(eb))
        k3.metric("Y-Junction Mean Loss",
                  f"{yj['simulated_loss_db'].mean():.3f} dB" if len(yj) else "—")
        k4.metric("Euler Bend Mean Loss",
                  f"{eb['simulated_loss_db'].mean():.3f} dB" if len(eb) else "—")

        st.markdown("<hr style='margin:1.2rem 0;border-color:#e2dff5;'>", unsafe_allow_html=True)

        # ── Loss comparison box plot ─────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            fig_box = px.box(
                sim_designs, x="device_type", y="simulated_loss_db",
                color="device_type",
                color_discrete_map={"y_junction": "#7c3aed", "euler_bend": "#06b6d4"},
                title="Insertion Loss Distribution by Device Type",
                labels={"device_type": "", "simulated_loss_db": "Simulated Loss (dB)"},
                points="all",
            )
            style_fig(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

        with col2:
            fig_bw = px.box(
                sim_designs, x="device_type", y="simulated_bandwidth_ghz",
                color="device_type",
                color_discrete_map={"y_junction": "#7c3aed", "euler_bend": "#06b6d4"},
                title="Bandwidth Distribution by Device Type",
                labels={"device_type": "", "simulated_bandwidth_ghz": "Bandwidth (GHz)"},
                points="all",
            )
            style_fig(fig_bw)
            st.plotly_chart(fig_bw, use_container_width=True)

        st.markdown("<hr style='margin:1.2rem 0;border-color:#e2dff5;'>", unsafe_allow_html=True)

        # ── Parameter sweep scatter ──────────────────────────────────────
        section("Parameter Sweep", "How geometry parameters affect simulated loss")

        dev_choice = st.radio(
            "Device type", ["Y-Junction", "Euler Bend", "Both"],
            horizontal=True, key="sim_dev_choice",
        )

        sweep_df = sim_designs.copy()
        if dev_choice == "Y-Junction":
            sweep_df = yj.copy()
            x_col    = "waveguide_width_nm"
            x_label  = "Input Width w1 (nm)"
        elif dev_choice == "Euler Bend":
            sweep_df = eb.copy()
            x_col    = "ring_radius_um"
            x_label  = "Effective Radius R_eff (µm)"
        else:
            x_col   = "waveguide_width_nm"
            x_label = "Waveguide Width (nm)"

        if not sweep_df.empty:
            fig_sweep = px.scatter(
                sweep_df,
                x=x_col, y="simulated_loss_db",
                color="device_type",
                color_discrete_map={"y_junction": "#7c3aed", "euler_bend": "#06b6d4"},
                hover_data=["design_id"],
                title=f"Simulated Loss vs {x_label}",
                labels={x_col: x_label, "simulated_loss_db": "Insertion Loss (dB)"},
            )
            style_fig(fig_sweep)
            st.plotly_chart(fig_sweep, use_container_width=True)

        st.markdown("<hr style='margin:1.2rem 0;border-color:#e2dff5;'>", unsafe_allow_html=True)

        # ── Spectrum viewer ──────────────────────────────────────────────
        section("Transmission Spectrum", "Select a design to view its wavelength-resolved loss")

        spectra_dir = ROOT / "data" / "spectra"
        available   = [p.stem for p in spectra_dir.glob("*.json")] if spectra_dir.exists() else []
        sim_with_spectrum = sim_designs[sim_designs["design_id"].isin(available)]

        if sim_with_spectrum.empty:
            st.info("No spectrum files found yet.")
        else:
            sel_id = st.selectbox(
                "Design ID", sim_with_spectrum["design_id"].tolist(), key="spectrum_sel"
            )
            if sel_id:
                spec = json.loads((spectra_dir / f"{sel_id}.json").read_text())
                ldas = spec["wavelengths_um"]
                dev_type = sim_designs[sim_designs["design_id"] == sel_id]["device_type"].iloc[0]

                fig_spec = go.Figure()

                if dev_type == "y_junction":
                    fig_spec.add_trace(go.Scatter(
                        x=ldas, y=spec["insertion_loss_db"],
                        mode="lines", name="Insertion Loss",
                        line=dict(color="#7c3aed", width=2),
                    ))
                    fig_spec.update_layout(yaxis_title="Insertion Loss (dB)")

                elif dev_type == "euler_bend":
                    fig_spec.add_trace(go.Scatter(
                        x=ldas, y=spec.get("loss_euler_db", []),
                        mode="lines", name="Euler Bend",
                        line=dict(color="#06b6d4", width=2),
                    ))
                    if "loss_circular_db" in spec:
                        fig_spec.add_trace(go.Scatter(
                            x=ldas, y=spec["loss_circular_db"],
                            mode="lines", name="Circular Bend (baseline)",
                            line=dict(color="#ec4899", width=2, dash="dash"),
                        ))
                    fig_spec.update_layout(yaxis_title="Bending Loss (dB)")

                fig_spec.update_layout(
                    **PLOT_LAYOUT,
                    title=f"Transmission Spectrum — {sel_id}",
                    xaxis_title="Wavelength (µm)",
                    height=380,
                )
                st.plotly_chart(fig_spec, use_container_width=True)

        st.markdown("<hr style='margin:1.2rem 0;border-color:#e2dff5;'>", unsafe_allow_html=True)
        section("All Simulation Records")
        render_table(sim_designs)



# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    section("Fabrication & Metrology", "As-fabricated wafer data and measured geometry")

    fig = px.bar(
        fabrication, x="wafer_id", y="defect_count",
        color="foundry", color_discrete_sequence=PALETTE,
        title="Defect Count by Wafer",
        labels={"wafer_id": "Wafer", "defect_count": "Defects"},
    )
    fig.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-60)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.box(
            fabrication, x="foundry", y="sidewall_roughness_nm",
            color="foundry", color_discrete_sequence=PALETTE,
            title="Sidewall Roughness by Foundry",
        )
        style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.scatter(
            fabrication, x="film_thickness_nm", y="etch_depth_nm",
            color="foundry", color_discrete_sequence=PALETTE,
            title="Film Thickness vs Etch Depth",
        )
        style_fig(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    render_table(fabrication)


# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    section("Test Measurements", "Optical measurements and pass/fail outcomes")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            tests, x="insertion_loss_db", y="bandwidth_ghz",
            color="passed", color_discrete_map={1: "#7c3aed", 0: "#ec4899"},
            hover_data=["device_id", "wafer_id"],
            title="Insertion Loss vs Bandwidth",
            opacity=0.6,
        )
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            tests, x="insertion_loss_db",
            color_discrete_sequence=["#a855f7"],
            title="Insertion Loss Distribution",
            nbins=40,
        )
        style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    render_table(tests)


# ═══════════════════════════════════════════════════════════════════════════
with tab6:
    section("AI Anomaly Detection", "IsolationForest flags devices with unusual parameter combinations")

    fig = px.scatter(
        results,
        x="insertion_loss_db", y="bandwidth_ghz",
        color="ai_flagged_anomaly",
        color_discrete_map={False: "#7c3aed", True: "#f97316"},
        symbol="passed",
        hover_data=["device_id", "design_id", "foundry", "defect_count"],
        title="AI-Flagged Anomalies · Insertion Loss vs Bandwidth",
        opacity=0.7,
        labels={"ai_flagged_anomaly": "Anomaly", "insertion_loss_db": "Insertion Loss (dB)", "bandwidth_ghz": "Bandwidth (GHz)"},
    )
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div style="
          background:rgba(255,255,255,0.88); border:1px solid #e2dff5;
          border-left:3px solid #f97316; backdrop-filter:blur(6px);
          border-radius:12px; padding:1.4rem 1.6rem; margin-top:0.4rem;
          box-shadow:0 2px 12px rgba(124,58,237,0.07);">
          <div style="color:#7168a8;font-size:0.78rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.6rem;">
            Engineer Summary
          </div>
          <div style="font-size:1.8rem;font-weight:700;color:#f97316;">{anomaly_count}</div>
          <div style="color:#1a1040;margin:.3rem 0;">anomalous devices flagged</div>
          <hr style="border-color:#e2dff5;margin:.8rem 0;">
          <div style="font-size:1.8rem;font-weight:700;color:#7c3aed;">{overall_yield:.1f}%</div>
          <div style="color:#1a1040;margin:.3rem 0;">overall yield</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP explainability ─────────────────────────────────────────────
    st.markdown("<hr style='margin:1.4rem 0;border-color:#e2dff5;'>", unsafe_allow_html=True)
    section("Why Were They Flagged?", "SHAP values explain which features drove each anomaly flag")

    if not shap_ready:
        st.info("Re-run `python3 scripts/run_ai.py` to generate SHAP explanations.", icon="ℹ️")
    else:
        shap_feature_cols = [c for c in shap_values.columns if c.startswith("shap_")]
        feature_labels    = [c.replace("shap_", "") for c in shap_feature_cols]

        # ── Overall: mean |SHAP| for anomalous devices ──────────────────
        anomaly_shap = shap_values[shap_values["ai_flagged_anomaly"] == True]
        mean_abs_shap = anomaly_shap[shap_feature_cols].abs().mean()
        importance_df = pd.DataFrame({
            "feature":    feature_labels,
            "importance": mean_abs_shap.values,
        }).sort_values("importance", ascending=True)

        fig_imp = px.bar(
            importance_df,
            x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=[[0, "#a855f7"], [1, "#f97316"]],
            title="Mean |SHAP| for Flagged Devices — Which Features Matter Most",
            labels={"importance": "Mean |SHAP value|", "feature": ""},
        )
        fig_imp.update_coloraxes(showscale=False)
        style_fig(fig_imp)
        fig_imp.update_layout(height=420)
        st.plotly_chart(fig_imp, use_container_width=True)

        # ── Per-device: pick a flagged device and see its breakdown ─────
        st.markdown(
            "<p style='color:#9591c4;font-size:0.88rem;margin:0.4rem 0 0.6rem;'>"
            "Select a flagged device to see exactly which features pushed it into anomaly territory:</p>",
            unsafe_allow_html=True,
        )
        flagged_ids = anomaly_shap["device_id"].tolist()
        selected_id = st.selectbox("Flagged device", flagged_ids, label_visibility="collapsed")

        if selected_id:
            row     = anomaly_shap[anomaly_shap["device_id"] == selected_id].iloc[0]
            dev_row = results[results["device_id"] == selected_id].iloc[0]

            shap_vals  = [row[c] for c in shap_feature_cols]
            device_df  = pd.DataFrame({
                "feature": feature_labels,
                "shap":    shap_vals,
                "actual":  [round(dev_row[f], 3) for f in feature_labels],
            }).sort_values("shap", key=lambda s: s.abs(), ascending=True)

            device_df["direction"] = device_df["shap"].apply(
                lambda v: "Pushed toward anomaly" if v < 0 else "Pushed toward normal"
            )

            fig_dev = px.bar(
                device_df,
                x="shap", y="feature",
                orientation="h",
                color="direction",
                color_discrete_map={
                    "Pushed toward anomaly": "#f97316",
                    "Pushed toward normal":  "#7c3aed",
                },
                hover_data={"actual": True, "shap": ":.4f"},
                title=f"SHAP Breakdown · {selected_id}",
                labels={"shap": "SHAP value", "feature": "", "direction": ""},
            )
            style_fig(fig_dev)
            fig_dev.update_layout(height=420)
            fig_dev.add_vline(x=0, line_width=1, line_color="#9591c4")
            st.plotly_chart(fig_dev, use_container_width=True)


    st.markdown("<hr style='margin:1.4rem 0;border-color:#e2dff5;'>", unsafe_allow_html=True)
    section("Anomalous Devices")
    anomalies = results[results["ai_flagged_anomaly"] == 1]
    render_table(anomalies)


# ═══════════════════════════════════════════════════════════════════════════
with tab7:
    section("Process–Performance Analysis", "Fabrication metrology correlated with device performance")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            results, x="measured_width_nm", y="insertion_loss_db",
            color="foundry", color_discrete_sequence=PALETTE,
            hover_data=["device_id", "design_id", "passed"],
            title="Measured Width vs Insertion Loss",
            opacity=0.6,
        )
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            results, x="defect_count", y="bandwidth_ghz",
            color="foundry", color_discrete_sequence=PALETTE,
            hover_data=["device_id", "design_id", "passed"],
            title="Defect Count vs Bandwidth",
            opacity=0.6,
        )
        style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    section("Foundry Performance Summary")
    render_table(foundry_summary)

    section("Strongest Process–Performance Correlations")

    # Heatmap of correlations
    pivot = correlations.pivot(index="process_feature", columns="performance_metric", values="correlation")
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale=[[0, "#ec4899"], [0.5, "#f8f7ff"], [1, "#7c3aed"]],
        zmid=0,
        text=pivot.values.round(2),
        texttemplate="%{text}",
        showscale=True,
    ))
    fig_heat.update_layout(
        **PLOT_LAYOUT,
        title="Process × Performance Correlation Heatmap",
        height=320,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    strongest = correlations.iloc[0]
    st.markdown(f"""
    <div style="
      background:rgba(255,255,255,0.88); border:1px solid #e2dff5;
      border-left:3px solid #06b6d4; backdrop-filter:blur(6px);
      border-radius:12px; padding:1.2rem 1.6rem; margin-top:0.6rem;
      box-shadow:0 2px 12px rgba(124,58,237,0.07);">
      <span style="color:#7168a8;font-size:0.8rem;text-transform:uppercase;letter-spacing:.08em;">
        Strongest Relationship Detected
      </span><br>
      <span style="color:#7c3aed;font-weight:600;">{strongest['process_feature']}</span>
      <span style="color:#7168a8;"> × </span>
      <span style="color:#06b6d4;font-weight:600;">{strongest['performance_metric']}</span>
      <span style="color:#7168a8;"> · r = </span>
      <span style="color:#1a1040;font-weight:700;">{strongest['correlation']}</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
with tab8:
    section("Predictive Model", "Neural network trained to predict insertion loss from fabrication parameters")

    # ── Prediction form ───────────────────────────────────────────────────
    pred_model_result = _load_pred_model()

    if pred_model_result is None:
        st.info(
            "Run `python3 scripts/train_model.py` first to enable the prediction form.",
            icon="ℹ️",
        )
    else:
        _pred_model, _X_mean, _X_std = pred_model_result

        section("Try a Prediction", "Enter fabrication parameters and get an estimated insertion loss")

        # Defaults (typical mid-range values from the synthetic dataset)
        _PRED_DEFAULTS = {
            "pred_defect_count":         5.0,
            "pred_sidewall_roughness_nm": 3.5,
            "pred_width_error_nm":        2.0,
            "pred_film_thickness_nm":   220.0,
            "pred_etch_depth_nm":        70.0,
            "pred_measured_width_nm":   482.0,
        }
        for _k, _v in _PRED_DEFAULTS.items():
            if _k not in st.session_state:
                st.session_state[_k] = _v

        # Load Example button — fills fields from a random wafer in fabrication table
        if "pred_example_wafer" not in st.session_state:
            st.session_state["pred_example_wafer"] = fabrication.sample(1).iloc[0]
        _first = st.session_state["pred_example_wafer"]
        _ex_col, _ = st.columns([1, 5])
        with _ex_col:
            if st.button("Load example wafer", use_container_width=True):
                _first = fabrication.sample(1).iloc[0]
                st.session_state["pred_example_wafer"]         = _first
                st.session_state["pred_defect_count"]          = float(_first["defect_count"])
                st.session_state["pred_sidewall_roughness_nm"] = float(_first["sidewall_roughness_nm"])
                st.session_state["pred_width_error_nm"]        = float(_first["width_error_nm"])
                st.session_state["pred_film_thickness_nm"]     = float(_first["film_thickness_nm"])
                st.session_state["pred_etch_depth_nm"]         = float(_first["etch_depth_nm"])
                st.session_state["pred_measured_width_nm"]     = float(_first["measured_width_nm"])

        st.caption(
            f"Example loads wafer **{_first['wafer_id']}** "
            f"({_first['foundry']}, defects: {int(_first['defect_count'])}, "
            f"roughness: {_first['sidewall_roughness_nm']} nm)"
        )

        with st.form("predict_form"):
            _c1, _c2, _c3 = st.columns(3)

            with _c1:
                st.markdown(
                    "<p style='font-size:0.8rem;color:#9591c4;text-transform:uppercase;"
                    "letter-spacing:.06em;margin-bottom:0.3rem;'>Defect Count</p>",
                    unsafe_allow_html=True,
                )
                _defect = st.number_input(
                    "Defect count", min_value=0, max_value=30, step=1,
                    key="pred_defect_count", label_visibility="collapsed",
                )
                st.markdown(
                    "<p style='font-size:0.8rem;color:#9591c4;text-transform:uppercase;"
                    "letter-spacing:.06em;margin:0.8rem 0 0.3rem;'>Sidewall Roughness (nm)</p>",
                    unsafe_allow_html=True,
                )
                _roughness = st.number_input(
                    "Sidewall roughness", min_value=0.5, max_value=10.0, step=0.1, format="%.2f",
                    key="pred_sidewall_roughness_nm", label_visibility="collapsed",
                )

            with _c2:
                st.markdown(
                    "<p style='font-size:0.8rem;color:#9591c4;text-transform:uppercase;"
                    "letter-spacing:.06em;margin-bottom:0.3rem;'>Width Error (nm)</p>",
                    unsafe_allow_html=True,
                )
                _width_err = st.number_input(
                    "Width error", min_value=-60.0, max_value=60.0, step=0.5, format="%.2f",
                    key="pred_width_error_nm", label_visibility="collapsed",
                )
                st.markdown(
                    "<p style='font-size:0.8rem;color:#9591c4;text-transform:uppercase;"
                    "letter-spacing:.06em;margin:0.8rem 0 0.3rem;'>Film Thickness (nm)</p>",
                    unsafe_allow_html=True,
                )
                _film = st.number_input(
                    "Film thickness", min_value=190.0, max_value=250.0, step=0.5, format="%.2f",
                    key="pred_film_thickness_nm", label_visibility="collapsed",
                )

            with _c3:
                st.markdown(
                    "<p style='font-size:0.8rem;color:#9591c4;text-transform:uppercase;"
                    "letter-spacing:.06em;margin-bottom:0.3rem;'>Etch Depth (nm)</p>",
                    unsafe_allow_html=True,
                )
                _etch = st.number_input(
                    "Etch depth", min_value=50.0, max_value=95.0, step=0.5, format="%.2f",
                    key="pred_etch_depth_nm", label_visibility="collapsed",
                )
                st.markdown(
                    "<p style='font-size:0.8rem;color:#9591c4;text-transform:uppercase;"
                    "letter-spacing:.06em;margin:0.8rem 0 0.3rem;'>Measured Width (nm)</p>",
                    unsafe_allow_html=True,
                )
                _meas_w = st.number_input(
                    "Measured width", min_value=380.0, max_value=580.0, step=0.5, format="%.2f",
                    key="pred_measured_width_nm", label_visibility="collapsed",
                )

            _predict_clicked = st.form_submit_button("Predict →", type="primary")

        if _predict_clicked:
            _x_raw = np.array([[
                _defect, _roughness, _width_err, _film, _etch, _meas_w,
            ]], dtype=np.float32)
            _x_norm = (_x_raw - _X_mean) / _X_std
            with torch.no_grad():
                _y_hat = _pred_model(torch.FloatTensor(_x_norm.tolist())).item()

            _pass_threshold = 4.0
            _passes = _y_hat < _pass_threshold
            _color  = "#10b981" if _passes else "#f97316"
            _label  = "Expected to PASS" if _passes else "Expected to FAIL"
            _icon   = "✓" if _passes else "✗"

            st.markdown(f"""
            <div style="
              background:rgba(255,255,255,0.92); border:1px solid #e2dff5;
              border-left:4px solid {_color}; border-radius:14px;
              padding:1.4rem 1.8rem; margin-top:0.8rem;
              box-shadow:0 2px 16px rgba(124,58,237,0.08);
              display:flex; align-items:center; gap:2rem; flex-wrap:wrap;">
              <div>
                <div style="color:#9591c4;font-size:0.75rem;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:.3rem;">Predicted Insertion Loss</div>
                <div style="font-size:2.4rem;font-weight:700;color:#1a1040;line-height:1;">
                  {_y_hat:.3f} <span style="font-size:1.1rem;font-weight:400;color:#9591c4;">dB</span>
                </div>
              </div>
              <div style="border-left:1px solid #e2dff5; padding-left:2rem;">
                <div style="color:#9591c4;font-size:0.75rem;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:.3rem;">Outcome (IL &lt; 4.0 dB)</div>
                <div style="font-size:1.5rem;font-weight:700;color:{_color};">
                  {_icon} {_label}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:1.8rem 0;border-color:#e2dff5;'>", unsafe_allow_html=True)

    # Load predictions — only available after running train_model.py
    try:
        model_preds = load_table("model_predictions")
        model_ready = True
    except Exception:
        model_ready = False

    if not model_ready:
        st.info(
            "Run `python3 scripts/train_model.py` to train the model and populate this tab.",
            icon="ℹ️",
        )
    else:
        # ── KPIs ──────────────────────────────────────────────────────────
        mae  = model_preds["residual_db"].abs().mean()
        rmse = (model_preds["residual_db"] ** 2).mean() ** 0.5

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",            f"{mae:.3f} dB")
        m2.metric("RMSE",           f"{rmse:.3f} dB")
        m3.metric("Devices",        len(model_preds))

        st.markdown("<hr style='margin:1rem 0;border-color:#e2dff5;'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            section("Predicted vs Actual", "Points on the diagonal = perfect prediction")
            fig_pva = px.scatter(
                model_preds,
                x="insertion_loss_db",
                y="predicted_insertion_loss_db",
                color="foundry",
                color_discrete_sequence=PALETTE,
                hover_data=["device_id", "foundry", "residual_db"],
                labels={
                    "insertion_loss_db": "Actual (dB)",
                    "predicted_insertion_loss_db": "Predicted (dB)",
                },
                opacity=0.55,
            )
            # Perfect-prediction diagonal
            lo = model_preds["insertion_loss_db"].min()
            hi = model_preds["insertion_loss_db"].max()
            fig_pva.add_shape(
                type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                line=dict(color="#ec4899", dash="dash", width=1.5),
            )
            style_fig(fig_pva)
            st.plotly_chart(fig_pva, use_container_width=True)

        with col2:
            section("Residuals Distribution", "Residual = actual − predicted")
            fig_res = px.histogram(
                model_preds,
                x="residual_db",
                color="foundry",
                color_discrete_sequence=PALETTE,
                nbins=40,
                barmode="overlay",
                opacity=0.7,
                labels={"residual_db": "Residual (dB)"},
            )
            style_fig(fig_res)
            st.plotly_chart(fig_res, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Prediction Table")
        render_table(model_preds[[
            "device_id", "foundry",
            "insertion_loss_db", "predicted_insertion_loss_db", "residual_db",
        ]])


# ═══════════════════════════════════════════════════════════════════════════
with tab9:
    section("Ask Your Data", "Natural language → SQL → Claude interpretation")

    # ── API key check ────────────────────────────────────────────────────
    _api_key = _get_api_key()
    if not _api_key or _api_key.startswith("sk-ant-..."):
        st.warning(
            "**Add your Anthropic API key to get started.**\n\n"
            "1. Get a key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)\n"
            "2. Open `.streamlit/secrets.toml` in your project folder\n"
            "3. Replace `sk-ant-...` with your real key\n"
            "4. Save and refresh this page"
        )
        st.stop()

    # ── Session history ──────────────────────────────────────────────────
    if "ask_history" not in st.session_state:
        st.session_state.ask_history = []

    # ── Example chips ───────────────────────────────────────────────────
    st.markdown(
        "<p style='color:#9591c4;font-size:0.83rem;margin-bottom:0.4rem;'>"
        "Try an example:</p>",
        unsafe_allow_html=True,
    )
    examples = [
        "Which foundry has the highest yield rate?",
        "What process parameter most strongly predicts insertion loss?",
        "Show the top 10 devices with the most anomalies by foundry",
        "Compare average bandwidth across foundries",
        "Which wafers have defect counts above 8?",
    ]
    ex_cols = st.columns(len(examples))
    chosen_example = None
    for col, ex in zip(ex_cols, examples):
        if col.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            chosen_example = ex

    # ── Input form ──────────────────────────────────────────────────────
    with st.form("ask_form", clear_on_submit=True):
        user_q = st.text_input(
            "Question",
            value=chosen_example or "",
            placeholder="e.g. Which foundry has the lowest insertion loss?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask →", type="primary", use_container_width=False)

    # ── Process question ────────────────────────────────────────────────
    if submitted and user_q.strip():
        question = user_q.strip()

        with st.spinner("Generating SQL…"):
            try:
                sql = _ask_generate_sql(question, _api_key)
            except Exception as e:
                st.error(f"SQL generation failed: {e}")
                sql = None

        if sql:
            conn, tmp_path = _tmp_conn()
            try:
                raw_df = pd.read_sql_query(sql, conn)
                sql_error = None
            except Exception as e:
                raw_df = None
                sql_error = str(e)
            finally:
                conn.close()
                tmp_path.unlink(missing_ok=True)

            if sql_error:
                st.error(f"SQL error: {sql_error}")
            else:
                with st.spinner("Interpreting results…"):
                    try:
                        interpretation = _ask_interpret(
                            question, sql,
                            raw_df.to_string(index=False, max_rows=50),
                            _api_key,
                        )
                    except Exception as e:
                        interpretation = f"Interpretation failed: {e}"

                # Store in history (newest first)
                st.session_state.ask_history.insert(0, {
                    "question": question,
                    "sql": sql,
                    "df": raw_df,
                    "interpretation": interpretation,
                })

    # ── Render history ──────────────────────────────────────────────────
    for i, entry in enumerate(st.session_state.ask_history):
        st.markdown(
            f"<div style='"
            f"background:rgba(255,255,255,0.88);border:1px solid #e2dff5;"
            f"border-radius:14px;padding:1.4rem 1.6rem;margin:1.2rem 0;"
            f"box-shadow:0 2px 12px rgba(124,58,237,0.07);'>"
            f"<span style='color:#9591c4;font-size:0.75rem;text-transform:uppercase;"
            f"letter-spacing:.08em;'>Question</span><br>"
            f"<span style='font-weight:600;color:#1a1040;font-size:1.05rem;'>"
            f"{entry['question']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown(
                "<p style='color:#7c3aed;font-weight:600;font-size:0.9rem;"
                "text-transform:uppercase;letter-spacing:.06em;margin-bottom:0.4rem;'>"
                "Direct SQL Results</p>",
                unsafe_allow_html=True,
            )
            st.code(entry["sql"], language="sql")
            st.caption(f"{len(entry['df'])} row(s) returned")
            render_table(entry["df"])

        with right:
            st.markdown(
                "<p style='color:#06b6d4;font-weight:600;font-size:0.9rem;"
                "text-transform:uppercase;letter-spacing:.06em;margin-bottom:0.4rem;'>"
                "Claude's Analysis</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='background:rgba(248,247,255,0.95);"
                f"border:1px solid #e2dff5;border-left:3px solid #06b6d4;"
                f"border-radius:10px;padding:1.1rem 1.3rem;color:#1a1040;"
                f"font-size:0.93rem;line-height:1.65;'>"
                f"{entry['interpretation'].replace(chr(10), '<br>')}"
                f"</div>",
                unsafe_allow_html=True,
            )

        if i < len(st.session_state.ask_history) - 1:
            st.markdown("<hr style='margin:0.2rem 0;border-color:#e2dff5;'>",
                        unsafe_allow_html=True)
