"""
Streamlit UI for CS Literature Forecasting.

Multi-page app with three main workflows:
  1. Hypergraph Explorer - visualize keyword/topic network filtered by time and keywords
  2. GNN Forecast - predict top topics for a target year using THGNN
  3. Data Mining Explorer - frequent itemset mining and forecasting
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend_services import (
    DEFAULT_HYPERGRAPH_PATH,
    DEFAULT_PAPERS_CSV,
    DEFAULT_PHRASES_JSON,
    GNNDashboardResult,
    HypergraphSnapshot,
    MiningDashboardResult,
    build_hypergraph_snapshot,
    get_available_keywords,
    get_available_years,
    load_hypergraph,
    load_papers_dataframe,
    run_data_mining_dashboard,
    run_gnn_dashboard,
)


# ============================================================================
# Configuration and Styling
# ============================================================================

st.set_page_config(
    page_title="CS Literature Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
TOPIC_PALETTE = [
    "#0f766e", "#c2410c", "#1d4ed8", "#065f46", "#b45309",
    "#7c2d12", "#0f172a", "#be123c", "#0ea5e9", "#166534",
]


def inject_styles() -> None:
    """Inject custom CSS for clean, researcher-friendly UI."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --ink: rgba(255, 255, 255, 0.95);
            --muted: rgba(255, 255, 255, 0.60);
            --border: rgba(255, 255, 255, 0.12);
            --bg-light: rgba(255, 255, 255, 0.05);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--ink);
        }

        code, pre {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.85rem;
            background: var(--bg-light) !important;
            color: var(--ink) !important;
        }

        div[data-testid="stAppViewContainer"] { background: #0f0f0f; }
        div[data-testid="stSidebar"] {
            background: #1a1a1a;
            border-right: 1px solid var(--border);
        }

        h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }
        h2 { font-size: 1.3rem; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem; }
        h3 { font-size: 1.0rem; font-weight: 600; }

        .stat-card {
            background: var(--bg-light);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        .stat-label {
            font-family: 'IBM Plex Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.65rem;
            color: var(--muted);
            margin-bottom: 0.4rem;
            font-weight: 600;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--ink);
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 6px;
            border: 1px solid var(--border);
            background: rgba(100, 150, 255, 0.15);
            color: var(--ink) !important;
            font-weight: 500;
            padding: 0.5rem 1.2rem !important;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            background: rgba(100, 150, 255, 0.25) !important;
        }

        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
        .stTabs [data-baseweb="tab"] {
            background: var(--bg-light);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
            color: var(--muted) !important;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(100, 150, 255, 0.20) !important;
            border-color: rgba(100, 150, 255, 0.40) !important;
            color: var(--ink) !important;
        }

        .stDataFrame { background: var(--bg-light) !important; }
        div[data-testid="stPlotlyChart"] { borderradius: 8px; }

        input, select, textarea {
            background: var(--bg-light) !important;
            color: var(--ink) !important;
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
        }

        input::placeholder { color: var(--muted) !important; }

        label, .stSelectbox label, .stSlider label, .stNumberInput label {
            color: var(--ink) !important;
            font-weight: 500;
        }

        p, li { font-size: 0.95rem; color: var(--ink); line-height: 1.5; }
        .stCaption { font-size: 0.8rem !important; color: var(--muted) !important; }

        .help-text {
            font-size: 0.85rem;
            color: var(--muted);
            font-style: italic;
            margin-top: 0.25rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Cache Wrappers
# ============================================================================

@st.cache_resource
def cached_hypergraph(path: str) -> dict:
    """Load and cache the hypergraph."""
    return load_hypergraph(Path(path))


@st.cache_data
def cached_papers(path: str) -> pd.DataFrame:
    """Load and cache papers dataframe."""
    return load_papers_dataframe(Path(path))


@st.cache_data(show_spinner=False)
def cached_mining_result(
    papers_csv: str,
    phrases_json: str,
    min_year: int,
    max_year: int,
    target_year: int,
    minsup_frac: float,
    min_count: int,
    min_lift: float,
    max_k: int,
    baseline: str,
    window: int,
    df_max_frac: float,
    df_min_count: int,
    topm: int,
    min_active_years: int,
    keyword: Optional[str],
) -> MiningDashboardResult:
    """Compute and cache data mining dashboard results."""
    return run_data_mining_dashboard(
        papers_csv=Path(papers_csv),
        phrases_json=Path(phrases_json),
        min_year=min_year,
        max_year=max_year,
        target_year=target_year,
        minsup_frac=minsup_frac,
        min_count=min_count,
        min_lift=min_lift,
        max_k=max_k,
        baseline=baseline,
        window=window,
        df_max_frac=df_max_frac,
        df_min_count=df_min_count,
        topm=topm,
        min_active_years=min_active_years,
        keyword=keyword,
    )


@st.cache_data(show_spinner=False)
def cached_gnn_result(
    hypergraph_pt: str,
    papers_csv: Optional[str],
    train_year_min: int,
    train_year_max: int,
    gnn_window: int,
    baseline_window: int,
    families: Tuple[str, ...],
    hidden: int,
    layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    device: str,
    target_year: int,
    keyword: Optional[str],
    topn: int,
) -> GNNDashboardResult:
    """Compute and cache GNN dashboard results."""
    return run_gnn_dashboard(
        hypergraph_pt=Path(hypergraph_pt),
        papers_csv=Path(papers_csv) if papers_csv else None,
        train_year_min=train_year_min,
        train_year_max=train_year_max,
        gnn_window=gnn_window,
        baseline_window=baseline_window,
        families=families,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        seed=seed,
        device=device,
        target_year=target_year,
        keyword=keyword,
        topn=topn,
    )


# ============================================================================
# UI Helper Functions
# ============================================================================

def format_value(value: Any) -> str:
    """Format a value for display."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:,.2f}"
        return f"{value:,.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(item) for item in value)
    return str(value)


def render_stat_cards(items: Sequence[Tuple[str, Any]]) -> None:
    """Render a row of stat cards."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{format_value(value)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def apply_dark_graph_theme(fig: go.Figure) -> go.Figure:
    """Apply dark theme to Plotly figure."""
    fig.update_layout(
        font=dict(color="#ffffff", size=11),
        title_font=dict(color="#ffffff", size=15),
        title=dict(pad=dict(t=12, b=8)),
        paper_bgcolor="rgba(15, 15, 15, 1)",
        plot_bgcolor="rgba(25, 25, 25, 1)",
        hoverlabel=dict(font_color="#ffffff", bgcolor="rgba(10, 10, 10, 0.9)"),
    )
    fig.update_xaxes(
        title_font=dict(color="#ffffff"),
        tickfont=dict(color="#ffffff"),
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(255, 255, 255, 0.1)",
    )
    fig.update_yaxes(
        title_font=dict(color="#ffffff"),
        tickfont=dict(color="#ffffff"),
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(255, 255, 255, 0.1)",
    )
    if fig.layout.legend:
        fig.update_layout(legend=dict(font=dict(color="#ffffff")))
    return fig


# ============================================================================
# Page: Hypergraph Explorer
# ============================================================================

def page_hypergraph():
    """Hypergraph Explorer page."""
    st.title("🌐 Hypergraph Explorer")
    st.caption("Explore the keyword-topic network from Computer Science literature")

    # Load data
    try:
        hg = cached_hypergraph(str(DEFAULT_HYPERGRAPH_PATH))
        papers_df = cached_papers(str(DEFAULT_PAPERS_CSV))
        available_years = get_available_years(hg)
        available_keywords = get_available_keywords(hg)
    except Exception as e:
        st.error(f"Error loading hypergraph data: {e}")
        return

    if not available_years:
        st.warning("No years available in the hypergraph.")
        return

    # Sidebar filters
    st.sidebar.subheader("Hypergraph Filters")
    
    year = st.sidebar.selectbox(
        "Year",
        available_years,
        index=len(available_years) - 1,
        help="Single year to visualize"
    )

    keyword = st.sidebar.selectbox(
        "Keyword (optional)",
        [""] + available_keywords,
        help="Focus on a specific keyword/phrase"
    )

    with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
        max_papers = st.number_input("Max papers to display", value=50, min_value=5, max_value=200)
        max_keywords = st.number_input("Max topics to display", value=20, min_value=3, max_value=100)
        show_year_range = st.checkbox("Use year range", value=False)
        min_year_for_range = None
        if show_year_range:
            min_year_for_range = st.selectbox("Minimum year in range", available_years)

    # Build snapshot
    try:
        with st.spinner("Building hypergraph snapshot..."):
            snapshot = build_hypergraph_snapshot(
                hypergraph=hg,
                papers_df=papers_df,
                year=year,
                keyword=keyword if keyword else None,
                max_papers=max_papers,
                max_keywords=max_keywords,
                min_year=min_year_for_range,
            )
    except Exception as e:
        st.error(f"Error building hypergraph snapshot: {e}")
        return

    # Display stats
    st.subheader("Overview")
    render_stat_cards([
        ("Topics", snapshot.stats["selected_keywords"]),
        ("Papers", snapshot.stats["selected_papers"]),
        ("Connections", snapshot.stats["incidence_count"]),
        ("Year", snapshot.stats["year"]),
    ])

    # Display graph
    st.subheader("Network Visualization")
    if not snapshot.keyword_nodes.empty or not snapshot.paper_nodes.empty:
        fig = _build_hypergraph_figure(snapshot)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to display for the selected filters.")

    # Display topic table
    if not snapshot.keyword_nodes.empty:
        st.subheader("Topics in View")
        display_df = snapshot.keyword_nodes[["topic_name", "degree_in_view", "paper_count_in_view", "phrase_score"]].copy()
        display_df.columns = ["Topic", "Degree", "Papers", "Score"]
        st.dataframe(display_df, use_container_width=True)

    # Display paper table
    if not snapshot.paper_nodes.empty:
        st.subheader("Papers in View")
        display_df = snapshot.paper_nodes[["paper_id", "title", "year", "citation_count", "venue"]].copy()
        display_df.columns = ["Paper ID", "Title", "Year", "Citations", "Venue"]
        st.dataframe(display_df, use_container_width=True)


def _build_hypergraph_figure(snapshot: HypergraphSnapshot) -> go.Figure:
    """Build Plotly figure for hypergraph visualization."""
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=8, r=8, t=16, b=8),
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.1, 1.1]),
        showlegend=False,
        height=500,
        hovermode="closest",
    )

    if snapshot.edges.empty and snapshot.keyword_nodes.empty:
        fig.add_annotation(
            x=0.5, y=0.5,
            text="No data to display",
            showarrow=False,
            font=dict(size=16, color="#ffffff"),
        )
        return apply_dark_graph_theme(fig)

    # Position topics and papers
    keyword_pos: Dict[int, Tuple[float, float]] = {}
    paper_pos: Dict[int, Tuple[float, float]] = {}

    if not snapshot.keyword_nodes.empty:
        for idx, row in enumerate(snapshot.keyword_nodes.itertuples(index=False)):
            degree = max(1, int(row.degree_in_view))
            max_kw_degree = max(1, snapshot.keyword_nodes["degree_in_view"].max())
            degree_scale = 0.0 if max_kw_degree <= 1 else (degree - 1) / (max_kw_degree - 1)
            x = 0.15 + 0.20 * degree_scale
            y = float(row.y) + 0.01 * math.sin(idx * 1.9)
            keyword_pos[int(row.node_id)] = (x, max(0, min(1, y)))

    if not snapshot.paper_nodes.empty:
        for idx, row in enumerate(snapshot.paper_nodes.itertuples(index=False)):
            degree = max(1, int(row.num_keywords))
            max_paper_degree = max(1, snapshot.paper_nodes["num_keywords"].max())
            degree_scale = 0.0 if max_paper_degree <= 1 else (degree - 1) / (max_paper_degree - 1)
            x = 0.85 - 0.20 * degree_scale
            y = float(row.y) + 0.01 * math.cos(idx * 1.6)
            paper_pos[int(row.edge_id)] = (x, max(0, min(1, y)))

    # Draw edges
    edge_x, edge_y = [], []
    for idx, row in enumerate(snapshot.edges.itertuples(index=False)):
        x0, y0 = keyword_pos.get(int(row.node_id), (0.15, 0.5))
        x1, y1 = paper_pos.get(int(row.edge_id), (0.85, 0.5))
        curve_mag = 0.01 + 0.005 * ((idx % 6) / 5.0)
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2 + curve_mag
        edge_x.extend([x0, mid_x, x1, None])
        edge_y.extend([y0, mid_y, y1, None])

    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(color="rgba(100, 150, 255, 0.2)", width=0.8),
            hoverinfo="skip",
        )
    )

    # Draw topic nodes
    if not snapshot.keyword_nodes.empty:
        topic_ids = sorted(snapshot.keyword_nodes["topic_id"].unique())
        topic_colors = {tid: TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i, tid in enumerate(topic_ids)}
        kw_x = [keyword_pos.get(int(nid), (0.15, 0.5))[0] for nid in snapshot.keyword_nodes["node_id"]]
        kw_y = [keyword_pos.get(int(nid), (0.15, 0.5))[1] for nid in snapshot.keyword_nodes["node_id"]]
        fig.add_trace(
            go.Scatter(
                x=kw_x, y=kw_y,
                mode="markers+text",
                text=[str(l) if f else "" for l, f in zip(snapshot.keyword_nodes["label"], snapshot.keyword_nodes["is_focus"])],
                textposition="middle left",
                marker=dict(
                    size=8 + snapshot.keyword_nodes["degree_in_view"],
                    color=[topic_colors[int(t)] for t in snapshot.keyword_nodes["topic_id"]],
                    line=dict(color="#0f0f0f", width=1.5),
                    opacity=0.9,
                ),
                customdata=snapshot.keyword_nodes[["label", "topic_name"]].to_numpy(),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
            )
        )

    # Draw paper nodes
    if not snapshot.paper_nodes.empty:
        paper_x = [paper_pos.get(int(eid), (0.85, 0.5))[0] for eid in snapshot.paper_nodes["edge_id"]]
        paper_y = [paper_pos.get(int(eid), (0.85, 0.5))[1] for eid in snapshot.paper_nodes["edge_id"]]
        fig.add_trace(
            go.Scatter(
                x=paper_x, y=paper_y,
                mode="markers",
                marker=dict(
                    size=6 + snapshot.paper_nodes["num_keywords"] * 0.5,
                    symbol="diamond",
                    color="#ff9500",
                    opacity=0.8,
                    line=dict(color="#0f0f0f", width=1),
                ),
                customdata=snapshot.paper_nodes[["paper_id", "title"]].to_numpy(),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
            )
        )

    return apply_dark_graph_theme(fig)


# ============================================================================
# Page: GNN Forecast
# ============================================================================

def page_gnn_forecast():
    """GNN Forecast page."""
    st.title("🤖 GNN Forecast")
    st.caption("Predict top topics for a target year using the THGNN model")

    try:
        hg = cached_hypergraph(str(DEFAULT_HYPERGRAPH_PATH))
        available_years = get_available_years(hg)
        available_keywords = get_available_keywords(hg)
    except Exception as e:
        st.error(f"Error loading hypergraph: {e}")
        return

    if len(available_years) < 2:
        st.warning("Not enough years in the dataset to train a model.")
        return

    # Sidebar controls
    st.sidebar.subheader("Forecast Settings")
    
    target_year = st.sidebar.selectbox(
        "Target year for prediction",
        available_years[::-1],
        help="The year you want to forecast topics for"
    )

    keyword = st.sidebar.selectbox(
        "Focus keyword (optional)",
        [""] + available_keywords,
        help="Show historical trend for a specific keyword"
    )

    simple_mode = not st.sidebar.checkbox("Enable Advanced Settings")

    if simple_mode:
        # Use safe defaults
        train_year_min = max(min(available_years), target_year - 10)
        train_year_max = target_year - 1
        gnn_window = 3
        hidden = 32
        layers = 2
        dropout = 0.3
        lr = 0.001
        weight_decay = 1e-5
        epochs = 50
        baseline_window = 3
        families = ("exp", "power")
        topn = 10
        device = "cuda" if st.session_state.get("use_cuda", False) else "cpu"
    else:
        with st.sidebar.expander("Training Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                train_year_min = st.selectbox("Training year start", available_years)
                hidden = st.slider("Hidden dimension", 16, 128, 32, step=16)
                dropout = st.slider("Dropout", 0.0, 0.5, 0.3, step=0.1)
            with col2:
                train_year_max = st.selectbox("Training year end", available_years)
                layers = st.slider("Layers", 1, 5, 2)
                lr = st.slider("Learning rate", 1e-5, 1e-2, 1e-3, step=1e-4, format="%e")

        with st.sidebar.expander("Advanced Hyperparameters", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                weight_decay = st.slider("Weight decay", 0.0, 1e-3, 1e-5, step=1e-5, format="%e")
                gnn_window = st.slider("GNN window size", 1, 10, 3)
                epochs = st.slider("Epochs", 10, 200, 50, step=10)
            with col2:
                baseline_window = st.slider("Baseline window", 1, 10, 3)
                families = st.multiselect("Curve families", ["exp", "power", "logistic"], default=["exp", "power"])
                topn = st.slider("Top topics to show", 5, 30, 10)

        device = "cuda" if st.checkbox("Use CUDA (if available)") else "cpu"

    # Validate settings
    if train_year_min >= train_year_max:
        st.error("Training year start must be before end.")
        return

    if target_year <= train_year_max:
        st.error("Target year must be after training period.")
        return

    if not families:
        st.error("Select at least one curve family.")
        return

    # Run forecast
    if st.button("▶️ Run Forecast", key="gnn_forecast_button"):
        with st.spinner("Running GNN forecast..."):
            try:
                result = cached_gnn_result(
                    hypergraph_pt=str(DEFAULT_HYPERGRAPH_PATH),
                    papers_csv=str(DEFAULT_PAPERS_CSV),
                    train_year_min=train_year_min,
                    train_year_max=train_year_max,
                    gnn_window=gnn_window,
                    baseline_window=baseline_window,
                    families=tuple(families),
                    hidden=hidden,
                    layers=layers,
                    dropout=dropout,
                    lr=lr,
                    weight_decay=weight_decay,
                    epochs=epochs,
                    seed=42,
                    device=device,
                    target_year=target_year,
                    keyword=keyword if keyword else None,
                    topn=topn,
                )

                # Display results
                st.success("Forecast complete!")

                st.subheader("Overview")
                render_stat_cards([
                    ("Forecast Year", result.stats["forecast_year"]),
                    ("Top Topic", result.stats["focus_topic_name"]),
                    ("Training Years", result.stats["training_years"]),
                    ("Device", result.stats["device"].upper()),
                ])

                # Top topics table
                st.subheader("Top Predicted Topics")
                display_df = result.top_topics[["rank", "topic_name", "fused_score"]].copy()
                display_df.columns = ["Rank", "Topic", "Score"]
                display_df["Score"] = display_df["Score"].round(4)
                st.dataframe(display_df, use_container_width=True)

                # Historical trend chart
                if not result.focus_topic_history.empty:
                    st.subheader("Historical Trend for Focus Topic")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result.focus_topic_history["year"],
                        y=result.focus_topic_history["usage"],
                        mode="lines+markers",
                        name="Usage",
                    ))
                    fig.update_layout(
                        xaxis_title="Year",
                        yaxis_title="Usage",
                        height=350,
                    )
                    fig = apply_dark_graph_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)

                # Store in session state for potential comparison
                st.session_state["last_gnn_result"] = result
                st.session_state["last_gnn_target_year"] = target_year

            except Exception as e:
                st.error(f"Error running forecast: {e}")
                import traceback
                st.error(traceback.format_exc())


# ============================================================================
# Page: Data Mining Explorer
# ============================================================================

def page_data_mining():
    """Data Mining Explorer page."""
    st.title("⛏️ Data Mining Explorer")
    st.caption("Discover and forecast frequent itemsets from keyword transactions")

    try:
        hg = cached_hypergraph(str(DEFAULT_HYPERGRAPH_PATH))
        available_years = sorted(get_available_years(hg))
        available_keywords = get_available_keywords(hg)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    if not available_years or len(available_years) < 2:
        st.warning("Not enough years in the dataset.")
        return

    # Sidebar controls
    st.sidebar.subheader("Mining Settings")

    target_year = st.sidebar.selectbox(
        "Forecast year",
        available_years[::-1],
        help="Year to forecast itemsets for"
    )

    min_year = st.sidebar.selectbox(
        "Analysis start year",
        available_years,
        help="Earliest year to include"
    )

    max_year = st.sidebar.selectbox(
        "Analysis end year",
        available_years,
        index=len(available_years) - 1,
        help="Latest year to include"
    )

    keyword = st.sidebar.selectbox(
        "Keyword filter (optional)",
        [""] + available_keywords,
    )

    simple_mode = not st.sidebar.checkbox("Enable Advanced Settings")

    if simple_mode:
        minsup_frac = 0.05
        min_count = 5
        min_lift = 1.2
        window = 3
        baseline = "exp"
        topm = 10
    else:
        with st.sidebar.expander("Mining Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                minsup_frac = st.slider("Min support fraction", 0.01, 0.20, 0.05, step=0.01)
                min_count = st.number_input("Min item count", 1, 100, 5)
            with col2:
                min_lift = st.slider("Min lift", 1.0, 3.0, 1.2, step=0.1)
                window = st.slider("Forecast window (years)", 1, 10, 3)

        with st.sidebar.expander("Advanced Options", expanded=False):
            baseline = st.selectbox("Baseline method", ["exp", "power", "logistic"])
            topm = st.slider("Top itemsets to show", 5, 50, 10)

    # Validate
    if min_year > max_year:
        st.error("Start year must be before end year.")
        return

    if target_year <= max_year:
        st.error("Forecast year must be after analysis period.")
        return

    # Run mining
    if st.button("▶️ Run Mining Analysis", key="mining_button"):
        with st.spinner("Running data mining analysis..."):
            try:
                result = cached_mining_result(
                    papers_csv=str(DEFAULT_PAPERS_CSV),
                    phrases_json=str(DEFAULT_PHRASES_JSON),
                    min_year=min_year,
                    max_year=max_year,
                    target_year=target_year,
                    minsup_frac=minsup_frac,
                    min_count=min_count,
                    min_lift=min_lift,
                    max_k=3,
                    baseline=baseline,
                    window=window,
                    df_max_frac=0.95,
                    df_min_count=3,
                    topm=topm,
                    min_active_years=2,
                    keyword=keyword if keyword else None,
                )

                st.success("Mining complete!")

                # Display stats
                st.subheader("Dataset Overview")
                render_stat_cards([
                    ("Papers analyzed", result.stats["papers_with_phrases"]),
                    ("Phrase vocabulary", result.stats["phrase_vocabulary"]),
                    ("Papers in year", result.stats["documents_in_year"]),
                    ("Modeled itemsets", result.stats["modeled_itemsets"]),
                ])

                # Tabs for results
                tab1, tab2, tab3, tab4 = st.tabs(["Top Itemsets", "Pairs", "Triples", "Trends"])

                with tab1:
                    st.subheader("Predicted Top Itemsets for Next Year")
                    if not result.predicted_itemsets.empty:
                        display_df = result.predicted_itemsets[["itemset", "k", "score", "score_weighted"]].copy()
                        display_df.columns = ["Itemset", "Size", "Score", "Weighted Score"]
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.info("No predicted itemsets.")

                with tab2:
                    st.subheader("Top Pairs by Support")
                    if not result.top_support_pairs.empty:
                        st.dataframe(result.top_support_pairs, use_container_width=True)
                    else:
                        st.info("No pairs found.")

                with tab3:
                    st.subheader("Top Triples")
                    if not result.top_triples.empty:
                        st.dataframe(result.top_triples, use_container_width=True)
                    else:
                        st.info("No triples found.")

                with tab4:
                    st.subheader("Lead Itemset Trend Over Time")
                    if not result.lead_itemset_series.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=result.lead_itemset_series["year"],
                            y=result.lead_itemset_series["support"],
                            mode="lines+markers",
                            name="Support",
                        ))
                        fig.update_layout(
                            xaxis_title="Year",
                            yaxis_title="Support",
                            height=350,
                        )
                        fig = apply_dark_graph_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No trend data available.")

                st.session_state["last_mining_result"] = result
                st.session_state["last_mining_target_year"] = target_year

            except Exception as e:
                st.error(f"Error running mining: {e}")
                import traceback
                st.error(traceback.format_exc())


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main app entry point."""
    inject_styles()

    # Sidebar navigation
    st.sidebar.title("📚 CS Forecasting")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Select a page:",
        ["Hypergraph Explorer", "GNN Forecast", "Data Mining Explorer"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption("Multi-workflow exploratory analysis for CS literature trends")

    # Route to page
    if page == "Hypergraph Explorer":
        page_hypergraph()
    elif page == "GNN Forecast":
        page_gnn_forecast()
    elif page == "Data Mining Explorer":
        page_data_mining()


if __name__ == "__main__":
    main()
