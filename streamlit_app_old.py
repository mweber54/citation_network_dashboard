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
    """Inject custom CSS for clean, researcher-friendly styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

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
            background: rgba(255, 255, 255, 0.06) !important;
            color: var(--ink) !important;
        }

        div[data-testid="stAppViewContainer"] {
            background: #0f0f0f;
        }

        div[data-testid="stSidebar"] {
            background: #1a1a1a;
            border-right: 1px solid var(--border);
        }

        h1 { font-size: 1.22rem !important; letter-spacing: -0.03em; margin-bottom: 0.15rem !important; color: var(--ink) !important; }
        h2 { font-size: 1.00rem !important; letter-spacing: -0.02em; margin-bottom: 0.10rem !important; color: var(--ink) !important; }
        h3 { font-size: 0.93rem !important; letter-spacing: -0.02em; color: var(--ink) !important; }

        .page-header {
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
            padding: 0.45rem 0 0.35rem 0;
            border-bottom: 1px solid var(--line);
            margin-bottom: 0.6rem;
        }
        .page-header-title {
            font-size: 1.12rem;
            font-weight: 700;
            color: var(--ink);
            letter-spacing: -0.03em;
        }
        .page-header-sub {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.5rem 0.7rem;
            box-shadow: 0 6px 18px rgba(90, 110, 255, 0.05);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }

        .stat-label {
            font-family: 'IBM Plex Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-size: 0.60rem;
            color: var(--muted);
            margin-bottom: 0.15rem;
        }

        .stat-value {
            font-size: 1.05rem;
            line-height: 1.15;
            font-weight: 700;
            color: var(--ink);
        }

        .section-title {
            margin-top: 0.25rem;
            margin-bottom: 0.10rem;
            font-size: 0.94rem !important;
            color: var(--ink) !important;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.79rem;
            margin-bottom: 0.5rem;
        }

        /* Style the actual Streamlit iframe element that follows the graph anchor marker. */
        div[data-testid="stElementContainer"]:has(.graph-anchor) + div[data-testid="stElementContainer"] {
            width: 100% !important;
        }

        div[data-testid="stElementContainer"]:has(.graph-anchor) + div[data-testid="stElementContainer"] iframe {
            width: 100% !important;
            max-width: none !important;
            display: block !important;
            margin: 0 auto !important;
            border: 1px solid var(--line) !important;
            border-radius: 20px !important;
            box-shadow: 0 10px 30px rgba(90, 110, 255, 0.08) !important;
            background: rgba(14, 18, 32, 0.50) !important;
            overflow: hidden !important;
        }

        .panel-note {
            font-size: 0.77rem;
            color: var(--muted);
            margin-top: -0.10rem;
            margin-bottom: 0.5rem;
        }

        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            background: linear-gradient(180deg, rgba(90, 110, 255, 0.20), rgba(100, 240, 220, 0.12));
            color: var(--ink) !important;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            padding: 0.3rem 1rem !important;
            box-shadow: 0 6px 16px rgba(90, 110, 255, 0.12);
        }

        .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
            background: linear-gradient(180deg, rgba(90, 110, 255, 0.30), rgba(100, 240, 220, 0.20)) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            border: 1px solid var(--line);
            padding: 0.22rem 0.7rem;
            font-size: 0.80rem;
            color: var(--muted) !important;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(90, 110, 255, 0.20) !important;
            border: 1px solid rgba(90, 110, 255, 0.30) !important;
            color: var(--ink) !important;
        }

        .stDataFrame, div[data-testid="stPlotlyChart"] {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 14px;
        }

        input, select, textarea {
            background: rgba(255, 255, 255, 0.08) !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
        }

        input::placeholder {
            color: var(--muted) !important;
        }

        label, .stSelectbox label, .stSlider label, .stNumberInput label,
        .stMultiSelect label, .stToggle label {
            font-size: 0.79rem !important;
            color: var(--ink) !important;
        }

        div[data-testid="stForm"] label,
        div[data-testid="stForm"] label p,
        div[data-testid="stForm"] label span,
        div[data-testid="stForm"] [data-testid="stWidgetLabel"] p,
        div[data-testid="stForm"] [data-testid="stWidgetLabel"] span,
        div[data-testid="stForm"] .stMarkdown p,
        div[data-testid="stForm"] .stRadio p,
        div[data-testid="stForm"] .stCheckbox p,
        div[data-testid="stForm"] .stSelectbox p,
        div[data-testid="stForm"] .stMultiSelect p,
        div[data-testid="stForm"] .stSlider p,
        div[data-testid="stForm"] .stNumberInput p,
        div[data-testid="stForm"] .stToggle p {
            color: var(--ink) !important;
        }

        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] {
            font-size: 0.69rem;
            color: var(--muted) !important;
        }

        p, li { font-size: 0.84rem; color: var(--ink); }
        .stCaption { font-size: 0.74rem !important; color: var(--muted) !important; }

        .stMarkdown, .stText, .stCaption, .stSelectbox, .stMultiSelect,
        .stSlider, .stNumberInput, .stToggle, .stRadio, .stCheckbox {
            color: var(--ink) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def cached_hypergraph(path: str, _version: int) -> dict:
    return load_hypergraph(Path(path))


@st.cache_data
def cached_papers(path: str):
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


def format_value(value: Any) -> str:
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


def apply_black_graph_text(fig: go.Figure) -> go.Figure:
    """Force all Plotly text elements to white for readability on dark backgrounds."""
    fig.update_layout(
        font=dict(color="#ffffff", size=12),
        title_font=dict(color="#ffffff", size=17),
        title=dict(pad=dict(t=12, b=8)),
        hoverlabel=dict(font_color="#ffffff", bgcolor="rgba(10,13,20,0.92)"),
    )
    fig.update_xaxes(
        title_font=dict(color="#ffffff", size=12),
        tickfont=dict(color="#ffffff", size=11),
        title_standoff=12,
        automargin=True,
    )
    fig.update_yaxes(
        title_font=dict(color="#ffffff", size=12),
        tickfont=dict(color="#ffffff", size=11),
        title_standoff=14,
        automargin=True,
    )
    legend_cfg = fig.layout.legend.to_plotly_json() if fig.layout.legend else {}
    legend_cfg["font"] = dict(color="#ffffff")
    legend_cfg["title"] = dict(font=dict(color="#ffffff"))
    fig.update_layout(legend=legend_cfg)
    return fig


def build_hypergraph_figure(snapshot: HypergraphSnapshot) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=8, r=8, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.24)",
        xaxis=dict(visible=False, range=[-0.02, 1.02]),
        yaxis=dict(visible=False, range=[-0.06, 1.06]),
        showlegend=False,
        height=520,
        hoverlabel=dict(bgcolor="rgba(10,13,20,0.92)", font_size=12, font_family="Space Grotesk", font_color="#ffffff"),
    )

    if snapshot.edges.empty and snapshot.keyword_nodes.empty:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="No hypergraph slice matches the current year and keyword filter.",
            showarrow=False,
            font=dict(size=16, color="#ffffff"),
        )
        return apply_black_graph_text(fig)

    keyword_pos: dict[int, Tuple[float, float]] = {}
    paper_pos: dict[int, Tuple[float, float]] = {}

    if not snapshot.keyword_nodes.empty:
        max_kw_degree = max(int(v) for v in snapshot.keyword_nodes["degree_in_view"].tolist())
        for idx, row in enumerate(snapshot.keyword_nodes.itertuples(index=False)):
            degree = max(1, int(row.degree_in_view))
            degree_scale = 0.0 if max_kw_degree <= 1 else (degree - 1) / float(max_kw_degree - 1)
            x = 0.12 + 0.30 * degree_scale
            y = min(1.0, max(0.0, float(row.y) + 0.012 * math.sin(idx * 1.9)))
            keyword_pos[int(row.node_id)] = (x, y)

    if not snapshot.paper_nodes.empty:
        max_paper_degree = max(int(v) for v in snapshot.paper_nodes["num_keywords"].tolist())
        for idx, row in enumerate(snapshot.paper_nodes.itertuples(index=False)):
            degree = max(1, int(row.num_keywords))
            degree_scale = 0.0 if max_paper_degree <= 1 else (degree - 1) / float(max_paper_degree - 1)
            x = 0.88 - 0.30 * degree_scale
            y = min(1.0, max(0.0, float(row.y) + 0.012 * math.cos(idx * 1.6)))
            paper_pos[int(row.edge_id)] = (x, y)

    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    for idx, row in enumerate(snapshot.edges.itertuples(index=False)):
        x0, y0 = keyword_pos.get(int(row.node_id), (float(row.x0), float(row.y0)))
        x1, y1 = paper_pos.get(int(row.edge_id), (float(row.x1), float(row.y1)))
        curve_sign = 1.0 if ((idx % 2) == 0) else -1.0
        curve_mag = 0.015 + 0.010 * ((idx % 6) / 5.0)
        mid_x = (x0 + x1) / 2.0
        mid_y = min(1.0, max(0.0, (y0 + y1) / 2.0 + curve_sign * curve_mag))
        edge_x.extend([x0, mid_x, x1, None])
        edge_y.extend([y0, mid_y, y1, None])

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(15, 118, 110, 0.24)", width=1.1, shape="spline", smoothing=0.95),
            hoverinfo="skip",
        )
    )

    if not snapshot.keyword_nodes.empty:
        topic_ids = sorted(snapshot.keyword_nodes["topic_id"].unique().tolist())
        topic_colors = {
            topic_id: TOPIC_PALETTE[idx % len(TOPIC_PALETTE)]
            for idx, topic_id in enumerate(topic_ids)
        }
        marker_sizes = [5.5 + 1.3 * max(1, degree) for degree in snapshot.keyword_nodes["degree_in_view"]]
        marker_colors = [topic_colors[topic_id] for topic_id in snapshot.keyword_nodes["topic_id"]]
        marker_lines = [2.0 if is_focus else 0.7 for is_focus in snapshot.keyword_nodes["is_focus"]]
        top_keyword_ids = set(
            int(v)
            for v in snapshot.keyword_nodes.sort_values("degree_in_view", ascending=False).head(6)["node_id"].tolist()
        )
        keyword_x = [keyword_pos.get(int(node_id), (x, y))[0] for node_id, x, y in zip(snapshot.keyword_nodes["node_id"], snapshot.keyword_nodes["x"], snapshot.keyword_nodes["y"])]
        keyword_y = [keyword_pos.get(int(node_id), (x, y))[1] for node_id, x, y in zip(snapshot.keyword_nodes["node_id"], snapshot.keyword_nodes["x"], snapshot.keyword_nodes["y"])]
        keyword_labels = [
            str(label) if (int(node_id) in top_keyword_ids or bool(is_focus)) else ""
            for node_id, label, is_focus in zip(
                snapshot.keyword_nodes["node_id"],
                snapshot.keyword_nodes["label"],
                snapshot.keyword_nodes["is_focus"],
            )
        ]
        fig.add_trace(
            go.Scatter(
                x=keyword_x,
                y=keyword_y,
                mode="markers+text",
                text=keyword_labels,
                textposition="middle left",
                textfont=dict(size=10, color="#ffffff"),
                marker=dict(
                    size=marker_sizes,
                    color=marker_colors,
                    line=dict(color="#0f172a", width=marker_lines),
                    opacity=0.94,
                ),
                customdata=snapshot.keyword_nodes[["label", "topic_name", "degree_in_view", "phrase_score"]].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Topic: %{customdata[1]}<br>"
                    "Degree in view: %{customdata[2]}<br>"
                    "Score norm: %{customdata[3]:.3f}<extra></extra>"
                ),
            )
        )

    if not snapshot.paper_nodes.empty:
        paper_sizes = [5.0 + 1.1 * max(1, size) for size in snapshot.paper_nodes["num_keywords"]]
        paper_x = [paper_pos.get(int(edge_id), (x, y))[0] for edge_id, x, y in zip(snapshot.paper_nodes["edge_id"], snapshot.paper_nodes["x"], snapshot.paper_nodes["y"])]
        paper_y = [paper_pos.get(int(edge_id), (x, y))[1] for edge_id, x, y in zip(snapshot.paper_nodes["edge_id"], snapshot.paper_nodes["x"], snapshot.paper_nodes["y"])]
        fig.add_trace(
            go.Scatter(
                x=paper_x,
                y=paper_y,
                mode="markers",
                marker=dict(
                    size=paper_sizes,
                    symbol="diamond",
                    color="#b45309",
                    opacity=0.86,
                    line=dict(color="#0f172a", width=0.9),
                ),
                customdata=snapshot.paper_nodes[["paper_id", "title", "year", "citation_count", "venue"]].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}<br>"
                    "Year: %{customdata[2]}<br>"
                    "Citations: %{customdata[3]}<br>"
                    "Venue: %{customdata[4]}<extra></extra>"
                ),
            )
        )

    fig.add_annotation(x=0.20, y=1.03, text="Topic nodes", showarrow=False, font=dict(size=11, color="#ffffff"))
    fig.add_annotation(x=0.80, y=1.03, text="Paper nodes", showarrow=False, font=dict(size=11, color="#ffffff"))
    return apply_black_graph_text(fig)


def build_3d_hypergraph_html(snapshot: HypergraphSnapshot) -> str:
    """Build a stationary interactive hypergraph visualization with topic nodes only."""
    import json

    if snapshot.edges.empty and snapshot.keyword_nodes.empty:
        return "<div style='width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:rgba(255,255,255,0.65);background:linear-gradient(180deg,#0e1220,#0a0d14);'>No data</div>"

    nodes_data: List[dict] = []
    node_id_map: dict[int, int] = {}
    if not snapshot.keyword_nodes.empty:
        for row in snapshot.keyword_nodes.itertuples(index=False):
            topic_id = int(row.node_id)
            node_id_map[topic_id] = len(nodes_data)
            nodes_data.append(
                {
                    "id": topic_id,
                    "type": "topic",
                    "label": str(row.label),
                    "topic": str(row.topic_name),
                    "degree": int(row.degree_in_view),
                    "paperCount": int(getattr(row, "paper_count_in_view", row.degree_in_view)),
                }
            )

    paper_meta: dict[int, dict] = {}
    if not snapshot.paper_nodes.empty:
        for row in snapshot.paper_nodes.itertuples(index=False):
            edge_id = int(row.edge_id)
            paper_meta[edge_id] = {
                "paper_id": str(row.paper_id),
                "paper_label": str(row.title),
            }

    # Convert each paper hyperedge into topic-topic links for rendering.
    edges_data: List[dict] = []
    if not snapshot.edges.empty:
        grouped = snapshot.edges.groupby("edge_id")["node_id"].apply(lambda values: sorted({int(v) for v in values}))
        for edge_id, topic_ids in grouped.items():
            valid_topic_indices = [node_id_map[topic_id] for topic_id in topic_ids if topic_id in node_id_map]
            if len(valid_topic_indices) < 2:
                continue
            meta = paper_meta.get(int(edge_id), {})
            for i in range(len(valid_topic_indices)):
                for j in range(i + 1, len(valid_topic_indices)):
                    edges_data.append(
                        {
                            "a": int(valid_topic_indices[i]),
                            "b": int(valid_topic_indices[j]),
                            "edge_id": int(edge_id),
                            "paper_id": str(meta.get("paper_id", "n/a")),
                            "paper_label": str(meta.get("paper_label", f"paper_{int(edge_id)}")),
                        }
                    )

    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
        * {{ box-sizing: border-box; }}
        html, body {{
            margin: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background:
                radial-gradient(circle at 50% 48%, rgba(90,110,255,0.11), transparent 33%),
                radial-gradient(circle at 50% 56%, rgba(100,240,220,0.05), transparent 42%),
                linear-gradient(180deg, #0e1220, #0a0d14);
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
            color: rgba(255,255,255,0.92);
        }}
        #root {{ position: relative; width: 100%; height: 100%; overflow: hidden; }}
        canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; display: block; cursor: grab; }}
        canvas.dragging {{ cursor: grabbing; }}
        .badge {{
            position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
            background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.13);
            border-radius: 12px; padding: 6px 10px; font-size: 11px; color: rgba(255,255,255,0.62);
            pointer-events: none; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
        }}
        .info {{
            position: absolute; left: 10px; bottom: 10px; right: 10px;
            padding: 8px 10px; border-radius: 10px;
            background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14);
            color: rgba(255,255,255,0.86); font-size: 12px;
            backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <div id="root">
        <canvas id="graph"></canvas>
        <div class="badge">Stationary 3D Hypergraph · links show papers shared across topics · drag background to rotate</div>
        <div class="info" id="info">Click a topic node or a hyperedge link for details.</div>
    </div>

    <script>
        const nodesData = {nodes_json};
        const rawEdges = {edges_json};
        const canvas = document.getElementById("graph");
        const info = document.getElementById("info");
        const ctx = canvas.getContext("2d");

        let w = 0, h = 0, dpr = 1;
        let yaw = 0.8, pitch = -0.32, zoom = 2.2;
        let pointerDown = false;
        let draggingNode = null;
        let rotating = false;
        let moved = false;
        let lastX = 0, lastY = 0;
        let selectedNode = null;
        let selectedEdgeUid = null;

        const nodeScreenOffset = new Map();
        const TAU = Math.PI * 2;
        const CLUSTER_COUNT = Math.max(8, Math.min(96, Math.ceil(Math.sqrt(Math.max(1, nodesData.length)) * 2.2)));

        function rand(a = 0, b = 1) {{ return a + Math.random() * (b - a); }}
        function rotateY(p, a) {{ const c = Math.cos(a), s = Math.sin(a); return {{ x: p.x * c + p.z * s, y: p.y, z: -p.x * s + p.z * c }}; }}
        function rotateX(p, a) {{ const c = Math.cos(a), s = Math.sin(a); return {{ x: p.x, y: p.y * c - p.z * s, z: p.y * s + p.z * c }}; }}

        function resizeCanvas() {{
            w = canvas.parentElement.clientWidth;
            h = canvas.parentElement.clientHeight;
            dpr = Math.min(window.devicePixelRatio || 1, 2);
            canvas.width = Math.max(1, Math.floor(w * dpr));
            canvas.height = Math.max(1, Math.floor(h * dpr));
            canvas.style.width = w + "px";
            canvas.style.height = h + "px";
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            render();
        }}

        function buildScene() {{
            const baseRadius = Math.max(130, Math.min(w, h) * 0.46);
            const spread = Math.max(26, Math.min(w, h) * 0.11);
            const clusters = [];
            const nodes = [];

            for (let i = 0; i < CLUSTER_COUNT; i++) {{
                const a = (i / CLUSTER_COUNT) * TAU;
                clusters.push({{
                    x: Math.cos(a) * baseRadius,
                    y: Math.sin(a * 1.6) * baseRadius * 0.25,
                    z: Math.sin(a) * baseRadius * 0.5,
                }});
            }}

            for (let i = 0; i < nodesData.length; i++) {{
                const c = clusters[i % CLUSTER_COUNT];
                const ringIdx = Math.floor(i / CLUSTER_COUNT);
                const ringA = (ringIdx * 2.399963229728653) % TAU;
                const ringR = Math.min(spread, 6 + Math.sqrt(ringIdx + 1) * 3.4);
                nodes.push({{
                    data: nodesData[i],
                    base: {{
                        x: c.x + Math.cos(ringA) * ringR + rand(-2.2, 2.2),
                        y: c.y + Math.sin(ringA * 1.1) * ringR + rand(-2.2, 2.2),
                        z: c.z + Math.sin(ringA) * ringR * 0.8 + rand(-2.2, 2.2),
                    }},
                }});
                nodeScreenOffset.set(i, {{ x: 0, y: 0 }});
            }}
            return nodes;
        }}

        function getProjection(nodes) {{
            const cameraZ = Math.max(680, Math.min(w, h) * 1.6) / zoom;
            const projection = Math.max(460, Math.min(w, h) * 1.06);
            const transformed = [];
            const projected = [];

            for (let i = 0; i < nodes.length; i++) {{
                let p = nodes[i].base;
                p = rotateY(p, yaw);
                p = rotateX(p, pitch);
                transformed.push(p);
                const dz = cameraZ - p.z;
                if (dz < 140) {{ projected.push(null); continue; }}
                const f = Math.min(2.2, projection / dz);
                projected.push({{ x: w * 0.5 + p.x * f, y: h * 0.5 + p.y * f, s: f, z: p.z }});
            }}
            return {{ transformed, projected }};
        }}

        function fitProjected(projected) {{
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (const p of projected) {{
                if (!p) continue;
                if (p.x < minX) minX = p.x;
                if (p.x > maxX) maxX = p.x;
                if (p.y < minY) minY = p.y;
                if (p.y > maxY) maxY = p.y;
            }}
            if (!isFinite(minX) || !isFinite(maxX) || !isFinite(minY) || !isFinite(maxY)) return {{ scale: 1, dx: 0, dy: 0 }};
            const bw = Math.max(1, maxX - minX), bh = Math.max(1, maxY - minY);
            const scale = Math.max(0.30, Math.min(2.8, Math.min((w * 0.95) / bw, (h * 0.90) / bh)));
            const cx = (minX + maxX) * 0.5, cy = (minY + maxY) * 0.5;
            return {{ scale, dx: w * 0.5 - cx * scale, dy: h * 0.5 - cy * scale }};
        }}

        function toScreen(p, fit, idx) {{
            const off = nodeScreenOffset.get(idx) || {{ x: 0, y: 0 }};
            return {{ x: p.x * fit.scale + fit.dx + off.x, y: p.y * fit.scale + fit.dy + off.y, s: p.s * fit.scale }};
        }}

        function distPointToSegment(px, py, ax, ay, bx, by) {{
            const dx = bx - ax, dy = by - ay;
            const denom = dx * dx + dy * dy;
            if (denom === 0) return Math.hypot(px - ax, py - ay);
            const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / denom));
            const cx = ax + t * dx, cy = ay + t * dy;
            return Math.hypot(px - cx, py - cy);
        }}

        const sceneNodes = buildScene();
        let cached = null;

        function computeCached() {{
            const {{ transformed, projected }} = getProjection(sceneNodes);
            const fit = fitProjected(projected);
            const screenNodes = projected.map((p, idx) => (p ? toScreen(p, fit, idx) : null));
            const edges = [];
            for (let i = 0; i < rawEdges.length; i++) {{
                const e = rawEdges[i];
                const a = Number(e.a);
                const b = Number(e.b);
                if (!screenNodes[a] || !screenNodes[b]) continue;
                edges.push({{
                    a,
                    b,
                    edgeId: Number(e.edge_id),
                    paperId: String(e.paper_id || "n/a"),
                    paperLabel: String(e.paper_label || `paper_${{e.edge_id}}`),
                    uid: `${{e.edge_id}}:${{a}}:${{b}}:${{i}}`,
                    az: transformed[a].z,
                    bz: transformed[b].z,
                }});
            }}
            cached = {{ transformed, projected, fit, screenNodes, edges }};
        }}

        function drawGlow() {{
            const r = Math.min(w, h) * 0.33;
            const g = ctx.createRadialGradient(w * 0.5, h * 0.5, 0, w * 0.5, h * 0.5, r);
            g.addColorStop(0, "rgba(95,115,255,0.09)");
            g.addColorStop(0.46, "rgba(95,115,255,0.025)");
            g.addColorStop(1, "rgba(95,115,255,0)");
            ctx.fillStyle = g; ctx.beginPath(); ctx.arc(w * 0.5, h * 0.5, r, 0, TAU); ctx.fill();
        }}

        function render() {{
            if (!w || !h) return;
            computeCached();
            ctx.clearRect(0, 0, w, h);
            drawGlow();

            const edgeItems = cached.edges.slice().sort((e1, e2) => (e1.az + e1.bz) - (e2.az + e2.bz));
            for (const e of edgeItems) {{
                const a = cached.screenNodes[e.a], b = cached.screenNodes[e.b];
                const selected = selectedEdgeUid === e.uid;
                ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
                ctx.strokeStyle = selected ? "rgba(255,208,120,0.95)" : "rgba(135,155,255,0.30)";
                ctx.lineWidth = selected ? 2.2 : Math.max(0.75, 0.95 * cached.fit.scale);
                ctx.stroke();
            }}

            const nodeItems = [];
            for (let i = 0; i < cached.screenNodes.length; i++) {{
                const p = cached.screenNodes[i];
                if (!p) continue;
                nodeItems.push({{ idx: i, z: cached.transformed[i].z, p, data: nodesData[i] }});
            }}
            nodeItems.sort((a, b) => a.z - b.z);

            for (const n of nodeItems) {{
                const densityScale = nodesData.length > 600 ? 0.66 : (nodesData.length > 300 ? 0.78 : 1.0);
                const r = Math.max(2.6, Math.min(9.8, n.p.s * 0.024 * densityScale));
                const selected = selectedNode === n.idx;
                ctx.beginPath();
                ctx.arc(n.p.x, n.p.y, r, 0, TAU); ctx.fillStyle = "rgba(100,240,220,0.95)";
                ctx.fill();
                if (selected) {{
                    ctx.beginPath(); ctx.arc(n.p.x, n.p.y, r + 3.5, 0, TAU);
                    ctx.strokeStyle = "rgba(255,220,140,0.95)"; ctx.lineWidth = 1.6; ctx.stroke();
                }}
            }}

            // Draw topic label only for the currently selected topic.
            ctx.font = '600 11px "Space Grotesk", "Segoe UI", sans-serif';
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            for (const n of nodeItems) {{
                if (n.data.type !== "topic") continue;
                const selected = selectedNode === n.idx;
                if (!selected) continue;
                const label = String(n.data.label || "");
                if (!label) continue;
                const x = n.p.x + 7;
                const y = n.p.y;
                ctx.lineWidth = 3.2;
                ctx.strokeStyle = "rgba(10,13,20,0.95)";
                ctx.strokeText(label, x, y);
                ctx.fillStyle = selected ? "rgba(255,240,180,0.98)" : "rgba(235,242,255,0.95)";
                ctx.fillText(label, x, y);
            }}
        }}

        function hitNode(x, y) {{
            let best = null;
            for (let i = 0; i < cached.screenNodes.length; i++) {{
                const p = cached.screenNodes[i];
                if (!p) continue;
                const densityScale = nodesData.length > 600 ? 0.72 : (nodesData.length > 300 ? 0.85 : 1.0);
                const r = Math.max(5.5, Math.min(12.5, (p.s * 0.018 + 6.0) * densityScale));
                const d = Math.hypot(x - p.x, y - p.y);
                if (d <= r && (!best || d < best.d)) best = {{ idx: i, d }};
            }}
            return best;
        }}

        function hitEdge(x, y) {{
            let best = null;
            for (const e of cached.edges) {{
                const a = cached.screenNodes[e.a], b = cached.screenNodes[e.b];
                const d = distPointToSegment(x, y, a.x, a.y, b.x, b.y);
                if (d <= 7 && (!best || d < best.d)) best = {{ ...e, d }};
            }}
            return best;
        }}

        function setInfoText(text) {{ info.textContent = text; }}

        canvas.addEventListener("pointerdown", (e) => {{
            pointerDown = true; moved = false; lastX = e.clientX; lastY = e.clientY;
            canvas.setPointerCapture(e.pointerId);
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left, y = e.clientY - rect.top;
            const nodeHit = hitNode(x, y);
            if (nodeHit) {{
                draggingNode = nodeHit.idx;
                rotating = false;
                selectedNode = nodeHit.idx;
                selectedEdgeUid = null;
            }} else {{
                draggingNode = null;
                rotating = true;
            }}
            canvas.classList.add("dragging");
        }});

        canvas.addEventListener("pointermove", (e) => {{
            if (!pointerDown) return;
            const dx = e.clientX - lastX, dy = e.clientY - lastY;
            if (Math.abs(dx) + Math.abs(dy) > 1) moved = true;
            lastX = e.clientX; lastY = e.clientY;

            if (draggingNode !== null) {{
                const off = nodeScreenOffset.get(draggingNode) || {{ x: 0, y: 0 }};
                off.x += dx; off.y += dy;
                nodeScreenOffset.set(draggingNode, off);
                render();
            }} else if (rotating) {{
                yaw += dx * 0.005;
                pitch = Math.max(-1.05, Math.min(1.05, pitch + dy * 0.0045));
                render();
            }}
        }});

        canvas.addEventListener("pointerup", (e) => {{
            if (!pointerDown) return;
            pointerDown = false;
            canvas.classList.remove("dragging");
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left, y = e.clientY - rect.top;

            if (!moved) {{
                const nodeHit = hitNode(x, y);
                if (nodeHit) {{
                    selectedNode = nodeHit.idx;
                    selectedEdgeUid = null;
                    const n = nodesData[nodeHit.idx];
                    setInfoText(`Topic: ${{n.topic || n.label}} | Papers in view: ${{n.paperCount || 0}}`);
                }} else {{
                    const edgeHit = hitEdge(x, y);
                    if (edgeHit) {{
                        selectedNode = null;
                        selectedEdgeUid = edgeHit.uid;
                        setInfoText(`Hyperedge paper: ${{edgeHit.paperLabel}} | Paper ID: ${{edgeHit.paperId}} | Hyperedge ID: ${{edgeHit.edgeId}}`);
                    }} else {{
                        selectedNode = null;
                        selectedEdgeUid = null;
                        setInfoText("Click a topic node or a hyperedge link for details.");
                    }}
                }}
            }}

            draggingNode = null;
            rotating = false;
            render();
        }});

        canvas.addEventListener("pointercancel", () => {{
            pointerDown = false;
            draggingNode = null;
            rotating = false;
            moved = false;
            canvas.classList.remove("dragging");
        }});

        canvas.addEventListener("wheel", (e) => {{
            zoom *= Math.exp(-e.deltaY * 0.0018);
            zoom = Math.max(0.60, Math.min(5.0, zoom));
            render();
        }}, {{ passive: true }});

        window.addEventListener("resize", resizeCanvas);
        resizeCanvas();
        render();
    </script>
</body>
</html>"""


def build_line_figure(
    title: str,
    x_values: Iterable[Any],
    y_values: Iterable[float],
    line_color: str,
    y_title: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(x_values),
            y=list(y_values),
            mode="lines+markers",
            line=dict(color=line_color, width=3),
            marker=dict(size=7, color=line_color),
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=14, r=14, t=52, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        xaxis=dict(title="Year", gridcolor="rgba(15,23,42,0.08)"),
        yaxis=dict(title=y_title, gridcolor="rgba(15,23,42,0.08)"),
        height=280,
    )
    return apply_black_graph_text(fig)


def build_training_figure(history) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=history["train_loss"],
            mode="lines+markers",
            name="Train",
            line=dict(color="#0f766e", width=3),
        )
    )
    if "val_loss" in history and history["val_loss"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=history["epoch"],
                y=history["val_loss"],
                mode="lines+markers",
                name="Validation",
                line=dict(color="#c2410c", width=3),
            )
        )
    fig.update_layout(
        title="Training loss",
        margin=dict(l=14, r=14, t=52, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        xaxis=dict(title="Epoch", gridcolor="rgba(15,23,42,0.08)"),
        yaxis=dict(title="Weighted MSE", gridcolor="rgba(15,23,42,0.08)"),
        height=280,
    )
    return apply_black_graph_text(fig)


def build_topic_history_figure(history, columns: Sequence[Tuple[str, str, str]], title: str) -> go.Figure:
    fig = go.Figure()
    for column, label, color in columns:
        fig.add_trace(
            go.Scatter(
                x=history["year"],
                y=history[column],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
            )
        )
    fig.update_layout(
        title=title,
        margin=dict(l=14, r=14, t=52, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        xaxis=dict(title="Year", gridcolor="rgba(15,23,42,0.08)"),
        yaxis=dict(gridcolor="rgba(15,23,42,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        height=300,
    )
    return apply_black_graph_text(fig)


def render_header() -> None:
    st.markdown(
        """
        <div class="page-header">
            <span class="page-header-title">Data Mining vs GNN</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_graph_section(snapshot: HypergraphSnapshot, available_years: Sequence[int]) -> None:
    st.markdown('<div class="graph-anchor"></div>', unsafe_allow_html=True)
    html_viz = build_3d_hypergraph_html(snapshot)
    st.components.v1.html(html_viz, height=620, scrolling=False)

    render_stat_cards(
        [
            ("Year", snapshot.stats.get("year")),
            ("Papers in year", snapshot.stats.get("available_papers")),
            ("Papers in view", snapshot.stats.get("selected_papers")),
            ("Topics in view", snapshot.stats.get("selected_keywords")),
            ("Incidence links", snapshot.stats.get("incidence_count")),
        ]
    )
    if snapshot.focus_topic:
        st.caption(
            f"Focused keyword: {snapshot.focus_topic['keyword']}  \u00b7  Topic: {snapshot.focus_topic['topic_name']}"
        )
    else:
        display_min_year = max(available_years[0], 2000)
        display_max_year = min(available_years[-1], 2025)
        if display_min_year > display_max_year:
            display_min_year, display_max_year = available_years[0], available_years[-1]
        st.caption(f"Years in hypergraph: {display_min_year}–{display_max_year}")


def render_mining_panel(result: Optional[MiningDashboardResult], keyword: Optional[str]) -> None:
    st.markdown('<h2 class="section-title">Data Mining</h2>', unsafe_allow_html=True)

    if result is None:
        st.info("Run the mining form to populate pair, triple, and next-year trend results.")
        return

    render_stat_cards(
        [
            ("Docs in year", result.stats["documents_in_year"]),
            ("Frequent pairs", result.stats["frequent_pairs"]),
            ("Strong triples", result.stats["strong_triples"]),
            ("Forecast year", result.stats["forecast_year"]),
        ]
    )

    if not result.keyword_series.empty:
        st.plotly_chart(
            build_line_figure(
                title=f"Keyword support over time: {keyword}",
                x_values=result.keyword_series["year"],
                y_values=result.keyword_series["support"],
                line_color="#0f766e",
                y_title="Support",
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )

    tabs = st.tabs(["Pairs by support", "Pairs by lift", "Triples", "Next-year forecast"])
    with tabs[0]:
        st.dataframe(result.top_support_pairs, use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(result.top_lift_pairs, use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(result.top_triples, use_container_width=True, hide_index=True)
    with tabs[3]:
        st.dataframe(result.predicted_itemsets, use_container_width=True, hide_index=True)

    if not result.lead_itemset_series.empty:
        chart_tabs = st.tabs(["Lead itemset support", "Lead itemset residual"])
        with chart_tabs[0]:
            st.plotly_chart(
                build_line_figure(
                    title=f"Top forecast itemset support: {result.lead_itemset_series['itemset'].iloc[0]}",
                    x_values=result.lead_itemset_series["year"],
                    y_values=result.lead_itemset_series["support"],
                    line_color="#c2410c",
                    y_title="Support",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
        with chart_tabs[1]:
            st.plotly_chart(
                build_line_figure(
                    title=f"Top forecast itemset residual: {result.lead_itemset_series['itemset'].iloc[0]}",
                    x_values=result.lead_itemset_series["year"],
                    y_values=result.lead_itemset_series["residual"],
                    line_color="#1d4ed8",
                    y_title="Residual",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )


def render_gnn_panel(result: Optional[GNNDashboardResult]) -> None:
    st.markdown('<h2 class="section-title">GNN Forecast</h2>', unsafe_allow_html=True)

    if result is None:
        st.info("Run the GNN form to train the model and surface topic forecasts.")
        return

    render_stat_cards(
        [
            ("Forecast year", result.stats["forecast_year"]),
            ("Training years", result.stats["training_years"]),
            ("Device", result.stats["device"]),
            ("Focus rank", result.stats["focus_topic_rank"]),
        ]
    )

    if not result.training_history.empty:
        st.plotly_chart(build_training_figure(result.training_history), use_container_width=True, config={"displaylogo": False})

    st.dataframe(result.top_topics, use_container_width=True, hide_index=True)
    st.caption(
        f"Focus topic: {result.stats['focus_topic_name']} | Citation source: {result.stats['citation_source']}"
    )
    if result.stats.get("focus_keyword"):
        st.caption(f"Focused keyword mapped to topic: {result.stats['focus_keyword']}")

    history_2000_plus = result.focus_topic_history[result.focus_topic_history["year"] >= 2000].copy()
    if history_2000_plus.empty:
        st.info("No focus-topic history is available for years 2000 and later.")
        return

    signal_tab, residual_tab = st.tabs(["Topic signals", "Topic residuals"])
    with signal_tab:
        st.plotly_chart(
            build_topic_history_figure(
                history_2000_plus,
                [
                    ("usage", "Usage", "#0f766e"),
                    ("citation", "Citation mass", "#c2410c"),
                    ("intensity", "Intensity", "#1d4ed8"),
                ],
                title="Focus topic history",
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with residual_tab:
        st.plotly_chart(
            build_topic_history_figure(
                history_2000_plus,
                [
                    ("usage_residual", "Usage residual", "#0f766e"),
                    ("citation_residual", "Citation residual", "#c2410c"),
                    ("intensity_residual", "Intensity residual", "#1d4ed8"),
                ],
                title="Focus topic residual history",
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )


def _year_graph_capacity(hypergraph: dict, year: int, min_year: Optional[int] = None) -> Tuple[int, int]:
    """Return (paper_hyperedges, topic_nodes) available for a selected year."""
    try:
        hyperedge_index = hypergraph["hyperedge_index"].long().cpu().numpy()
        node_ids = hyperedge_index[0]
        edge_ids = hyperedge_index[1]
        node_labels = hypergraph["node_labels"].long().cpu().numpy()
        year_per_edge = hypergraph["year_per_edge"].long().cpu().numpy()
        if min_year is None:
            year_mask = year_per_edge[edge_ids] == int(year)
        else:
            year_mask = (year_per_edge[edge_ids] >= int(min_year)) & (year_per_edge[edge_ids] <= int(year))
        if not np.any(year_mask):
            return 0, 0
        year_topics = np.unique(node_labels[node_ids[year_mask]])
        return int(np.unique(edge_ids[year_mask]).size), int(year_topics.size)
    except Exception:
        return 0, 0


def render_graph_controls(hypergraph: dict) -> Tuple[int, Optional[str], int, int, bool]:
    """Compact inline controls rendered above the graph."""
    available_years = get_available_years(hypergraph)
    slider_years = [y for y in available_years if 2000 <= y <= 2025]
    if not slider_years:
        slider_years = available_years

    if "graph_playing" not in st.session_state:
        st.session_state["graph_playing"] = False
    if "graph_year" not in st.session_state or st.session_state["graph_year"] not in slider_years:
        st.session_state["graph_year"] = slider_years[-1]

    play_start_year = 2000 if 2000 in slider_years else slider_years[0]
    play_end_year = 2025 if 2025 in slider_years else slider_years[-1]

    available_keywords = get_available_keywords(hypergraph)
    c1, c2, c3 = st.columns([1.4, 2.2, 1.0])
    with c1:
        inspect_year = st.select_slider("Year", options=slider_years, value=st.session_state["graph_year"])
        st.session_state["graph_year"] = inspect_year
    with c2:
        keyword = st.selectbox(
            "Keyword (Mining/GNN only)",
            options=[None] + available_keywords,
            format_func=lambda item: "— all —" if item is None else item,
            index=0,
        )
    with c3:
        play_clicked = st.button("Play 2000–2025", use_container_width=True)
        stop_clicked = st.button("Stop", use_container_width=True)

    if play_clicked:
        st.session_state["graph_playing"] = True
        st.session_state["graph_year"] = play_start_year
        inspect_year = play_start_year
    elif stop_clicked:
        st.session_state["graph_playing"] = False

    # Always use cumulative slicing from start year through selected year.
    capacity_min_year = play_start_year
    year_paper_count, year_keyword_count = _year_graph_capacity(
        hypergraph,
        inspect_year,
        min_year=capacity_min_year,
    )
    max_papers = max(1, year_paper_count)
    max_keywords = max(1, year_keyword_count)

    st.caption(
        f"Displaying cumulative slice: {capacity_min_year}-{inspect_year} · {year_keyword_count:,} topic nodes · {year_paper_count:,} paper hyperedges"
    )
    st.caption("Graph is not keyword-filtered. Selected keyword applies to Mining and GNN panels only.")

    should_advance_playback = False
    if st.session_state.get("graph_playing", False):
        st.caption(f"Auto-playing years: {play_start_year} → {play_end_year} (current: {inspect_year})")
        if inspect_year >= play_end_year:
            st.session_state["graph_playing"] = False
        else:
            should_advance_playback = True

    return inspect_year, keyword, max_papers, max_keywords, should_advance_playback


def main() -> None:
    inject_styles()
    render_header()

    hypergraph_path = Path(DEFAULT_HYPERGRAPH_PATH)
    hypergraph_version = int(hypergraph_path.stat().st_mtime_ns) if hypergraph_path.exists() else 0
    hypergraph = cached_hypergraph(str(hypergraph_path), hypergraph_version)
    papers_df = cached_papers(str(DEFAULT_PAPERS_CSV))
    available_years = get_available_years(hypergraph)

    render_stat_cards(
        [
            ("Total keyword nodes", len(hypergraph.get("node_id_to_phrase", {}))),
            ("Total paper hyperedges", len(hypergraph.get("edge_id_to_paper", {}))),
            ("Total topics", len(hypergraph.get("topic_names", {}))),
            ("Year span", "2000\u20132025"),
        ]
    )

    st.markdown("<div style='margin-top:0rem'></div>", unsafe_allow_html=True)
    left_col, center_col, right_col = st.columns([0.65, 2.7, 0.65])

    with center_col:
        inspect_year, keyword, max_papers, max_keywords, should_advance_playback = render_graph_controls(hypergraph)
        play_start_year = 2000 if 2000 in available_years else available_years[0]
        cumulative_start_year = play_start_year

        snapshot = build_hypergraph_snapshot(
            hypergraph=hypergraph,
            papers_df=papers_df,
            year=inspect_year,
            keyword=None,
            max_papers=max_papers,
            max_keywords=max_keywords,
            min_year=cumulative_start_year,
        )
        render_graph_section(snapshot, available_years)

        if should_advance_playback:
            slider_years = [y for y in available_years if 2000 <= y <= 2025]
            if not slider_years:
                slider_years = available_years
            curr_idx = slider_years.index(int(inspect_year))
            next_idx = min(curr_idx + 1, len(slider_years) - 1)
            st.session_state["graph_year"] = int(slider_years[next_idx])
            time.sleep(0.45)
            st.rerun()

    with left_col:
        mining_menu = st.radio(
            "Mining menu",
            options=["Quick setup", "Hyperparameters"],
            horizontal=True,
            label_visibility="collapsed",
        )

        with st.form("data_mining_form"):
            mining_min_year = max(available_years[0], 2000)
            mining_max_year = min(available_years[-1], 2025)
            if mining_min_year > mining_max_year:
                mining_min_year, mining_max_year = available_years[0], available_years[-1]

            default_target_year = min(max(inspect_year, mining_min_year), mining_max_year)
            mining_range = st.slider(
                "Mining corpus range",
                min_value=mining_min_year,
                max_value=mining_max_year,
                value=(max(mining_min_year, default_target_year - 20), default_target_year),
            )
            mining_target_options = [y for y in available_years if mining_min_year <= y <= mining_max_year]
            target_year = st.select_slider("Target mining year", options=mining_target_options, value=default_target_year)

            minsup_frac = st.slider("Minimum support fraction", min_value=0.001, max_value=0.10, value=0.01, step=0.001)
            min_count = st.number_input("Minimum support count", min_value=1, max_value=20, value=3)
            min_lift = st.slider("Minimum lift", min_value=1.0, max_value=6.0, value=2.0, step=0.1)
            max_k = st.segmented_control("Max itemset size", options=[2, 3], default=3)
            mining_topm = st.slider("Rows to keep", min_value=10, max_value=100, value=25, step=5)

            baseline = "drift"
            window = 5
            df_max_frac = 0.25
            df_min_count = 2
            min_active_years = 3
            if mining_menu == "Hyperparameters":
                baseline = st.selectbox("Residual baseline", options=["drift", "ar1"], index=0)
                window = st.slider("Residual window", min_value=3, max_value=10, value=5)
                df_max_frac = st.slider("Keyword DF max fraction", min_value=0.05, max_value=0.60, value=0.25, step=0.01)
                df_min_count = st.number_input("Keyword DF min count", min_value=1, max_value=20, value=2)
                min_active_years = st.slider("Min active years", min_value=1, max_value=10, value=3)

            run_mining = st.form_submit_button("Run Data Mining")

        if run_mining:
            target_year = max(min(target_year, mining_range[1]), mining_range[0])
            with st.spinner("Running mining pipeline..."):
                st.session_state["mining_result"] = cached_mining_result(
                    papers_csv=str(DEFAULT_PAPERS_CSV),
                    phrases_json=str(DEFAULT_PHRASES_JSON),
                    min_year=mining_range[0],
                    max_year=mining_range[1],
                    target_year=target_year,
                    minsup_frac=float(minsup_frac),
                    min_count=int(min_count),
                    min_lift=float(min_lift),
                    max_k=int(max_k or 3),
                    baseline=baseline,
                    window=int(window),
                    df_max_frac=float(df_max_frac),
                    df_min_count=int(df_min_count),
                    topm=int(mining_topm),
                    min_active_years=int(min_active_years),
                    keyword=keyword,
                )

    with right_col:
        gnn_menu = st.radio(
            "GNN menu",
            options=["Quick setup", "Hyperparameters"],
            horizontal=True,
            label_visibility="collapsed",
        )

        cuda_available = False
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

        with st.form("gnn_form"):
            training_min_year = max(available_years[0], 2000)
            training_max_year = min(available_years[-1], 2025)
            if training_min_year > training_max_year:
                training_min_year, training_max_year = available_years[0], available_years[-1]

            training_range = st.slider(
                "Training year range",
                min_value=training_min_year,
                max_value=training_max_year,
                value=(max(2005, training_min_year), training_max_year),
            )
            forecast_year = st.number_input("Forecast year", min_value=training_max_year, max_value=training_max_year + 10, value=training_max_year + 1)
            gnn_window = st.slider("GNN window", min_value=2, max_value=10, value=5)
            baseline_window = st.slider("Baseline window", min_value=2, max_value=10, value=5)
            families = st.multiselect("Curve families", options=["logistic", "exp", "power"], default=["logistic", "exp", "power"])
            topn = st.slider("Top topics to show", min_value=5, max_value=20, value=10)

            hidden = 128
            layers = 2
            dropout = 0.2
            lr = 2e-3
            weight_decay = 1e-4
            epochs = 10
            seed = 7
            device = "cpu"
            if gnn_menu == "Hyperparameters":
                hidden = st.slider("Hidden width", min_value=32, max_value=256, value=128, step=32)
                layers = st.slider("HypergraphConv layers", min_value=1, max_value=4, value=2)
                dropout = st.slider("Dropout", min_value=0.0, max_value=0.6, value=0.2, step=0.05)
                lr = st.number_input("Learning rate", min_value=1e-4, max_value=1e-2, value=2e-3, step=1e-4, format="%.4f")
                weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1e-2, value=1e-4, step=1e-4, format="%.4f")
                epochs = st.slider("Epochs", min_value=0, max_value=60, value=10)
                seed = st.number_input("Random seed", min_value=0, max_value=9999, value=7)
                device = st.selectbox("Device", options=["cpu", "cuda"] if cuda_available else ["cpu"], index=0)

            run_gnn = st.form_submit_button("Run GNN Forecast")

        if run_gnn:
            if not families:
                st.error("Pick at least one curve family for the rolling baseline.")
            else:
                with st.spinner("Training temporal hypergraph model..."):
                    st.session_state["gnn_result"] = cached_gnn_result(
                        hypergraph_pt=str(DEFAULT_HYPERGRAPH_PATH),
                        papers_csv=str(DEFAULT_PAPERS_CSV),
                        train_year_min=training_range[0],
                        train_year_max=training_range[1],
                        gnn_window=int(gnn_window),
                        baseline_window=int(baseline_window),
                        families=tuple(families),
                        hidden=int(hidden),
                        layers=int(layers),
                        dropout=float(dropout),
                        lr=float(lr),
                        weight_decay=float(weight_decay),
                        epochs=int(epochs),
                        seed=int(seed),
                        device=device,
                        target_year=int(forecast_year),
                        keyword=keyword,
                        topn=int(topn),
                    )

    st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
    render_mining_panel(st.session_state.get("mining_result"), keyword)

    st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
    render_gnn_panel(st.session_state.get("gnn_result"))

if __name__ == "__main__":
    main()