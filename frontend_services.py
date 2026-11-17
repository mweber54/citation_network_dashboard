from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import run as gnn_pipeline
from data_mining.baseline import compute_all_itemset_residuals
from data_mining.forecast_2026 import predict_residual_2026
from data_mining.mine_pairs import mine_pairs_for_year
from data_mining.mine_triples import mine_triples_for_year
from data_mining.preprocess import normalize_phrase, preprocess_phrases
from data_mining.series import build_itemset_universe, build_yearly_metrics_rows, format_itemset
from data_mining.transactions import build_transactions_by_year


ROOT = Path(__file__).resolve().parent
DEFAULT_HYPERGRAPH_PATH = ROOT / "hypergraph.pt"
DEFAULT_PAPERS_CSV = ROOT / "cs_citation_network" / "papers.csv"
DEFAULT_PHRASES_JSON = ROOT / "topics_out" / "paper_phrases.json"


@dataclass
class HypergraphSnapshot:
    keyword_nodes: pd.DataFrame
    paper_nodes: pd.DataFrame
    edges: pd.DataFrame
    stats: Dict[str, Any]
    focus_keyword: Optional[str]
    focus_topic: Optional[Dict[str, Any]]


@dataclass
class MiningDashboardResult:
    stats: Dict[str, Any]
    top_support_pairs: pd.DataFrame
    top_lift_pairs: pd.DataFrame
    top_triples: pd.DataFrame
    predicted_itemsets: pd.DataFrame
    keyword_series: pd.DataFrame
    lead_itemset_series: pd.DataFrame


@dataclass
class GNNDashboardResult:
    stats: Dict[str, Any]
    top_topics: pd.DataFrame
    training_history: pd.DataFrame
    focus_topic_history: pd.DataFrame


def load_hypergraph(path: Path = DEFAULT_HYPERGRAPH_PATH) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def load_papers_dataframe(path: Path = DEFAULT_PAPERS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["paper_id"] = df["paper_id"].astype(str)
    if "title" not in df.columns:
        df["title"] = ""
    else:
        df["title"] = df["title"].fillna("")
    if "venue" not in df.columns:
        df["venue"] = ""
    else:
        df["venue"] = df["venue"].fillna("")
    df["year"] = pd.to_numeric(df["year"] if "year" in df.columns else 0, errors="coerce").fillna(0).astype(int)
    df["citation_count"] = pd.to_numeric(
        df["citation_count"] if "citation_count" in df.columns else 0,
        errors="coerce",
    ).fillna(0).astype(int)
    return df


def get_available_years(hypergraph: Dict[str, Any]) -> List[int]:
    years = hypergraph["year_per_edge"].long().cpu().numpy().astype(int).tolist()
    return sorted(set(years))


def get_available_keywords(hypergraph: Dict[str, Any]) -> List[str]:
    node_id_to_phrase = hypergraph.get("node_id_to_phrase", {})
    return sorted(str(phrase) for phrase in node_id_to_phrase.values())


def _resolve_keyword_node_id(hypergraph: Dict[str, Any], keyword: Optional[str]) -> Optional[int]:
    if not keyword:
        return None

    phrase_to_node = hypergraph.get("phrase_to_node_id", {})
    if keyword in phrase_to_node:
        return int(phrase_to_node[keyword])

    normalized_lookup = {
        normalize_phrase(str(phrase)): int(node_id)
        for node_id, phrase in hypergraph.get("node_id_to_phrase", {}).items()
    }
    return normalized_lookup.get(normalize_phrase(keyword))


def _series_positions(ids: Sequence[int]) -> Dict[int, float]:
    if not ids:
        return {}
    if len(ids) == 1:
        return {int(ids[0]): 0.5}
    step = 1.0 / max(len(ids) - 1, 1)
    return {int(item_id): 1.0 - idx * step for idx, item_id in enumerate(ids)}


def _compact_topic_label(topic_name: str, max_parts: int = 2, max_chars: int = 52) -> str:
    """Build a concise on-graph label from a multi-phrase topic name."""
    raw = str(topic_name or "").strip()
    if not raw:
        return ""
    parts = [part.strip() for part in raw.split("/") if part.strip()]
    if parts:
        compact = " / ".join(parts[: max(1, int(max_parts))])
    else:
        compact = raw
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def build_hypergraph_snapshot(
    hypergraph: Dict[str, Any],
    papers_df: pd.DataFrame,
    year: int,
    keyword: Optional[str],
    max_papers: int,
    max_keywords: int,
    min_year: Optional[int] = None,
) -> HypergraphSnapshot:
    hyperedge_index = hypergraph["hyperedge_index"].long().cpu().numpy()
    node_ids = hyperedge_index[0]
    edge_ids = hyperedge_index[1]
    year_per_edge = hypergraph["year_per_edge"].long().cpu().numpy()
    edge_features = hypergraph["hyperedge_features"].float().cpu().numpy()
    node_features = hypergraph["node_features"].float().cpu().numpy()
    node_labels = hypergraph["node_labels"].long().cpu().numpy()

    papers_meta = load_papers_dataframe() if papers_df.empty else papers_df
    papers_lookup = papers_meta.set_index("paper_id", drop=False)
    node_id_to_phrase = hypergraph.get("node_id_to_phrase", {})
    edge_id_to_paper = hypergraph.get("edge_id_to_paper", {})
    topic_names = hypergraph.get("topic_names", {})

    edge_degree = Counter(int(edge_id) for edge_id in edge_ids.tolist())
    if min_year is None:
        year_mask = year_per_edge[edge_ids] == year
    else:
        year_mask = (year_per_edge[edge_ids] >= int(min_year)) & (year_per_edge[edge_ids] <= int(year))
    year_edge_ids = np.unique(edge_ids[year_mask]).tolist()

    focus_node_id = _resolve_keyword_node_id(hypergraph, keyword)
    focus_keyword = None
    focus_topic = None

    def paper_rank(edge_id: int) -> Tuple[float, float, int]:
        paper_id = str(edge_id_to_paper.get(int(edge_id), ""))
        citation_count = 0.0
        if paper_id in papers_lookup.index:
            citation_count = float(papers_lookup.at[paper_id, "citation_count"])
        if citation_count <= 0.0:
            citation_count = float(edge_features[int(edge_id), 1])
        return (
            citation_count,
            float(edge_degree.get(int(edge_id), 0)),
            -int(edge_id),
        )

    if focus_node_id is not None:
        focus_keyword = str(node_id_to_phrase.get(int(focus_node_id), keyword or ""))
        focus_edges = np.unique(edge_ids[year_mask & (node_ids == focus_node_id)]).tolist()
        focus_edges.sort(key=paper_rank, reverse=True)
        selected_edges = focus_edges[:max_papers]
        topic_id = int(node_labels[int(focus_node_id)])
        focus_topic = {
            "topic_id": topic_id,
            "topic_name": str(topic_names.get(topic_id, f"topic_{topic_id}")),
            "keyword": focus_keyword,
        }
    else:
        selected_edges = sorted(year_edge_ids, key=paper_rank, reverse=True)[:max_papers]

    if not selected_edges:
        stats = {
            "year": year,
            "available_papers": len(year_edge_ids),
            "selected_papers": 0,
            "selected_keywords": 0,
            "incidence_count": 0,
            "focus_keyword_hits": 0,
        }
        return HypergraphSnapshot(
            keyword_nodes=pd.DataFrame(),
            paper_nodes=pd.DataFrame(),
            edges=pd.DataFrame(),
            stats=stats,
            focus_keyword=focus_keyword,
            focus_topic=focus_topic,
        )

    incidence_mask = year_mask & np.isin(edge_ids, np.asarray(selected_edges))
    view_node_ids = node_ids[incidence_mask]
    view_edge_ids = edge_ids[incidence_mask]
    view_topic_ids = node_labels[view_node_ids]
    topic_degree_in_view = Counter(int(topic_id) for topic_id in view_topic_ids.tolist())

    focus_topic_id = int(node_labels[int(focus_node_id)]) if focus_node_id is not None else None
    if focus_topic_id is not None:
        ranked_topics = [
            int(topic_id)
            for topic_id, _ in sorted(
                topic_degree_in_view.items(),
                key=lambda item: (
                    item[0] != focus_topic_id,
                    -item[1],
                    str(topic_names.get(int(item[0]), f"topic_{int(item[0])}")),
                ),
            )
        ]
    else:
        ranked_topics = [
            int(topic_id)
            for topic_id, _ in sorted(
                topic_degree_in_view.items(),
                key=lambda item: (-item[1], str(topic_names.get(int(item[0]), f"topic_{int(item[0])}"))),
            )
        ]

    kept_topics = ranked_topics[:max_keywords]
    if kept_topics:
        incidence_mask = incidence_mask & np.isin(node_labels[node_ids], np.asarray(kept_topics))
    else:
        incidence_mask = incidence_mask & False

    view_node_ids = node_ids[incidence_mask]
    view_edge_ids = edge_ids[incidence_mask]
    # Keep only paper hyperedges that connect at least two visible topics.
    # The 3D renderer projects hyperedges as topic-to-topic links, so singleton
    # hyperedges would otherwise inflate counts while producing no visible links.
    edge_to_topics: Dict[int, set[int]] = defaultdict(set)
    for node_id, edge_id in zip(view_node_ids.tolist(), view_edge_ids.tolist()):
        edge_to_topics[int(edge_id)].add(int(node_labels[int(node_id)]))
    renderable_edges = {edge_id for edge_id, topics in edge_to_topics.items() if len(topics) >= 2}

    if renderable_edges:
        keep_mask = np.isin(view_edge_ids, np.asarray(sorted(renderable_edges), dtype=np.int64))
        view_node_ids = view_node_ids[keep_mask]
        view_edge_ids = view_edge_ids[keep_mask]
    else:
        view_node_ids = np.asarray([], dtype=np.int64)
        view_edge_ids = np.asarray([], dtype=np.int64)

    selected_edges = sorted(set(int(edge_id) for edge_id in view_edge_ids.tolist()), key=paper_rank, reverse=True)
    view_topic_ids = node_labels[view_node_ids]
    topic_degree_in_view = Counter(int(topic_id) for topic_id in view_topic_ids.tolist())

    # Track unique papers per topic separately from raw incidence counts.
    topic_unique_papers: Dict[int, set[int]] = defaultdict(set)
    for node_id, edge_id in zip(view_node_ids.tolist(), view_edge_ids.tolist()):
        topic_id = int(node_labels[int(node_id)])
        topic_unique_papers[topic_id].add(int(edge_id))

    topic_positions = _series_positions(kept_topics)
    paper_positions = _series_positions(selected_edges)

    topic_to_score: Dict[int, float] = {}
    if view_node_ids.size > 0:
        topic_nodes: Dict[int, List[int]] = {}
        for node_id in view_node_ids.tolist():
            topic_id = int(node_labels[int(node_id)])
            topic_nodes.setdefault(topic_id, []).append(int(node_id))
        for topic_id, nodes in topic_nodes.items():
            score = float(np.mean([float(node_features[node_id, 1]) for node_id in nodes]))
            topic_to_score[topic_id] = score

    keyword_rows: List[Dict[str, Any]] = []
    for topic_id in kept_topics:
        if topic_id not in topic_degree_in_view:
            continue
        topic_name = str(topic_names.get(int(topic_id), f"topic_{int(topic_id)}"))
        keyword_rows.append(
            {
                "node_id": int(topic_id),
                "label": _compact_topic_label(topic_name),
                "topic_id": int(topic_id),
                "topic_name": topic_name,
                "degree_in_view": int(topic_degree_in_view.get(int(topic_id), 0)),
                "paper_count_in_view": int(len(topic_unique_papers.get(int(topic_id), set()))),
                "phrase_score": float(topic_to_score.get(int(topic_id), 0.0)),
                "x": 0.18,
                "y": float(topic_positions.get(int(topic_id), 0.5)),
                "is_focus": bool(focus_topic_id == int(topic_id)),
            }
        )

    paper_rows: List[Dict[str, Any]] = []
    for edge_id in selected_edges:
        paper_id = str(edge_id_to_paper.get(int(edge_id), f"paper_{edge_id}"))
        title = ""
        venue = ""
        citation_count = 0
        if paper_id in papers_lookup.index:
            title = str(papers_lookup.at[paper_id, "title"])
            venue = str(papers_lookup.at[paper_id, "venue"])
            citation_count = int(papers_lookup.at[paper_id, "citation_count"])
        paper_rows.append(
            {
                "edge_id": int(edge_id),
                "paper_id": paper_id,
                "title": title,
                "venue": venue,
                "year": int(year_per_edge[int(edge_id)]),
                "citation_count": citation_count,
                "num_keywords": int(edge_degree.get(int(edge_id), 0)),
                "x": 0.82,
                "y": float(paper_positions.get(int(edge_id), 0.5)),
            }
        )

    keyword_lookup = {row["node_id"]: row for row in keyword_rows}
    paper_lookup = {row["edge_id"]: row for row in paper_rows}
    edge_rows: List[Dict[str, Any]] = []
    seen_topic_paper_pairs: set[Tuple[int, int]] = set()
    for node_id, edge_id in zip(view_node_ids.tolist(), view_edge_ids.tolist()):
        topic_id = int(node_labels[int(node_id)])
        pair_key = (topic_id, int(edge_id))
        if pair_key in seen_topic_paper_pairs:
            continue
        seen_topic_paper_pairs.add(pair_key)

        if topic_id not in keyword_lookup or edge_id not in paper_lookup:
            continue
        kw = keyword_lookup[int(topic_id)]
        paper = paper_lookup[int(edge_id)]
        edge_rows.append(
            {
                "node_id": int(topic_id),
                "edge_id": int(edge_id),
                "keyword_label": kw["label"],
                "paper_id": paper["paper_id"],
                "x0": kw["x"],
                "y0": kw["y"],
                "x1": paper["x"],
                "y1": paper["y"],
            }
        )

    stats = {
        "year": year,
        "available_papers": len(year_edge_ids),
        "selected_papers": len(paper_rows),
        "selected_keywords": len(keyword_rows),
        "incidence_count": len(edge_rows),
        "focus_keyword_hits": sum(1 for row in edge_rows if row["node_id"] == focus_topic_id),
    }
    return HypergraphSnapshot(
        keyword_nodes=pd.DataFrame(keyword_rows),
        paper_nodes=pd.DataFrame(paper_rows),
        edges=pd.DataFrame(edge_rows),
        stats=stats,
        focus_keyword=focus_keyword,
        focus_topic=focus_topic,
    )


def _keyword_id_from_preprocess(keyword: Optional[str], phrase_to_id: Dict[str, int]) -> Optional[int]:
    if not keyword:
        return None
    return phrase_to_id.get(normalize_phrase(keyword))


def _pair_dataframe(
    pairs: Dict[Tuple[int, int], Dict[str, float]],
    id_to_phrase: Dict[int, str],
    keyword_id: Optional[int],
    sort_key: str,
    topn: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (a, b), metrics in pairs.items():
        if keyword_id is not None and keyword_id not in (a, b):
            continue
        rows.append(
            {
                "itemset": format_itemset((a, b), id_to_phrase),
                "support": float(metrics.get("supp_frac", 0.0)),
                "count": int(metrics.get("sup_count", 0.0)),
                "lift": float(metrics.get("lift", 0.0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["itemset", "support", "count", "lift"])
    return pd.DataFrame(rows).sort_values(sort_key, ascending=False).head(topn).reset_index(drop=True)


def _triple_dataframe(
    triples: Dict[Tuple[int, int, int], Dict[str, float]],
    id_to_phrase: Dict[int, str],
    keyword_id: Optional[int],
    topn: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for itemset, metrics in triples.items():
        if keyword_id is not None and keyword_id not in itemset:
            continue
        rows.append(
            {
                "itemset": format_itemset(itemset, id_to_phrase),
                "support": float(metrics.get("supp_frac", 0.0)),
                "count": int(metrics.get("sup_count", 0.0)),
                "lift_min": float(metrics.get("lift_min", 0.0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["itemset", "support", "count", "lift_min"])
    return pd.DataFrame(rows).sort_values(["lift_min", "support"], ascending=False).head(topn).reset_index(drop=True)


def run_data_mining_dashboard(
    papers_csv: Path,
    phrases_json: Path,
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
    prep = preprocess_phrases(
        papers_csv=papers_csv,
        phrases_json=phrases_json,
        min_year=min_year,
        max_year=max_year,
        df_min_count=df_min_count,
        df_max_frac=df_max_frac,
    )
    tx = build_transactions_by_year(
        paper_year=prep.paper_year,
        paper_phrase_ids=prep.paper_phrase_ids,
        min_year=min_year,
        max_year=max_year,
    )
    keyword_id = _keyword_id_from_preprocess(keyword, prep.phrase_to_id)

    pair_metrics_by_year: Dict[int, Dict[Tuple[int, int], Dict[str, float]]] = {}
    strong_pairs_by_year: Dict[int, Dict[Tuple[int, int], Dict[str, float]]] = {}
    strong_triples_by_year: Dict[int, Dict[Tuple[int, int, int], Dict[str, float]]] = {}

    for year in tx.years:
        pair_result = mine_pairs_for_year(
            transactions=tx.transactions_by_year.get(year, []),
            single_counts=tx.single_counts_by_year.get(year, {}),
            n_year=tx.n_by_year.get(year, 0),
            minsup_frac=minsup_frac,
            min_count=min_count,
            min_lift=min_lift,
        )
        pair_metrics_by_year[year] = pair_result["frequent_pairs"]
        strong_pairs_by_year[year] = pair_result["strong_pairs"]

        if max_k >= 3:
            strong_triples_by_year[year] = mine_triples_for_year(
                transactions=tx.transactions_by_year.get(year, []),
                pair_metrics=strong_pairs_by_year[year],
                n_year=tx.n_by_year.get(year, 0),
                minsup_frac=minsup_frac,
                min_count=min_count,
                min_lift=min_lift,
            )
        else:
            strong_triples_by_year[year] = {}

    target_pairs = pair_metrics_by_year.get(target_year, {})
    target_strong_pairs = strong_pairs_by_year.get(target_year, {})
    target_triples = strong_triples_by_year.get(target_year, {})

    itemsets = build_itemset_universe(
        strong_pairs_by_year=strong_pairs_by_year,
        strong_triples_by_year=strong_triples_by_year,
        min_active_years=min_active_years,
        years=tx.years,
        recent_years=list(range(max(min_year, target_year - 2), target_year + 1)),
    )
    _, z_series, supp_series = build_yearly_metrics_rows(
        itemsets=itemsets,
        years=tx.years,
        n_by_year=tx.n_by_year,
        pair_metrics_by_year=pair_metrics_by_year,
        triple_metrics_by_year=strong_triples_by_year,
        id_to_phrase=prep.id_to_phrase,
    )
    _, residual_by_itemset = compute_all_itemset_residuals(
        z_series=z_series,
        window=window,
        method=baseline,
    )

    forecast_window = (max(min_year, target_year - window + 1), target_year)
    predicted_rows = predict_residual_2026(
        itemsets=itemsets,
        years=tx.years,
        residual_by_itemset=residual_by_itemset,
        supp_series=supp_series,
        residual_window_years=forecast_window,
    )

    prediction_frame_rows: List[Dict[str, Any]] = []
    lead_itemset = None
    for row in predicted_rows:
        itemset = tuple(row["itemset"])
        if keyword_id is not None and keyword_id not in itemset:
            continue
        if lead_itemset is None:
            lead_itemset = itemset
        prediction_frame_rows.append(
            {
                "itemset": format_itemset(itemset, prep.id_to_phrase),
                "k": len(itemset),
                "predicted_next_year_residual": float(row["pred_residual_2026"]),
                "recent_support": float(row["recent_support"]),
                "score": float(row["score"]),
                "score_weighted": float(row["score_weighted"]),
            }
        )
        if len(prediction_frame_rows) >= topm:
            break

    keyword_series_rows: List[Dict[str, Any]] = []
    if keyword_id is not None:
        for year in tx.years:
            doc_count = int(tx.n_by_year.get(year, 0))
            keyword_count = int(tx.single_counts_by_year.get(year, {}).get(keyword_id, 0))
            keyword_series_rows.append(
                {
                    "year": year,
                    "count": keyword_count,
                    "documents": doc_count,
                    "support": (keyword_count / doc_count) if doc_count > 0 else 0.0,
                    "keyword": keyword,
                }
            )

    lead_itemset_rows: List[Dict[str, Any]] = []
    if lead_itemset is not None:
        label = format_itemset(lead_itemset, prep.id_to_phrase)
        for idx, year in enumerate(tx.years):
            lead_itemset_rows.append(
                {
                    "year": year,
                    "support": float(supp_series[lead_itemset][idx]),
                    "residual": float(residual_by_itemset[lead_itemset][idx]),
                    "itemset": label,
                }
            )

    stats = {
        "papers_with_phrases": int(prep.total_docs),
        "phrase_vocabulary": int(len(prep.phrase_to_id)),
        "documents_in_year": int(tx.n_by_year.get(target_year, 0)),
        "frequent_pairs": int(len(target_pairs)),
        "strong_pairs": int(len(target_strong_pairs)),
        "strong_triples": int(len(target_triples)),
        "modeled_itemsets": int(len(itemsets)),
        "forecast_year": int(target_year + 1),
        "keyword_filter": keyword or "All keywords",
    }

    return MiningDashboardResult(
        stats=stats,
        top_support_pairs=_pair_dataframe(target_pairs, prep.id_to_phrase, keyword_id, "support", topm),
        top_lift_pairs=_pair_dataframe(target_pairs, prep.id_to_phrase, keyword_id, "lift", topm),
        top_triples=_triple_dataframe(target_triples, prep.id_to_phrase, keyword_id, topm),
        predicted_itemsets=pd.DataFrame(prediction_frame_rows),
        keyword_series=pd.DataFrame(keyword_series_rows),
        lead_itemset_series=pd.DataFrame(lead_itemset_rows),
    )


def run_gnn_dashboard(
    hypergraph_pt: Path,
    papers_csv: Optional[Path],
    train_year_min: int,
    train_year_max: int,
    gnn_window: int,
    baseline_window: int,
    families: Sequence[str],
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
    torch.manual_seed(seed)
    np.random.seed(seed)

    hypergraph = load_hypergraph(hypergraph_pt)
    hyperedge_index = hypergraph["hyperedge_index"].long()
    node_features = hypergraph["node_features"].float()
    hyperedge_features = hypergraph["hyperedge_features"].float()
    node_labels = hypergraph["node_labels"].long()
    year_per_edge = hypergraph["year_per_edge"].long()
    edge_id_to_paper = hypergraph.get("edge_id_to_paper", {})
    topic_names = hypergraph.get("topic_names", {})

    years = sorted(set(year_per_edge.detach().cpu().numpy().astype(int).tolist()))
    year_to_idx = {year: idx for idx, year in enumerate(years)}
    C = int(node_labels.max().item()) + 1
    node_score = node_features[:, 1].detach().cpu().numpy()

    if papers_csv is not None and edge_id_to_paper:
        paper_citations = gnn_pipeline.load_raw_citations_from_papers_csv(
            str(papers_csv),
            edge_id_to_paper,
            year_per_edge.numel(),
        )
        citation_source = "raw citation_count"
    else:
        paper_citations = hyperedge_features[:, 1].clone()
        citation_source = "normalized citation proxy"

    u_k_t, m_k_t = gnn_pipeline.compute_keyword_year_u_m(
        hyperedge_index=hyperedge_index,
        year_per_edge=year_per_edge,
        paper_citations=paper_citations,
        years=years,
    )
    u_c_t, m_c_t, a_c_t = gnn_pipeline.compute_topic_aggregates(
        u_k_t=u_k_t,
        m_k_t=m_k_t,
        node_labels=node_labels.detach().cpu().numpy(),
        node_scores=node_score,
        C=C,
        eps=1e-6,
    )

    Z_u = np.log1p(np.maximum(u_c_t, 0.0))
    Z_m = np.log1p(np.maximum(m_c_t, 0.0))
    Z_a = np.log1p(np.maximum(a_c_t, 0.0))

    family_tuple = tuple(families)
    T_u = gnn_pipeline.rolling_baseline_residuals(Z_u, years, L=baseline_window, families=family_tuple)
    T_m = gnn_pipeline.rolling_baseline_residuals(Z_m, years, L=baseline_window, families=family_tuple)
    T_a = gnn_pipeline.rolling_baseline_residuals(Z_a, years, L=baseline_window, families=family_tuple)

    train_years = [
        year
        for year in years
        if train_year_min <= year <= train_year_max and year_to_idx[year] >= baseline_window
    ]
    if not train_years:
        raise ValueError("No training years remain after applying the selected range and baseline window.")

    targets: List[np.ndarray] = []
    kept_years: List[int] = []
    for year in train_years:
        idx = year_to_idx[year]
        target = np.stack([T_u[:, idx], T_m[:, idx], T_a[:, idx]], axis=1).astype(np.float32)
        if np.any(np.isnan(target)):
            continue
        targets.append(target)
        kept_years.append(year)

    if not targets:
        raise ValueError("Residual targets are empty for the selected training range.")

    train_years = kept_years
    Y = np.stack(targets, axis=0)
    Ny = Y.shape[0]

    variance = np.var(Y.reshape(-1, 3), axis=0) + 1e-8
    omega = 1.0 / variance
    omega = omega / omega.sum()

    actual_device = device
    if device == "cuda" and not torch.cuda.is_available():
        actual_device = "cpu"
    omega_t = torch.tensor(omega, device=actual_device).view(1, 1, 3)

    windows: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for year in train_years:
        start_year = year - gnn_window
        end_year = year - 1
        windows[year] = gnn_pipeline.build_window_subhypergraph(hyperedge_index, year_per_edge, start_year, end_year)

    model = gnn_pipeline.THGNN(
        node_in=node_features.size(1),
        edge_in=hyperedge_features.size(1),
        hidden=hidden,
        out_dim=3,
        layers=layers,
        dropout=dropout,
    ).to(actual_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    node_features = node_features.to(actual_device)
    hyperedge_features = hyperedge_features.to(actual_device)
    node_labels = node_labels.to(actual_device)
    Y_t = torch.tensor(Y, device=actual_device)

    split = max(int(0.8 * Ny), 1)
    train_idx = list(range(0, split))
    val_idx = list(range(split, Ny))
    training_history: List[Dict[str, Any]] = []

    def run_epoch(indices: Sequence[int], train_mode: bool) -> float:
        model.train(train_mode)
        total_loss = 0.0
        count = 0
        for idx in indices:
            year = train_years[idx]
            sub_idx, edge_map = windows[year]
            sub_idx = sub_idx.to(actual_device)
            edge_map = edge_map.to(actual_device)

            x_edge = hyperedge_features[edge_map]
            pred = model(node_features, sub_idx, x_edge, node_labels, C=C)
            target = Y_t[idx]
            loss = (((pred - target) ** 2) * omega_t[0]).mean()

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += float(loss.item())
            count += 1

        return total_loss / max(count, 1)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_idx, train_mode=True)
        val_loss = run_epoch(val_idx, train_mode=False) if val_idx else float("nan")
        training_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_idx and val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    inference_start = target_year - gnn_window
    inference_end = target_year - 1
    sub_idx, edge_map = gnn_pipeline.build_window_subhypergraph(
        hyperedge_index,
        year_per_edge,
        inference_start,
        inference_end,
    )
    if edge_map.numel() == 0:
        raise ValueError("The selected forecast year has no supporting sub-hypergraph window.")

    model.eval()
    with torch.no_grad():
        pred_resid = model(
            node_features,
            sub_idx.to(actual_device),
            hyperedge_features[edge_map.to(actual_device)],
            node_labels,
            C=C,
        )
        fused = (pred_resid * omega_t[0]).sum(dim=1)

    fused_np = fused.detach().cpu().numpy()
    pred_np = pred_resid.detach().cpu().numpy()
    ranked_topic_ids = np.argsort(-fused_np)

    top_topic_rows: List[Dict[str, Any]] = []
    topic_rank_lookup: Dict[int, int] = {}
    for rank, topic_id in enumerate(ranked_topic_ids.tolist(), start=1):
        topic_rank_lookup[int(topic_id)] = rank
        if rank <= topn:
            top_topic_rows.append(
                {
                    "rank": rank,
                    "topic_id": int(topic_id),
                    "topic_name": str(topic_names.get(int(topic_id), f"topic_{int(topic_id)}")),
                    "fused_score": float(fused_np[int(topic_id)]),
                    "usage_residual": float(pred_np[int(topic_id), 0]),
                    "citation_residual": float(pred_np[int(topic_id), 1]),
                    "intensity_residual": float(pred_np[int(topic_id), 2]),
                }
            )

    focus_node_id = _resolve_keyword_node_id(hypergraph, keyword)
    if focus_node_id is not None:
        focus_topic_id = int(hypergraph["node_labels"][int(focus_node_id)].item())
        focus_label = str(hypergraph["node_id_to_phrase"].get(int(focus_node_id), keyword or ""))
    else:
        focus_topic_id = int(ranked_topic_ids[0])
        focus_label = None

    focus_topic_history_rows: List[Dict[str, Any]] = []
    for year in years:
        idx = year_to_idx[year]
        focus_topic_history_rows.append(
            {
                "year": year,
                "usage": float(u_c_t[focus_topic_id, idx]),
                "citation": float(m_c_t[focus_topic_id, idx]),
                "intensity": float(a_c_t[focus_topic_id, idx]),
                "usage_residual": float(T_u[focus_topic_id, idx]),
                "citation_residual": float(T_m[focus_topic_id, idx]),
                "intensity_residual": float(T_a[focus_topic_id, idx]),
            }
        )

    stats = {
        "keywords": int(node_features.size(0)),
        "papers": int(year_per_edge.numel()),
        "topics": int(C),
        "training_years": int(len(train_years)),
        "forecast_year": int(target_year),
        "device": actual_device,
        "citation_source": citation_source,
        "best_val_loss": None if best_val == float("inf") else float(best_val),
        "channel_weights": omega.tolist(),
        "focus_topic_id": int(focus_topic_id),
        "focus_topic_name": str(topic_names.get(int(focus_topic_id), f"topic_{int(focus_topic_id)}")),
        "focus_topic_rank": int(topic_rank_lookup.get(int(focus_topic_id), 0)),
        "focus_keyword": focus_label,
    }

    return GNNDashboardResult(
        stats=stats,
        top_topics=pd.DataFrame(top_topic_rows),
        training_history=pd.DataFrame(training_history),
        focus_topic_history=pd.DataFrame(focus_topic_history_rows),
    )