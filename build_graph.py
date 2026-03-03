#!/usr/bin/env python
"""
build_hypergraph.py
===================
Build a keyword–paper hypergraph from topic-builder outputs and export it
in formats ready for GNN training (PyTorch Geometric).

Hypergraph structure
--------------------
* **Nodes** = quality-filtered keyword phrases  (from phrase_to_topic.csv)
* **Hyperedges** = papers  (each paper connects the keywords it contains)

Outputs  (all written to --out directory)
-----------------------------------------
* hypergraph.pt          – PyTorch file containing:
      hyperedge_index   : LongTensor [2, E]  (node_idx, hyperedge_idx pairs)
      node_features     : FloatTensor [N, d_node]
      hyperedge_features: FloatTensor [M, d_edge]
      node_labels       : LongTensor [N]  (topic id per keyword)
      year_per_edge     : LongTensor [M]  (year per paper — for temporal splits)
      node_id_to_phrase : dict  {int: str}
      edge_id_to_paper  : dict  {int: str}
      phrase_to_node_id : dict  {str: int}
      paper_to_edge_id  : dict  {str: int}
      topic_names       : dict  {int: str}

* hypergraph_summary.txt – human-readable summary statistics
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch


# ──────────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────────

def load_phrase_to_topic(path: str) -> pd.DataFrame:
    """Load phrase_to_topic.csv → DataFrame with columns [phrase, topic_id, score]."""
    df = pd.read_csv(path, encoding="utf-8")
    df["topic_id"] = df["topic_id"].astype(int)
    df["score"] = df["score"].astype(float)
    return df


def load_paper_phrases(path: str) -> dict:
    """Load paper_phrases.json → dict {paper_id: [phrase, …]}."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_papers(path: str) -> pd.DataFrame:
    """Load papers.csv → DataFrame."""
    df = pd.read_csv(path, encoding="utf-8")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0).astype(int)
    return df


def load_topics(path: str) -> dict:
    """Load topics.json → dict keyed by topic_id (int)."""
    with open(path, encoding="utf-8") as f:
        topics = json.load(f)
    return {int(t["topic_id"]): t for t in topics}


# ──────────────────────────────────────────────────────────────────
# Build hypergraph
# ──────────────────────────────────────────────────────────────────

def build_hypergraph(
    phrase_df: pd.DataFrame,
    paper_phrases: dict,
    papers_df: pd.DataFrame,
    topics: dict,
):
    """
    Returns
    -------
    data : dict   with all tensors and metadata for the hypergraph
    stats : dict  summary statistics
    """

    # --- 1. Build node (keyword) index ---
    # Only keep phrases present in phrase_to_topic (quality-filtered)
    valid_phrases = set(phrase_df["phrase"].values)
    phrase_to_nid = {p: i for i, p in enumerate(sorted(valid_phrases))}
    nid_to_phrase = {i: p for p, i in phrase_to_nid.items()}
    num_nodes = len(phrase_to_nid)

    # Phrase→topic / score lookup
    phrase_topic = dict(zip(phrase_df["phrase"], phrase_df["topic_id"]))
    phrase_score = dict(zip(phrase_df["phrase"], phrase_df["score"]))

    # --- 2. Filter paper_phrases to valid phrases & build edge index ---
    paper_ids_ordered = []
    paper_to_eid = {}
    incidence_node = []   # node indices
    incidence_edge = []   # hyperedge indices

    papers_meta = papers_df.set_index("paper_id")

    for pid, raw_phrases in paper_phrases.items():
        filtered = [p for p in raw_phrases if p in valid_phrases]
        if not filtered:
            continue
        eid = len(paper_ids_ordered)
        paper_ids_ordered.append(pid)
        paper_to_eid[pid] = eid
        for p in filtered:
            incidence_node.append(phrase_to_nid[p])
            incidence_edge.append(eid)

    num_edges = len(paper_ids_ordered)
    eid_to_paper = {i: pid for i, pid in enumerate(paper_ids_ordered)}

    hyperedge_index = torch.tensor(
        [incidence_node, incidence_edge], dtype=torch.long
    )

    # --- 3. Node features (keywords) ---
    # Features per keyword:
    #   0: topic_id  (integer, but stored as float for stacking)
    #   1: phrase_score  (from topic builder weighting)
    #   2: word_count  (number of tokens in the phrase)
    #   3: degree  (number of papers containing this keyword)

    node_degree = Counter(incidence_node)

    node_feats = np.zeros((num_nodes, 4), dtype=np.float32)
    node_labels = np.zeros(num_nodes, dtype=np.int64)
    for phrase, nid in phrase_to_nid.items():
        tid = phrase_topic.get(phrase, -1)
        sc  = phrase_score.get(phrase, 0.0)
        wc  = len(phrase.split())
        deg = node_degree.get(nid, 0)
        node_feats[nid] = [float(tid), sc, float(wc), float(deg)]
        node_labels[nid] = tid

    # Normalise continuous features (score, degree) to [0, 1]
    for col in [1, 3]:
        lo, hi = node_feats[:, col].min(), node_feats[:, col].max()
        if hi > lo:
            node_feats[:, col] = (node_feats[:, col] - lo) / (hi - lo)

    # --- 4. Hyperedge features (papers) ---
    # Features per paper:
    #   0: year  (normalised)
    #   1: citation_count  (log1p, then normalised)
    #   2: num_keywords  (how many keyword nodes this paper connects)
    #   3: mean_keyword_score  (average phrase score of connected keywords)
    #   4: is_recent  (1 if year >= 2023, else 0)

    edge_degree = Counter(incidence_edge)  # keywords per paper

    edge_feats = np.zeros((num_edges, 5), dtype=np.float32)
    year_per_edge = np.zeros(num_edges, dtype=np.int64)

    for eid, pid in enumerate(paper_ids_ordered):
        if pid in papers_meta.index:
            row = papers_meta.loc[pid]
            yr = int(row.get("year", 0)) if not isinstance(row, pd.DataFrame) else 0
            cc = int(row.get("citation_count", 0)) if not isinstance(row, pd.DataFrame) else 0
        else:
            yr, cc = 0, 0

        nk = edge_degree.get(eid, 0)

        # Mean keyword score for this paper
        connected_nodes = [incidence_node[j] for j in range(len(incidence_node)) if incidence_edge[j] == eid]
        mean_ks = np.mean([phrase_score.get(nid_to_phrase[n], 0.0) for n in connected_nodes]) if connected_nodes else 0.0

        edge_feats[eid] = [float(yr), math.log1p(cc), float(nk), mean_ks, float(yr >= 2023)]
        year_per_edge[eid] = yr

    # Normalise year and log-citation to [0, 1]
    for col in [0, 1, 3]:
        lo, hi = edge_feats[:, col].min(), edge_feats[:, col].max()
        if hi > lo:
            edge_feats[:, col] = (edge_feats[:, col] - lo) / (hi - lo)

    # --- 5. Topic name lookup ---
    topic_names = {}
    for tid, t in topics.items():
        topic_names[tid] = t.get("label", f"topic_{tid}")

    # --- 6. Assemble output ---
    data = {
        "hyperedge_index":    hyperedge_index,
        "node_features":      torch.from_numpy(node_feats),
        "hyperedge_features": torch.from_numpy(edge_feats),
        "node_labels":        torch.from_numpy(node_labels),
        "year_per_edge":      torch.from_numpy(year_per_edge),
        "node_id_to_phrase":  nid_to_phrase,
        "edge_id_to_paper":   eid_to_paper,
        "phrase_to_node_id":  phrase_to_nid,
        "paper_to_edge_id":   paper_to_eid,
        "topic_names":        topic_names,
    }

    # Stats
    stats = {
        "num_nodes":          num_nodes,
        "num_hyperedges":     num_edges,
        "num_incidence_pairs": len(incidence_node),
        "avg_keywords_per_paper": np.mean(list(edge_degree.values())),
        "avg_papers_per_keyword": np.mean(list(node_degree.values())) if node_degree else 0,
        "num_topics":         len(topic_names),
        "year_range":         (int(year_per_edge.min()), int(year_per_edge.max())),
        "papers_2023_plus":   int((year_per_edge >= 2023).sum()),
    }

    return data, stats


# ──────────────────────────────────────────────────────────────────
# Clique expansion (optional, for standard GNN baselines)
# ──────────────────────────────────────────────────────────────────

def clique_expand(hyperedge_index: torch.Tensor) -> torch.Tensor:
    """
    Convert hypergraph to pairwise graph via clique expansion.
    Each hyperedge becomes a clique connecting all its member nodes.
    Returns edge_index [2, E_pairwise] (undirected, deduplicated).
    """
    from collections import defaultdict
    edge_to_nodes = defaultdict(list)
    for i in range(hyperedge_index.size(1)):
        eid = hyperedge_index[1, i].item()
        nid = hyperedge_index[0, i].item()
        edge_to_nodes[eid].append(nid)

    edges = set()
    for nodes in edge_to_nodes.values():
        for a in nodes:
            for b in nodes:
                if a < b:
                    edges.add((a, b))

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)

    src, dst = zip(*edges)
    # Make undirected
    src = list(src) + list(dst)
    dst = list(dst) + list(src[:len(edges)])
    # Actually easier:
    edge_list = list(edges)
    src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
    dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]

    return torch.tensor([src, dst], dtype=torch.long)


# ──────────────────────────────────────────────────────────────────
# Summary & export
# ──────────────────────────────────────────────────────────────────

def write_summary(stats: dict, data: dict, out_path: str):
    lines = [
        "=" * 60,
        "  HYPERGRAPH SUMMARY",
        "=" * 60,
        f"  Keyword nodes        : {stats['num_nodes']:,}",
        f"  Paper hyperedges     : {stats['num_hyperedges']:,}",
        f"  Incidence pairs      : {stats['num_incidence_pairs']:,}",
        f"  Avg keywords / paper : {stats['avg_keywords_per_paper']:.2f}",
        f"  Avg papers / keyword : {stats['avg_papers_per_keyword']:.2f}",
        f"  Topics               : {stats['num_topics']}",
        f"  Year range           : {stats['year_range'][0]} – {stats['year_range'][1]}",
        f"  Papers ≥ 2023        : {stats['papers_2023_plus']:,}",
        "",
        "  Node features  (per keyword)  : [topic_id_norm, score_norm, word_count, degree_norm]",
        "  Edge features  (per paper)    : [year_norm, log_cit_norm, num_kw, mean_kw_score_norm, is_recent]",
        "",
        "  Topic labels:",
    ]
    for tid in sorted(data["topic_names"]):
        lines.append(f"    {tid:3d}: {data['topic_names'][tid]}")
    lines.append("=" * 60)

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def export_gephi_graph(
    data: dict,
    phrase_df: pd.DataFrame,
    papers_df: pd.DataFrame,
    out_path: str,
    fmt: str,
):
    """
    Export a bipartite graph (keywords <-> papers) for Gephi.
    Nodes are prefixed to avoid id collisions: kw_<nid>, paper_<eid>.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise SystemExit(
            "networkx is required for Gephi export. Install with: pip install networkx"
        ) from exc

    hyperedge_index = data["hyperedge_index"]
    nid_to_phrase = data["node_id_to_phrase"]
    eid_to_paper = data["edge_id_to_paper"]

    phrase_score = dict(zip(phrase_df["phrase"], phrase_df["score"]))
    phrase_topic = dict(zip(phrase_df["phrase"], phrase_df["topic_id"]))
    topic_names = data.get("topic_names", {})
    papers_meta = papers_df.set_index("paper_id")

    node_degree = Counter(hyperedge_index[0].tolist())
    edge_degree = Counter(hyperedge_index[1].tolist())

    score_sum = Counter()
    score_count = Counter()

    G = nx.Graph()

    for i in range(hyperedge_index.size(1)):
        nid = int(hyperedge_index[0, i].item())
        eid = int(hyperedge_index[1, i].item())
        phrase = nid_to_phrase[nid]
        paper_id = eid_to_paper[eid]

        kw_node_id = f"kw_{nid}"
        paper_node_id = f"paper_{eid}"

        if kw_node_id not in G:
            tid = int(phrase_topic.get(phrase, -1))
            G.add_node(
                kw_node_id,
                label=phrase,
                type="keyword",
                bipartite=0,
                phrase=phrase,
                topic_id=tid,
                topic_name=topic_names.get(tid, f"topic_{tid}"),
                phrase_score=float(phrase_score.get(phrase, 0.0)),
                word_count=int(len(phrase.split())),
                degree=int(node_degree.get(nid, 0)),
            )

        if paper_node_id not in G:
            if paper_id in papers_meta.index:
                row = papers_meta.loc[paper_id]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                year = int(row.get("year", 0))
                citation_count = int(row.get("citation_count", 0))
            else:
                year = 0
                citation_count = 0

            G.add_node(
                paper_node_id,
                label=str(paper_id),
                type="paper",
                bipartite=1,
                paper_id=str(paper_id),
                year=int(year),
                citation_count=int(citation_count),
                num_keywords=int(edge_degree.get(eid, 0)),
                mean_keyword_score=0.0,
            )

        G.add_edge(kw_node_id, paper_node_id, weight=1.0)
        score_sum[eid] += float(phrase_score.get(phrase, 0.0))
        score_count[eid] += 1

    for eid, total in score_sum.items():
        paper_node_id = f"paper_{eid}"
        if paper_node_id in G:
            cnt = max(score_count.get(eid, 1), 1)
            G.nodes[paper_node_id]["mean_keyword_score"] = float(total) / float(cnt)

    fmt = fmt.lower()
    if fmt == "gexf":
        nx.write_gexf(G, out_path)
    elif fmt == "gml":
        nx.write_gml(G, out_path)
    elif fmt == "graphml":
        nx.write_graphml(G, out_path)
    else:
        raise ValueError(f"Unsupported Gephi format: {fmt}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build keyword–paper hypergraph")
    ap.add_argument("--topics_dir", default=r"c:\Users\User\data\topics_out",
                    help="Directory with topic-builder outputs")
    ap.add_argument("--papers_csv", default=r"c:\Users\User\data\cs_citation_network\papers.csv",
                    help="Path to papers.csv")
    ap.add_argument("--out", default=r"c:\Users\User\data\hypergraph_out",
                    help="Output directory")
    ap.add_argument("--clique", action="store_true",
                    help="Also export clique-expanded pairwise edge_index")
    ap.add_argument("--gephi_format", choices=["gexf", "gml", "graphml"], default="gexf",
                    help="Gephi export format (bipartite graph)")
    ap.add_argument("--gephi_path", default=None,
                    help="Optional output path for Gephi file")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- Load data ---
    print("Loading data …")
    phrase_df     = load_phrase_to_topic(os.path.join(args.topics_dir, "phrase_to_topic.csv"))
    paper_phrases = load_paper_phrases(os.path.join(args.topics_dir, "paper_phrases.json"))
    papers_df     = load_papers(args.papers_csv)
    topics        = load_topics(os.path.join(args.topics_dir, "topics.json"))

    print(f"  {len(phrase_df)} quality phrases, {len(paper_phrases)} papers with phrases, "
          f"{len(papers_df)} papers total, {len(topics)} topics")

    # --- Build hypergraph ---
    print("\nBuilding hypergraph …")
    data, stats = build_hypergraph(phrase_df, paper_phrases, papers_df, topics)

    # --- Optional clique expansion ---
    if args.clique:
        print("Computing clique expansion …")
        clique_edge_index = clique_expand(data["hyperedge_index"])
        data["clique_edge_index"] = clique_edge_index
        print(f"  Clique edges: {clique_edge_index.size(1):,}")

    # --- Save ---
    pt_path = os.path.join(args.out, "hypergraph.pt")
    torch.save(data, pt_path)
    print(f"\nSaved  {pt_path}  ({os.path.getsize(pt_path) / 1024:.0f} KB)")

    summary_path = os.path.join(args.out, "hypergraph_summary.txt")
    write_summary(stats, data, summary_path)
    print(f"Saved  {summary_path}")

    gephi_path = args.gephi_path
    if not gephi_path:
        gephi_path = os.path.join(args.out, f"hypergraph.{args.gephi_format}")
    export_gephi_graph(data, phrase_df, papers_df, gephi_path, args.gephi_format)
    print(f"Saved  {gephi_path}")


if __name__ == "__main__":
    main()
