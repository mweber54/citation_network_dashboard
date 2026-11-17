#!/usr/bin/env python
"""
Research topic builder v2.

This refactor focuses on topic quality:
- less destructive normalization (surface + canonical forms)
- phrase quality scoring over brittle hard rejections
- reduced clique inflation in co-occurrence edges
- topic diagnostics, hierarchy metadata, and trend-oriented stats
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import string
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import pandas as pd


CURRENT_YEAR = 2026

# Lightweight generic terms used as soft penalties (not hard blocks).
GENERIC_TOKENS: Set[str] = {
    "method",
    "approach",
    "model",
    "analysis",
    "result",
    "study",
    "paper",
    "system",
    "framework",
    "technique",
    "task",
    "application",
    "applications",
    "problem",
    "new",
    "novel",
    "effective",
    "efficient",
    "survey",
    "review",
    "overview",
    "data",
    "state",
    "art",
    "use",
    "based",
}

ARTICLE_TOKENS: Set[str] = {"a", "an", "the"}

VENUE_STYLE_TOKENS: Set[str] = {
    "ieee",
    "acm",
    "arxiv",
    "conference",
    "journal",
    "proceedings",
    "transaction",
    "transactions",
    "workshop",
    "symposium",
    "meeting",
    "annual",
    "international",
}

ACRONYM_ALLOWLIST: Set[str] = {
    "llm",
    "rag",
    "gnn",
    "cnn",
    "rnn",
    "nlp",
    "vqa",
    "bert",
    "gpt",
    "asr",
    "ner",
    "rl",
    "ssl",
    "gan",
}

# Canonical phrase fixes for malformed outputs and frequent variants.
PHRASE_NORMALIZATION_MAP: Dict[str, str] = {
    "natural language process": "natural language processing",
    "federate learning": "federated learning",
    "semantics segmentation": "semantic segmentation",
    "physics inform": "physics informed",
    "pretraine model": "pretrained model",
    "graph neural network": "graph neural networks",
    "large language model": "large language models",
    "retrieval augmented generation": "retrieval-augmented generation",
    "retrieval augmented language generation": "retrieval-augmented generation",
    "vision language model": "vision-language models",
    "state art": "state of the art",
}

BLOCKLIST_PHRASES: Set[str] = {
    "state of the art",
    "state art",
}

TOKEN_NORMALIZATION_MAP: Dict[str, str] = {
    "pretraine": "pretrained",
    "federate": "federated",
    "semantics": "semantic",
    "inform": "informed",
    "modelling": "modeling",
}

PHRASE_NORMALIZATION_REGEX: Sequence[Tuple[re.Pattern[str], str]] = (
    (re.compile(r"\bretrieval\s+augmented\s+generation\b"), "retrieval-augmented generation"),
    (re.compile(r"\bgraph\s+neural\s+network\b"), "graph neural networks"),
    (re.compile(r"\blarge\s+language\s+model\b"), "large language models"),
    (re.compile(r"\bvision\s+language\s+model\b"), "vision-language models"),
)

BROAD_AREA_KEYWORDS: Dict[str, Set[str]] = {
    "Computer Vision": {"object", "detection", "segmentation", "image", "vision", "video", "point cloud", "3d"},
    "Natural Language Processing": {"language", "nlp", "llm", "bert", "gpt", "translation", "retrieval-augmented generation"},
    "Machine Learning": {"learning", "optimization", "representation", "generalization", "reinforcement"},
    "Security and Privacy": {"security", "privacy", "adversarial", "intrusion", "attack", "defense"},
    "Databases and Systems": {"query", "database", "cloud", "distributed", "system", "index"},
}


_ENGLISH_STOPS: Set[str] = set()
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation if c not in {"-"}})


def _load_english_stopwords() -> Set[str]:
    global _ENGLISH_STOPS
    if _ENGLISH_STOPS:
        return _ENGLISH_STOPS
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        _ENGLISH_STOPS = set(ENGLISH_STOP_WORDS)
    except ImportError:
        _ENGLISH_STOPS = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "in",
            "on",
            "of",
            "to",
            "for",
            "with",
            "is",
            "are",
            "was",
            "were",
            "this",
            "that",
            "it",
            "as",
            "by",
            "from",
        }
    _ENGLISH_STOPS |= {"et", "al", "ie", "eg", "vs", "etc"}
    return _ENGLISH_STOPS


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"paper_id": str})
    if "title" not in df.columns:
        df["title"] = ""
    if "abstract" not in df.columns:
        df["abstract"] = ""
    if "fields_of_study" not in df.columns:
        df["fields_of_study"] = ""
    if "venue" not in df.columns:
        df["venue"] = ""
    if "publication_date" not in df.columns:
        df["publication_date"] = ""

    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["fields_of_study"] = df["fields_of_study"].fillna("")
    df["venue"] = df["venue"].fillna("")
    df["publication_date"] = df["publication_date"].fillna("")
    df["year"] = pd.to_numeric(df.get("year", 0), errors="coerce").fillna(0).astype(int)
    df["citation_count"] = pd.to_numeric(df.get("citation_count", 0), errors="coerce").fillna(0).astype(int)
    print(f"  Loaded {len(df)} papers from {path}")
    return df


def normalize_text(text: str) -> str:
    """Normalize text without blind phrase-level lemmatization."""
    text = str(text or "").lower()
    text = re.sub(r"[\u2013\u2014\u2212]", "-", text)
    text = text.replace("/", " ")
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_phrase(phrase: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", phrase.lower()) if tok]


def _extract_acronym_map(text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for full, acro in re.findall(r"([A-Za-z][A-Za-z\- ]{4,80})\s*\(([A-Z]{2,10})\)", text or ""):
        full_clean = normalize_text(full)
        if full_clean:
            mapping[acro.lower()] = full_clean
    return mapping


def canonicalize_phrase(raw_phrase: str, acronym_map: Optional[Dict[str, str]] = None) -> str:
    phrase = normalize_text(raw_phrase)
    if not phrase:
        return ""

    tokens = tokenize_phrase(phrase)
    if not tokens:
        return ""

    if len(tokens) == 1 and acronym_map and tokens[0] in acronym_map:
        phrase = acronym_map[tokens[0]]
        tokens = tokenize_phrase(phrase)

    fixed_tokens = [TOKEN_NORMALIZATION_MAP.get(tok, tok) for tok in tokens]
    phrase = " ".join(fixed_tokens)

    if phrase in PHRASE_NORMALIZATION_MAP:
        phrase = PHRASE_NORMALIZATION_MAP[phrase]

    for patt, replacement in PHRASE_NORMALIZATION_REGEX:
        phrase = patt.sub(replacement, phrase)

    phrase = re.sub(r"\s+", " ", phrase).strip()
    if phrase in BLOCKLIST_PHRASES:
        return ""
    return phrase


def _is_malformed_phrase(phrase: str) -> bool:
    if len(phrase) < 3:
        return True
    if re.fullmatch(r"[\d\s\-]+", phrase):
        return True
    if any(ch * 4 in phrase for ch in "abcdefghijklmnopqrstuvwxyz"):
        return True
    toks = tokenize_phrase(phrase)
    if not toks:
        return True
    if all(len(t) <= 1 for t in toks):
        return True
    return False


def _extract_noun_chunks(doc: Any, stops: Set[str], acronym_map: Dict[str, str]) -> List[Tuple[str, str, float]]:
    mentions: List[Tuple[str, str, float]] = []

    for chunk in doc.noun_chunks:
        cleaned_tokens: List[str] = []
        for tok in chunk:
            t = normalize_text(tok.text)
            if not t:
                continue
            if tok.pos_ in {"DET", "PRON", "ADP", "CCONJ", "SCONJ", "PART"}:
                continue
            if t in stops and t not in ACRONYM_ALLOWLIST:
                continue
            if len(t) <= 1 and t not in ACRONYM_ALLOWLIST:
                continue
            cleaned_tokens.append(t)

        if not cleaned_tokens:
            continue
        if len(cleaned_tokens) == 1 and cleaned_tokens[0] in GENERIC_TOKENS and cleaned_tokens[0] not in ACRONYM_ALLOWLIST:
            continue

        surface = " ".join(cleaned_tokens)
        canonical = canonicalize_phrase(surface, acronym_map=acronym_map)
        if not canonical or _is_malformed_phrase(canonical):
            continue
        toks = tokenize_phrase(canonical)
        if len(toks) == 1 and toks[0] in stops and toks[0] not in ACRONYM_ALLOWLIST:
            continue
        mentions.append((canonical, surface, 1.0))

    for ent in doc.ents:
        if ent.label_ in {"PERSON", "DATE", "TIME", "MONEY", "PERCENT", "ORDINAL", "CARDINAL"}:
            continue
        surface = normalize_text(ent.text)
        canonical = canonicalize_phrase(surface, acronym_map=acronym_map)
        if canonical and not _is_malformed_phrase(canonical):
            mentions.append((canonical, surface, 1.0))

    return mentions


def _extract_ngrams(text: str, stops: Set[str], acronym_map: Dict[str, str]) -> List[Tuple[str, str, float]]:
    mentions: List[Tuple[str, str, float]] = []
    if not text.strip():
        return mentions

    try:
        from sklearn.feature_extraction.text import CountVectorizer

        cv = CountVectorizer(
            ngram_range=(2, 5),
            stop_words=list(stops),
            token_pattern=r"(?u)\b[a-z][a-z0-9\-]{1,}\b",
        )
        cv.fit_transform([text])
        for ngram in cv.get_feature_names_out().tolist():
            canonical = canonicalize_phrase(ngram, acronym_map=acronym_map)
            if canonical and not _is_malformed_phrase(canonical):
                mentions.append((canonical, normalize_text(ngram), 0.8))
    except Exception:
        pass

    return mentions


def extract_phrase_mentions(
    title: str,
    abstract: str,
    fields_of_study: str,
    venue: str,
    use_spacy: bool,
    nlp: Any,
    stops: Set[str],
    include_metadata: bool,
) -> List[Tuple[str, str, float]]:
    """Return (canonical_phrase, surface_phrase, source_weight) mentions."""
    title = str(title or "").strip()
    abstract = str(abstract or "").strip()

    # Primary text: title + abstract when abstract exists.
    if abstract:
        primary = f"{title}. {abstract}"
    else:
        primary = title
    primary_norm = normalize_text(primary)
    acronym_map = _extract_acronym_map(primary)

    mentions: List[Tuple[str, str, float]] = []
    if use_spacy and nlp is not None and primary_norm:
        doc = nlp(primary_norm)
        mentions.extend(_extract_noun_chunks(doc, stops, acronym_map))
    else:
        mentions.extend(_extract_ngrams(primary_norm, stops, acronym_map))

    # Metadata is auxiliary signal only.
    if include_metadata:
        meta_parts: List[str] = []
        for f in str(fields_of_study or "").split(";"):
            f = normalize_text(f)
            if f and f != "computer science":
                meta_parts.append(f)
        venue_norm = normalize_text(venue)
        if venue_norm:
            meta_parts.append(venue_norm)
        meta_text = " . ".join(meta_parts)
        if meta_text:
            if use_spacy and nlp is not None:
                meta_doc = nlp(meta_text)
                for canon, surface, _ in _extract_noun_chunks(meta_doc, stops, acronym_map):
                    mentions.append((canon, surface, 0.35))
            else:
                for canon, surface, _ in _extract_ngrams(meta_text, stops, acronym_map):
                    mentions.append((canon, surface, 0.25))

    # Keep unique mention per canonical form, retaining best source weight.
    best: Dict[str, Tuple[str, float]] = {}
    for canon, surface, src_w in mentions:
        prev = best.get(canon)
        if prev is None or src_w > prev[1]:
            best[canon] = (surface, src_w)

    return [(canon, surface, src_w) for canon, (surface, src_w) in best.items()]


def compute_phrase_quality(
    paper_mentions: Dict[str, List[Tuple[str, str, float]]],
    n_docs: int,
    min_doc_freq: int,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, int]]:
    """Score phrases with soft quality signals rather than hard reject rules."""
    doc_freq: Counter = Counter()
    source_weight_sum: Counter = Counter()
    surface_counts: Dict[str, Counter] = defaultdict(Counter)

    for mentions in paper_mentions.values():
        seen: Set[str] = set()
        for canon, surface, src_w in mentions:
            source_weight_sum[canon] += float(src_w)
            surface_counts[canon][surface] += 1
            if canon not in seen:
                doc_freq[canon] += 1
                seen.add(canon)

    canonical_to_surface: Dict[str, str] = {}
    phrase_quality: Dict[str, float] = {}
    phrase_df: Dict[str, int] = {}

    for canon, df in doc_freq.items():
        phrase_df[canon] = int(df)
        canonical_to_surface[canon] = surface_counts[canon].most_common(1)[0][0]

        toks = tokenize_phrase(canon)
        if not toks:
            continue
        if _is_malformed_phrase(canon):
            continue

        idf = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        df_ratio = df / max(1, n_docs)
        length = len(toks)
        multiword_bonus = 0.65 if length >= 2 else -0.4
        specificity = 1.0 - (sum(1 for t in toks if t in GENERIC_TOKENS) / max(1, length))
        venue_ratio = sum(1 for t in toks if t in VENUE_STYLE_TOKENS) / max(1, length)
        article_ratio = sum(1 for t in toks if t in ARTICLE_TOKENS) / max(1, length)
        acronym_bonus = 0.22 if any(t in ACRONYM_ALLOWLIST for t in toks) else 0.0
        source_bonus = min(0.35, 0.04 * math.log1p(source_weight_sum[canon]))

        malformed_pen = 0.0
        if any(len(t) <= 1 for t in toks):
            malformed_pen += 0.6
        if any(re.search(r"\d{4,}", t) for t in toks):
            malformed_pen += 0.5

        commonness_pen = 0.0
        if df_ratio > 0.20:
            commonness_pen += 1.0 + (df_ratio - 0.20) * 2.0
        elif df_ratio > 0.10:
            commonness_pen += 0.35

        venue_pen = 1.8 * venue_ratio
        article_pen = 1.3 * article_ratio

        generic_ratio = sum(1 for t in toks if t in GENERIC_TOKENS) / max(1, length)
        generic_pen = 1.1 * generic_ratio

        quality = (
            1.05 * idf
            + 0.95 * specificity
            + multiword_bonus
            + acronym_bonus
            + source_bonus
            - malformed_pen
            - commonness_pen
            - venue_pen
            - article_pen
            - generic_pen
        )

        if df < min_doc_freq:
            quality -= 1.0

        phrase_quality[canon] = float(quality)

    return phrase_quality, canonical_to_surface, phrase_df


def select_paper_phrases(
    paper_mentions: Dict[str, List[Tuple[str, str, float]]],
    phrase_quality: Dict[str, float],
    phrase_df: Dict[str, int],
    k_phrases: int,
    quality_threshold: float,
    min_doc_freq: int,
) -> Dict[str, List[str]]:
    paper_phrases: Dict[str, List[str]] = {}
    for pid, mentions in paper_mentions.items():
        scored: List[Tuple[float, str]] = []
        for canon, _surface, src_w in mentions:
            q = phrase_quality.get(canon, -999.0)
            if q < quality_threshold:
                continue
            if phrase_df.get(canon, 0) < min_doc_freq:
                continue
            toks = tokenize_phrase(canon)
            local = q * float(src_w) * (1.0 + 0.08 * max(0, len(toks) - 2))
            scored.append((local, canon))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected: List[str] = []
        seen: Set[str] = set()
        for _score, canon in scored:
            if canon in seen:
                continue
            selected.append(canon)
            seen.add(canon)
            if len(selected) >= k_phrases:
                break
        if selected:
            paper_phrases[pid] = selected
    return paper_phrases


def compute_weights(df: pd.DataFrame, use_recency: bool = True, decay_lambda: float = 0.03) -> pd.Series:
    w = df["citation_count"].apply(lambda c: math.log1p(max(0, int(c))))
    if use_recency:
        ages = df["year"].apply(lambda y: max(CURRENT_YEAR - int(y), 0) if pd.notna(y) else 10)
        recency = ages.apply(lambda a: math.exp(-decay_lambda * a))
        w = w * recency
    return w


def build_cooccurrence_graph(
    paper_phrases: Dict[str, List[str]],
    paper_weights: Dict[str, float],
    phrase_quality: Dict[str, float],
    min_freq: float,
    min_edge_weight: float,
    max_pairs_per_paper: int,
) -> nx.Graph:
    """
    Build normalized co-occurrence graph with reduced clique inflation.

    - Node contribution is normalized by paper phrase count.
    - Pair contribution is normalized and optionally capped per paper.
    """
    node_weight: Counter = Counter()
    edge_weight: Counter = Counter()
    edge_support: Counter = Counter()

    for pid, phrases_raw in paper_phrases.items():
        phrases = sorted(set(phrases_raw))
        if not phrases:
            continue

        w = float(paper_weights.get(pid, 1.0))
        k = len(phrases)
        node_norm = w / max(1.0, math.sqrt(float(k)))

        for p in phrases:
            node_weight[p] += node_norm * (1.0 + 0.18 * max(0.0, phrase_quality.get(p, 0.0)))

        if k < 2:
            continue

        pair_candidates: List[Tuple[float, str, str]] = []
        for a, b in combinations(phrases, 2):
            local_strength = 0.5 + 0.5 * max(0.0, phrase_quality.get(a, 0.0) + phrase_quality.get(b, 0.0))
            pair_candidates.append((local_strength, a, b))

        pair_candidates.sort(key=lambda item: item[0], reverse=True)
        kept = pair_candidates[: max(1, int(max_pairs_per_paper))]
        pair_norm = w / max(1.0, float(k - 1))

        for local_strength, a, b in kept:
            key = tuple(sorted((a, b)))
            edge_weight[key] += pair_norm * local_strength
            edge_support[key] += 1

    keep_nodes = {p for p, wt in node_weight.items() if wt >= float(min_freq)}

    G = nx.Graph()
    for p in keep_nodes:
        G.add_node(p, weight=float(node_weight[p]), quality=float(phrase_quality.get(p, 0.0)))

    for (a, b), wt in edge_weight.items():
        if a not in keep_nodes or b not in keep_nodes:
            continue
        assoc = wt / math.sqrt(max(1e-9, node_weight[a] * node_weight[b]))
        score = wt * (0.55 + 0.45 * assoc)
        if score >= float(min_edge_weight):
            G.add_edge(
                a,
                b,
                weight=float(score),
                raw_weight=float(wt),
                assoc=float(assoc),
                support=int(edge_support[(a, b)]),
            )

    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)

    print(f"  Co-occurrence graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def _louvain_partition(G: nx.Graph, resolution: float) -> Dict[str, int]:
    if G.number_of_nodes() == 0:
        return {}
    try:
        import community as community_louvain
    except ImportError as exc:
        raise ImportError("Louvain clustering requires python-louvain") from exc
    return community_louvain.best_partition(G, weight="weight", random_state=42, resolution=resolution)


def detect_communities(G: nx.Graph, resolution: float = 1.0) -> Dict[str, int]:
    partition = _louvain_partition(G, resolution=resolution)
    print(f"  Community detection (Louvain): {len(set(partition.values()))} topics")
    return partition


def _reindex_partition(partition: Dict[str, int]) -> Dict[str, int]:
    ids = sorted(set(partition.values()))
    remap = {old: new for new, old in enumerate(ids)}
    return {p: remap[t] for p, t in partition.items()}


def build_phrase_papers(paper_phrases: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    phrase_papers: Dict[str, Set[str]] = defaultdict(set)
    for pid, phrases in paper_phrases.items():
        for p in phrases:
            phrase_papers[p].add(pid)
    return phrase_papers


def _topic_phrases(partition: Dict[str, int]) -> Dict[int, List[str]]:
    tp: Dict[int, List[str]] = defaultdict(list)
    for phrase, tid in partition.items():
        tp[int(tid)].append(phrase)
    return tp


def topic_coherence_score(topic_id: int, partition: Dict[str, int], G: nx.Graph) -> float:
    phrases = [p for p, tid in partition.items() if int(tid) == int(topic_id)]
    if len(phrases) < 2:
        return 0.0

    in_w = 0.0
    out_w = 0.0
    phrase_set = set(phrases)
    for p in phrases:
        for nb in G.neighbors(p):
            w = float(G[p][nb].get("weight", 0.0))
            if nb in phrase_set:
                in_w += w
            else:
                out_w += w

    # Internal coherence as ratio of internal edge mass.
    total = in_w + out_w + 1e-9
    return float(in_w / total)


def refine_large_topics(
    partition: Dict[str, int],
    G: nx.Graph,
    base_resolution: float,
    max_topic_phrases: int,
    incoherence_threshold: float,
) -> Dict[str, int]:
    """Split very large and incoherent topics by recursive reclustering."""
    refined = dict(partition)
    tp = _topic_phrases(refined)
    next_tid = max(tp.keys()) + 1 if tp else 0

    for tid, phrases in sorted(tp.items()):
        if len(phrases) <= max_topic_phrases:
            continue
        coh = topic_coherence_score(tid, refined, G)
        if coh >= incoherence_threshold:
            continue

        sub = G.subgraph(phrases).copy()
        if sub.number_of_edges() < 4 or sub.number_of_nodes() < 8:
            continue

        sub_part = _louvain_partition(sub, resolution=max(1.05, base_resolution * 1.35))
        local_ids = sorted(set(sub_part.values()))
        if len(local_ids) <= 1:
            continue

        local_to_global: Dict[int, int] = {}
        for idx, local_tid in enumerate(local_ids):
            if idx == 0:
                local_to_global[local_tid] = int(tid)
            else:
                local_to_global[local_tid] = next_tid
                next_tid += 1

        for phrase in phrases:
            refined[phrase] = local_to_global[int(sub_part[phrase])]

    return _reindex_partition(refined)


def _topic_paper_set(topic_id: int, topic_phrases: Dict[int, List[str]], phrase_papers: Dict[str, Set[str]]) -> Set[str]:
    papers: Set[str] = set()
    for p in topic_phrases.get(int(topic_id), []):
        papers |= phrase_papers.get(p, set())
    return papers


def _topic_token_vector(phrases: Iterable[str], phrase_quality: Dict[str, float]) -> Dict[str, float]:
    vec: Counter = Counter()
    for phrase in phrases:
        weight = max(0.01, float(phrase_quality.get(phrase, 0.1)))
        for tok in tokenize_phrase(phrase):
            vec[tok] += weight
    return dict(vec)


def _cosine_dict(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    dot = 0.0
    for key, val in v1.items():
        dot += val * v2.get(key, 0.0)
    n1 = math.sqrt(sum(v * v for v in v1.values()))
    n2 = math.sqrt(sum(v * v for v in v2.values()))
    if n1 <= 0 or n2 <= 0:
        return 0.0
    return float(dot / (n1 * n2))


def merge_small_topics(
    partition: Dict[str, int],
    G: nx.Graph,
    paper_phrases: Dict[str, List[str]],
    phrase_quality: Dict[str, float],
    min_topic_papers: int,
    merge_threshold: float = 0.28,
) -> Dict[str, int]:
    """
    Merge tiny topics using structural similarity, paper overlap, and centroid similarity.

    Coherent niche topics can remain independent even if small.
    """
    phrase_papers = build_phrase_papers(paper_phrases)
    part = dict(partition)

    def _split(partition_map: Dict[str, int]) -> Tuple[Set[int], Set[int], Dict[int, List[str]], Dict[int, Set[str]]]:
        tp = _topic_phrases(partition_map)
        topic_papers = {tid: _topic_paper_set(tid, tp, phrase_papers) for tid in tp}
        small = {tid for tid, pids in topic_papers.items() if len(pids) < min_topic_papers}
        large = {tid for tid in tp if tid not in small}
        return small, large, tp, topic_papers

    small, large, tp, topic_papers = _split(part)
    if not small or not large:
        return _reindex_partition(part)

    topic_vec = {tid: _topic_token_vector(tp[tid], phrase_quality) for tid in tp}
    merge_map: Dict[int, int] = {}
    preserved_niche = 0

    for stid in sorted(small):
        coh = topic_coherence_score(stid, part, G)
        if coh >= 0.34 and len(tp.get(stid, [])) >= 6:
            preserved_niche += 1
            continue

        candidates: List[Tuple[float, int]] = []
        sphrases = set(tp.get(stid, []))
        s_papers = topic_papers.get(stid, set())

        for ltid in sorted(large):
            lphrases = set(tp.get(ltid, []))
            if not lphrases:
                continue

            inter_w = 0.0
            for p in sphrases:
                if p not in G:
                    continue
                for nb in G.neighbors(p):
                    if nb in lphrases:
                        inter_w += float(G[p][nb].get("weight", 0.0))

            jacc = 0.0
            l_papers = topic_papers.get(ltid, set())
            if s_papers or l_papers:
                inter = len(s_papers & l_papers)
                union = len(s_papers | l_papers)
                jacc = inter / max(1, union)

            sim = _cosine_dict(topic_vec.get(stid, {}), topic_vec.get(ltid, {}))
            score = 0.50 * math.tanh(inter_w / 5.0) + 0.25 * jacc + 0.25 * max(0.0, sim)
            candidates.append((score, ltid))

        if not candidates:
            continue

        best_score, best_tid = max(candidates, key=lambda item: item[0])
        if best_score >= merge_threshold:
            merge_map[stid] = int(best_tid)
        else:
            preserved_niche += 1

    merged = {phrase: merge_map.get(tid, tid) for phrase, tid in part.items()}
    merged = _reindex_partition(merged)

    print(
        "  Merged small topics with structural criteria: "
        f"{len(set(partition.values()))} -> {len(set(merged.values()))} topics "
        f"({len(merge_map)} merged, {preserved_niche} niche preserved)"
    )
    return merged


def _topic_phrase_score(phrase: str, G: nx.Graph, phrase_quality: Dict[str, float]) -> float:
    nw = float(G.nodes[phrase].get("weight", 0.0)) if phrase in G else 0.0
    deg = 0.0
    if phrase in G:
        deg = float(sum(float(d.get("weight", 0.0)) for _, _, d in G.edges(phrase, data=True)))
    return nw + 0.45 * deg + 0.75 * max(0.0, phrase_quality.get(phrase, 0.0))


def _pretty_phrase(phrase: str) -> str:
    acronym_caps = {a.upper() for a in ACRONYM_ALLOWLIST}
    words = []
    for tok in phrase.split():
        t = tok.strip()
        if t.upper() in acronym_caps:
            words.append(t.upper())
        elif t in {"3d", "2d"}:
            words.append(t.upper())
        else:
            words.append(t.capitalize())
    return " ".join(words)


def _label_specificity(label: str) -> float:
    toks = tokenize_phrase(label)
    if not toks:
        return 0.0
    generic_ratio = sum(1 for t in toks if t in GENERIC_TOKENS) / max(1, len(toks))
    return max(0.0, 1.0 - generic_ratio)


def _generate_human_label(rep_phrases: List[str]) -> str:
    if not rep_phrases:
        return "Unlabeled Topic"

    candidates: List[str] = []
    for p in rep_phrases:
        toks = tokenize_phrase(p)
        if not toks:
            continue
        if toks[0] in ARTICLE_TOKENS:
            continue
        generic_ratio = sum(1 for t in toks if t in GENERIC_TOKENS) / max(1, len(toks))
        if generic_ratio >= 0.6:
            continue
        candidates.append(p)
    if candidates:
        rep_phrases = candidates

    p1 = rep_phrases[0]
    p2 = rep_phrases[1] if len(rep_phrases) > 1 else ""

    t1 = set(tokenize_phrase(p1))
    t2 = set(tokenize_phrase(p2))

    # Small set of useful templates before generic fallback.
    if "object" in t1 and "detection" in t1 and ("3d" in t1 or "point" in t1 or "3d" in t2 or "point" in t2):
        return "3D Vision and Object Detection"
    if "large" in t1 and "language" in t1 and "models" in t1:
        return "Large Language Models"
    if "graph" in t1 and "neural" in t1 and "networks" in t1:
        return "Graph Neural Networks"
    if "retrieval" in t1 and "augmented" in t1:
        return "Retrieval-Augmented Generation"

    pretty1 = _pretty_phrase(p1)
    if not p2:
        return pretty1

    # Avoid synthetic "X and Y" labels for weakly related phrases.
    # If overlap is low, prefer the strongest representative phrase only.
    label = pretty1

    return label[:78]


def _assign_broad_area(top_phrases: List[str]) -> str:
    phrase_text = " ".join(top_phrases).lower()
    scores: Dict[str, int] = {}
    for area, keys in BROAD_AREA_KEYWORDS.items():
        score = 0
        for key in keys:
            if key in phrase_text:
                score += 1
        scores[area] = score
    best_area, best_score = max(scores.items(), key=lambda item: item[1]) if scores else ("General AI", 0)
    return best_area if best_score > 0 else "General AI"


def label_topics(
    G: nx.Graph,
    partition: Dict[str, int],
    paper_phrases: Dict[str, List[str]],
    df: pd.DataFrame,
    phrase_quality: Dict[str, float],
    canonical_to_surface: Dict[str, str],
    top_n_label: int = 3,
    top_n_phrases: int = 20,
) -> List[Dict[str, Any]]:
    phrase_papers = build_phrase_papers(paper_phrases)
    topic_phrases = _topic_phrases(partition)

    papers_meta = df.set_index("paper_id", drop=False)

    topics: List[Dict[str, Any]] = []
    for tid in sorted(topic_phrases):
        phrases = topic_phrases[tid]
        ranked = sorted(phrases, key=lambda p: _topic_phrase_score(p, G, phrase_quality), reverse=True)

        representative_surfaces: List[str] = []
        for p in ranked:
            representative_surfaces.append(canonical_to_surface.get(p, p))

        # Build topic paper coverage and representative papers.
        paper_cov: Counter = Counter()
        for p in phrases:
            for pid in phrase_papers.get(p, set()):
                paper_cov[pid] += 1
        topic_pids = set(paper_cov.keys())

        rep_papers = sorted(
            topic_pids,
            key=lambda pid: (
                paper_cov[pid],
                int(papers_meta.at[pid, "citation_count"]) if pid in papers_meta.index else 0,
            ),
            reverse=True,
        )[:5]

        years = [int(papers_meta.at[pid, "year"]) for pid in topic_pids if pid in papers_meta.index]
        cites = [int(papers_meta.at[pid, "citation_count"]) for pid in topic_pids if pid in papers_meta.index]
        year_counter = Counter(years)
        birth_year = min(years) if years else 0
        peak_year = max(year_counter.items(), key=lambda item: item[1])[0] if year_counter else 0

        # Trend-friendly summary signals.
        recent_years = sorted(year_counter.keys())[-3:]
        prev_years = sorted(year_counter.keys())[-6:-3]
        recent_mean = sum(year_counter[y] for y in recent_years) / max(1, len(recent_years))
        prev_mean = sum(year_counter[y] for y in prev_years) / max(1, len(prev_years))
        growth = (recent_mean - prev_mean) / max(1.0, prev_mean)

        top_surfaces = representative_surfaces[: max(1, top_n_label)]
        label = _generate_human_label(top_surfaces)
        coherence = topic_coherence_score(tid, partition, G)

        internal = 0.0
        external = 0.0
        phrase_set = set(phrases)
        for p in phrases:
            if p not in G:
                continue
            for nb in G.neighbors(p):
                w = float(G[p][nb].get("weight", 0.0))
                if nb in phrase_set:
                    internal += w
                else:
                    external += w
        separation = internal / max(1e-9, internal + external)

        low_quality = bool(
            len(topic_pids) < 8
            or coherence < 0.12
            or separation < 0.18
            or _label_specificity(label) < 0.35
        )

        fos_counter: Counter = Counter()
        venue_counter: Counter = Counter()
        for pid in topic_pids:
            if pid not in papers_meta.index:
                continue
            for f in str(papers_meta.at[pid, "fields_of_study"] or "").split(";"):
                f = f.strip()
                if f:
                    fos_counter[f] += 1
            v = str(papers_meta.at[pid, "venue"] or "").strip()
            if v:
                venue_counter[v] += 1

        broad_area = _assign_broad_area(ranked[:8])

        topics.append(
            {
                "topic_id": int(tid),
                "label": label,
                "top_phrases": representative_surfaces[:top_n_phrases],
                "canonical_top_phrases": ranked[:top_n_phrases],
                "phrase_count": len(phrases),
                "paper_count": len(topic_pids),
                "paper_ids": sorted(topic_pids),
                "representative_paper_ids": rep_papers,
                "fields_of_study_dist": dict(fos_counter.most_common(10)),
                "venue_dist": dict(venue_counter.most_common(10)),
                "broad_area": broad_area,
                "hierarchy_level": 1,
                "parent_topic_id": None,
                "coherence": round(float(coherence), 4),
                "separation": round(float(separation), 4),
                "label_specificity": round(float(_label_specificity(label)), 4),
                "low_quality": low_quality,
                "birth_year": int(birth_year),
                "peak_year": int(peak_year),
                "yearly_paper_counts": dict(sorted(year_counter.items())),
                "growth_score": round(float(growth), 4),
                "citation_momentum": round(float(sum(cites[-20:]) / max(1, min(20, len(cites)))), 4) if cites else 0.0,
            }
        )

    # Simple hierarchy assignment: largest topic in each broad area becomes parent.
    by_area: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in topics:
        by_area[str(t.get("broad_area", "General AI"))].append(t)

    for area_topics in by_area.values():
        area_topics.sort(key=lambda t: int(t.get("paper_count", 0)), reverse=True)
        if not area_topics:
            continue
        root_tid = int(area_topics[0]["topic_id"])
        area_topics[0]["hierarchy_level"] = 1
        area_topics[0]["parent_topic_id"] = None
        for child in area_topics[1:]:
            child["hierarchy_level"] = 2
            child["parent_topic_id"] = root_tid

    return sorted(topics, key=lambda t: int(t["topic_id"]))


def build_topic_centroids(partition: Dict[str, int], phrase_quality: Dict[str, float]) -> Dict[int, Dict[str, float]]:
    topic_phrases = _topic_phrases(partition)
    return {tid: _topic_token_vector(phrases, phrase_quality) for tid, phrases in topic_phrases.items()}


def assign_papers_to_topics(
    paper_phrases: Dict[str, List[str]],
    partition: Dict[str, int],
    phrase_scores: Dict[str, float],
    topic_centroids: Dict[int, Dict[str, float]],
    paper_top_k: int,
    min_relative_score: float,
) -> List[Tuple[str, int, float, float, float, str, str]]:
    rows: List[Tuple[str, int, float, float, float, str, str]] = []

    for pid, phrases in paper_phrases.items():
        phrase_evidence: Counter = Counter()
        paper_vec: Counter = Counter()

        for p in phrases:
            for tok in tokenize_phrase(p):
                paper_vec[tok] += max(0.01, phrase_scores.get(p, 0.0))
            if p in partition:
                phrase_evidence[int(partition[p])] += float(phrase_scores.get(p, 0.0))

        if not phrase_evidence:
            rows.append((pid, -1, 0.0, 0.0, 0.0, "", ""))
            continue

        total_evidence = sum(phrase_evidence.values())
        ranked_topics: List[Tuple[int, float]] = []

        for tid, ev in phrase_evidence.items():
            sim = _cosine_dict(dict(paper_vec), topic_centroids.get(int(tid), {}))
            ev_norm = ev / max(1e-9, total_evidence)
            combined = 0.72 * ev_norm + 0.28 * max(0.0, sim)
            ranked_topics.append((int(tid), float(combined)))

        ranked_topics.sort(key=lambda item: item[1], reverse=True)
        best_tid, best_score = ranked_topics[0]
        second_score = ranked_topics[1][1] if len(ranked_topics) > 1 else 0.0
        confidence = best_score / max(1e-9, sum(score for _, score in ranked_topics))
        margin = best_score - second_score

        kept = [
            (tid, score)
            for tid, score in ranked_topics[: max(1, int(paper_top_k))]
            if score >= best_score * float(min_relative_score)
        ]
        top_topic_ids = ";".join(str(tid) for tid, _ in kept)
        top_topic_scores = ";".join(f"{score:.4f}" for _, score in kept)

        rows.append(
            (
                pid,
                int(best_tid),
                round(float(best_score), 4),
                round(float(confidence), 4),
                round(float(margin), 4),
                top_topic_ids,
                top_topic_scores,
            )
        )

    return rows


def export_outputs(
    topics: List[Dict[str, Any]],
    partition: Dict[str, int],
    G: nx.Graph,
    paper_phrases: Dict[str, List[str]],
    out_dir: str,
    paper_top_k: int,
    min_relative_score: float,
    phrase_quality: Dict[str, float],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    topics_path = os.path.join(out_dir, "topics.json")
    topics_slim = []
    for t in topics:
        topics_slim.append(
            {
                "topic_id": t["topic_id"],
                "label": t["label"],
                "top_phrases": t["top_phrases"],
                "phrase_count": t["phrase_count"],
                "paper_count": t["paper_count"],
                "fields_of_study_dist": t["fields_of_study_dist"],
                "venue_dist": t["venue_dist"],
                "broad_area": t["broad_area"],
                "hierarchy_level": t["hierarchy_level"],
                "parent_topic_id": t["parent_topic_id"],
                "coherence": t["coherence"],
                "separation": t["separation"],
                "label_specificity": t["label_specificity"],
                "low_quality": t["low_quality"],
                "birth_year": t["birth_year"],
                "peak_year": t["peak_year"],
                "yearly_paper_counts": t["yearly_paper_counts"],
                "growth_score": t["growth_score"],
                "citation_momentum": t["citation_momentum"],
                "representative_paper_ids": t["representative_paper_ids"],
            }
        )

    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(topics_slim, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {len(topics_slim)} topics -> {topics_path}")

    phrase_path = os.path.join(out_dir, "phrase_to_topic.csv")
    with open(phrase_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phrase", "topic_id", "score"])
        rows = []
        for phrase, tid in partition.items():
            nw = float(G.nodes[phrase].get("weight", 0.0)) if phrase in G else 0.0
            wd = float(sum(float(d.get("weight", 0.0)) for _, _, d in G.edges(phrase, data=True))) if phrase in G else 0.0
            score = round(nw + 0.45 * wd + phrase_quality.get(phrase, 0.0), 4)
            rows.append((phrase, int(tid), score))
        rows.sort(key=lambda r: (-r[2], r[1], r[0]))
        for row in rows:
            writer.writerow(row)
    print(f"  Wrote {len(rows)} phrases -> {phrase_path}")

    phrase_scores = {p: s for p, _tid, s in rows}
    topic_centroids = build_topic_centroids(partition, phrase_quality)
    paper_rows = assign_papers_to_topics(
        paper_phrases=paper_phrases,
        partition=partition,
        phrase_scores=phrase_scores,
        topic_centroids=topic_centroids,
        paper_top_k=paper_top_k,
        min_relative_score=min_relative_score,
    )

    paper_topic_path = os.path.join(out_dir, "paper_to_topic.csv")
    with open(paper_topic_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "paper_id",
                "topic_id",
                "coverage_score",
                "assignment_confidence",
                "score_margin",
                "top_topic_ids",
                "top_topic_scores",
            ]
        )
        for row in paper_rows:
            writer.writerow(row)
    print(f"  Wrote {len(paper_rows)} paper assignments -> {paper_topic_path}")

    pp_path = os.path.join(out_dir, "paper_phrases.json")
    with open(pp_path, "w", encoding="utf-8") as f:
        json.dump(paper_phrases, f, ensure_ascii=False)
    print(f"  Wrote {len(paper_phrases)} paper->phrase lists -> {pp_path}")


def run_pipeline(
    input_csv: str,
    out_dir: str,
    use_spacy: bool = True,
    k_phrases: int = 12,
    min_freq: float = 4.0,
    min_edge_weight: float = 1.2,
    use_recency: bool = True,
    decay_lambda: float = 0.03,
    include_metadata: bool = False,
    resolution: float = 1.0,
    min_topic_papers: int = 10,
    label_phrases: int = 3,
    paper_top_k: int = 3,
    paper_min_relative_score: float = 0.35,
    phrase_quality_min: float = 1.15,
    min_doc_freq: int = 2,
    max_pairs_per_paper: int = 28,
    max_topic_phrases: int = 220,
    incoherence_split_threshold: float = 0.22,
) -> None:
    print("=" * 60)
    print("  Topic Builder v2 - quality-aware research topic discovery")
    print("=" * 60)

    print("\n[1/8] Loading data ...")
    df = load_data(input_csv)

    nlp = None
    if use_spacy:
        try:
            import spacy

            # Keep parser+ner enabled for noun chunks/entities.
            nlp = spacy.load("en_core_web_sm")
            print("  spaCy en_core_web_sm loaded")
        except Exception as exc:
            print(f"  spaCy unavailable ({exc}); falling back to n-grams")
            nlp = None
            use_spacy = False

    stops = _load_english_stopwords()

    print("\n[2/8] Extracting candidate phrases (title+abstract preferred) ...")
    paper_mentions: Dict[str, List[Tuple[str, str, float]]] = {}
    empty_count = 0
    for row in df.itertuples(index=False):
        pid = str(getattr(row, "paper_id"))
        title = str(getattr(row, "title", ""))
        abstract = str(getattr(row, "abstract", ""))
        if not title.strip() and not abstract.strip():
            empty_count += 1
            continue
        mentions = extract_phrase_mentions(
            title=title,
            abstract=abstract,
            fields_of_study=str(getattr(row, "fields_of_study", "")),
            venue=str(getattr(row, "venue", "")),
            use_spacy=use_spacy,
            nlp=nlp,
            stops=stops,
            include_metadata=include_metadata,
        )
        if mentions:
            paper_mentions[pid] = mentions

    print(f"  Candidate mentions extracted for {len(paper_mentions)} papers ({empty_count} empty skipped)")

    print("\n[3/8] Scoring phrase quality ...")
    phrase_quality, canonical_to_surface, phrase_df = compute_phrase_quality(
        paper_mentions=paper_mentions,
        n_docs=len(df),
        min_doc_freq=min_doc_freq,
    )

    paper_phrases = select_paper_phrases(
        paper_mentions=paper_mentions,
        phrase_quality=phrase_quality,
        phrase_df=phrase_df,
        k_phrases=k_phrases,
        quality_threshold=phrase_quality_min,
        min_doc_freq=min_doc_freq,
    )
    print(f"  Selected quality phrases for {len(paper_phrases)} papers")

    print("\n[4/8] Computing paper weights and graph edges ...")
    weight_series = compute_weights(df, use_recency=use_recency, decay_lambda=decay_lambda)
    paper_weights = dict(zip(df["paper_id"].astype(str), weight_series))
    if paper_weights:
        print(f"  Weight range: {min(paper_weights.values()):.3f} - {max(paper_weights.values()):.3f}")

    G = build_cooccurrence_graph(
        paper_phrases=paper_phrases,
        paper_weights=paper_weights,
        phrase_quality=phrase_quality,
        min_freq=min_freq,
        min_edge_weight=min_edge_weight,
        max_pairs_per_paper=max_pairs_per_paper,
    )

    if G.number_of_nodes() == 0:
        print("  WARNING: graph is empty after quality and pruning filters")
        return

    print("\n[5/8] Clustering and coherence-aware splitting ...")
    partition = detect_communities(G, resolution=resolution)
    partition = refine_large_topics(
        partition=partition,
        G=G,
        base_resolution=resolution,
        max_topic_phrases=max_topic_phrases,
        incoherence_threshold=incoherence_split_threshold,
    )

    print("\n[6/8] Structure-aware small-topic merging ...")
    partition = merge_small_topics(
        partition=partition,
        G=G,
        paper_phrases=paper_phrases,
        phrase_quality=phrase_quality,
        min_topic_papers=min_topic_papers,
    )

    print("\n[7/8] Human-friendly labeling + diagnostics + hierarchy/trends ...")
    topics = label_topics(
        G=G,
        partition=partition,
        paper_phrases=paper_phrases,
        df=df,
        phrase_quality=phrase_quality,
        canonical_to_surface=canonical_to_surface,
        top_n_label=label_phrases,
    )

    print(f"\n  {'ID':>4}  {'Phrases':>7}  {'Papers':>7}  {'Coh':>6}  Label")
    print(f"  {'--':>4}  {'-------':>7}  {'------':>7}  {'---':>6}  -----")
    for t in sorted(topics, key=lambda x: -x["paper_count"])[:25]:
        print(
            f"  {t['topic_id']:>4}  {t['phrase_count']:>7}  {t['paper_count']:>7}  {t['coherence']:>6.3f}  {t['label'][:72]}"
        )
    if len(topics) > 25:
        print(f"  ... and {len(topics) - 25} more topics")

    low_quality_count = sum(1 for t in topics if t.get("low_quality"))
    print(f"  Low-quality topics flagged: {low_quality_count}/{len(topics)}")

    print(f"\n[8/8] Exporting outputs to {out_dir} ...")
    export_outputs(
        topics=topics,
        partition=partition,
        G=G,
        paper_phrases=paper_phrases,
        out_dir=out_dir,
        paper_top_k=paper_top_k,
        min_relative_score=paper_min_relative_score,
        phrase_quality=phrase_quality,
    )

    print(f"\nDone! {len(topics)} topics from {G.number_of_nodes()} phrases across {len(paper_phrases)} papers.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build quality-aware research topics from paper metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", required=True, help="Path to papers CSV")
    parser.add_argument("--out", default="topics_out", help="Output directory")

    parser.add_argument("--use_spacy", type=int, default=1, choices=[0, 1], help="1=spaCy phrases, 0=ngram fallback")
    parser.add_argument("--k_phrases", type=int, default=10, help="Max kept phrases per paper")
    parser.add_argument("--min_freq", type=float, default=5.0, help="Min node weight to keep phrase")
    parser.add_argument("--min_edge", type=float, default=1.6, help="Min normalized edge score")
    parser.add_argument("--max_pairs_per_paper", type=int, default=18, help="Cap pair edges contributed per paper")

    parser.add_argument("--use_recency", type=int, default=1, choices=[0, 1], help="Use recency in paper weights")
    parser.add_argument("--decay_lambda", type=float, default=0.03, help="Recency decay")
    parser.add_argument("--include_metadata", type=int, default=0, choices=[0, 1], help="Use venue/FoS as auxiliary phrase signals")

    parser.add_argument("--resolution", type=float, default=1.35, help="Louvain resolution")
    parser.add_argument("--min_topic_papers", type=int, default=12, help="Tiny-topic merge threshold")
    parser.add_argument("--max_topic_phrases", type=int, default=180, help="Split overly large incoherent topics above this size")
    parser.add_argument("--incoherence_split_threshold", type=float, default=0.30, help="Split topic if coherence is below threshold")

    parser.add_argument("--label_phrases", type=int, default=2, help="Representative phrase count for label generation")
    parser.add_argument("--paper_top_k", type=int, default=2, help="Max retained topic assignments per paper")
    parser.add_argument("--paper_min_relative", type=float, default=0.45, help="Keep assignments within this fraction of best paper-topic score")

    parser.add_argument("--phrase_quality_min", type=float, default=1.45, help="Minimum phrase quality score")
    parser.add_argument("--min_doc_freq", type=int, default=4, help="Minimum phrase document frequency")

    args = parser.parse_args()

    run_pipeline(
        input_csv=args.input,
        out_dir=args.out,
        use_spacy=bool(args.use_spacy),
        k_phrases=int(args.k_phrases),
        min_freq=float(args.min_freq),
        min_edge_weight=float(args.min_edge),
        use_recency=bool(args.use_recency),
        decay_lambda=float(args.decay_lambda),
        include_metadata=bool(args.include_metadata),
        resolution=float(args.resolution),
        min_topic_papers=int(args.min_topic_papers),
        label_phrases=int(args.label_phrases),
        paper_top_k=int(args.paper_top_k),
        paper_min_relative_score=float(args.paper_min_relative),
        phrase_quality_min=float(args.phrase_quality_min),
        min_doc_freq=int(args.min_doc_freq),
        max_pairs_per_paper=int(args.max_pairs_per_paper),
        max_topic_phrases=int(args.max_topic_phrases),
        incoherence_split_threshold=float(args.incoherence_split_threshold),
    )


if __name__ == "__main__":
    main()
