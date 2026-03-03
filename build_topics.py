
# topic_builder.py — Extract keyword topics from a research-papers CSV.

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import string
import sys
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CURRENT_YEAR: int = 2026  # for recency computation

DOMAIN_STOPLIST: Set[str] = {
    "method", "approach", "tool", "new", "introduction", "synopsis",
    "notes", "hints", "selection", "problem", "paper", "study",
    "analysis", "results", "based", "using", "via", "towards",
    "overview", "survey", "review", "tutorial", "brief", "short",
    "improved", "novel", "efficient", "effective", "simple",
    "propose", "present", "show", "demonstrate", "achieve",
    "recent", "investigate", "existing", "current", "previous",
    "address", "explore", "focus", "consider", "experiment",
}

# Fix spaCy lemma artifacts (e.g. "data" -> "datum")
LEMMA_FIXES: Dict[str, str] = {
    "datum": "data", "medium": "media", "mathematic": "mathematics",
    "statistic": "statistics", "robotic": "robotics",
    "linguistic": "linguistics", "bioinformatic": "bioinformatics",
    "informatic": "informatics", "physic": "physics",
    "dynamic": "dynamics", "mechanic": "mechanics",
    "heuristic": "heuristics", "logistic": "logistics",
    "semantic": "semantics", "genomic": "genomics",
    "analytic": "analytics", "graphic": "graphics",
    "diagnostic": "diagnostics", "therapeutic": "therapeutics",
    "kinematic": "kinematics", "aerodynamic": "aerodynamics",
}

# Single-word phrases rejected as standalone keyphrases
# (perfectly fine inside multi-word compounds)
SINGLE_WORD_REJECT: Set[str] = {
    # ── Too vague / generic standalone ──
    "end", "self", "multi", "base", "pre", "art", "state",
    "role", "power", "depth", "gap", "loss", "step", "point",
    "edge", "case", "work", "open", "close", "look", "face",
    "thing", "use", "limit", "print", "letter", "chain",
    "scratch", "anchor", "game", "story", "book", "movie",
    "nature", "expert", "advance", "comprehensive", "learn",
    "machine", "pattern", "object", "knowledge", "data", "datum",
    "way", "type", "level", "form", "set", "part", "order",
    "key", "kind", "term", "need", "idea", "line", "view",
    "class", "group", "pair", "unit", "field", "piece",
    # ── Generic academic / methodological ──
    "model", "algorithm", "detection", "network", "transfer",
    "prediction", "image", "technique", "identification",
    "application", "vision", "language", "evaluation",
    "generation", "benchmark", "reasoning", "performance",
    "domain", "accuracy", "sequence", "development", "scale",
    "query", "task", "perspective", "instance", "research",
    "context", "code", "program", "extraction", "text",
    "potential", "empirical", "training", "guide", "difficulty",
    "importance", "initialization", "momentum", "architecture",
    "progress", "baseline", "semantics", "inference",
    "regression", "time", "confidence", "segment",
    "augmentation", "deployment", "localization", "information",
    "convergence", "target", "improvement", "exploration",
    "document", "policy", "alignment", "reconstruction",
    "geometry", "era", "resource", "integration", "mask",
    "calibration", "cross", "holistic", "event", "optimizer",
    "cascade", "detector", "pillar", "video", "stage",
    "parameter", "kernel", "methodology", "paradigm", "hybrid",
    "cost", "effect", "design", "structure", "implementation",
    "shot", "sparse", "content", "noise", "future",
    "test", "feature", "word", "representation", "learning",
    "classification", "optimization", "framework", "systematic",
    "attention", "comparison", "comparative", "sentiment",
    "opportunity", "challenge", "dataset", "internet",
    "critical", "relation", "understanding", "generation",
    "literature", "mapping", "bridge", "recommendation",
    "prompt", "robustness", "defense", "attack", "distribute",
    "taxonomy", "cloud", "net", "hand", "update", "conduct",
    "diagram", "strategy", "discussion", "proposal", "source",
    "service", "utility", "vehicle", "intersection", "scan",
    "rgb", "transformation", "modelling", "modeling", "estimate",
    "estimation", "inverse", "batch", "resolution", "ensemble",
    "consistency", "adaptive", "supervision", "perception",
    "dynamics", "backbone", "detector", "region", "tracking",
    "correspondence", "habitat", "simulation", "acceleration",
    "augmentation", "siamese", "fine", "tuning", "graph",
    "precision", "validity", "correlation", "suite", "roc",
    "big", "danger", "acquisition", "entailment", "scene",
    "focal", "bind", "excitation", "union", "instruction",
    "norm", "hardware", "property", "modal", "control",
    "sample", "reduction", "regret", "platform", "sweep",
    "measurement", "measure", "theory", "guide", "rate",
    "introduction", "conclusion", "result", "table", "figure",
    "section", "abstract", "chapter", "appendix",
    "pretraine", "pretrained", "benchmarke", "fouri",
    "process", "rest", "rise",
    "linear", "forecast", "forecasting", "trend", "operator",
    "classifier", "crowdsource", "crowdsourcing", "sentence",
    "subword", "plan", "chess", "ocean", "introduce",
    "freebie", "normality", "represent", "span", "harm",
    "usage", "tune", "optimize", "fact", "verification",
    "checklist", "reliability", "meaning", "variance",
    "examination", "histogram", "library", "modify",
    "generate", "unified", "narrative", "photograph",
    "don", "fog", "dimensionality", "testing", "winter",
    "world", "formula", "search", "gradient", "clarity",
    "mobile", "equality", "pixel", "interpretation",
    "approximation", "student", "foundation", "interface",
    "antenna", "overthinking", "gift", "rout", "connection",
    "difference", "imbalance", "explainable", "detailed",
    "induction", "mixture", "translation", "track",
    "bell", "experimentation", "rethink", "faithfulness",
    "scalability", "advancement", "compositionality",
    "generative", "syntax", "human", "speech",
    "nb15", "train", "google", "microsoft", "stanford",
    "maximization", "precipitation", "bag", "trick",
    "question", "explanation", "accurate",
    "pairwise", "squeeze", "lesson", "decade",
    "replication", "snowball", "caption", "programming",
    "package", "rouge", "summary", "detect",
    "reason", "environment", "ben", "dab", "arc",
    "judge", "bench", "overfitte",
    "feedback", "primer", "phrase", "evolution", "align",
    "fate", "inclusion", "evaluate", "prune", "taming",
    "variational", "habit", "tree", "disciplined", "patience",
    "sparsity", "hallucination", "forgetting", "forget",
    "associate", "learner", "robot",
    "convolution", "segmentation", "privacy", "guideline",
    "uncertainty", "perturbation", "comprehension",
    "transferability", "memorization", "coreference",
    # ── Academic / venue / publisher ──
    "ieee", "acm", "arxiv", "org", "springer", "elsevier",
    "conference", "journal", "transaction", "proceeding",
    "workshop", "symposium", "association", "society",
    "committee", "meeting", "encyclopedia", "access",
    "frontier", "select", "report", "communication",
    "automation", "publication",
    # ── Location / demonym / language ──
    "italian", "chinese", "american", "british", "english",
    "french", "german", "japanese", "korean", "north",
    "european", "international", "national", "annual",
    "united", "states", "china", "arabic", "brazilian",
    "portuguese", "spanish", "russian", "indian", "turkish",
    "iranian", "iran", "germany", "italy", "sweden",
    # ── Non-CS domain terms (standalone) ──
    "mathematics", "medicine", "physics", "engineering",
    "biology", "chemistry", "geology", "psychology",
    "sociology", "economics", "geography", "philosophy",
    "history", "education", "neuroscience", "science",
    "computing", "informatics", "geoscience",
    "photogrammetry", "instrumentation", "management",
    "technology", "intelligent",
    "disease", "patient", "healthcare", "health", "clinical",
    "agriculture", "diabetes", "covid", "covid19", "gene",
    "emotion", "risk", "chest", "ray", "woman", "man",
    "gender", "people", "person", "secret",
    "year", "hour", "minute", "day", "week", "month",
    "diagnosis", "severity", "innovation", "hypothesis",
    "obstacle", "reality", "recognition", "relationship",
    "tweet", "crowd", "safety", "threat", "belief",
    "sensitivity", "specificity", "activation", "formula",
    "signal", "awareness", "experience", "impact",
    "bias", "fairness",
}

# Will be filled by _load_english_stopwords()
_ENGLISH_STOPS: Set[str] = set()

# ---------------------------------------------------------------------------
# Helpers — stopwords
# ---------------------------------------------------------------------------

def _load_english_stopwords() -> Set[str]:
    """Return a broad English stopword set (sklearn's list + extras)."""
    global _ENGLISH_STOPS
    if _ENGLISH_STOPS:
        return _ENGLISH_STOPS
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        _ENGLISH_STOPS = set(ENGLISH_STOP_WORDS)
    except ImportError:
        # minimal fallback
        _ENGLISH_STOPS = {
            "a", "an", "the", "and", "or", "but", "in", "on", "of", "to",
            "for", "with", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "shall", "should", "may", "might", "can", "could", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "out", "up", "down", "about",
            "this", "that", "these", "those", "it", "its", "they", "them",
            "their", "we", "our", "you", "your", "he", "she", "him", "her",
            "not", "no", "nor", "so", "if", "then", "than", "too", "very",
            "just", "also", "more", "most", "other", "some", "such", "only",
            "same", "how", "what", "which", "who", "whom", "when", "where",
            "why", "all", "each", "every", "both", "few", "many", "much",
            "own", "any",
        }
    extras = {"et", "al", "ie", "eg", "vs", "etc", "de", "la", "le", "el"}
    _ENGLISH_STOPS |= extras
    return _ENGLISH_STOPS


# ---------------------------------------------------------------------------
# 1) Load data
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load papers CSV; coerce types and handle missing values."""
    df = pd.read_csv(path, dtype={"paper_id": str, "year": "Int64"})
    df["title"] = df["title"].fillna("")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0).astype(int)
    df["fields_of_study"] = df["fields_of_study"].fillna("")
    df["venue"] = df["venue"].fillna("")
    df["publication_date"] = df["publication_date"].fillna("")
    print(f"  Loaded {len(df)} papers from {path}")
    return df


# ---------------------------------------------------------------------------
# 2) Text normalization
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))


def normalize_text(text: str, lemmatizer: Any = None) -> str:
    """Lowercase, strip punctuation/quotes, normalize hyphens, collapse spaces.

    If *lemmatizer* is a spaCy Language, each token is lemmatized.
    """
    text = text.lower()
    # hyphens / dashes → space
    text = re.sub(r"[-–—]+", " ", text)
    # quotes
    text = re.sub(r"[\"'`'""]", " ", text)
    # punctuation → space
    text = text.translate(_PUNCT_TABLE)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if lemmatizer is not None:
        doc = lemmatizer(text)
        text = " ".join(LEMMA_FIXES.get(tok.lemma_, tok.lemma_) for tok in doc)

    return text


# ---------------------------------------------------------------------------
# 3a) Keyphrase extraction — spaCy noun phrases
# ---------------------------------------------------------------------------

def _phrase_ok(phrase: str, stops: Set[str]) -> bool:
    """Return True if phrase passes quality filters."""
    if len(phrase) < 3:
        return False
    # all digits / year-like
    if re.fullmatch(r"[\d\s]+", phrase):
        return False
    tokens = phrase.split()
    # every token is a stopword or in domain stoplist
    if all(t in stops or t in DOMAIN_STOPLIST for t in tokens):
        return False
    # single char tokens only
    if all(len(t) < 3 for t in tokens):
        return False
    # single-word phrases must not be in SINGLE_WORD_REJECT
    if len(tokens) == 1 and tokens[0] in SINGLE_WORD_REJECT:
        return False
    return True


def _extract_noun_phrases(doc: Any, stops: Set[str]) -> List[str]:
    """Pull cleaned noun-phrase candidates from a spaCy Doc."""
    candidates: List[str] = []
    for chunk in doc.noun_chunks:
        tokens = []
        for tok in chunk:
            lemma = tok.lemma_.lower().strip()
            lemma = LEMMA_FIXES.get(lemma, lemma)
            if lemma in stops or lemma in DOMAIN_STOPLIST or len(lemma) < 3:
                continue
            if re.fullmatch(r"\d+", lemma):
                continue
            tokens.append(lemma)
        if tokens:
            phrase = " ".join(tokens)
            if _phrase_ok(phrase, stops):
                candidates.append(phrase)

    for ent in doc.ents:
        if ent.label_ in ("PERSON", "DATE", "TIME", "MONEY", "PERCENT",
                          "QUANTITY", "ORDINAL", "CARDINAL"):
            continue
        phrase = ent.text.lower().strip()
        phrase = re.sub(r"[^a-z0-9 ]", "", phrase).strip()
        if _phrase_ok(phrase, stops):
            candidates.append(phrase)
    return candidates


def extract_keyphrases_spacy(
    title: str,
    nlp: Any,
    stops: Set[str],
    k: int = 10,
    extra_text: str = "",
) -> List[str]:
    """Extract noun-phrase keyphrases using spaCy.

    Title and metadata are parsed *separately* so that venue/field names
    don't merge into title noun chunks.
    Returns up to *k* phrases sorted by length (longer = more specific).
    """
    # extract from title
    doc_title = nlp(title)
    candidates = _extract_noun_phrases(doc_title, stops)

    # extract from metadata separately (if provided)
    if extra_text and extra_text.strip():
        doc_meta = nlp(extra_text)
        candidates.extend(_extract_noun_phrases(doc_meta, stops))

    # deduplicate, rank by length (longer phrases are more specific)
    seen: Set[str] = set()
    unique: List[str] = []
    for c in candidates:
        norm = re.sub(r"\s+", " ", c).strip()
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(norm)

    unique.sort(key=lambda p: (-len(p.split()), -len(p)))
    return unique[:k]


# ---------------------------------------------------------------------------
# 3b) Keyphrase extraction — n-gram fallback
# ---------------------------------------------------------------------------

def extract_keyphrases_ngrams(
    title: str,
    stops: Set[str],
    k: int = 10,
    extra_text: str = "",
) -> List[str]:
    """Extract keyphrases via n-grams (bigrams–4-grams) + unigrams."""
    from sklearn.feature_extraction.text import CountVectorizer

    combined = title
    if extra_text:
        combined = f"{title} {extra_text}"
    if not combined.strip():
        return []

    all_stops = list(stops | DOMAIN_STOPLIST)

    candidates: List[str] = []
    # n-grams 2–4
    try:
        cv = CountVectorizer(
            ngram_range=(2, 4),
            stop_words=all_stops,
            token_pattern=r"(?u)\b[a-z][a-z0-9]{2,}\b",
        )
        cv.fit_transform([combined])
        candidates.extend(cv.get_feature_names_out().tolist())
    except ValueError:
        pass

    # unigrams (only if not enough multi-word phrases)
    if len(candidates) < k:
        tokens = combined.split()
        for t in tokens:
            t = t.strip()
            if len(t) >= 3 and t not in stops and t not in DOMAIN_STOPLIST and not re.fullmatch(r"\d+", t):
                candidates.append(t)

    seen: Set[str] = set()
    unique: List[str] = []
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip()
        if c and c not in seen and _phrase_ok(c, stops):
            seen.add(c)
            unique.append(c)

    unique.sort(key=lambda p: (-len(p.split()), -len(p)))
    return unique[:k]


# ---------------------------------------------------------------------------
# 4) Paper weighting
# ---------------------------------------------------------------------------

def compute_weights(
    df: pd.DataFrame,
    use_recency: bool = True,
    decay_lambda: float = 0.03,
) -> pd.Series:
    """Return per-paper weight: log1p(citation_count) * optional recency."""
    w = df["citation_count"].apply(lambda c: math.log1p(c))
    if use_recency:
        ages = df["year"].apply(lambda y: max(CURRENT_YEAR - int(y), 0) if pd.notna(y) else 10)
        recency = ages.apply(lambda a: math.exp(-decay_lambda * a))
        w = w * recency
    return w


# ---------------------------------------------------------------------------
# 5) Co-occurrence graph
# ---------------------------------------------------------------------------

def build_cooccurrence_graph(
    paper_phrases: Dict[str, List[str]],
    paper_weights: Dict[str, float],
    min_freq: float = 5.0,
    min_edge_weight: float = 2.0,
) -> nx.Graph:
    """Build a weighted keyword co-occurrence graph.

    Nodes = keyphrases, edges = co-occurrence weighted by paper weight.
    """
    node_weight: Counter = Counter()
    edge_weight: Counter = Counter()

    for pid, phrases in paper_phrases.items():
        w = paper_weights.get(pid, 1.0)
        for p in phrases:
            node_weight[p] += w
        for a, b in combinations(phrases, 2):
            key = tuple(sorted([a, b]))
            edge_weight[key] += w

    # prune low-frequency nodes
    keep_nodes = {n for n, wt in node_weight.items() if wt >= min_freq}

    G = nx.Graph()
    for n in keep_nodes:
        G.add_node(n, weight=node_weight[n])

    for (a, b), wt in edge_weight.items():
        if a in keep_nodes and b in keep_nodes and wt >= min_edge_weight:
            G.add_edge(a, b, weight=wt)

    # remove isolated nodes after edge pruning
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    print(f"  Co-occurrence graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ---------------------------------------------------------------------------
# 6) Community detection
# ---------------------------------------------------------------------------

def detect_communities(G: nx.Graph, resolution: float = 1.0) -> Dict[str, int]:
    """Assign each node a topic_id via Louvain (python-louvain).

    Lower *resolution* produces fewer, larger communities.
    Returns mapping phrase -> topic_id.
    """
    if G.number_of_nodes() == 0:
        return {}

    try:
        import community as community_louvain  # pip install python-louvain
    except ImportError as e:
        raise ImportError(
            "Louvain community detection requires the 'python-louvain' package"
        ) from e

    partition = community_louvain.best_partition(
        G,
        weight="weight",
        random_state=42,
        resolution=resolution,
    )

    n_topics = len(set(partition.values()))
    print(f"  Community detection (Louvain): {n_topics} topics")
    return partition
# ---------------------------------------------------------------------------
# 6b) Merge tiny topics into nearest large neighbour
# ---------------------------------------------------------------------------

def merge_small_topics(
    partition: Dict[str, int],
    G: nx.Graph,
    paper_phrases: Dict[str, List[str]],
    min_topic_papers: int = 10,
) -> Dict[str, int]:
    """Merge small communities into larger ones.

    A topic is small if its phrases collectively cover fewer than
    *min_topic_papers* papers.

    Pass 1 — edge-based: merge by total inter-community edge weight.
    Pass 2 — text-based: for still-small (disconnected) topics, find
             the large topic with the most word overlap in phrases.
    """
    # phrase -> set of paper_ids
    phrase_papers: Dict[str, Set[str]] = defaultdict(set)
    for pid, phrases in paper_phrases.items():
        for p in phrases:
            phrase_papers[p].add(pid)

    def _topic_paper_count(tid: int, tp_map: Dict[int, List[str]]) -> int:
        pids: Set[str] = set()
        for p in tp_map.get(tid, []):
            pids |= phrase_papers.get(p, set())
        return len(pids)

    def _reindex(part: Dict[str, int]) -> Dict[str, int]:
        old_tids = sorted(set(part.values()))
        remap = {old: new for new, old in enumerate(old_tids)}
        return {p: remap[t] for p, t in part.items()}

    def _split_topics(part: Dict[str, int]) -> Tuple[Set[int], Set[int], Dict[int, List[str]]]:
        tp: Dict[int, List[str]] = defaultdict(list)
        for phrase, tid in part.items():
            tp[tid].append(phrase)
        small = {tid for tid in tp if _topic_paper_count(tid, tp) < min_topic_papers}
        large = {tid for tid in tp if tid not in small}
        return small, large, tp

    # ── Pass 1: merge by edge weight ──
    small_tids, large_tids, tp_map = _split_topics(partition)
    if not small_tids or not large_tids:
        return _reindex(partition)

    merge_map: Dict[int, int] = {}
    for stid in small_tids:
        scores: Counter = Counter()
        for phrase in tp_map[stid]:
            for nb in G.neighbors(phrase):
                ntid = partition.get(nb)
                if ntid is not None and ntid in large_tids:
                    scores[ntid] += G[phrase][nb].get("weight", 0)
        if scores:
            merge_map[stid] = scores.most_common(1)[0][0]

    part1: Dict[str, int] = {p: merge_map.get(t, t) for p, t in partition.items()}
    part1 = _reindex(part1)
    edge_merged = len(merge_map)

    # ── Pass 2: text-similarity fallback for still-disconnected topics ──
    small_tids2, large_tids2, tp_map2 = _split_topics(part1)
    merge_map2: Dict[int, int] = {}
    if small_tids2 and large_tids2:
        # build word-set for each large topic
        large_wordsets: Dict[int, Set[str]] = {}
        for ltid in large_tids2:
            words: Set[str] = set()
            for p in tp_map2[ltid]:
                words.update(p.split())
            large_wordsets[ltid] = words

        for stid in small_tids2:
            small_words: Set[str] = set()
            for p in tp_map2[stid]:
                small_words.update(p.split())
            best_tid = None
            best_overlap = 0
            for ltid, lwords in large_wordsets.items():
                overlap = len(small_words & lwords)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_tid = ltid
            if best_tid is not None and best_overlap > 0:
                merge_map2[stid] = best_tid
            else:
                # no word overlap at all — assign to the largest topic
                largest_tid = max(large_tids2,
                                  key=lambda t: _topic_paper_count(t, tp_map2))
                merge_map2[stid] = largest_tid

    part2: Dict[str, int] = {p: merge_map2.get(t, t) for p, t in part1.items()}
    part2 = _reindex(part2)
    text_merged = len(merge_map2)

    n_orig = len(set(partition.values()))
    n_final = len(set(part2.values()))
    print(f"  Merged small topics: {n_orig} -> {n_final} topics "
          f"({edge_merged} edge-merged, {text_merged} text-merged)")
    return part2


# ---------------------------------------------------------------------------
# 7) Topic labeling
# ---------------------------------------------------------------------------

def label_topics(
    G: nx.Graph,
    partition: Dict[str, int],
    paper_phrases: Dict[str, List[str]],
    df: pd.DataFrame,
    top_n_label: int = 1,
    top_n_phrases: int = 20,
) -> List[Dict[str, Any]]:
    """Build rich topic metadata."""

    # phrase -> set of paper_ids
    phrase_papers: Dict[str, Set[str]] = defaultdict(set)
    for pid, phrases in paper_phrases.items():
        for p in phrases:
            phrase_papers[p].add(pid)

    # group phrases by topic
    topic_phrases: Dict[int, List[str]] = defaultdict(list)
    for phrase, tid in partition.items():
        topic_phrases[tid].append(phrase)

    # score each phrase: node_weight + weighted_degree
    def _score(phrase: str) -> float:
        nw = G.nodes[phrase].get("weight", 0) if phrase in G else 0
        wd = sum(d.get("weight", 0) for _, _, d in G.edges(phrase, data=True)) if phrase in G else 0
        return nw + wd

    # fields_of_study and venue lookups
    pid_to_fos: Dict[str, str] = dict(zip(df["paper_id"], df["fields_of_study"]))
    pid_to_venue: Dict[str, str] = dict(zip(df["paper_id"], df["venue"]))

    topics: List[Dict[str, Any]] = []
    for tid in sorted(topic_phrases.keys()):
        phrases = topic_phrases[tid]
        ranked = sorted(phrases, key=_score, reverse=True)

        # papers that contain any phrase in this topic
        pids: Set[str] = set()
        for p in ranked:
            pids |= phrase_papers.get(p, set())

        # field / venue distributions
        fos_counter: Counter = Counter()
        venue_counter: Counter = Counter()
        for pid in pids:
            for f in pid_to_fos.get(pid, "").split(";"):
                f = f.strip()
                if f:
                    fos_counter[f] += 1
            v = pid_to_venue.get(pid, "").strip()
            if v:
                venue_counter[v] += 1

        label = " / ".join(ranked[:top_n_label])
        topics.append({
            "topic_id": tid,
            "label": label,
            "top_phrases": ranked[:top_n_phrases],
            "phrase_count": len(ranked),
            "paper_count": len(pids),
            "paper_ids": sorted(pids),
            "fields_of_study_dist": dict(fos_counter.most_common(10)),
            "venue_dist": dict(venue_counter.most_common(10)),
        })

    return topics


# ---------------------------------------------------------------------------
# 8) Export outputs
# ---------------------------------------------------------------------------

def export_outputs(
    topics: List[Dict[str, Any]],
    partition: Dict[str, int],
    G: nx.Graph,
    paper_phrases: Dict[str, List[str]],
    out_dir: str,
) -> None:
    """Write topics.json, phrase_to_topic.csv, paper_to_topic.csv."""

    os.makedirs(out_dir, exist_ok=True)

    # ── topics.json ──
    topics_slim = []
    for t in topics:
        topics_slim.append({
            "topic_id": t["topic_id"],
            "label": t["label"],
            "top_phrases": t["top_phrases"],
            "phrase_count": t["phrase_count"],
            "paper_count": t["paper_count"],
            "fields_of_study_dist": t["fields_of_study_dist"],
            "venue_dist": t["venue_dist"],
        })
    topics_path = os.path.join(out_dir, "topics.json")
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(topics_slim, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {len(topics_slim)} topics -> {topics_path}")

    # ── phrase_to_topic.csv ──
    def _score(phrase: str) -> float:
        nw = G.nodes[phrase].get("weight", 0) if phrase in G else 0
        wd = sum(d.get("weight", 0) for _, _, d in G.edges(phrase, data=True)) if phrase in G else 0
        return round(nw + wd, 4)

    phrase_path = os.path.join(out_dir, "phrase_to_topic.csv")
    with open(phrase_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phrase", "topic_id", "score"])
        rows = [(p, tid, _score(p)) for p, tid in partition.items()]
        rows.sort(key=lambda r: (-r[2], r[1], r[0]))
        for row in rows:
            w.writerow(row)
    print(f"  Wrote {len(rows)} phrases -> {phrase_path}")

    # ── paper_to_topic.csv ──
    # Assign each paper to the topic whose phrases cover it most (sum of scores).
    phrase_tid = partition
    paper_topic_path = os.path.join(out_dir, "paper_to_topic.csv")
    with open(paper_topic_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "topic_id", "coverage_score"])
        for pid, phrases in paper_phrases.items():
            tid_scores: Counter = Counter()
            for p in phrases:
                if p in phrase_tid:
                    tid_scores[phrase_tid[p]] += _score(p)
            if tid_scores:
                best_tid = tid_scores.most_common(1)[0][0]
                best_score = round(tid_scores[best_tid], 4)
            else:
                best_tid = -1
                best_score = 0.0
            w.writerow([pid, best_tid, best_score])
    papers_written = len(paper_phrases)
    print(f"  Wrote {papers_written} paper assignments -> {paper_topic_path}")

    # ── paper_phrases.json ── (for downstream hypergraph construction etc.)
    pp_path = os.path.join(out_dir, "paper_phrases.json")
    with open(pp_path, "w", encoding="utf-8") as f:
        json.dump(paper_phrases, f, ensure_ascii=False)
    print(f"  Wrote {len(paper_phrases)} paper->phrase lists -> {pp_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_csv: str,
    out_dir: str,
    use_spacy: bool = True,
    k_phrases: int = 10,
    min_freq: float = 5.0,
    min_edge_weight: float = 2.0,
    use_recency: bool = True,
    decay_lambda: float = 0.03,
    include_metadata: bool = False,
    resolution: float = 0.8,
) -> None:
    """End-to-end pipeline."""

    print("=" * 60)
    print("  Topic Builder — keyword co-occurrence community detection")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading data ...")
    df = load_data(input_csv)

    # ── spaCy setup ───────────────────────────────────────────────────
    nlp = None
    lemmatizer = None
    if use_spacy:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")  # full pipeline for noun chunks + NER
            lemmatizer = nlp  # reuse same model for lemmatization
            print("  spaCy en_core_web_sm loaded")
        except (ImportError, OSError) as e:
            print(f"  spaCy unavailable ({e}); falling back to n-gram extraction")
            use_spacy = False
            nlp = None
            lemmatizer = None

    stops = _load_english_stopwords()

    # ── Normalize + extract keyphrases ────────────────────────────────
    print("\n[2/7] Normalizing text & extracting keyphrases ...")
    paper_phrases: Dict[str, List[str]] = {}
    empty_count = 0

    for _, row in df.iterrows():
        pid: str = str(row["paper_id"])
        title_raw: str = str(row["title"])
        if not title_raw.strip():
            empty_count += 1
            continue

        title_norm = normalize_text(title_raw, lemmatizer=lemmatizer)

        # optionally add fields_of_study + venue as extra standalone phrases
        extra = ""
        if include_metadata:
            # Build a cleaner metadata string: separate field names
            # so they become distinct noun phrases, not merged with title
            parts = []
            fos_raw = str(row["fields_of_study"])
            for f in fos_raw.split(";"):
                f = f.strip()
                if f and f.lower() != "computer science":  # too generic, skip
                    parts.append(f.lower())
            venue_raw = str(row["venue"]).strip()
            if venue_raw:
                parts.append(normalize_text(venue_raw))
            extra = " . ".join(parts)  # period separator keeps them as separate chunks

        if use_spacy and nlp is not None:
            phrases = extract_keyphrases_spacy(title_norm, nlp, stops, k=k_phrases, extra_text=extra)
        else:
            phrases = extract_keyphrases_ngrams(title_norm, stops, k=k_phrases, extra_text=extra)

        if phrases:
            paper_phrases[pid] = phrases

    print(f"  Extracted phrases for {len(paper_phrases)} papers ({empty_count} empty titles skipped)")

    # ── Weights ───────────────────────────────────────────────────────
    print("\n[3/7] Computing paper weights ...")
    weight_series = compute_weights(df, use_recency=use_recency, decay_lambda=decay_lambda)
    paper_weights: Dict[str, float] = dict(zip(df["paper_id"].astype(str), weight_series))
    print(f"  Weight range: {min(paper_weights.values()):.3f} – {max(paper_weights.values()):.3f}")

    # ── Co-occurrence graph ───────────────────────────────────────────
    print("\n[4/7] Building co-occurrence graph ...")
    G = build_cooccurrence_graph(paper_phrases, paper_weights, min_freq=min_freq, min_edge_weight=min_edge_weight)

    if G.number_of_nodes() == 0:
        print("  WARNING: Graph is empty after pruning. Try lower --min_freq / --min_edge values.")
        return

    # ── Community detection ───────────────────────────────────────────
    print("\n[5/7] Detecting communities ...")
    partition = detect_communities(G, resolution=resolution)

    # ── Merge tiny communities ────────────────────────────────────────
    partition = merge_small_topics(partition, G, paper_phrases, min_topic_papers=10)

    # ── Topic labeling ────────────────────────────────────────────────
    print("\n[6/7] Labeling topics ...")
    topics = label_topics(G, partition, paper_phrases, df)

    # quick summary
    print(f"\n  {'ID':>4}  {'Phrases':>7}  {'Papers':>7}  Label")
    print(f"  {'--':>4}  {'-------':>7}  {'------':>7}  -----")
    for t in sorted(topics, key=lambda x: -x["paper_count"])[:25]:
        label_short = t["label"][:72]
        print(f"  {t['topic_id']:>4}  {t['phrase_count']:>7}  {t['paper_count']:>7}  {label_short}")
    if len(topics) > 25:
        print(f"  ... and {len(topics) - 25} more topics")

    # ── Export ────────────────────────────────────────────────────────
    print(f"\n[7/7] Exporting to {out_dir} ...")
    export_outputs(topics, partition, G, paper_phrases, out_dir)

    print(f"\nDone!  {len(topics)} topics from {G.number_of_nodes()} phrases across {len(paper_phrases)} papers.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract keyword-based research topics from a papers CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to papers CSV")
    parser.add_argument("--out", default="topics_out", help="Output directory")
    parser.add_argument("--use_spacy", type=int, default=1, choices=[0, 1],
                        help="1=use spaCy noun phrases, 0=use n-gram fallback")
    parser.add_argument("--k_phrases", type=int, default=10, help="Max keyphrases per paper")
    parser.add_argument("--min_freq", type=float, default=5.0, help="Min node frequency to keep")
    parser.add_argument("--min_edge", type=float, default=2.0, help="Min edge weight to keep")
    parser.add_argument("--use_recency", type=int, default=1, choices=[0, 1],
                        help="1=apply recency weighting, 0=citation count only")
    parser.add_argument("--decay_lambda", type=float, default=0.03,
                        help="Exponential decay rate for recency weighting")
    parser.add_argument("--include_metadata", type=int, default=0, choices=[0, 1],
                        help="1=include fields_of_study+venue in phrase extraction")
    parser.add_argument("--resolution", type=float, default=0.8,
                        help="Louvain resolution: lower = fewer larger topics")

    args = parser.parse_args()

    run_pipeline(
        input_csv=args.input,
        out_dir=args.out,
        use_spacy=bool(args.use_spacy),
        k_phrases=args.k_phrases,
        min_freq=args.min_freq,
        min_edge_weight=args.min_edge,
        use_recency=bool(args.use_recency),
        decay_lambda=args.decay_lambda,
        include_metadata=bool(args.include_metadata),
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
