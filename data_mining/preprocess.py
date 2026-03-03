from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


@dataclass
class PreprocessResult:
    paper_year: Dict[str, int]
    paper_phrase_ids: Dict[str, List[int]]
    phrase_to_id: Dict[str, int]
    id_to_phrase: Dict[int, str]
    phrase_df: Dict[int, int]
    total_docs: int


def normalize_phrase(phrase: str) -> str:
    text = unicodedata.normalize("NFKC", phrase).lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _load_paper_years(papers_csv: Path, min_year: int, max_year: int) -> Dict[str, int]:
    paper_year: Dict[str, int] = {}
    with papers_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            paper_id = (row.get("paper_id") or "").strip()
            year_text = (row.get("year") or "").strip()
            if not paper_id or not year_text.isdigit():
                continue
            year = int(year_text)
            if min_year <= year <= max_year:
                paper_year[paper_id] = year
    return paper_year


def _load_paper_phrases(phrases_json: Path) -> Dict[str, List[str]]:
    with phrases_json.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    result: Dict[str, List[str]] = {}
    for paper_id, phrases in raw.items():
        if not isinstance(phrases, list):
            continue
        phrase_list = [str(p) for p in phrases if str(p).strip()]
        result[str(paper_id)] = phrase_list
    return result


def preprocess_phrases(
    papers_csv: Path,
    phrases_json: Path,
    min_year: int,
    max_year: int,
    df_min_count: int,
    df_max_frac: float,
) -> PreprocessResult:
    paper_year = _load_paper_years(papers_csv, min_year, max_year)
    paper_phrases = _load_paper_phrases(phrases_json)

    normalized_by_paper: Dict[str, Set[str]] = {}
    global_df: Dict[str, int] = {}

    for paper_id, year in paper_year.items():
        _ = year
        phrases = paper_phrases.get(paper_id)
        if not phrases:
            continue
        norm_set = {normalize_phrase(p) for p in phrases}
        norm_set = {p for p in norm_set if p}
        if not norm_set:
            continue
        normalized_by_paper[paper_id] = norm_set
        for phrase in norm_set:
            global_df[phrase] = global_df.get(phrase, 0) + 1

    total_docs = len(normalized_by_paper)
    max_df_count = int(total_docs * df_max_frac)
    kept_phrases = {
        phrase
        for phrase, df in global_df.items()
        if df >= df_min_count and (max_df_count <= 0 or df <= max_df_count)
    }

    phrase_to_id: Dict[str, int] = {}
    id_to_phrase: Dict[int, str] = {}
    phrase_df: Dict[int, int] = {}

    for phrase in sorted(kept_phrases):
        pid = len(phrase_to_id)
        phrase_to_id[phrase] = pid
        id_to_phrase[pid] = phrase
        phrase_df[pid] = global_df[phrase]

    paper_phrase_ids: Dict[str, List[int]] = {}
    for paper_id, phrase_set in normalized_by_paper.items():
        ids = sorted(phrase_to_id[p] for p in phrase_set if p in phrase_to_id)
        if ids:
            paper_phrase_ids[paper_id] = ids

    filtered_paper_year = {pid: y for pid, y in paper_year.items() if pid in paper_phrase_ids}

    return PreprocessResult(
        paper_year=filtered_paper_year,
        paper_phrase_ids=paper_phrase_ids,
        phrase_to_id=phrase_to_id,
        id_to_phrase=id_to_phrase,
        phrase_df=phrase_df,
        total_docs=len(filtered_paper_year),
    )
