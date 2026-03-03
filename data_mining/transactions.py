from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List


@dataclass
class TransactionsResult:
    years: List[int]
    transactions_by_year: Dict[int, List[List[int]]]
    n_by_year: Dict[int, int]
    single_counts_by_year: Dict[int, Dict[int, int]]


def build_transactions_by_year(
    paper_year: Dict[str, int],
    paper_phrase_ids: Dict[str, List[int]],
    min_year: int,
    max_year: int,
) -> TransactionsResult:
    transactions_by_year: DefaultDict[int, List[List[int]]] = defaultdict(list)
    single_counter_by_year: DefaultDict[int, Counter[int]] = defaultdict(Counter)

    for paper_id, year in paper_year.items():
        if year < min_year or year > max_year:
            continue
        items = paper_phrase_ids.get(paper_id)
        if not items:
            continue
        tx = sorted(set(items))
        if not tx:
            continue
        transactions_by_year[year].append(tx)
        single_counter_by_year[year].update(tx)

    years = list(range(min_year, max_year + 1))
    n_by_year = {year: len(transactions_by_year.get(year, [])) for year in years}
    single_counts_by_year: Dict[int, Dict[int, int]] = {
        year: dict(single_counter_by_year.get(year, Counter())) for year in years
    }
    tx_by_year_full = {year: transactions_by_year.get(year, []) for year in years}

    return TransactionsResult(
        years=years,
        transactions_by_year=tx_by_year_full,
        n_by_year=n_by_year,
        single_counts_by_year=single_counts_by_year,
    )
