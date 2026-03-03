from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple


Itemset = Tuple[int, ...]


def format_itemset(itemset: Itemset, id_to_phrase: Dict[int, str]) -> str:
    return " | ".join(id_to_phrase[i] for i in itemset)


def build_itemset_universe(
    strong_pairs_by_year: Dict[int, Dict[Tuple[int, int], Dict[str, float]]],
    strong_triples_by_year: Dict[int, Dict[Tuple[int, int, int], Dict[str, float]]],
    min_active_years: int,
    years: Iterable[int],
    recent_years: Iterable[int],
) -> List[Itemset]:
    activity: Dict[Itemset, int] = {}
    recent_activity: Dict[Itemset, float] = {}

    for year in years:
        for itemset, metrics in strong_pairs_by_year.get(year, {}).items():
            key: Itemset = tuple(itemset)
            activity[key] = activity.get(key, 0) + 1
            if year in recent_years:
                recent_activity[key] = recent_activity.get(key, 0.0) + metrics.get("supp_frac", 0.0)
        for itemset, metrics in strong_triples_by_year.get(year, {}).items():
            key = tuple(itemset)
            activity[key] = activity.get(key, 0) + 1
            if year in recent_years:
                recent_activity[key] = recent_activity.get(key, 0.0) + metrics.get("supp_frac", 0.0)

    kept = []
    for itemset, active_years in activity.items():
        if active_years >= min_active_years or recent_activity.get(itemset, 0.0) > 0.0:
            kept.append(itemset)
    kept.sort(key=lambda x: (len(x), x))
    return kept


def build_yearly_metrics_rows(
    itemsets: List[Itemset],
    years: List[int],
    n_by_year: Dict[int, int],
    pair_metrics_by_year: Dict[int, Dict[Tuple[int, int], Dict[str, float]]],
    triple_metrics_by_year: Dict[int, Dict[Tuple[int, int, int], Dict[str, float]]],
    id_to_phrase: Dict[int, str],
) -> Tuple[List[Dict[str, float]], Dict[Itemset, List[float]], Dict[Itemset, List[float]]]:
    rows: List[Dict[str, float]] = []
    z_series: Dict[Itemset, List[float]] = {itemset: [] for itemset in itemsets}
    supp_series: Dict[Itemset, List[float]] = {itemset: [] for itemset in itemsets}

    for year in years:
        n_year = n_by_year.get(year, 0)
        pair_map = pair_metrics_by_year.get(year, {})
        triple_map = triple_metrics_by_year.get(year, {})

        for itemset in itemsets:
            k = len(itemset)
            if k == 2:
                metrics = pair_map.get((itemset[0], itemset[1]))
                sup_count = metrics.get("sup_count", 0.0) if metrics else 0.0
                supp_frac = metrics.get("supp_frac", 0.0) if metrics else 0.0
                lift_min = metrics.get("lift", 0.0) if metrics else 0.0
            else:
                metrics = triple_map.get((itemset[0], itemset[1], itemset[2]))
                sup_count = metrics.get("sup_count", 0.0) if metrics else 0.0
                supp_frac = metrics.get("supp_frac", 0.0) if metrics else 0.0
                lift_min = metrics.get("lift_min", 0.0) if metrics else 0.0

            z = math.log1p(supp_frac)
            z_series[itemset].append(z)
            supp_series[itemset].append(supp_frac)

            rows.append(
                {
                    "year": year,
                    "itemset": format_itemset(itemset, id_to_phrase),
                    "k": k,
                    "sup_count": sup_count,
                    "supp_frac": supp_frac,
                    "z": z,
                    "z_hat": float("nan"),
                    "residual": float("nan"),
                    "lift_min": lift_min,
                    "N_year": n_year,
                }
            )

    return rows, z_series, supp_series
