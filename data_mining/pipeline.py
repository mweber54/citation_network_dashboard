from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from baseline import compute_all_itemset_residuals
from forecast_2026 import evaluate_next_year_alignment, predict_residual_2026
from mine_pairs import mine_pairs_for_year
from mine_triples import mine_triples_for_year
from preprocess import preprocess_phrases
from series import build_itemset_universe, build_yearly_metrics_rows, format_itemset
from transactions import build_transactions_by_year


def _write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_top_phrases(phrase_df: Dict[int, int], id_to_phrase: Dict[int, str], topn: int = 10) -> None:
    top = sorted(phrase_df.items(), key=lambda x: x[1], reverse=True)[:topn]
    print("\nTop 10 phrases by global DF")
    for pid, df in top:
        print(f"  {id_to_phrase[pid]}\tDF={df}")


def _print_top_pairs_2025(
    pair_metrics_2025: Dict[Tuple[int, int], Dict[str, float]],
    id_to_phrase: Dict[int, str],
    topn: int = 10,
) -> None:
    print("\nTop 10 pairs in 2025 by support")
    top_support = sorted(pair_metrics_2025.items(), key=lambda x: x[1]["supp_frac"], reverse=True)[:topn]
    for (a, b), m in top_support:
        print(
            f"  {id_to_phrase[a]} | {id_to_phrase[b]}\t"
            f"supp={m['supp_frac']:.4f}\tcount={int(m['sup_count'])}\tlift={m['lift']:.3f}"
        )

    print("\nTop 10 pairs in 2025 by lift")
    top_lift = sorted(pair_metrics_2025.items(), key=lambda x: x[1]["lift"], reverse=True)[:topn]
    for (a, b), m in top_lift:
        print(
            f"  {id_to_phrase[a]} | {id_to_phrase[b]}\t"
            f"lift={m['lift']:.3f}\tsupp={m['supp_frac']:.4f}\tcount={int(m['sup_count'])}"
        )


def _write_top_2026_report(
    path: Path,
    top_rows: List[Dict[str, float]],
    id_to_phrase: Dict[int, str],
    topm: int,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Top predicted itemsets for 2026\n")
        handle.write("rank\titemset\tk\tpred_residual_2026\trecent_support\tscore_weighted\n")
        for rank, row in enumerate(top_rows[:topm], start=1):
            itemset = row["itemset"]
            if not isinstance(itemset, tuple):
                continue
            itemset_str = " | ".join(id_to_phrase[i] for i in itemset)
            handle.write(
                f"{rank}\t{itemset_str}\t{len(itemset)}\t"
                f"{row['pred_residual_2026']:.6f}\t{row['recent_support']:.6f}\t{row['score_weighted']:.6f}\n"
            )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    default_papers = root / "cs_citation_network" / "papers.csv"
    default_phrases = root / "topics_out" / "paper_phrases.json"
    default_output = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Data mining + time series trend pipeline for keyword itemsets")
    parser.add_argument("--papers_csv", type=Path, default=default_papers)
    parser.add_argument("--phrases_json", type=Path, default=default_phrases)
    parser.add_argument("--output_dir", type=Path, default=default_output)

    parser.add_argument("--min_year", type=int, default=1950)
    parser.add_argument("--max_year", type=int, default=2025)
    parser.add_argument("--minsup_frac", type=float, default=0.01)
    parser.add_argument("--min_count", type=int, default=3)
    parser.add_argument("--min_lift", type=float, default=2.0)
    parser.add_argument("--max_k", type=int, choices=[2, 3], default=3)

    parser.add_argument("--baseline", choices=["drift", "ar1"], default="drift")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--df_max_frac", type=float, default=0.25)
    parser.add_argument("--df_min_count", type=int, default=2)
    parser.add_argument("--topm", type=int, default=100)
    parser.add_argument("--min_active_years", type=int, default=3)
    parser.add_argument("--eval_k", type=int, default=50)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prep = preprocess_phrases(
        papers_csv=args.papers_csv,
        phrases_json=args.phrases_json,
        min_year=args.min_year,
        max_year=args.max_year,
        df_min_count=args.df_min_count,
        df_max_frac=args.df_max_frac,
    )

    print(f"Loaded papers with phrases: {prep.total_docs}")
    print(f"Retained phrase vocabulary size: {len(prep.phrase_to_id)}")
    _print_top_phrases(prep.phrase_df, prep.id_to_phrase, topn=10)

    tx = build_transactions_by_year(
        paper_year=prep.paper_year,
        paper_phrase_ids=prep.paper_phrase_ids,
        min_year=args.min_year,
        max_year=args.max_year,
    )

    pair_metrics_by_year: Dict[int, Dict[Tuple[int, int], Dict[str, float]]] = {}
    strong_pairs_by_year: Dict[int, Dict[Tuple[int, int], Dict[str, float]]] = {}
    strong_triples_by_year: Dict[int, Dict[Tuple[int, int, int], Dict[str, float]]] = {}

    for year in tx.years:
        n_year = tx.n_by_year.get(year, 0)
        year_transactions = tx.transactions_by_year.get(year, [])
        year_singles = tx.single_counts_by_year.get(year, {})

        pair_result = mine_pairs_for_year(
            transactions=year_transactions,
            single_counts=year_singles,
            n_year=n_year,
            minsup_frac=args.minsup_frac,
            min_count=args.min_count,
            min_lift=args.min_lift,
        )
        pair_metrics_by_year[year] = pair_result["frequent_pairs"]
        strong_pairs_by_year[year] = pair_result["strong_pairs"]

        if args.max_k >= 3:
            triples = mine_triples_for_year(
                transactions=year_transactions,
                pair_metrics=strong_pairs_by_year[year],
                n_year=n_year,
                minsup_frac=args.minsup_frac,
                min_count=args.min_count,
                min_lift=args.min_lift,
            )
            strong_triples_by_year[year] = triples
        else:
            strong_triples_by_year[year] = {}

    _print_top_pairs_2025(pair_metrics_by_year.get(2025, {}), prep.id_to_phrase, topn=10)

    itemsets = build_itemset_universe(
        strong_pairs_by_year=strong_pairs_by_year,
        strong_triples_by_year=strong_triples_by_year,
        min_active_years=args.min_active_years,
        years=tx.years,
        recent_years=[2023, 2024, 2025],
    )
    print(f"\nRetained itemsets for time-series modeling: {len(itemsets)}")

    rows, z_series, supp_series = build_yearly_metrics_rows(
        itemsets=itemsets,
        years=tx.years,
        n_by_year=tx.n_by_year,
        pair_metrics_by_year=pair_metrics_by_year,
        triple_metrics_by_year=strong_triples_by_year,
        id_to_phrase=prep.id_to_phrase,
    )

    z_hat_by_itemset, residual_by_itemset = compute_all_itemset_residuals(
        z_series=z_series,
        window=args.window,
        method=args.baseline,
    )

    idx = 0
    for year in tx.years:
        for itemset in itemsets:
            z_hat = z_hat_by_itemset[itemset][tx.years.index(year)]
            residual = residual_by_itemset[itemset][tx.years.index(year)]
            rows[idx]["z_hat"] = z_hat
            rows[idx]["residual"] = residual
            idx += 1

    metrics_path = args.output_dir / "itemsets_yearly_metrics.csv"
    _write_csv(
        metrics_path,
        rows,
        fieldnames=["year", "itemset", "k", "sup_count", "supp_frac", "z", "z_hat", "residual", "lift_min", "N_year"],
    )

    pred_rows = predict_residual_2026(
        itemsets=itemsets,
        years=tx.years,
        residual_by_itemset=residual_by_itemset,
        supp_series=supp_series,
        residual_window_years=(2021, 2025),
    )

    report_path = args.output_dir / "top_itemsets_2026.txt"
    _write_top_2026_report(report_path, pred_rows, prep.id_to_phrase, topm=args.topm)

    print("\nTop 20 predicted 2026 itemsets")
    for row in pred_rows[:20]:
        itemset = row["itemset"]
        if not isinstance(itemset, tuple):
            continue
        itemset_text = format_itemset(itemset, prep.id_to_phrase)
        print(
            f"  {itemset_text}\t"
            f"pred_resid_2026={row['pred_residual_2026']:.6f}\t"
            f"recent_support={row['recent_support']:.6f}"
        )

    eval_rows = evaluate_next_year_alignment(
        itemsets=itemsets,
        years=tx.years,
        residual_by_itemset=residual_by_itemset,
        k=args.eval_k,
        start_year=2005,
        end_year=2024,
    )
    eval_path = args.output_dir / "evaluation_2005_2025.csv"
    _write_csv(eval_path, eval_rows, fieldnames=["year", "precision_at_k", "ndcg_at_k"])

    per_year_top_path = args.output_dir / "top_trending_itemsets_2005_2025.csv"
    per_year_rows: List[Dict[str, float]] = []
    year_to_idx = {year: i for i, year in enumerate(tx.years)}
    for year in range(2005, 2026):
        if year not in year_to_idx:
            continue
        i = year_to_idx[year]
        year_rank = []
        for itemset in itemsets:
            resid = residual_by_itemset[itemset][i]
            if math.isnan(resid):
                continue
            year_rank.append((itemset, resid))
        year_rank.sort(key=lambda x: x[1], reverse=True)
        for rank, (itemset, resid) in enumerate(year_rank[: args.topm], start=1):
            per_year_rows.append(
                {
                    "year": float(year),
                    "rank": float(rank),
                    "itemset": format_itemset(itemset, prep.id_to_phrase),
                    "residual": resid,
                }
            )
    _write_csv(per_year_top_path, per_year_rows, fieldnames=["year", "rank", "itemset", "residual"])

    print(f"\nWrote: {metrics_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {eval_path}")
    print(f"Wrote: {per_year_top_path}")


if __name__ == "__main__":
    run()
