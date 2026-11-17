# Citation Network & Literature Forecasting Dashboard

An interactive Streamlit dashboard and modeling suite for exploring computer science citation networks, topic dynamics, and predicting emergent research trends using Temporal Hypergraph Neural Networks (THGNN) and frequent itemset time-series mining.

---

## Overview

This repository provides an end-to-end framework and visual interface for:
- **Interactive Hypergraph Exploration**: 3D and 2D temporal hypergraph visualization connecting papers, author topics, and citation dynamics.
- **GNN Residual Forecasting**: Neural network predictions using Temporal Hypergraph Neural Networks (THGNN) to uncover unexpected growth and trending research frontiers.
- **Frequent Itemset & Keyword Mining**: Discovering co-occurring research phrases and forecasting multi-year trend trajectories with baseline drift and AR(1) models.
- **Full Citation Network Analytics**: Graph metrics, degree distributions, community structures, and topic-paper cross-references.

---

## Repository Structure

```
├── streamlit_app.py               # Main Streamlit dashboard application
├── frontend_services.py           # Backend caching, graph layout, and metric services
├── test.html                      # Standalone 3D interactive hypergraph visualization
│
├── cs_citation_network/           # Citation network graph dataset
│   ├── papers.csv                 # Paper metadata (ID, title, year, venue)
│   └── citations.csv              # Directed citation pairs
│
├── topics_out/                    # Extracted topics & phrase mappings
│   ├── topics.json                # Topic clusters and definitions
│   ├── paper_phrases.json         # Normalized keyphrases per paper
│   ├── paper_to_topic.csv         # Paper-topic assignment weights
│   ├── phrase_to_topic.csv        # Phrase-topic memberships
│   └── topics_trend_targets.csv   # Historical and target trend series
│
├── data_mining/                   # Data mining & residual forecasting pipeline
│   ├── preprocess.py              # Phrase extraction & DF filtering
│   ├── transactions.py            # Yearly transaction construction
│   ├── mine_pairs.py              # Pair mining with lift filtering
│   ├── mine_triples.py            # Frequent triple mining
│   ├── series.py                  # Time-series support transformations
│   ├── baseline.py                # Drift and AR(1) rolling baselines
│   ├── forecast_2026.py           # Trend residual ranking for 2026
│   └── pipeline.py                # Data mining CLI pipeline orchestrator
│
├── build_graph.py                 # Hypergraph compilation script
├── build_topics.py                # Topic model builder & extractor
├── run.py                         # THGNN model training & pipeline runner
│
├── hypergraph.gexf                # Exported hypergraph in GEXF format
├── hypergraph.pt                  # PyTorch hypergraph data tensor
├── thgnn_residual_model.pt        # Trained THGNN residual checkpoint
│
├── APP_STRUCTURE.md               # Technical UI architecture & component hierarchy
├── QUICK_START.md                 # User guide and workflow documentation
├── RUN_INSTRUCTIONS.md            # Execution flags, deployment & benchmarks
└── DELIVERABLES.md                # System deliverables and validation notes
```

---

## Quick Start

### 1. Prerequisites

Ensure Python 3.8+ is installed with the required dependencies:

```bash
pip install streamlit pandas numpy torch plotly networkx scipy
```

### 2. Launching the Dashboard

Run the Streamlit application from the root directory:

```bash
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## Dashboard Pages & Workflows

1. **Hypergraph Explorer**:
   - Filter by topic, year range, and minimum citation threshold.
   - Interactive 2D/3D hyperedge visualizer highlighting paper-topic communities.
   - Node-level metrics (degree centrality, citation counts, topic associations).

2. **GNN Trend Forecasting**:
   - Inspect Temporal Hypergraph Neural Network (THGNN) residual predictions.
   - Identify papers and concepts exhibiting positive residual momentum.
   - Top-ranked emerging frontier predictions for 2026.

3. **Data Mining & Keyword Trajectories**:
   - Explore mined pairs and triples of research terms with high lift and support.
   - Multi-year trend curves comparing empirical log-support against rolling drift and AR(1) baselines.
   - Download forecast metrics and evaluated rankings.

---

## Documentation

For in-depth technical details, refer to:
- [QUICK_START.md](file:///c:/Users/User/data2/QUICK_START.md) — Comprehensive user walkthrough.
- [APP_STRUCTURE.md](file:///c:/Users/User/data2/APP_STRUCTURE.md) — Frontend architecture and data caching pipeline.
- [RUN_INSTRUCTIONS.md](file:///c:/Users/User/data2/RUN_INSTRUCTIONS.md) — Step-by-step pipeline execution and retraining guides.
- [DELIVERABLES.md](file:///c:/Users/User/data2/DELIVERABLES.md) — Project deliverables and evaluation summaries.
