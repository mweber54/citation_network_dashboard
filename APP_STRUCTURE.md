# CS Literature Forecasting App - Structure & Architecture

## Overview

The refactored application is a **multi-page Streamlit UI** designed for researchers to explore Computer Science literature trends using three complementary workflows:

1. **Hypergraph Explorer** - Interactive visualization of keyword/topic networks
2. **GNN Forecast** - AI-powered topic prediction for target years
3. **Data Mining Explorer** - Frequent itemset discovery and forecasting

## Application Architecture

### File Organization

```
streamlit_app.py         # Main multi-page app (refactored)
frontend_services.py     # Core business logic (unchanged)
run.py                   # THGNN model pipeline (unchanged)
data_mining/             # Itemset mining utilities (unchanged)
build_graph.py           # Hypergraph construction (unchanged)
build_topics.py          # Topic extraction (unchanged)
```

### Key Components

#### 1. **Styling & Configuration** (Lines 1-130)
- Clean, researcher-friendly CSS with dark theme
- Sensible typography using Inter + IBM Plex Mono
- Responsive stat cards, buttons, and data tables
- `inject_styles()` - Centralized CSS injection

#### 2. **Caching Layer** (Lines 130-220)
- `@st.cache_resource` - Hypergraph loaded once per session
- `@st.cache_data` - Papers and results cached by parameter hash
- `cached_mining_result()` - Heavy computation cached
- `cached_gnn_result()` - GNN predictions cached

#### 3. **UI Helper Functions** (Lines 220-320)
- `format_value()` - Consistent numerical formatting
- `render_stat_cards()` - Reusable card layout
- `apply_dark_graph_theme()` - Unified Plotly styling
- `_build_hypergraph_figure()` - Graph visualization

#### 4. **Page Functions** (Lines 325-850)

##### Page 1: Hypergraph Explorer (`page_hypergraph()`)
**Purpose**: Visualize keyword-topic networks with temporal and keyword filters

**Features**:
- Year selector with optional year-range mode
- Keyword search/focus
- Advanced settings: max papers, max topics display
- Network visualization (Plotly 2D graph)
- Summary statistics cards
- Detailed topic and paper tables

**Workflow**:
1. Load hypergraph + papers via cache
2. Apply filters (year, keyword, ranges)
3. Build filtered snapshot via `build_hypergraph_snapshot()`
4. Render network visualization
5. Display detailed table views

**Data Flow**:
```
hypergraph.pt → load_hypergraph() → build_hypergraph_snapshot() 
  → HypergraphSnapshot → visualization + tables
```

##### Page 2: GNN Forecast (`page_gnn_forecast()`)
**Purpose**: Predict top topics for a target year using THGNN model

**Features**:
- Target year selector
- Optional keyword focus
- **Simple Mode** (default):
  - One-click forecastwith safe defaults
  - Training: target_year - 10 to target_year - 1
  - Hyperparameters pre-tuned for stability
- **Advanced Mode** (opt-in):
  - Full hyperparameter control
  - Learning rate, hidden dim, dropout, epochs
  - Curve fitting families (exp, power, logistic)
  - Custom window sizes
  - Device selection (CPU/CUDA)

**Output**:
- Top predicted topics ranked by fused score
- Historical trend chart for focus topic
- Training metrics (validation loss, channel weights)

**Workflow**:
1. Validate year ranges
2. Call `run_gnn_dashboard()` with params
3. THGNN trains on windowed subhypergraph
4. Inference on target year window
5. Display results + trend history

##### Page 3: Data Mining Explorer (`page_data_mining()`)
**Purpose**: Discover and forecast frequent itemsets across times

**Features**:
- Analysis year range selector
- Forecast year selector
- Optional keyword filter
- **Simple Mode**:
  - Auto-configured mining thresholds
  - Support, lift, baseline method pre-tuned
  - Window: 3 years
- **Advanced Mode**:
  - Support fraction, minimum count, lift threshold
  - Baseline method (exp, power, logistic)
  - Forecast window size
  - Top-k output tuning

**Output** (4 tabs):
1. **Top Itemsets** - Predicted for next year, ranked by score
2. **Pairs** - Top keyword pairs by support/lift
3. **Triples** - Top 3-itemsets by lift
4. **Trends** - Lead itemset support over time

**Workflow**:
1. Preprocess phrases + build transactions by year
2. Mine pairs/triples per year using A-Priori + lift
3. Compute residuals (signal - baseline fit)
4. Forecast next year residuals
5. Display results in tabs + trend charts

#### 5. **Main Entry Point** (Lines 850-915)
- Sidebar navigation with 3 radio buttons
- Page routing based on selection
- Style injection on load
- Simple, clean UX

## Data Flow & Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit App Entry                      │
│  (inject_styles → sidebar nav → page routing)               │
└─────────────┬───────────────────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
    ▼         ▼         ▼
┌──────┐  ┌──────┐  ┌────────┐
│Graph │  │ GNN  │  │ Mining │
│      │  │      │  │        │
└──┬───┘  └──┬───┘  └────┬───┘
   │         │           │
   │         │           ▼
   │         │      ┌─────────────────┐
   │         │      │ Data Mining     │
   │         │      │ Pipeline        │
   │         │      │ (run mining)    │
   │         │      └────────┬────────┘
   │         │               │
   │         ▼               │
   │    ┌─────────────────┐  │
   │    │ GNN Dashboard   │  │
   │    │ (run_gnn)       │  │
   │    └────────┬────────┘  │
   │             │           │
   ▼             ▼           ▼
┌───────────────────────────────────────┐
│    frontend_services.py               │
│  (Business Logic Layer)               │
└─┬───────┬──────────────┬──────────┬───┘
  │       │              │          │
  ▼       ▼              ▼          ▼
hypergraph topics papers mining   run.py
data       metadata  CSV    funcs   THGNN
```

## Design Decisions

### Safety-by-Default
- Simple mode requires **zero hyperparameter tuning**
- Sensible defaults derived from data size
- Year ranges auto-validated
- Errors fail gracefully with clear messages

### Performance & Caching
- Results cached by exact parameters → no re-computation on UI reuse
- Large hypergraph filtered to viewport-size before rendering
- Spinner feedback for long operations
- Session state stores last results for quick access

### Modularity & Maintainability
- Each page is a self-contained function
- Helper utilities separated (`format_value`, `render_stat_cards`, etc.)
- No code duplication between pages
- Styling centralized in `inject_styles()`
- Business logic remains in `frontend_services.py`

### UX Principles
- **Progressive disclosure**: Advanced settings hidden by default
- **Consistent labeling**: All inputs have help text
- **Visual hierarchy**: Headers, stat cards, tabs organize content
- **Fast feedback**: Caching + spinners prevent UI lock
- **Error resilience**: Missing/stale files handled gracefully

## Key Functions & Signatures

### Core Data Loading
```python
cached_hypergraph(path: str) -> dict
cached_papers(path: str) -> pd.DataFrame
```

### Business Logic (from frontend_services.py)
```python
build_hypergraph_snapshot(
    hypergraph, papers_df, year, keyword, 
    max_papers, max_keywords, min_year=None
) -> HypergraphSnapshot

run_data_mining_dashboard(
    papers_csv, phrases_json, min_year, max_year, target_year,
    minsup_frac, min_count, min_lift, max_k, baseline, window, 
    df_max_frac, df_min_count, topm, min_active_years, keyword
) -> MiningDashboardResult

run_gnn_dashboard(
    hypergraph_pt, papers_csv, train_year_min, train_year_max,
    gnn_window, baseline_window, families, hidden, layers, dropout,
    lr, weight_decay, epochs, seed, device, target_year, keyword, topn
) -> GNNDashboardResult
```

### UI Rendering
```python
render_stat_cards(items: Sequence[Tuple[str, Any]]) -> None
apply_dark_graph_theme(fig: go.Figure) -> go.Figure
_build_hypergraph_figure(snapshot: HypergraphSnapshot) -> go.Figure
```

## Configuration & Defaults

### Hypergraph Explorer
- Default year: latest available
- Default max_papers: 50
- Default max_topics: 20
- Optional year-range mode for temporal analysis

### GNN Forecast - Simple Mode Defaults
- Training: [target_year - 10, target_year - 1]
- GNN window: 3 years
- Hidden dim: 32
- Layers: 2
- Dropout: 0.3
- Learning rate: 0.001
- Epochs: 50
- Baseline window: 3
- Families: exp + power
- Device: GPU if available, else CPU

### Data Mining - Simple Mode Defaults
- Support fraction: 5%
- Min count: 5
- Min lift: 1.2
- Window: 3 years
- Baseline: exponential fit
- Top-k: 10

## Running the App

```bash
streamlit run streamlit_app.py
```

The app will start on `http://localhost:8501` by default.

## Future Extensions

Potential areas for enhancement:
- Export filtered data/results as CSV/JSON
- Comparison views (GNN vs. Mining for same year)
- Interactive parameter recommendations
- Real-time progress indicators
- Caching dashboard (memory usage, cache hits)
- Result history/session replay
- API endpoints for programmatic access

## Technical Notes

- **Streamlit version**: 1.x+
- **Python version**: 3.8+
- **Key dependencies**: torch, plotly, pandas, numpy
- **Data files**: Assumes hypergraph.pt, papers.csv, topics_out/ present
- **Styling**: Custom CSS injected at runtime (no external stylesheets)
- **State management**: Streamlit session_state for cross-page data sharing
