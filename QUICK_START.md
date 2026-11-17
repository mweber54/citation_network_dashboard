# CS Literature Forecasting - Quick Start Guide

## Installation

### 1. Prerequisites
- Python 3.8+
- Dependencies as specified in environment (torch, streamlit, pandas, plotly, numpy, scipy)

### 2. Verify Data Files
The app expects these files in the working directory:
```
hypergraph.pt              # Compiled hypergraph
papers.csv                 # Paper metadata
topics_out/
  ├─ paper_phrases.json    # Paper-to-phrase associations
  ├─ paper_to_topic.csv    # Paper-to-topic mapping
  └─ topics.json           # Topic metadata
```

Check they exist:
```bash
ls hypergraph.pt papers.csv cs_citation_network/papers.csv
ls topics_out/paper_phrases.json topics_out/topics.json
```

### 3. Run the App
```bash
streamlit run streamlit_app.py
```

The app will launch at `http://localhost:8501`.

---

## Using the App

### **Page 1: Hypergraph Explorer**

#### Purpose
Explore and visualize the keyword-topic network from CS literature across time and topics.

#### Quick Start
1. Open the **Hypergraph Explorer** page from the sidebar
2. Select a year using the year dropdown
3. (Optional) Search for a keyword by name
4. Click "Run" to build and display the network  
5. Hover over nodes to see details
6. Scroll down to see topic and paper tables

#### Workflow
```
Select Year → (Optional: Select Keyword) → View Graph → Explore Details
```

#### Options

| Option | Default | Purpose |
|--------|---------|---------|
| **Year** | Latest | Single year to visualize |
| **Keyword (optional)** | None | Focus the graph on one keyword/phrase |
| **Use year range** | OFF | Enable to include papers from a time window |
| **Max papers** | 50 | Limit papers shown (improves performance) |
| **Max topics** | 20 | Limit topics shown |

#### Output
- **Overview**: 4 key stats (# topics, # papers, # connections, year)
- **Network Graph**: Interactive 2D visualization
- **Topics Table**: Details of each topic node
- **Papers Table**: Details of each paper node

#### Tips
- If the graph is slow, reduce "Max papers" or "Max topics"
- Use keyword search to focus analysis on specific concepts
- The "Use year range" option is useful for temporal trends
- Nodes colored by topic ID; larger nodes = more connections

---

### **Page 2: GNN Forecast**

#### Purpose
Use AI (THGNN model) to predict which topics will be most important in a target year.

#### Quick Start (Simple Mode - Recommended)
1. Open the **GNN Forecast** page
2. Select a **target year** (must be in the future relative to your data)
3. (Optional) Select a **keyword** to track
4. Click **▶️ Run Forecast**
5. View top-10 predicted topics
6. See historical trend for focus topic

#### Workflow (Simple Mode)
```
Select Target Year → (Optional: Select Keyword) → Click Run → View Predictions
```

#### Advanced Mode (Optional)
Enable "Advanced Settings" in the sidebar to tune:

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| Training year start | Any | target - 10 | Earliest training year |
| Training year end | Any | target - 1 | Latest training year |
| Hidden dimension | 16-128 | 32 | Model capacity |
| Layers | 1-5 | 2 | Network depth |
| Dropout | 0.0-0.5 | 0.3 | Regularization |
| Learning rate | 1e-5 to 1e-2 | 1e-3 | Optimization speed |
| Weight decay | 0 to 1e-3 | 1e-5 | L2 regularization |
| Epochs | 10-200 | 50 | Training iterations |
| GNN window | 1-10 | 3 | Temporal window for model input |
| Baseline window | 1-10 | 3 | Rolling window for drift removal |
| Curve families | exp, power, logistic | exp + power | Baseline fitting method |
| Use CUDA | - | auto-detect | GPU acceleration (if available) |

#### Output
- **Overview**: 4 key stats (forecast year, top topic, training period, device)
- **Top Predicted Topics table**: Ranked by fused score (combined importance)
- **Historical Trend chart**: Usage/citation signals over time for focus topic
- Results cached in session for quick re-access

#### Tips
- **Simple mode** is recommended for most users
- Start with default settings and only tune if results seem off
- Training period should be at least 5-10 years
- Target year should be 1-3 years ahead of your data  
- CUDA device is ~10x faster if you have an NVIDIA GPU
- If results are noisy, increase epochs or adjust baseline window

---

### **Page 3: Data Mining Explorer**

#### Purpose
Discover frequent keyword combinations (itemsets) and forecast which will be important next year.

#### Quick Start (Simple Mode - Recommended)
1. Open the **Data Mining Explorer** page
2. Select **analysis year range** (start and end years)
3. Select **forecast year** (the target year to predict for)
4. (Optional) Filter by **keyword**
5. Click **▶️ Run Mining Analysis**
6. View results in 4 tabs: Itemsets, Pairs, Triples, Trends

#### Workflow (Simple Mode)
```
Select Years → (Optional: Filter by Keyword) → Click Run → View Results Tabs
```

#### Advanced Mode (Optional)
Enable "Advanced Settings" to tune:

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| Min support fraction | 1%-20% | 5% | Minimum co-occurrence frequency |
| Min item count | 1-100 | 5 | Minimum number of papers with itemset |
| Min lift | 1.0-3.0 | 1.2 | Minimum association strength |
| Forecast window | 1-10 years | 3 | Historical window for prediction |
| Baseline method | exp, power, logistic | exp | Trend fitting method |
| Top itemsets | 5-50 | 10 | Number of results to display |

#### Output Tabs

**Tab 1: Top Itemsets**
- Predicted itemsets for next year
- Ranked by combined score (residual + support)
- Shows itemset, size (2 or 3), and scores

**Tab 2: Top Pairs**
- Keyword pairs that frequently co-occur
- Ranked by support (frequency) or lift (correlation strength)
- Key metrics: support, count, lift

**Tab 3: Top Triples**
- Keyword triples (3-item associations)
- Ranked by lift and support
- Useful for finding emerging technology combinations

**Tab 4: Trends**
- Support and residual trends over time for the leading itemset
- Shows if a pattern is growing, declining, or stable

#### Tips
- **Simple mode** works well for most datasets
- If you get "no results", increase "Min support fraction" or "Min item count"
- Use keyword filter to focus on specific research areas
- Compare Pairs/Triples results across years to see evolution
- High lift (>2.0) indicates strong, non-obvious associations

---

## Typical Workflows

### Workflow A: Exploring Recent Trends
1. **Hypergraph Explorer**: Select latest year, no filters
2. **GNN Forecast**: Predict 1-2 years ahead
3. **Data Mining**: Analyze last 5 years to see rising itemsets
→ **Insight**: Emerging topics and their relationships

### Workflow B: Deep-Dive into Specific Topic
1. **Hypergraph Explorer**: Search for keyword, use year range
2. **GNN Forecast**: Focus on keyword (optional), forecast
3. **Data Mining**: Filter by keyword, analyze co-occurrences
→ **Insight**: Topic evolution, related keywords, prediction

### Workflow C: Comparing Prediction Methods
1. **GNN Forecast**: Run simple mode forecast
2. **Data Mining**: Run simple mode forecast for same year
3. Compare top topics from both methods
4. (Optional) Adjust advanced settings on each to align approaches
→ **Insight**: GNN often captures emerging signals; Mining shows consensus

---

## Troubleshooting

### App won't start
```bash
# Check Streamlit installation
pip list | grep streamlit

# Try running in verbose mode
streamlit run streamlit_app.py --logger.level=debug
```

### Data loading errors
- Verify files exist in current directory
- Check file permissions (readable)
- Ensure file formats are correct (CSV, JSON, .pt)
- Look at error message for missing column names

### "No data to display"
- Check year/keyword filters are realistic
- Try removing filters (year range mode, keywords)
- Verify hypergraph has data for selected year
- Increase "Max papers" / "Max topics" if limit is too low

### Slow performance
- **Hypergraph**: Reduce "Max papers" or "Max topics"
- **GNN**: Reduce epochs, use CPU if CUDA not available
- **Mining**: Increase support threshold (fewer itemsets = faster)

### Memory errors
- Reduce number of papers/topics to display
- Use "Simple Mode" (safer defaults)
- Close other applications
- If running on low-RAM system, try reducing batch sizes (advanced)

### Results look wrong
- Double-check year selections (training must be before target)
- Verify input data integrity (papers.csv, hypergraph.pt)
- Try simple mode first (defaults are validated)
- Check console output for warnings

---

## Performance Tips

| Action | Impact | How |
|--------|--------|-----|
| Use Simple Mode | ⚡⚡ Faster | Don't expand Advanced Settings |
| Reduce max papers | ⚡ Faster | Set to 20 or less |
| Shorter year range | ⚡ Faster | 3-5 years instead of 20+ |
| Mine fewer itemsets | ⚡ Faster | Increase min support % |
| Use GPU (CUDA) | ⚡⚡⚡ Much Faster | Enable in Advanced Settings |
| First run takes time | Normal | Data is cached after |

---

## File Changes Summary

### Modified Files
| File | Changes | Impact |
|------|---------|--------|
| `streamlit_app.py` | **Complete refactor**: Multi-page structure, clean data flow, better UX | ✅ Now easy to extend and maintain |

### Unchanged Files
| File | Reason | Notes |
|------|--------|-------|
| `frontend_services.py` | Core business logic untouched | Called by streamlit_app.py |
| `run.py` | THGNN implementation stable | No changes needed |
| `data_mining/*` | Mining algorithms stable | No changes needed |
| `build_graph.py`, `build_topics.py` | Data prep tools | No changes needed |

---

## For Developers

### Adding a New Page
```python
def page_new_analysis():
    """My new analysis page."""
    st.title("📊 New Analysis")
    # Your code here
    
# Then add to main():
if page == "New Analysis":
    page_new_analysis()
```

### Reusing Components
```python
# Stat cards
render_stat_cards([
    ("Label 1", value1),
    ("Label 2", value2),
])

# Dark-themed graphs
fig = go.Figure(...)
fig = apply_dark_graph_theme(fig)
st.plotly_chart(fig, use_container_width=True)
```

### Caching New Results
```python
@st.cache_data(show_spinner=False)
def my_cached_function(param1, param2):
    return expensive_computation(param1, param2)
```

---

## Questions?

Refer to:
- `APP_STRUCTURE.md` - Detailed architecture and code organization
- `streamlit_app.py` - Well-commented source code
- `frontend_services.py` - Business logic and data processing
- `run.py` - THGNN model implementation
