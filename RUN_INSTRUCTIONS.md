# Run Instructions

## Quick Start (TL;DR)

```bash
cd c:\Users\User\data2
streamlit run streamlit_app.py
```

Then open browser to `http://localhost:8501`

---

## Detailed Setup

### Step 1: Verify Data Files
Make sure you're in the correct directory and all required files exist:

```bash
cd c:\Users\User\data2

# Check files exist
dir hypergraph.pt
dir papers.csv
dir topics_out\paper_phrases.json
```

If any files are missing, run the data preparation pipelines first:
```bash
python build_graph.py --papers_csv cs_citation_network/papers.csv --topics_dir topics_out --out hypergraph.pt
python build_topics.py --input topics.json --out topics_out
```

### Step 2: Check Python Environment
Ensure you have the required packages:

```bash
python -m pip list | grep -E "streamlit|torch|pandas|plotly|numpy"
```

If missing, install them:
```bash
pip install streamlit torch pandas plotly numpy scipy
```

### Step 3: Run the App

```bash
streamlit run streamlit_app.py
```

You should see:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Step 4: Open in Browser
- Click the link above, or manually open `http://localhost:8501`
- You should see the app with a sidebar
- Start with **Hypergraph Explorer** to verify data loads correctly

---

## Using Each Page

### Page 1: Hypergraph Explorer

**Initial Load**:
1. Sidebar appears with "Year" dropdown
2. Year defaults to latest available
3. Select a year and filter options
4. Graph should render below
5. Tables appear at bottom

**If graph is slow**:
- Reduce "Max papers" to 20 or less
- Reduce "Max topics" to 10 or less
- Use "Advanced Settings" to limit data

### Page 2: GNN Forecast

**Initial Load**:
1. Select "Target year for prediction"
2. (Optional) Select a keyword
3. Click "▶️ Run Forecast"
4. First run takes 30-60 seconds
5. Results appear: predictions table + trend chart
6. Subsequent runs are instant (cached)

**Advanced Options**:
- Click "Enable Advanced Settings" to tune hyperparameters
- Most users should **not** need this

### Page 3: Data Mining Explorer

**Initial Load**:
1. Select "Analysis start year" and "Analysis end year"
2. Select "Forecast year"
3. (Optional) Filter by keyword
4. Click "▶️ Run Mining Analysis"
5. First run takes 15-30 seconds
6. Results appear in 4 tabs: Itemsets, Pairs, Triples, Trends

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
**Fix**: Install Streamlit
```bash
pip install streamlit
```

### "FileNotFoundError: hypergraph.pt"
**Fix**: Regenerate the hypergraph
```bash
python build_graph.py --papers_csv cs_citation_network/papers.csv --topics_dir topics_out --out hypergraph.pt
```

### Graph doesn't load / "No data to display"
1. Check that hypergraph.pt exists and is not corrupted
2. Try reloading the page (Ctrl+R)
3. Check browser console for JavaScript errors (F12)
4. Restart app:
   - Press Ctrl+C in terminal
   - Run `streamlit run streamlit_app.py` again

### App freezes after clicking "Run"
1. This is normal if it's computing
2. Wait 30-60 seconds (first GNN run will be slow)
3. If it's taking >2 minutes, your data might be very large
4. Try reducing parameters in Advanced Settings

### "CUDA out of memory" error
- Restart the app
- In GNN Forecast, disable "Use CUDA" in advanced settings
- Or reduce "Hidden dimension" and "Epochs"

### Port 8501 already in use
Run on a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

---

## Environment Variables (Optional)

```bash
# Run faster (no logger output)
streamlit run streamlit_app.py --logger.level=error

# Run with max theme (dark mode)
streamlit run streamlit_app.py --theme.base=dark

# Disable file watcher (for remote systems)
streamlit run streamlit_app.py --client.showErrorDetails=false
```

---

## Performance Tips

### For Slow Systems
```
Hypergraph Explorer:
  - Max papers: 20
  - Max topics: 10
  - Don't use year range mode

GNN Forecast:
  - Use CPU (disable CUDA)
  - Reduce epochs to 20-30
  - Simple mode only

Data Mining:
  - Min support: 0.10 (10%)
  - Min count: 10
```

### For Fast Systems
```
Hypergraph Explorer:
  - Max papers: 100+
  - Max topics: 50+
  - Use year range freely

GNN Forecast:
  - Use CUDA if available
  - Can tune all parameters

Data Mining:
  - Min support: 0.01 (1%)
  - Can include small datasets
```

---

## Testing the App (Quick Validation)

After starting, verify each page loads:

```bash
# In a new terminal, after app starts:

# Test that Hypergraph page works
curl http://localhost:8501/

# (App will return HTML - just checking it's running)
```

Or manually test in browser:
1. Load http://localhost:8501
2. See "Hypergraph Explorer" selected in sidebar
3. Graph renders with no errors
4. Click "GNN Forecast" → no errors
5. Click "Data Mining Explorer" → no errors

---

## Data Backup

Before running for the first time, backup your data:

```bash
# Make a backup of important files
cp hypergraph.pt hypergraph.pt.backup
cp papers.csv papers.csv.backup
```

### If Something Goes Wrong
```bash
# Restore backups
cp hypergraph.pt.backup hypergraph.pt
cp papers.csv.backup papers.csv

# Restart app
streamlit run streamlit_app.py
```

---

## Stopping the App

```bash
# In the terminal, press:
Ctrl+C

# You should see:
# Keyboard interrupt received (or similar)
```

---

## Next Steps

1. **First Run**: Start with Hypergraph Explorer to explore data
2. **Verify Data**: Generate a graph for current year
3. **Test GNN**: Run a forecast for next year (should complete in <1 min)
4. **Test Mining**: Run mining analysis (should complete in <1 min)
5. **Explore Advanced**: Try adjusting parameters to see effects

---

## Getting Help

### Check Documentation
- `QUICK_START.md` - User guide
- `APP_STRUCTURE.md` - Technical architecture
- `IMPLEMENTATION_SUMMARY.md` - What changed

### Check Logs
```bash
# In terminal where app runs, look for error messages
# Most errors are shown in console and in browser
```

### Common Issues & Solutions
See "Troubleshooting" section above.

---

## System Requirements

### Minimum
- Python 3.8+
- 4GB RAM
- 2GB disk space (for model data)
- CPU (will be slow for GNN)

### Recommended
- Python 3.9+
- 8GB+ RAM
- SSD (faster data loading)
- NVIDIA GPU (for GNN training)

### Tested On
- Windows 10/11
- Python 3.9-3.11
- Streamlit 1.20+

---

## Performance Benchmarks

Typical times on recommended system:

| Operation | Time |
|-----------|------|
| App startup | 2-5 seconds |
| Hypergraph load | 1-2 seconds |
| Graph render | 0.5-1 second |
| GNN forecast (first) | 30-60 seconds |
| GNN forecast (cached) | 0.2 seconds |
| Mining analysis (first) | 15-30 seconds |
| Mining analysis (cached) | 0.2 seconds |

---

## Frequently Asked Questions

**Q: Is the app safe to run repeatedly?**
A: Yes, it's read-only. It doesn't modify your data files.

**Q: Can I run multiple instances?**
A: Yes, but use different ports: `streamlit run ... --server.port 8502`

**Q: Why is the first run slow?**
A: Data loading + model training/computation. Cached afterward.

**Q: Can I use this on a remote server?**
A: Yes, see Remote Deployment section below.

**Q: What if I want to retrain the model?**
A: Clear cache in Advanced Settings or restart app.

---

## Remote Deployment (Advanced)

To run on a remote server:

```bash
# On remote server, start app (background)
nohup streamlit run streamlit_app.py --server.headless true --server.port 8501 &

# On local machine, SSH tunnel
ssh -L 8501:localhost:8501 user@remote-server

# Then open browser to http://localhost:8501
```

---

## Reporting Issues

When reporting bugs, include:
1. Steps to reproduce
2. Error message (full stack trace)
3. Your system info (OS, Python version, packages)
4. Data size (# papers, years)

See error messages in app for clues:
- Red box = Data problem
- Orange warning = Non-critical issue
- Terminal shows detailed logs (useful for debugging)

---

## Uninstalling

To remove the app (but keep data):

```bash
# Deactivate environment (if using venv)
deactivate

# Or manually delete:
rm -r venv/
```

To also remove cached files:

```bash
rm ~/.streamlit/secrets.toml  # If exists
```

Data files (hypergraph.pt, papers.csv, etc.) are untouched.

---

## Support

For issues:
1. Check Troubleshooting section above
2. Read `QUICK_START.md` and `APP_STRUCTURE.md`
3. Look at terminal output (detailed errors logged there)
4. Restart the app (clears session state)

---

**You're ready to go! Start with:**
```bash
streamlit run streamlit_app.py
```
