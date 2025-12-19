## Road & Traffic Simulation — Prototype

### Setup
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Run

```bash
python -m streamlit run app.py
```
### Use 
- Use the map's draw tools to sketch:
  - Polyline for the road alignment.
  - Markers or polygons for obstacles (potholes, barriers, etc.).
- The drawn layers are captured as GeoJSON and displayed on the right.
- Next iterations will parse this GeoJSON and run a simple traffic model to reveal congestion and recommended fixes.



