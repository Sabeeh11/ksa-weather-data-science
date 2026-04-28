# 🌍 The Kingdom’s Climate Pulse  
### Mining, Clustering, and Modeling Saudi Arabia’s Hourly Weather Data

## 🚀 Overview
This project analyzes large-scale hourly weather observations from Saudi Arabia to uncover hidden weather regimes and validate meteorological patterns using data science techniques.

The workflow combines:
- Real-world noisy sensor data
- Data cleaning and station quality filtering
- Feature engineering using meteorological relationships
- K-Means clustering
- Random Forest classification
- Apriori association rule mining

## 📊 Dataset
- Source: NOAA (National Oceanic and Atmospheric Administration)
- Region: Saudi Arabia air stations
- Size: approximately 900,000 observations
- Features: 35 numerical and categorical weather variables

> The full raw dataset is not included in this repository because of file size.  
> Place the original CSV inside the `data/` folder before running the pipeline.

Expected raw file name:

```text
data/saudi-hourly-weather-data_Historical.csv
```

## ⚙️ Project Pipeline

### 1. Data Cleaning and Preprocessing
- Converted observation timestamps
- Removed administrative columns
- Applied physical constraints to invalid readings
- Filtered stations based on missingness
- Imputed continuous values using station-specific means
- Encoded binary indicators for visibility and sky ceiling presence

### 2. Feature Engineering
Created meteorological features such as:
- Temperature-dewpoint spread
- Approximate relative humidity
- Wind vector components
- Pressure tendency by station

### 3. Unsupervised Learning
Used **K-Means clustering** to identify 4 dominant weather regimes.

### 4. Supervised Learning
Used a **Random Forest classifier** to validate whether the discovered clusters represented learnable structure.

### 5. Association Rule Mining
Used the **Apriori algorithm** to find recurring meteorological patterns.

Example strong rule:

```text
DRY_AIR → NO_CEILING
Confidence: 0.96
```

## 🔍 Key Results
- Identified 4 major weather patterns across Saudi air stations.
- Humid Stable Air Mass was the most common pattern.
- Random Forest achieved approximately 99% accuracy on real cluster labels.
- Shuffled-label validation dropped performance significantly, supporting that the model learned real structure.
- Cloud ceiling data showed systematic missingness across many stations.

## 📁 Repository Structure

```text
ksa-weather-data-science/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── weather_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── modeling.py
│   ├── pattern_mining.py
│   ├── visualization.py
│   └── run_pipeline.py
│
├── outputs/
│   ├── plots/
│   └── models/
│
├── report/
│   └── project_report.pdf
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 🛠️ Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Mlxtend

## ▶️ How to Run

1. Clone the repository.
2. Put the raw dataset in the `data/` folder.
3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Run the full pipeline:

```bash
python src/run_pipeline.py
```

## 👤 Author
Sabeeh Malik Ali Abdul Rahman
