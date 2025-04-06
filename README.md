# traffic-x
Traffic-x is an open-source software for law enforcement officers and researchers
# 🐾 TRAFFIC-X

**TRAFFIC-X** is a Python toolkit to analyze wildlife trafficking data.  
It supports statistical tests, trend analysis, anomaly detection, and criminal network reconstruction.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Features

- ✅ Pearson’s Chi-squared test for species co-occurrence
- 📈 Cumulative trend graphs (annual seizures)
- 📊 Linear and multiple regression
- 🕵️ Anomaly detection using:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - DBSCAN
- 🌐 Criminal network analysis (with weights)
- 🤖 GPT-4-powered result summarization

---

## 📦 Installation

Clone the repo and install locally:

```bash
git clone https://github.com/acarvalho-wcs/traffic-x.git
cd traffic-x
pip install .
```

---

## 📂 Example Usage

```python
import pandas as pd
from trafficx import TrafficAnalyzer

df = pd.read_csv("examples/sample_dataset.csv")
analyzer = TrafficAnalyzer(df)
analyzer.run_all()

# Plot trend
fig = analyzer.plot_trends()
fig.show()

# Generate network graph
fig = analyzer.plot_network(weighted=True)
fig.show()

# GPT-4 Summary (requires API key)
import openai
openai.api_key = "your-api-key"
summary = analyzer.generate_summary()
print(summary)
```

---

## 📁 Folder Structure

```
trafficx/
├── trafficx/                  ← Core module
│   ├── __init__.py
│   └── summaries.py
├── examples/
│   └── sample_dataset.csv
├── README.md
├── setup.py
├── pyproject.toml
├── LICENSE
└── .gitignore
```

---

## 🔐 License

This project is licensed under the [MIT License](LICENSE).

---

## ✨ Credits

Developed with ❤️ using Python, by [@acarvalho-wcs](https://github.com/acarvalho-wcs), with support from ChatGPT and WCS Brasil.
