# ₿ Bitcoin Price Predictor Server

Bitcoin price data analysis API server ⚡️

📦 Built with:
⚡️ FastAPI • 📊 pandas • 🐍 Python

---

## 📁 Data Requirements

This server requires `btcusd_1-min_data.csv` to function properly.

**Download from**: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data

Place the CSV file in this directory before running the server.

---

## 🚀 Quickstart

```bash
git clone https://github.com/open-webui/openapi-servers
cd openapi-servers/servers/bitcoin-price-predictor

# Download btcusd_1-min_data.csv and place it here
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --reload
```

You're live. 📈💰

---

## 🛠️ Available Tools

- `get_price_by_date`: Get Bitcoin price averages for a specific date
- `get_stat_by_date_range`: Get price statistics for a date range  
- `get_trend_by_date_range`: Get daily price trends (max 30 days)
- `get_current_date`: Get current date

🖥️ API Docs: http://localhost:8000/docs
