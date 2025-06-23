from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import pandas as pd

from datetime import datetime

app = FastAPI(
    title="Bitcoin Price Predictor",
    description="Bitcoin Price Predictor",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bitcoin_price_data = pd.read_csv("btcusd_1-min_data.csv")
bitcoin_price_data["Timestamp"] = pd.to_datetime(
    bitcoin_price_data["Timestamp"], unit='s')


class BitcoinPriceQueryForm(BaseModel):
    date: str = Field(
        ..., description="This is desired date time for bitcoin price and time format is YYYY-MM-DD (e.g., 2024-01-01)"
    )


class BitcoinPriceTrendQueryForm(BaseModel):
    start_date: str = Field(
        ..., description="This is desired start date time for bitcoin price trend and time format is YYYY-MM-DD (e.g., 2023-01-01)"
    )
    end_date: str = Field(
        ..., description="This is desired end date time for bitcoin price trend and time format is YYYY-MM-DD (e.g., 2024-01-01)"
    )


class BitcoinPriceStatQueryForm(BaseModel):
    start_date: str = Field(
        ..., description="This is desired start date time for bitcoin price state and time format is YYYY-MM-DD (e.g., 2023-01-01)"
    )
    end_date: str = Field(
        ..., description="This is desired end date time for bitcoin price state and time format is YYYY-MM-DD (e.g., 2024-01-01)"
    )


@app.post("/get_price_by_date", summary="Get Bitcoin price by date")
async def get_price_by_date(form_data: BitcoinPriceQueryForm):
    print(form_data)
    start_date = pd.to_datetime(form_data.date)
    end_date = start_date + pd.Timedelta(days=1)

    date_query_result = bitcoin_price_data[(
        bitcoin_price_data["Timestamp"] >= start_date) & (bitcoin_price_data["Timestamp"] < end_date)]

    if date_query_result.empty:
        raise HTTPException(
            status_code=404, detail="No data found for the specified date")

    date_query_result_mean = date_query_result.mean(
        numeric_only=True)
    return date_query_result_mean.to_dict()


@app.post("/get_stat_by_date_range", summary="Get Bitcoin price status by date range")
async def get_stat_by_date_range(form_data: BitcoinPriceStatQueryForm):
    start_date = pd.to_datetime(form_data.start_date)
    end_date = pd.to_datetime(form_data.end_date)
    date_query_result = bitcoin_price_data[(
        bitcoin_price_data["Timestamp"] >= start_date) & (bitcoin_price_data["Timestamp"] < end_date)]

    if date_query_result.empty:
        raise HTTPException(
            status_code=404, detail="No data found for the specified date range")

    highest_price = date_query_result["High"].max()
    lowest_price = date_query_result["Low"].min()
    average_price = date_query_result["Close"].mean()

    return {
        "highest_price": highest_price,
        "lowest_price": lowest_price,
        "average_price": average_price
    }


@app.post("/get_trend_by_date_range", summary="Get Bitcoin price status by date range")
async def get_trend_by_date_range(form_data: BitcoinPriceTrendQueryForm):
    start_date = pd.to_datetime(form_data.start_date)
    end_date = pd.to_datetime(form_data.end_date)
    date_difference = (end_date - start_date).days

    if date_difference > 30:
        raise HTTPException(
            status_code=400, detail="Date range ovred 30 days")

    date_query_result = bitcoin_price_data[(
        bitcoin_price_data["Timestamp"] >= start_date) & (bitcoin_price_data["Timestamp"] < end_date)]

    if date_query_result.empty:
        raise HTTPException(
            status_code=404, detail="No data found for the specified date range")

    daily_data = date_query_result.resample('D', on='Timestamp').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    daily_data = daily_data.dropna()

    result = {
        'Open': daily_data['Open'].tolist(),
        'High': daily_data['High'].tolist(),
        'Low': daily_data['Low'].tolist(),
        'Close': daily_data['Close'].tolist(),
        'Volume': daily_data['Volume'].tolist()
    }

    return result


@app.get("/get_current_date", summary="Get current date")
async def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")
