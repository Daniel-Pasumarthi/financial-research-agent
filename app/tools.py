import yfinance as yf


def get_financials(ticker: str) -> dict:
    """Fetch key live financial metrics via yFinance."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "company": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "revenue": info.get("totalRevenue", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"),
        }
    except Exception as e:
        return {"error": f"Could not fetch {ticker}: {e}"}


if __name__ == "__main__":
    result = get_financials("AAPL")
    for key, value in result.items():
        print(f"{key}: {value}")