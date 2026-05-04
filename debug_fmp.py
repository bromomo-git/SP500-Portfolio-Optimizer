import yfinance as yf
import pandas as pd

try:
    from curl_cffi import requests as curl_requests
    session = curl_requests.Session(impersonate="chrome")
    print("curl_cffi: OK")
    tk   = yf.Ticker("AAPL", session=session)
    hist = tk.history(period="5d", interval="1d", auto_adjust=True)
    print("AAPL data rows:", len(hist))
    print(hist.tail(3))
except Exception as e:
    print("ERROR:", e)
