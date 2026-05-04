"""
Tiingo API Debug Script
=======================
Tests that TIINGO_API_KEY is set and returns data from the API.
"""
import requests
import os
import sys

key = os.environ.get('TIINGO_API_KEY', '')

print("=" * 50)
print("Tiingo API Debug")
print("=" * 50)

if not key:
    print("ERROR: TIINGO_API_KEY is not set")
    print("Add it at: Settings -> Secrets -> Actions -> New secret")
    print("Get a free key at: tiingo.com")
    sys.exit(1)

print("Key length  :", len(key), "chars")
print("Key prefix  :", key[:8] + "...")

url = ("https://api.tiingo.com/tiingo/daily/AAPL/prices"
       "?startDate=2026-04-01&endDate=2026-05-01"
       f"&resampleFreq=daily&token={key}")

headers = {'Content-Type': 'application/json',
           'Authorization': f'Token {key}'}

print()
print("Testing Tiingo AAPL endpoint ...")

try:
    resp = requests.get(url, headers=headers, timeout=15)
    print("HTTP status :", resp.status_code)
    print()

    if resp.status_code == 200:
        data = resp.json()
        print(f"SUCCESS: Got {len(data)} days of AAPL data")
        if data:
            print("First record:", data[0])
    elif resp.status_code == 401:
        print("ERROR 401: Invalid API key")
    elif resp.status_code == 404:
        print("ERROR 404: Ticker not found")
    else:
        print("Response:", resp.text[:300])

except Exception as e:
    print("EXCEPTION:", type(e).__name__, str(e))
    sys.exit(1)

print("=" * 50)
