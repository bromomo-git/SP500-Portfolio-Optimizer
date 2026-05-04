"""
FMP API Debug Script
====================
Run by GitHub Actions to verify FMP API connectivity before the main script.
Checks: key is set, HTTP connection works, data is returned.
"""
import requests
import os
import sys

key = os.environ.get('FMP_API_KEY', '')

print("=" * 50)
print("FMP API Debug")
print("=" * 50)

if not key:
    print("ERROR: FMP_API_KEY is not set or empty")
    print("Go to: Settings -> Secrets and variables -> Actions")
    print("Add a secret named exactly: FMP_API_KEY")
    sys.exit(1)

print("Key length  :", len(key), "chars")
print("Key prefix  :", key[:8] + "...")

url = (
    "https://financialmodelingprep.com/api/v3"
    "/historical-price-full/AAPL"
    "?from=2026-04-01&to=2026-05-01"
    "&apikey=" + key
)

print()
print("Testing URL : financialmodelingprep.com/api/v3/historical-price-full/AAPL")

try:
    resp = requests.get(url, timeout=15)
    print("HTTP status :", resp.status_code)
    print()
    print("Response (first 600 chars):")
    print(resp.text[:600])
    print()

    if resp.status_code == 200:
        data = resp.json()
        historical = data.get("historical", [])
        if historical:
            print("SUCCESS: Got", len(historical), "days of AAPL data")
            print("First record:", historical[0])
        else:
            print("WARNING: Status 200 but historical list is empty")
            print("Full response:", resp.text[:300])
    elif resp.status_code == 401:
        print("ERROR 401: Invalid API key")
    elif resp.status_code == 403:
        print("ERROR 403: Endpoint not available on free tier")
        print("Check your FMP plan at financialmodelingprep.com/developer/docs")
    else:
        print("ERROR: Unexpected status code", resp.status_code)

except Exception as e:
    print("EXCEPTION:", type(e).__name__, str(e))
    sys.exit(1)

print("=" * 50)
