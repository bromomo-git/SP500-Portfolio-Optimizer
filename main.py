"""
SP500 Portfolio Optimizer — Google Cloud Function
==================================================
Triggered by GitHub Actions at 3:55 PM ET every trading day.
Runs on Google infrastructure so Yahoo Finance works perfectly.
Pushes results to GitHub repo via Personal Access Token.
"""

import functions_framework
import subprocess
import sys
import os
import json
import base64
import traceback
from datetime import date, datetime


@functions_framework.http
def run_portfolio(request):
    """
    HTTP trigger entry point.
    Called by GitHub Actions cron at 3:55 PM ET.
    Returns JSON with run status.
    """
    # ── Security check — simple token verification ─────────────
    # GitHub Actions passes this in the Authorization header
    expected_token = os.environ.get('TRIGGER_SECRET', '')
    auth_header    = request.headers.get('Authorization', '')

    if expected_token and auth_header != f'Bearer {expected_token}':
        return json.dumps({'status': 'error',
                           'message': 'Unauthorized'}), 401

    start_time = datetime.now()
    log(f"SP500 Portfolio Runner triggered at {start_time}")

    try:
        # ── Install dependencies ──────────────────────────────
        log("Installing dependencies...")
        _install([
            'yfinance==0.2.54',
            'pyomo==6.7.1',
            'scikit-learn==1.4.2',
            'pandas==2.2.2',
            'numpy==1.26.4',
            'matplotlib==3.8.4',
            'fear-and-greed==0.4',
            'lxml==5.2.2',
        ])
        log("Dependencies installed.")

        # ── Run the portfolio script ──────────────────────────
        log("Running portfolio optimization...")
        from portfolio_logic import run_daily
        result = run_daily()

        elapsed = (datetime.now() - start_time).seconds
        response = {
            'status':     'success',
            'date':       str(date.today()),
            'elapsed_s':  elapsed,
            'nav':        result.get('nav'),
            'pnl':        result.get('pnl'),
            'sharpe':     result.get('sharpe'),
            'stocks_held': result.get('stocks_held'),
        }
        log(f"Completed in {elapsed}s — NAV: {result.get('nav')}")
        return json.dumps(response), 200

    except Exception as e:
        error_msg = traceback.format_exc()
        log(f"ERROR: {error_msg}")
        return json.dumps({
            'status':  'error',
            'message': str(e),
            'trace':   error_msg[-500:]
        }), 500


def _install(packages):
    """Install pip packages at runtime."""
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '--quiet'] + packages
    )


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
