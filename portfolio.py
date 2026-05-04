"""
SP500 Portfolio Optimizer — Daily Production Script
====================================================
Runs automatically via GitHub Actions every trading day at 3:55 PM ET.

Full pipeline per run:
  1.  Load portfolio state (cash, positions, NAV history)
  2.  Check if today is a US trading day — exit cleanly if not
  3.  Scrape live S&P 500 universe from Wikipedia
  4.  Download 75 calendar days of daily OHLCV + SPY + VIX
  5.  Run daily filter → composite score → top 100 stocks
  6.  Estimate parameters (45-day mean + Ledoit-Wolf covariance)
  7.  Fetch CNN Fear & Greed Index → compute risk multiplier
  8.  Sweep efficient frontier (N_SWEEP MINLPs, dynamic risk range)
  9.  Auto-select max-Sharpe portfolio
  10. Execute trades at today's open + 0.01% slippage
  11. Compute end-of-day NAV and P&L
  12. Run Golden Cross strategy simulation (comparative baseline)
  13. Append row to results/performance_log.csv
  14. Update data/portfolio_state.json
  15. Regenerate dashboard/index.html with all charts

Files read/written:
  data/portfolio_state.json     — persistent portfolio state
  results/performance_log.csv   — daily performance log (append-only)
  dashboard/index.html          — live dashboard (overwritten daily)
"""

import os
import json
import sys
import warnings
import urllib.request
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless — no display needed in CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, ConstraintList,
    NonNegativeReals, Binary, maximize, value, SolverFactory
)
from pyomo.opt import TerminationCondition
from sklearn.covariance import LedoitWolf

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION  (mirrors Cell 2 of the Colab notebook)
# ══════════════════════════════════════════════════════════════

STARTING_CAPITAL   = 1_000_000.00
LOOKBACK           = 45       # rolling window for filter + estimation
TOP_N              = 100      # stocks passed to optimizer each day
N_STOCKS           = 10       # stocks held each day
N_PER_SECTOR       = 2        # exactly 2 per sector
WEIGHT_MIN         = 0.05     # 5%  floor — linking constraint minimum
WEIGHT_MAX         = 0.50     # 50% cap  — linking constraint maximum
N_SWEEP            = 6        # frontier points per day
FRONTIER_TIMELIMIT = 60       # seconds per individual frontier solve
SLIPPAGE_PCT       = 0.0001   # 0.01% of traded value, one-way

TARGET_SECTORS = [
    'Information Technology',
    'Health Care',
    'Utilities',
    'Energy',
    'Industrials',
]
SECTOR_SHORT = {
    'Information Technology': 'Technology',
    'Health Care':            'Healthcare',
    'Utilities':              'Utilities',
    'Energy':                 'Energy',
    'Industrials':            'Industrials',
}
SECTOR_COLOURS = {
    'Technology': '#2196F3',
    'Healthcare': '#F44336',
    'Utilities':  '#9C27B0',
    'Energy':     '#FF9800',
    'Industrials': '#4CAF50',
}

STATE_FILE = 'data/portfolio_state.json'
LOG_FILE   = 'results/performance_log.csv'
DASHBOARD  = 'docs/index.html'

BONMIN_PATH = os.path.expanduser('~/.idaes/bin/bonmin')
if not os.path.exists(BONMIN_PATH):
    BONMIN_PATH = 'bonmin'


# ══════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def is_trading_day(check_date=None):
    """
    Returns True if check_date is a US equity market trading day.
    Safe date construction — never raises ValueError for invalid days.
    """
    if check_date is None:
        check_date = date.today()

    # Weekend check
    if check_date.weekday() >= 5:
        return False

    year = check_date.year

    def safe_dates(month, weekday):
        """Return all valid dates in (year, month) whose weekday matches."""
        result = []
        for d in range(1, 32):
            try:
                dt = date(year, month, d)
                if dt.weekday() == weekday:
                    result.append(dt)
            except ValueError:
                continue   # skip invalid days (e.g. Feb 29 in non-leap year)
        return result

    holidays = set()

    # Fixed holidays
    holidays.add(date(year, 1, 1))    # New Year's Day
    holidays.add(date(year, 7, 4))    # Independence Day
    holidays.add(date(year, 12, 25))  # Christmas Day

    # MLK Day — 3rd Monday of January
    jan_mons = safe_dates(1, 0)
    if len(jan_mons) >= 3:
        holidays.add(jan_mons[2])

    # Presidents Day — 3rd Monday of February
    feb_mons = safe_dates(2, 0)
    if len(feb_mons) >= 3:
        holidays.add(feb_mons[2])

    # Memorial Day — last Monday of May
    may_mons = safe_dates(5, 0)
    if may_mons:
        holidays.add(may_mons[-1])

    # Labor Day — first Monday of September
    sep_mons = safe_dates(9, 0)
    if sep_mons:
        holidays.add(sep_mons[0])

    # Thanksgiving — 4th Thursday of November
    nov_thus = safe_dates(11, 3)
    if len(nov_thus) >= 4:
        holidays.add(nov_thus[3])

    return check_date not in holidays


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        log(f"State loaded — {len(state.get('nav_history', []))} days on record")
        return state
    log("No existing state — initialising fresh $1M portfolio")
    return {
        "cash": STARTING_CAPITAL, "positions": {},
        "last_run": None, "prev_weights": {},
        "nav_history": [], "trade_history": [],
    }


def save_state(state):
    os.makedirs('data', exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    log("Portfolio state saved.")


def append_log(row):
    os.makedirs('results', exist_ok=True)
    write_header = not os.path.exists(LOG_FILE)
    pd.DataFrame([row]).to_csv(LOG_FILE, mode='a',
                               header=write_header, index=False)
    log("Performance log updated.")


# ══════════════════════════════════════════════════════════════
#  UNIVERSE  (mirrors Cell 3)
# ══════════════════════════════════════════════════════════════

def build_universe():
    log("Scraping S&P 500 universe from Wikipedia ...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r:
        html = r.read()
    tables = pd.read_html(html)
    sp500  = tables[0][['Symbol', 'GICS Sector']].copy()
    sp500.columns = ['Ticker', 'Sector']
    sp500['Ticker'] = sp500['Ticker'].str.replace('.', '-', regex=False)
    universe = sp500[sp500['Sector'].isin(TARGET_SECTORS)].copy()
    universe['SectorShort'] = universe['Sector'].map(SECTOR_SHORT)
    ticker_sector = dict(zip(universe['Ticker'], universe['SectorShort']))
    all_tickers   = universe['Ticker'].tolist()
    log(f"Universe: {len(all_tickers)} stocks across {len(TARGET_SECTORS)} sectors")
    return all_tickers, ticker_sector


# ══════════════════════════════════════════════════════════════
#  DATA DOWNLOAD  (mirrors Cell 4)
# ══════════════════════════════════════════════════════════════

def download_data(all_tickers, ticker_sector):
    """
    Download price data using Tiingo API.
    Tiingo is a professional financial data provider that works reliably
    from any IP including GitHub Actions. Free tier: 500 requests/day.
    API key stored as GitHub Secret: TIINGO_API_KEY.

    Tiingo daily endpoint:
    https://api.tiingo.com/tiingo/daily/{ticker}/prices
    ?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&token=KEY
    """
    import time as _time
    import requests as _requests

    api_key = os.environ.get('TIINGO_API_KEY', '')
    if not api_key:
        raise RuntimeError(
            "TIINGO_API_KEY not set. Add it as a GitHub Secret: "
            "Settings -> Secrets and variables -> Actions -> New secret. "
            "Get a free key at tiingo.com"
        )

    start_str = (date.today() - timedelta(days=75)).strftime('%Y-%m-%d')
    end_str   = date.today().strftime('%Y-%m-%d')
    fetch     = all_tickers + ['SPY', 'VXX']   # VXX = VIX proxy on Tiingo

    HEADERS     = {'Content-Type': 'application/json',
                   'Authorization': f'Token {api_key}'}
    BASE_URL    = 'https://api.tiingo.com/tiingo/daily'
    BATCH_DELAY = 0.5   # Tiingo is fast — short delay is fine

    log(f"Downloading {len(fetch)} tickers via Tiingo API "
        f"({start_str} to {end_str}) ...")

    all_close  = {}
    all_open   = {}
    all_volume = {}

    for i, ticker in enumerate(fetch):
        if i % 50 == 0:
            log(f"  Progress: {i}/{len(fetch)} "
                f"({len(all_close)} tickers downloaded)")
        try:
            url  = (f"{BASE_URL}/{ticker}/prices"
                    f"?startDate={start_str}&endDate={end_str}"
                    f"&resampleFreq=daily&token={api_key}")
            resp = _requests.get(url, headers=HEADERS, timeout=15)

            if resp.status_code == 200:
                data = resp.json()
                if data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                    df = df.set_index('date').sort_index()

                    col_map = {'adjClose': 'Close',
                               'adjOpen':  'Open',
                               'adjVolume':'Volume',
                               'close':    'Close',
                               'open':     'Open',
                               'volume':   'Volume'}

                    for src, dst in col_map.items():
                        if src in df.columns and dst not in all_close:
                            if dst == 'Close':
                                all_close[ticker]  = df[src].rename(ticker)
                            elif dst == 'Open':
                                all_open[ticker]   = df[src].rename(ticker)
                            elif dst == 'Volume':
                                all_volume[ticker] = df[src].rename(ticker)

            elif resp.status_code == 404:
                pass   # ticker not found on Tiingo — skip silently
            elif resp.status_code == 401:
                raise RuntimeError(
                    "Tiingo API key invalid (401). "
                    "Check your TIINGO_API_KEY secret."
                )
            else:
                log(f"  {ticker}: HTTP {resp.status_code}")

        except RuntimeError:
            raise
        except Exception as e:
            pass   # skip individual ticker failures

        _time.sleep(BATCH_DELAY)

    downloaded = len(all_close)
    log(f"  Got data for {downloaded}/{len(fetch)} tickers")

    if downloaded < 10:
        raise RuntimeError(
            f"Only {downloaded} tickers returned data from Tiingo. "
            "Check TIINGO_API_KEY secret and tiingo.com account status."
        )

    # Build aligned DataFrames
    def build_df(data_dict, keys):
        series = {k: data_dict[k] for k in keys if k in data_dict}
        if not series:
            return pd.DataFrame()
        df = pd.DataFrame(series)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.sort_index().ffill().bfill()

    close_all  = build_df(all_close,  fetch)
    open_all   = build_df(all_open,   fetch)
    volume_all = build_df(all_volume, fetch)

    spy_close = close_all['SPY'].copy() if 'SPY' in close_all.columns else None
    vix_close = close_all['VXX'].copy() if 'VXX' in close_all.columns else None

    available = [t for t in all_tickers if t in close_all.columns]
    missing   = len(all_tickers) - len(available)
    if missing:
        log(f"  {missing} tickers not available on Tiingo — excluded")
    all_tickers   = available
    ticker_sector = {t: s for t, s in ticker_sector.items()
                     if t in available}

    if not all_tickers:
        raise RuntimeError("No universe tickers survived Tiingo download.")

    avail = close_all[all_tickers]
    bad   = avail.columns[(avail.isnull().mean() > 0.20)].tolist()
    if bad:
        log(f"  Dropped {len(bad)} tickers (>20% missing data)")
        all_tickers   = [t for t in all_tickers if t not in bad]
        ticker_sector = {t: s for t, s in ticker_sector.items()
                         if t not in bad}

    close_px  = close_all[all_tickers].copy()
    open_px   = (open_all[all_tickers].copy()
                 if all_tickers[0] in open_all.columns
                 else close_px.copy())
    volume_px = (volume_all[all_tickers].copy()
                 if all_tickers[0] in volume_all.columns
                 else pd.DataFrame(
                     np.ones_like(close_px),
                     index=close_px.index,
                     columns=all_tickers))

    daily_ret = close_px.pct_change().dropna(how='all')

    if len(daily_ret) == 0:
        raise RuntimeError("Tiingo returned 0 trading days of data.")

    log(f"Download complete — {len(daily_ret)} trading days, "
        f"{len(all_tickers)} stocks")

    return (close_px, open_px, volume_px, daily_ret,
            spy_close, vix_close, all_tickers, ticker_sector)


# ══════════════════════════════════════════════════════════════
#  DAILY FILTER  (mirrors Cell 5)
# ══════════════════════════════════════════════════════════════

def daily_filter(daily_ret, volume_px, ticker_sector, day_idx):
    if day_idx < LOOKBACK:
        return [], {}
    window_ret = daily_ret.iloc[day_idx - LOOKBACK: day_idx]
    window_vol = volume_px.iloc[day_idx - LOOKBACK: day_idx]
    cum_ret    = (1 + window_ret).prod() - 1
    recent_vol = window_vol.iloc[-5:].mean()
    avg_vol    = window_vol.mean()
    rel_vol    = recent_vol / avg_vol.replace(0, np.nan)
    vol_std    = window_ret.std() * np.sqrt(252)
    vol_std    = vol_std.replace(0, np.nan)
    score      = (cum_ret * rel_vol) / vol_std
    score      = score.dropna()
    score      = score[score.index.map(lambda t: ticker_sector.get(t) is not None)]
    sector_counts = pd.Series(
        {t: ticker_sector[t] for t in score.index}
    ).value_counts()
    valid_sectors = sector_counts[sector_counts >= N_PER_SECTOR].index.tolist()
    score = score[score.index.map(lambda t: ticker_sector[t] in valid_sectors)]
    top_tickers = score.nlargest(TOP_N).index.tolist()
    top_sectors = {t: ticker_sector[t] for t in top_tickers}
    return top_tickers, top_sectors


# ══════════════════════════════════════════════════════════════
#  OPTIMIZER  (mirrors Cell 6)
# ══════════════════════════════════════════════════════════════

def estimate_params(daily_ret, tickers, day_idx):
    window   = daily_ret[tickers].iloc[day_idx - LOOKBACK: day_idx].copy()
    window   = window.fillna(window.mean())
    mean_ret = window.mean()
    lw       = LedoitWolf().fit(window.values)
    cov_df   = pd.DataFrame(lw.covariance_, index=tickers, columns=tickers)
    return mean_ret, cov_df


def solve_portfolio(mean_ret, cov_matrix, sector_map,
                    risk_limit, warm_start=None):
    tickers     = mean_ret.index.tolist()
    sectors     = list(set(sector_map.values()))
    sec_tickers = {s: [t for t in tickers if sector_map.get(t) == s]
                   for s in sectors}
    for s in sectors:
        if len(sec_tickers[s]) < N_PER_SECTOR:
            return None, None
    m   = ConcreteModel()
    m.x = Var(tickers, domain=NonNegativeReals, bounds=(0.0, 1.0))
    m.y = Var(tickers, domain=Binary)
    if warm_start:
        for t in tickers:
            if t in warm_start:
                m.y[t].set_value(1 if warm_start[t] > 1e-4 else 0)
                m.x[t].set_value(float(warm_start[t]))
    m.obj = Objective(
        expr=sum(m.x[t] * float(mean_ret[t]) for t in tickers),
        sense=maximize
    )
    m.sum_w    = Constraint(expr=sum(m.x[t] for t in tickers) == 1.0)
    m.n_stocks = Constraint(expr=sum(m.y[t] for t in tickers) == N_STOCKS)
    m.cap   = Constraint(tickers, rule=lambda m, t: m.x[t] <= WEIGHT_MAX * m.y[t])
    m.floor = Constraint(tickers, rule=lambda m, t: m.x[t] >= WEIGHT_MIN * m.y[t])
    m.sector_exact = ConstraintList()
    for s in sectors:
        m.sector_exact.add(sum(m.y[t] for t in sec_tickers[s]) == N_PER_SECTOR)
    risk_expr = sum(
        m.x[t1] * float(cov_matrix.at[t1, t2]) * m.x[t2]
        for t1 in tickers for t2 in tickers
    )
    m.risk = Constraint(expr=risk_expr <= risk_limit)
    solver = SolverFactory('bonmin', executable=BONMIN_PATH)
    solver.options['bonmin.time_limit'] = FRONTIER_TIMELIMIT
    solver.options['bonmin.algorithm']  = 'B-BB'
    result = solver.solve(m, tee=False)
    tc = result.solver.termination_condition
    if tc in (TerminationCondition.optimal, TerminationCondition.maxTimeLimit):
        w     = {t: max(value(m.x[t]), 0.0) for t in tickers}
        total = sum(w.values())
        if total > 0:
            w = {t: v / total for t, v in w.items()}
        ret = sum(w[t] * float(mean_ret[t]) for t in tickers)
        return w, ret
    return None, None


def find_optimal_portfolio(mean_ret, cov_matrix, sector_map,
                            risk_multiplier=1.0, warm_start=None):
    tickers   = mean_ret.index.tolist()
    cov_array = cov_matrix.values
    n         = len(tickers)
    w_eq      = np.ones(n) / n
    eq_var    = float(w_eq @ cov_array @ w_eq)
    d_max     = float(np.diag(cov_array).max())
    r_min       = eq_var * 0.5 * risk_multiplier
    r_max       = d_max  * 1.2 * risk_multiplier
    risk_levels = np.linspace(r_min, r_max, N_SWEEP)
    frontier     = []
    best_weights = None
    best_ret     = None
    best_sharpe  = -np.inf
    best_risk    = None
    for r_limit in risk_levels:
        w, port_ret = solve_portfolio(
            mean_ret, cov_matrix, sector_map,
            risk_limit=r_limit, warm_start=warm_start
        )
        if w is None:
            continue
        w_vec    = np.array([w.get(t, 0.0) for t in tickers])
        port_var = float(w_vec @ cov_array @ w_vec)
        port_std = np.sqrt(max(port_var, 1e-12) * 252)
        ann_ret  = port_ret * 252
        sharpe   = ann_ret / port_std if port_std > 0 else -np.inf
        frontier.append({
            'Risk_Limit': r_limit, 'Port_Var': port_var,
            'Port_Std': port_std, 'Ann_Return': ann_ret, 'Sharpe': sharpe,
        })
        if sharpe > best_sharpe:
            best_sharpe  = sharpe
            best_weights = w
            best_ret     = port_ret
            best_risk    = port_var
    return best_weights, best_ret, best_sharpe, best_risk, frontier


# ══════════════════════════════════════════════════════════════
#  FEAR & GREED  (mirrors Cell 13)
# ══════════════════════════════════════════════════════════════

def get_fear_greed(vix_series=None):
    def score_to_label_mult(s):
        if   s <= 24: return 'Extreme Fear',  0.50
        elif s <= 44: return 'Fear',           0.75
        elif s <= 55: return 'Neutral',        1.00
        elif s <= 74: return 'Greed',          1.25
        else:         return 'Extreme Greed',  1.50
    try:
        import fear_and_greed
        data  = fear_and_greed.get()
        score = int(data.value)
        label, mult = score_to_label_mult(score)
        log(f"CNN Fear & Greed: {score} ({label})  [live]")
        return score, label, mult
    except Exception as e:
        log(f"CNN F&G unavailable ({e}) — using VIX fallback")
    if vix_series is not None and len(vix_series) > 0:
        vix_val = float(vix_series.dropna().iloc[-1])
        if   vix_val < 12: score = 85
        elif vix_val < 17: score = 65
        elif vix_val < 22: score = 50
        elif vix_val < 30: score = 30
        else:               score = 15
        label, mult = score_to_label_mult(score)
        log(f"VIX: {vix_val:.1f} → F&G proxy: {score} ({label})")
        return score, label, mult
    log("No sentiment data — using neutral (50)")
    return 50, 'Neutral', 1.00


# ══════════════════════════════════════════════════════════════
#  GOLDEN CROSS  (mirrors Cell 12)
# ══════════════════════════════════════════════════════════════

def run_golden_cross(spy_close, perf_df, starting_capital):
    start_long = (date.today() - timedelta(days=320)).strftime('%Y-%m-%d')
    end_long   = date.today().strftime('%Y-%m-%d')

    # Fetch 320 days of SPY from FMP for 200-day MA calculation
    # Fetch SPY via Tiingo for Golden Cross (same provider as main download)
    import requests as _req
    api_key  = os.environ.get('TIINGO_API_KEY', '')
    spy_long = None

    if api_key:
        try:
            url  = (f"https://api.tiingo.com/tiingo/daily/SPY/prices"
                    f"?startDate={start_long}&endDate={end_long}"
                    f"&resampleFreq=daily&token={api_key}")
            resp = _req.get(url,
                            headers={'Authorization': f'Token {api_key}'},
                            timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    df_spy = pd.DataFrame(data)
                    df_spy['date'] = pd.to_datetime(
                        df_spy['date']).dt.tz_localize(None)
                    df_spy = df_spy.set_index('date').sort_index()
                    col = 'adjClose' if 'adjClose' in df_spy.columns else 'close'
                    spy_long = df_spy[col]
                    log(f"Golden Cross: {len(spy_long)} days of SPY from Tiingo")
        except Exception as e:
            log(f"Golden Cross SPY fetch failed: {e}")

    if spy_long is None or len(spy_long) < 50:
        log("Golden Cross: using available SPY data (MAs may be approximate)")
        perf_idx = pd.to_datetime(perf_df['Date'])
        spy_long = spy_close.reindex(perf_idx).ffill()
    ma50       = spy_long.rolling(50).mean()
    ma200      = spy_long.rolling(200).mean()
    perf_idx   = pd.to_datetime(perf_df['Date'])
    spy_q2     = spy_long.reindex(perf_idx).ffill()
    signal_q2  = (ma50.reindex(perf_idx).ffill() >
                  ma200.reindex(perf_idx).ffill()).astype(int).fillna(0)
    spy_ret    = spy_q2.pct_change().fillna(0)
    gc_nav     = [starting_capital]
    gc_regime  = []
    for i in range(len(spy_ret)):
        prev = gc_nav[-1]
        if signal_q2.iloc[i]:
            gc_nav.append(prev * (1 + spy_ret.iloc[i]))
            gc_regime.append('Invested')
        else:
            gc_nav.append(prev)
            gc_regime.append('Cash')
    gc_series = pd.Series(gc_nav[1:], index=perf_idx)
    log(f"Golden Cross regime today: {gc_regime[-1] if gc_regime else 'N/A'}")
    return gc_series, gc_regime, spy_q2, ma50, ma200


# ══════════════════════════════════════════════════════════════
#  METRICS HELPER
# ══════════════════════════════════════════════════════════════

def compute_metrics(nav_series, label, starting_capital):
    r      = nav_series.pct_change().fillna(0)
    total  = nav_series.iloc[-1] / starting_capital - 1
    n      = len(r)
    ann_f  = 252 / n if n > 0 else 1
    ann_r  = (1 + total) ** ann_f - 1
    vol    = r.std() * np.sqrt(252)
    sharpe = ann_r / vol if vol > 0 else np.nan
    dd     = ((nav_series - nav_series.cummax()) / nav_series.cummax()).min()
    win_r  = (r > 0).mean()
    return {
        'Strategy':     label,
        'Final NAV':    f'${nav_series.iloc[-1]:,.0f}',
        'Total Return': f'{total:+.2%}',
        'Ann. Return':  f'{ann_r:+.2%}',
        'Volatility':   f'{vol:.2%}',
        'Sharpe':       f'{sharpe:.3f}',
        'Max Drawdown': f'{dd:.2%}',
        'Win Rate':     f'{win_r:.1%}',
    }


# ══════════════════════════════════════════════════════════════
#  DASHBOARD GENERATOR  (mirrors Cells 9 + 12 + 13)
# ══════════════════════════════════════════════════════════════

def build_dashboard(perf_df, today_str, frontier_data,
                    best_sharpe, fg_score, fg_label, fg_mult,
                    spy_close, gc_series, gc_regime, vix_series):
    os.makedirs('docs', exist_ok=True)

    dates   = perf_df['Date'].astype(str).tolist()
    navs    = perf_df['NAV'].tolist()
    pnls    = perf_df['Daily_PnL'].tolist()
    returns = (perf_df['Daily_Return'] * 100).round(4).tolist()
    sharpes = perf_df['Selected_Sharpe'].tolist()

    spy_aligned = spy_close.reindex(pd.to_datetime(perf_df['Date'])).ffill()
    spy_rb = []
    if len(spy_aligned) > 0 and spy_aligned.iloc[0] > 0:
        spy_rb = (spy_aligned / spy_aligned.iloc[0] * STARTING_CAPITAL).round(2).tolist()

    gc_navs = gc_series.round(2).tolist() if gc_series is not None and len(gc_series) > 0 else []

    vix_fg, mult_hist = [], []
    if vix_series is not None:
        vix_q2 = vix_series.reindex(pd.to_datetime(perf_df['Date'])).ffill()
        def vix_to_fg(v):
            if   v < 12: return 85
            elif v < 17: return 65
            elif v < 22: return 50
            elif v < 30: return 30
            else:        return 15
        vix_fg    = vix_q2.apply(vix_to_fg).tolist()
        mult_hist = [0.50 if s<=24 else 0.75 if s<=44 else 1.00 if s<=55 else 1.25 if s<=74 else 1.50
                     for s in vix_fg]

    f_risks   = [round(p['Port_Std'] * 100, 4) for p in frontier_data]
    f_returns = [round(p['Ann_Return'] * 100, 4) for p in frontier_data]
    best_idx  = (max(range(len(frontier_data)), key=lambda i: frontier_data[i]['Sharpe'])
                 if frontier_data else 0)
    b_risk    = f_risks[best_idx]   if frontier_data else 0
    b_ret     = f_returns[best_idx] if frontier_data else 0

    total_ret = (navs[-1] / STARTING_CAPITAL - 1) * 100 if navs else 0
    total_pnl = navs[-1] - STARTING_CAPITAL if navs else 0
    n_days    = len(navs)
    win_rate  = sum(1 for p in pnls if p > 0) / n_days * 100 if n_days > 0 else 0

    fg_colour = {
        'Extreme Fear': '#EF5350', 'Fear': '#FF7043',
        'Neutral': '#90A4AE', 'Greed': '#66BB6A',
        'Extreme Greed': '#43A047'
    }.get(fg_label, '#90A4AE')

    import json as _json

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SP500 Portfolio Optimizer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{box-sizing:border-box;margin:0;padding:0;}}
body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e0e0e0;padding:20px;}}
h1 {{font-size:1.55rem;color:#64B5F6;margin-bottom:4px;}}
.sub {{color:#90A4AE;font-size:0.80rem;margin-bottom:22px;}}
.kpis {{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:11px;margin-bottom:26px;}}
.kpi {{background:#1e2130;border-radius:10px;padding:15px;border-left:4px solid #2196F3;}}
.kpi .lbl {{font-size:0.68rem;color:#90A4AE;text-transform:uppercase;letter-spacing:.5px;}}
.kpi .val {{font-size:1.35rem;font-weight:700;margin-top:3px;}}
.pos {{color:#66BB6A;}}.neg {{color:#EF5350;}}.neu {{color:#64B5F6;}}
.grid {{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}}
.full {{grid-column:1/-1;}}
.card {{background:#1e2130;border-radius:12px;padding:16px;}}
.card h3 {{font-size:0.78rem;color:#90A4AE;margin-bottom:11px;text-transform:uppercase;letter-spacing:.5px;}}
.cw {{position:relative;height:225px;}}.cw.t {{height:285px;}}
footer {{text-align:center;color:#546E7A;font-size:0.70rem;margin-top:22px;}}
a {{color:#64B5F6;}}
.fg-badge {{display:inline-block;padding:3px 10px;border-radius:20px;font-weight:600;
            font-size:0.82rem;background:{fg_colour}22;color:{fg_colour};border:1px solid {fg_colour};margin-left:8px;}}
</style>
</head>
<body>
<h1>&#x1F4C8; SP500 Portfolio Optimizer
  <span class="fg-badge">&#x1F9E0; {fg_label} ({fg_score})</span>
</h1>
<p class="sub">
  Updated: <strong>{today_str}</strong> &nbsp;|&nbsp;
  $1M capital &nbsp;|&nbsp; {N_STOCKS} stocks/day &nbsp;|&nbsp;
  {LOOKBACK}-day window &nbsp;|&nbsp; {N_SWEEP} frontier solves/day &nbsp;|&nbsp;
  Risk multiplier: <strong>{fg_mult:.2f}&times;</strong>
</p>
<div class="kpis">
  <div class="kpi"><div class="lbl">Current NAV</div>
    <div class="val neu">${navs[-1]:,.0f}</div></div>
  <div class="kpi"><div class="lbl">Total P&amp;L</div>
    <div class="val {'pos' if total_pnl>=0 else 'neg'}">${total_pnl:+,.0f}</div></div>
  <div class="kpi"><div class="lbl">Total Return</div>
    <div class="val {'pos' if total_ret>=0 else 'neg'}">{total_ret:+.2f}%</div></div>
  <div class="kpi"><div class="lbl">Win Rate</div>
    <div class="val neu">{win_rate:.1f}%</div></div>
  <div class="kpi"><div class="lbl">Trading Days</div>
    <div class="val neu">{n_days}</div></div>
  <div class="kpi"><div class="lbl">Today Sharpe</div>
    <div class="val {'pos' if best_sharpe>=1 else 'neu'}">{best_sharpe:.3f}</div></div>
</div>
<div class="grid">
  <div class="card full">
    <h3>NAV — MINLP vs Golden Cross vs SPY Buy &amp; Hold</h3>
    <div class="cw t"><canvas id="navChart"></canvas></div>
  </div>
  <div class="card"><h3>Daily P&amp;L ($)</h3>
    <div class="cw"><canvas id="pnlChart"></canvas></div></div>
  <div class="card"><h3>Daily Return (%)</h3>
    <div class="cw"><canvas id="retChart"></canvas></div></div>
  <div class="card"><h3>Fear &amp; Greed Index (VIX-derived)</h3>
    <div class="cw"><canvas id="fgChart"></canvas></div></div>
  <div class="card"><h3>Sentiment Risk Multiplier</h3>
    <div class="cw"><canvas id="multChart"></canvas></div></div>
  <div class="card"><h3>Daily Frontier Sharpe Target</h3>
    <div class="cw"><canvas id="sharpeChart"></canvas></div></div>
  <div class="card"><h3>Today's Efficient Frontier &#x2605; Max-Sharpe</h3>
    <div class="cw"><canvas id="frontierChart"></canvas></div></div>
</div>
<footer>
  SP500 Portfolio Optimizer &mdash; GitHub Actions &mdash; Mon&ndash;Fri 3:55 PM ET &mdash;
  <a href="https://github.com/bromomo-git/SP500-Portfolio-Optimizer">View on GitHub</a>
</footer>
<script>
const labels={_json.dumps(dates)};
const navs={_json.dumps([round(v,2) for v in navs])};
const spy={_json.dumps([round(v,2) for v in spy_rb])};
const gc={_json.dumps([round(v,2) for v in gc_navs])};
const pnls={_json.dumps([round(v,2) for v in pnls])};
const rets={_json.dumps([round(v,4) for v in returns])};
const sharps={_json.dumps([round(v,4) for v in sharpes])};
const fg={_json.dumps(vix_fg)};
const mh={_json.dumps(mult_hist)};
const fR={_json.dumps(f_risks)};
const fRet={_json.dumps(f_returns)};
const bR={b_risk};const bRet2={b_ret};
const G={{color:'rgba(255,255,255,0.06)'}};
const F={{color:'#90A4AE',size:10}};
new Chart(document.getElementById('navChart'),{{type:'line',
  data:{{labels,datasets:[
    {{label:'MINLP Portfolio',data:navs,borderColor:'#2196F3',backgroundColor:'rgba(33,150,243,0.08)',borderWidth:2.2,pointRadius:2,fill:true,tension:0.3}},
    {{label:'Golden Cross',data:gc,borderColor:'#FF9800',borderDash:[6,3],borderWidth:1.8,pointRadius:0,fill:false,tension:0.3}},
    {{label:'SPY Buy & Hold',data:spy,borderColor:'#90A4AE',borderDash:[3,3],borderWidth:1.5,pointRadius:0,fill:false,tension:0.3}}
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{color:'#e0e0e0'}}}}}},
    scales:{{x:{{ticks:F,grid:G}},y:{{ticks:{{...F,callback:v=>'$'+v.toLocaleString()}},grid:G}}}}
  }}
}});
new Chart(document.getElementById('pnlChart'),{{type:'bar',
  data:{{labels,datasets:[{{label:'P&L $',data:pnls,
    backgroundColor:pnls.map(v=>v>=0?'rgba(102,187,106,0.8)':'rgba(239,83,80,0.8)'),borderRadius:3}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
    scales:{{x:{{ticks:F,grid:G}},y:{{ticks:{{...F,callback:v=>'$'+v.toLocaleString()}},grid:G}}}}
  }}
}});
new Chart(document.getElementById('retChart'),{{type:'bar',
  data:{{labels,datasets:[{{label:'Return %',data:rets,
    backgroundColor:rets.map(v=>v>=0?'rgba(102,187,106,0.8)':'rgba(239,83,80,0.8)'),borderRadius:3}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
    scales:{{x:{{ticks:F,grid:G}},y:{{ticks:{{...F,callback:v=>v.toFixed(2)+'%'}},grid:G}}}}
  }}
}});
new Chart(document.getElementById('fgChart'),{{type:'line',
  data:{{labels,datasets:[{{label:'F&G Score',data:fg,borderColor:'#CE93D8',
    backgroundColor:'rgba(206,147,216,0.15)',borderWidth:2,pointRadius:2,fill:true,tension:0.4}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
    scales:{{x:{{ticks:F,grid:G}},y:{{min:0,max:100,ticks:F,grid:G}}}}
  }}
}});
new Chart(document.getElementById('multChart'),{{type:'bar',
  data:{{labels,datasets:[{{label:'Multiplier',data:mh,
    backgroundColor:mh.map(v=>v>=1?'rgba(102,187,106,0.8)':'rgba(239,83,80,0.8)'),borderRadius:3}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
    scales:{{x:{{ticks:F,grid:G}},y:{{min:0,max:1.8,ticks:F,grid:G}}}}
  }}
}});
new Chart(document.getElementById('sharpeChart'),{{type:'line',
  data:{{labels,datasets:[{{label:'Sharpe',data:sharps,borderColor:'#64B5F6',
    backgroundColor:'rgba(100,181,246,0.1)',borderWidth:2,pointRadius:2,fill:true,tension:0.3}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
    scales:{{x:{{ticks:F,grid:G}},y:{{ticks:F,grid:G}}}}
  }}
}});
new Chart(document.getElementById('frontierChart'),{{type:'scatter',
  data:{{datasets:[
    {{label:'Frontier',data:fR.map((r,i)=>({{x:r,y:fRet[i]}})),borderColor:'#64B5F6',
      backgroundColor:'rgba(100,181,246,0.6)',pointRadius:6,showLine:true,tension:0.3}},
    {{label:'\u2605 Max Sharpe',data:[{{x:bR,y:bRet2}}],borderColor:'#F44336',
      backgroundColor:'#F44336',pointRadius:14,pointStyle:'star'}}
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{color:'#e0e0e0'}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Risk (Ann. Std %)',color:'#90A4AE'}},ticks:F,grid:G}},
      y:{{title:{{display:true,text:'Return (Ann. %)',color:'#90A4AE'}},ticks:F,grid:G}}
    }}
  }}
}});
</script>
</body>
</html>"""

    with open(DASHBOARD, 'w') as f:
        f.write(html)
    log(f"Dashboard written → {DASHBOARD}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    today = date.today()
    log("=" * 56)
    log(f"SP500 Portfolio Optimizer — {today}")
    log("=" * 56)

    if not is_trading_day(today):
        log(f"{today} is not a US trading day — exiting cleanly.")
        sys.exit(0)

    state = load_state()
    if state.get('last_run') == str(today):
        log(f"Already ran today ({today}) — exiting.")
        sys.exit(0)

    all_tickers, ticker_sector = build_universe()

    (close_px, open_px, volume_px, daily_ret,
     spy_close, vix_close, all_tickers,
     ticker_sector) = download_data(all_tickers, ticker_sector)

    today_ts = pd.Timestamp(today)
    all_days = daily_ret.index
    if today_ts in all_days:
        day_idx = all_days.get_loc(today_ts)
    else:
        day_idx = len(all_days) - 1
        log(f"Using most recent available day: {all_days[day_idx].date()}")

    if day_idx < LOOKBACK:
        log(f"Only {day_idx} days of history — need {LOOKBACK}. Exiting.")
        sys.exit(0)

    log(f"Running daily filter (top {TOP_N}) ...")
    top_tickers, top_sec_map = daily_filter(
        daily_ret, volume_px, ticker_sector, day_idx
    )
    if len(top_tickers) < N_STOCKS:
        log(f"Filter returned only {len(top_tickers)} stocks. Exiting.")
        sys.exit(1)
    log(f"Filter: {len(top_tickers)} stocks passed to optimizer")

    log("Estimating parameters (Ledoit-Wolf) ...")
    mean_r, cov_m = estimate_params(daily_ret, top_tickers, day_idx)

    fg_score, fg_label, fg_mult = get_fear_greed(vix_close)
    log(f"Sentiment: {fg_label} ({fg_score}) → multiplier {fg_mult:.2f}x")

    log(f"Frontier sweep ({N_SWEEP} solves, multiplier={fg_mult:.2f}x) ...")
    prev_w   = state.get('prev_weights', {})
    warm     = {t: prev_w.get(t, 0.0) for t in top_tickers}
    weights, est_ret, best_sharpe, best_risk, frontier = find_optimal_portfolio(
        mean_r, cov_m, top_sec_map,
        risk_multiplier=fg_mult, warm_start=warm
    )

    if weights is None:
        log("All frontier solves infeasible — carrying previous weights.")
        weights     = {t: prev_w.get(t, 0.0) for t in top_tickers}
        wsum        = sum(weights.values())
        if wsum > 0:
            weights = {t: v / wsum for t, v in weights.items()}
        est_ret     = 0.0
        best_sharpe = 0.0
        frontier    = []

    held = [t for t, w in weights.items() if w > 1e-4]
    log(f"Selected {len(held)} stocks | Sharpe {best_sharpe:.3f}")

    exec_day = today_ts if today_ts in open_px.index else open_px.index[-1]
    op = open_px.loc[exec_day]
    cp = close_px.loc[exec_day]

    positions = {t: state['positions'].get(t, 0.0) for t in all_tickers}
    cash      = float(state['cash'])

    equity_open = sum(positions.get(t, 0.0) * float(op.get(t, 0))
                      for t in all_tickers)
    nav_open = cash + equity_open
    log(f"NAV at open: ${nav_open:,.2f}")

    target_dollars  = {t: weights.get(t, 0.0) * nav_open for t in top_tickers}
    current_dollars = {t: positions.get(t, 0.0) * float(op.get(t, 0))
                       for t in all_tickers}
    for t in all_tickers:
        if t not in target_dollars and positions.get(t, 0.0) > 1e-6:
            target_dollars[t] = 0.0

    day_slippage = 0.0
    day_turnover = 0.0
    trades_today = []
    all_trade_t  = set(target_dollars.keys()) | {
        t for t in all_tickers if positions.get(t, 0.0) > 1e-6
    }

    for t in all_trade_t:
        delta = target_dollars.get(t, 0.0) - current_dollars.get(t, 0.0)
        if abs(delta) < 1.0:
            continue
        t_px = float(op.get(t, 0))
        if t_px <= 0:
            continue
        direction    = 1 if delta > 0 else -1
        slip_cost    = abs(delta) * SLIPPAGE_PCT
        effective_px = t_px * (1 + direction * SLIPPAGE_PCT)
        shares       = delta / effective_px
        positions[t]  = positions.get(t, 0.0) + shares
        cash         -= delta + direction * slip_cost
        day_slippage += slip_cost
        day_turnover += abs(delta)
        trades_today.append({
            'date': str(today), 'ticker': t,
            'sector': ticker_sector.get(t, top_sec_map.get(t, '')),
            'direction': 'BUY' if delta > 0 else 'SELL',
            'dollars': round(delta, 2), 'shares': round(shares, 6),
            'slippage': round(slip_cost, 4),
        })

    eod_equity    = sum(positions.get(t, 0.0) * float(cp.get(t, 0))
                        for t in all_tickers)
    eod_nav       = cash + eod_equity
    daily_pnl     = eod_nav - nav_open
    daily_ret_val = daily_pnl / nav_open if nav_open > 0 else 0

    log(f"EOD NAV: ${eod_nav:,.2f} | P&L: ${daily_pnl:+,.2f} ({daily_ret_val:+.4%})")

    append_log({
        'Date':            str(today),
        'NAV':             round(eod_nav, 2),
        'Daily_PnL':       round(daily_pnl, 2),
        'Daily_Return':    round(daily_ret_val, 6),
        'Slippage':        round(day_slippage, 4),
        'Turnover':        round(day_turnover, 2),
        'Selected_Sharpe': round(best_sharpe, 4),
        'FG_Score':        fg_score,
        'FG_Label':        fg_label,
        'FG_Multiplier':   fg_mult,
        'Held_Tickers':    ','.join(held),
    })

    state['nav_history'].append({
        'date': str(today), 'nav': round(eod_nav, 2),
        'pnl':  round(daily_pnl, 2)
    })
    state['cash']         = round(cash, 4)
    state['positions']    = {t: round(v, 6) for t, v in positions.items()
                              if abs(v) > 1e-8}
    state['last_run']     = str(today)
    state['prev_weights'] = weights
    state['trade_history'].extend(trades_today)
    save_state(state)

    log("Running Golden Cross comparative simulation ...")
    perf_df = pd.read_csv(LOG_FILE)
    gc_series, gc_regime = None, []
    try:
        gc_series, gc_regime, _, _, _ = run_golden_cross(
            spy_close, perf_df, STARTING_CAPITAL
        )
    except Exception as e:
        log(f"Golden Cross failed: {e} — skipping")

    if gc_series is not None and len(perf_df) > 1:
        spy_aligned = spy_close.reindex(
            pd.to_datetime(perf_df['Date'])
        ).ffill()
        spy_bh    = spy_aligned / spy_aligned.iloc[0] * STARTING_CAPITAL
        minlp_nav = pd.Series(perf_df['NAV'].values,
                               index=pd.to_datetime(perf_df['Date']))
        metrics = pd.DataFrame([
            compute_metrics(minlp_nav, 'MINLP Portfolio', STARTING_CAPITAL),
            compute_metrics(gc_series, 'Golden Cross',    STARTING_CAPITAL),
            compute_metrics(spy_bh,    'SPY Buy & Hold',  STARTING_CAPITAL),
        ])
        log("COMPARATIVE ANALYSIS")
        print(metrics.to_string(index=False))

    log("Rebuilding live dashboard ...")
    build_dashboard(
        perf_df=perf_df, today_str=str(today),
        frontier_data=frontier, best_sharpe=best_sharpe,
        fg_score=fg_score, fg_label=fg_label, fg_mult=fg_mult,
        spy_close=spy_close, gc_series=gc_series,
        gc_regime=gc_regime, vix_series=vix_close,
    )

    log("=" * 56)
    log(f"Run complete — {today}  NAV ${eod_nav:,.2f}")
    log("=" * 56)


if __name__ == '__main__':
    main()
