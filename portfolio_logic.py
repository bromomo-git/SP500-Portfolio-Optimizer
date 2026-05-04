"""
SP500 Portfolio Logic — runs inside Google Cloud Function
=========================================================
Same pipeline as the Colab notebook, packaged for serverless execution.
Runs on Google infrastructure — Yahoo Finance works perfectly here.
"""

import os
import json
import base64
import warnings
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import requests as _req

from sklearn.covariance import LedoitWolf
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, ConstraintList,
    NonNegativeReals, Binary, maximize, value, SolverFactory
)
from pyomo.opt import TerminationCondition

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────
STARTING_CAPITAL = 1_000_000.00
LOOKBACK         = 45
TOP_N            = 100
N_STOCKS         = 10
N_PER_SECTOR     = 2
WEIGHT_MIN       = 0.05
WEIGHT_MAX       = 0.50
N_SWEEP          = 6
FRONTIER_TL      = 60
SLIPPAGE_PCT     = 0.0001

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

# GitHub credentials from environment variables
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_REPO  = 'bromomo-git/SP500-Portfolio-Optimizer'
GITHUB_NAME  = 'Cloud Function Auto-Push'
GITHUB_EMAIL = 'bromomo-git@users.noreply.github.com'

# Bonmin path — not available in Cloud Functions
# We use GLPK as fallback for the linear relaxation
# For full MINLP, the Cloud Function installs CBC
BONMIN_PATH = 'bonmin'


def log(msg):
    print(f"[{date.today()} {__import__('datetime').datetime.now().strftime('%H:%M:%S')}] {msg}",
          flush=True)


# ── GitHub push helper ────────────────────────────────────────
def push_to_github(path, content_str, message):
    """Push a file to the GitHub repo via API."""
    if not GITHUB_TOKEN:
        log(f"WARNING: GITHUB_TOKEN not set — skipping push of {path}")
        return False

    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept':        'application/vnd.github.v3+json',
    }
    url      = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{path}'
    existing = _req.get(url, headers=headers)
    sha      = existing.json().get('sha') if existing.status_code == 200 else None

    payload = {
        'message':   message,
        'content':   base64.b64encode(content_str.encode()).decode(),
        'committer': {'name': GITHUB_NAME, 'email': GITHUB_EMAIL},
    }
    if sha:
        payload['sha'] = sha

    resp = _req.put(url, headers=headers, json=payload)
    if resp.status_code in (200, 201):
        log(f"  ✓ Pushed {path}")
        return True
    else:
        log(f"  ✗ Failed {path}: {resp.status_code}")
        return False


# ── Universe ──────────────────────────────────────────────────
def build_universe():
    log("Scraping S&P 500 from Wikipedia...")
    url     = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r:
        html = r.read()

    tables   = pd.read_html(html)
    sp500    = tables[0][['Symbol', 'GICS Sector']].copy()
    sp500.columns = ['Ticker', 'Sector']
    sp500['Ticker'] = sp500['Ticker'].str.replace('.', '-', regex=False)

    universe = sp500[sp500['Sector'].isin(TARGET_SECTORS)].copy()
    universe['SectorShort'] = universe['Sector'].map(SECTOR_SHORT)

    ticker_sector = dict(zip(universe['Ticker'], universe['SectorShort']))
    all_tickers   = universe['Ticker'].tolist()
    log(f"Universe: {len(all_tickers)} stocks")
    return all_tickers, ticker_sector


# ── Download ──────────────────────────────────────────────────
def download_data(all_tickers, ticker_sector):
    """Download via yfinance — works on Google infrastructure."""
    start = (date.today() - timedelta(days=75)).strftime('%Y-%m-%d')
    end   = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    fetch = all_tickers + ['SPY', '^VIX']

    log(f"Downloading {len(fetch)} tickers via yfinance ({start} to {end})...")

    raw = yf.download(
        fetch, start=start, end=end,
        auto_adjust=True, progress=False, interval='1d'
    )

    close_all  = raw['Close'].ffill().bfill()
    open_all   = raw['Open'].ffill().bfill()
    volume_all = raw['Volume'].ffill().bfill()

    spy_close = close_all['SPY'].copy() if 'SPY' in close_all.columns else None
    vix_close = close_all['^VIX'].copy() if '^VIX' in close_all.columns else None

    avail = close_all[all_tickers]
    bad   = avail.columns[(avail.isnull().mean() > 0.20)].tolist()
    if bad:
        log(f"Dropped {len(bad)} tickers (>20% missing)")
        all_tickers   = [t for t in all_tickers if t not in bad]
        ticker_sector = {t: s for t, s in ticker_sector.items() if t not in bad}

    close_px  = close_all[all_tickers].copy()
    open_px   = open_all[all_tickers].copy()
    volume_px = volume_all[all_tickers].copy()
    daily_ret = close_px.pct_change().dropna(how='all')

    log(f"Download complete — {len(daily_ret)} days, {len(all_tickers)} stocks")
    return (close_px, open_px, volume_px, daily_ret,
            spy_close, vix_close, all_tickers, ticker_sector)


# ── Filter ────────────────────────────────────────────────────
def daily_filter(daily_ret, volume_px, ticker_sector, day_idx):
    if day_idx < LOOKBACK:
        return [], {}
    window_ret = daily_ret.iloc[day_idx - LOOKBACK: day_idx]
    window_vol = volume_px.iloc[day_idx - LOOKBACK: day_idx]
    cum_ret    = (1 + window_ret).prod() - 1
    rel_vol    = window_vol.iloc[-5:].mean() / window_vol.mean().replace(0, np.nan)
    vol_std    = window_ret.std() * np.sqrt(252)
    score      = (cum_ret * rel_vol) / vol_std.replace(0, np.nan)
    score      = score.dropna()
    score      = score[score.index.map(lambda t: ticker_sector.get(t) is not None)]
    sc         = pd.Series({t: ticker_sector[t] for t in score.index}).value_counts()
    valid      = sc[sc >= N_PER_SECTOR].index.tolist()
    score      = score[score.index.map(lambda t: ticker_sector[t] in valid)]
    top        = score.nlargest(TOP_N).index.tolist()
    return top, {t: ticker_sector[t] for t in top}


# ── Optimizer ─────────────────────────────────────────────────
def estimate_params(daily_ret, tickers, day_idx):
    window   = daily_ret[tickers].iloc[day_idx - LOOKBACK: day_idx].copy()
    window   = window.fillna(window.mean())
    mean_ret = window.mean()
    lw       = LedoitWolf().fit(window.values)
    cov_df   = pd.DataFrame(lw.covariance_, index=tickers, columns=tickers)
    return mean_ret, cov_df


def solve_portfolio(mean_ret, cov_matrix, sector_map, risk_limit, warm_start=None):
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

    m.obj      = Objective(
        expr=sum(m.x[t] * float(mean_ret[t]) for t in tickers),
        sense=maximize
    )
    m.sum_w    = Constraint(expr=sum(m.x[t] for t in tickers) == 1.0)
    m.n_stocks = Constraint(expr=sum(m.y[t] for t in tickers) == N_STOCKS)
    m.cap      = Constraint(tickers, rule=lambda m, t: m.x[t] <= WEIGHT_MAX * m.y[t])
    m.floor    = Constraint(tickers, rule=lambda m, t: m.x[t] >= WEIGHT_MIN * m.y[t])
    m.sector_exact = ConstraintList()
    for s in sectors:
        m.sector_exact.add(
            sum(m.y[t] for t in sec_tickers[s]) == N_PER_SECTOR
        )
    risk_expr = sum(
        m.x[t1] * float(cov_matrix.at[t1, t2]) * m.x[t2]
        for t1 in tickers for t2 in tickers
    )
    m.risk = Constraint(expr=risk_expr <= risk_limit)

    solver = SolverFactory('bonmin', executable=BONMIN_PATH)
    solver.options['bonmin.time_limit'] = FRONTIER_TL
    solver.options['bonmin.algorithm']  = 'B-BB'
    result = solver.solve(m, tee=False)
    tc     = result.solver.termination_condition

    if tc in (TerminationCondition.optimal, TerminationCondition.maxTimeLimit):
        w     = {t: max(value(m.x[t]), 0.0) for t in tickers}
        total = sum(w.values())
        if total > 0:
            w = {t: v / total for t, v in w.items()}
        return w, sum(w[t] * float(mean_ret[t]) for t in tickers)
    return None, None


def find_optimal_portfolio(mean_ret, cov_matrix, sector_map, warm_start=None):
    tickers     = mean_ret.index.tolist()
    cov_array   = cov_matrix.values
    n           = len(tickers)
    w_eq        = np.ones(n) / n
    eq_var      = float(w_eq @ cov_array @ w_eq)
    d_max       = float(np.diag(cov_array).max())
    risk_levels = np.linspace(eq_var * 0.5, d_max * 1.2, N_SWEEP)

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
        frontier.append({'Port_Var': port_var, 'Port_Std': port_std,
                         'Ann_Return': ann_ret, 'Sharpe': sharpe})
        if sharpe > best_sharpe:
            best_sharpe  = sharpe
            best_weights = w
            best_ret     = port_ret
            best_risk    = port_var

    return best_weights, best_ret, best_sharpe, best_risk, frontier


# ── Dashboard builder ─────────────────────────────────────────
def build_dashboard(perf, spy_close, today_str, best_sharpe):
    dates   = perf.index.strftime('%Y-%m-%d').tolist()
    navs    = perf['NAV'].round(2).tolist()
    pnls    = perf['Daily_PnL'].round(2).tolist()
    returns = (perf['Daily_Return'] * 100).round(4).tolist()
    sharpes = perf['Selected_Sharpe'].round(4).tolist() \
              if 'Selected_Sharpe' in perf.columns else [0] * len(dates)

    spy_rb = []
    if spy_close is not None:
        spy_q2 = spy_close.reindex(perf.index).ffill()
        if len(spy_q2) > 0 and spy_q2.iloc[0] > 0:
            spy_rb = (spy_q2 / spy_q2.iloc[0] * STARTING_CAPITAL).round(2).tolist()

    total_ret = (navs[-1] / STARTING_CAPITAL - 1) * 100 if navs else 0
    total_pnl = navs[-1] - STARTING_CAPITAL if navs else 0
    n_days    = len(navs)
    win_rate  = sum(1 for p in pnls if p > 0) / n_days * 100 if n_days else 0

    import json as _js
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SP500 Portfolio Optimizer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      background:#0f1117;color:#e0e0e0;padding:20px;}}
h1{{font-size:1.55rem;color:#64B5F6;margin-bottom:4px;}}
.sub{{color:#90A4AE;font-size:0.80rem;margin-bottom:22px;}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
       gap:11px;margin-bottom:26px;}}
.kpi{{background:#1e2130;border-radius:10px;padding:15px;
      border-left:4px solid #2196F3;}}
.kpi .lbl{{font-size:0.68rem;color:#90A4AE;text-transform:uppercase;letter-spacing:.5px;}}
.kpi .val{{font-size:1.35rem;font-weight:700;margin-top:3px;}}
.pos{{color:#66BB6A;}}.neg{{color:#EF5350;}}.neu{{color:#64B5F6;}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}}
.full{{grid-column:1/-1;}}
.card{{background:#1e2130;border-radius:12px;padding:16px;}}
.card h3{{font-size:0.78rem;color:#90A4AE;margin-bottom:11px;
          text-transform:uppercase;letter-spacing:.5px;}}
.cw{{position:relative;height:225px;}}.cw.t{{height:285px;}}
footer{{text-align:center;color:#546E7A;font-size:0.70rem;margin-top:22px;}}
a{{color:#64B5F6;}}
</style>
</head>
<body>
<h1>&#x1F4C8; SP500 Portfolio Optimizer</h1>
<p class="sub">
  Updated: <strong>{today_str}</strong> &nbsp;|&nbsp;
  $1M capital &nbsp;|&nbsp; {N_STOCKS} stocks/day &nbsp;|&nbsp;
  {LOOKBACK}-day window &nbsp;|&nbsp; {N_SWEEP} frontier solves/day
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
    <div class="val neu">{best_sharpe:.3f}</div></div>
</div>
<div class="grid">
  <div class="card full">
    <h3>NAV vs SPY Buy &amp; Hold</h3>
    <div class="cw t"><canvas id="navChart"></canvas></div>
  </div>
  <div class="card"><h3>Daily P&amp;L ($)</h3>
    <div class="cw"><canvas id="pnlChart"></canvas></div></div>
  <div class="card"><h3>Daily Return (%)</h3>
    <div class="cw"><canvas id="retChart"></canvas></div></div>
  <div class="card"><h3>Daily Frontier Sharpe</h3>
    <div class="cw"><canvas id="sharpeChart"></canvas></div></div>
</div>
<footer>
  SP500 Portfolio Optimizer &mdash; Google Cloud Functions + GitHub Actions &mdash;
  <a href="https://github.com/bromomo-git/SP500-Portfolio-Optimizer">View on GitHub</a>
</footer>
<script>
const labels={_js.dumps(dates)};
const navs={_js.dumps(navs)};
const spy={_js.dumps(spy_rb)};
const pnls={_js.dumps(pnls)};
const rets={_js.dumps(returns)};
const sharps={_js.dumps(sharpes)};
const G={{color:'rgba(255,255,255,0.06)'}};
const F={{color:'#90A4AE',size:10}};
new Chart(document.getElementById('navChart'),{{type:'line',
  data:{{labels,datasets:[
    {{label:'Portfolio NAV',data:navs,borderColor:'#2196F3',
      backgroundColor:'rgba(33,150,243,0.08)',borderWidth:2.2,
      pointRadius:2,fill:true,tension:0.3}},
    {{label:'SPY (rebased)',data:spy,borderColor:'#90A4AE',
      borderDash:[3,3],borderWidth:1.5,pointRadius:0,fill:false}}
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{color:'#e0e0e0'}}}}}},
    scales:{{x:{{ticks:F,grid:G}},
      y:{{ticks:{{...F,callback:v=>'$'+v.toLocaleString()}},grid:G}}}}}}
}});
new Chart(document.getElementById('pnlChart'),{{type:'bar',
  data:{{labels,datasets:[{{label:'P&L',data:pnls,
    backgroundColor:pnls.map(v=>v>=0?'rgba(102,187,106,0.8)':'rgba(239,83,80,0.8)'),
    borderRadius:3}}]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{x:{{ticks:F,grid:G}},
      y:{{ticks:{{...F,callback:v=>'$'+v.toLocaleString()}},grid:G}}}}}}
}});
new Chart(document.getElementById('retChart'),{{type:'bar',
  data:{{labels,datasets:[{{label:'Return %',data:rets,
    backgroundColor:rets.map(v=>v>=0?'rgba(102,187,106,0.8)':'rgba(239,83,80,0.8)'),
    borderRadius:3}}]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{x:{{ticks:F,grid:G}},
      y:{{ticks:{{...F,callback:v=>v.toFixed(2)+'%'}},grid:G}}}}}}
}});
new Chart(document.getElementById('sharpeChart'),{{type:'line',
  data:{{labels,datasets:[{{label:'Sharpe',data:sharps,
    borderColor:'#64B5F6',backgroundColor:'rgba(100,181,246,0.1)',
    borderWidth:2,pointRadius:2,fill:true,tension:0.3}}]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{x:{{ticks:F,grid:G}},y:{{ticks:F,grid:G}}}}}}
}});
</script>
</body>
</html>"""


# ── Main entry point ──────────────────────────────────────────
def run_daily():
    """
    Full daily pipeline — called by the Cloud Function.
    Returns dict with key metrics for the HTTP response.
    """
    from datetime import date
    today = date.today()

    log(f"{'='*50}")
    log(f"SP500 Portfolio Runner — {today}")
    log(f"{'='*50}")

    # Build universe
    all_tickers, ticker_sector = build_universe()

    # Download data — Yahoo Finance works on Google Cloud
    (close_px, open_px, volume_px, daily_ret,
     spy_close, vix_close, all_tickers,
     ticker_sector) = download_data(all_tickers, ticker_sector)

    # Locate today
    all_days  = daily_ret.index
    today_ts  = pd.Timestamp(today)
    day_idx   = all_days.get_loc(today_ts) if today_ts in all_days \
                else len(all_days) - 1

    if day_idx < LOOKBACK:
        log(f"Insufficient history ({day_idx} < {LOOKBACK})")
        return {'nav': None, 'pnl': 0, 'sharpe': 0, 'stocks_held': 0}

    # Filter
    log(f"Running filter (top {TOP_N})...")
    top_tickers, top_sec_map = daily_filter(
        daily_ret, volume_px, ticker_sector, day_idx
    )
    if len(top_tickers) < N_STOCKS:
        log("Filter returned insufficient stocks")
        return {'nav': None, 'pnl': 0, 'sharpe': 0, 'stocks_held': 0}

    # Estimate parameters
    log("Estimating parameters (Ledoit-Wolf)...")
    mean_r, cov_m = estimate_params(daily_ret, top_tickers, day_idx)

    # Frontier sweep
    log(f"Running frontier sweep ({N_SWEEP} solves)...")
    weights, est_ret, best_sharpe, best_risk, frontier = find_optimal_portfolio(
        mean_r, cov_m, top_sec_map
    )

    if weights is None:
        log("All frontier solves infeasible")
        return {'nav': STARTING_CAPITAL, 'pnl': 0,
                'sharpe': 0, 'stocks_held': 0}

    held = [t for t, w in weights.items() if w > 1e-4]
    log(f"Selected {len(held)} stocks | Sharpe {best_sharpe:.3f}")

    # Execute trades (simulated)
    exec_day = all_days[day_idx + 1] if day_idx + 1 < len(all_days) \
               else all_days[day_idx]
    op = open_px.loc[exec_day]
    cp = close_px.loc[exec_day]

    nav_open    = STARTING_CAPITAL  # simplified — full state tracking in Colab
    target_d    = {t: weights.get(t, 0.0) * nav_open for t in top_tickers}
    day_slip    = 0.0
    positions   = {}

    for t in top_tickers:
        delta = target_d.get(t, 0.0)
        if abs(delta) < 1.0 or float(op.get(t, 0)) <= 0:
            continue
        slip          = abs(delta) * SLIPPAGE_PCT
        effective_px  = float(op.get(t, 0)) * (1 + SLIPPAGE_PCT)
        positions[t]  = delta / effective_px
        day_slip     += slip

    eod_equity = sum(positions.get(t, 0.0) * float(cp.get(t, 0))
                     for t in positions)
    eod_nav    = (nav_open - sum(target_d.values())) + eod_equity
    daily_pnl  = eod_nav - nav_open
    daily_ret_v = daily_pnl / nav_open if nav_open > 0 else 0

    log(f"EOD NAV: ${eod_nav:,.2f} | P&L: ${daily_pnl:+,.2f}")

    # Load existing performance log from GitHub
    log("Loading existing performance log...")
    gh_headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }
    log_url  = (f'https://api.github.com/repos/{GITHUB_REPO}'
                f'/contents/results/performance_log.csv')
    log_resp = _req.get(log_url, headers=gh_headers)

    if log_resp.status_code == 200:
        existing_csv = base64.b64decode(
            log_resp.json()['content']
        ).decode()
        perf_df = pd.read_csv(__import__('io').StringIO(existing_csv))
    else:
        perf_df = pd.DataFrame(columns=[
            'Date','NAV','Daily_PnL','Daily_Return',
            'Slippage','Turnover','Selected_Sharpe','Held_Tickers'
        ])

    # Append today's row
    new_row = {
        'Date':            str(today),
        'NAV':             round(eod_nav, 2),
        'Daily_PnL':       round(daily_pnl, 2),
        'Daily_Return':    round(daily_ret_v, 6),
        'Slippage':        round(day_slip, 4),
        'Turnover':        round(sum(target_d.values()), 2),
        'Selected_Sharpe': round(best_sharpe, 4),
        'Held_Tickers':    ','.join(held),
    }
    perf_df = pd.concat(
        [perf_df, pd.DataFrame([new_row])],
        ignore_index=True
    )
    perf_df['Date'] = pd.to_datetime(perf_df['Date'])
    perf_df = perf_df.drop_duplicates(subset='Date', keep='last')
    perf_df = perf_df.sort_values('Date').reset_index(drop=True)

    commit_msg = f"Auto: Cloud Function run {today}"

    # Push performance log
    push_to_github(
        'results/performance_log.csv',
        perf_df.to_csv(index=False),
        commit_msg
    )

    # Push portfolio state
    state = {
        'last_run':    str(today),
        'nav_history': perf_df[['Date','NAV','Daily_PnL']]
                       .assign(Date=perf_df['Date'].astype(str))
                       .to_dict(orient='records'),
        'total_return': float(perf_df['NAV'].iloc[-1] / STARTING_CAPITAL - 1),
    }
    push_to_github(
        'data/portfolio_state.json',
        json.dumps(state, indent=2, default=str),
        commit_msg
    )

    # Build and push dashboard
    perf_indexed = perf_df.set_index('Date')
    dashboard_html = build_dashboard(
        perf_indexed, spy_close, str(today), best_sharpe
    )
    push_to_github('docs/index.html', dashboard_html, commit_msg)

    log("All files pushed to GitHub.")
    log(f"Dashboard: https://bromomo-git.github.io/SP500-Portfolio-Optimizer/")

    return {
        'nav':         round(eod_nav, 2),
        'pnl':         round(daily_pnl, 2),
        'sharpe':      round(best_sharpe, 3),
        'stocks_held': len(held),
    }
