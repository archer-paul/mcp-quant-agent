"""Run baselines on the medium window (Jan 2023) with adapted short lookbacks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_quant_agent.backtest.baselines import (
    _load_prices, buy_and_hold, momentum_ts, mean_reversion_bb,
    COST_PER_TRADE, COMMISSION_BPS, SLIPPAGE_BPS,
)
from mcp_quant_agent.eval.financial import bootstrap_sharpe_ci

tickers = ["AAPL", "MSFT", "NVDA"]
start = "2023-01-03"
end = "2023-01-31"
initial_cash = 100_000.0

print("=== MEDIUM BASELINES (2023-01-03 to 2023-01-31, 10 bps costs) ===")
print(f"[!] MEDIUM-SCALE, NOT thesis-final. Short lookbacks adapted to 19-bar window.")
print(f"Transaction cost: {COMMISSION_BPS:.0f} bps commission + {SLIPPAGE_BPS:.0f} bps slippage = {(COMMISSION_BPS+SLIPPAGE_BPS):.0f} bps one-way")
print()

for ticker in tickers:
    df = _load_prices(ticker, start, end)
    if df is None:
        print(f"{ticker}: no data")
        continue
    n_bars = len(df)
    print(f"=== {ticker} ({n_bars} bars) ===")

    bh = buy_and_hold(df, initial_cash=initial_cash)
    m = bh["metrics"]
    print(f"  Buy&Hold   : ann={m['annualised_return']*100:+.1f}%  sharpe={m['sharpe']:.3f}  maxdd={m['max_drawdown']*100:.1f}%  cost_drag={m['cost_drag_bps']:.2f}bps  turnover={m['turnover_pct']:.1f}%")

    mom = momentum_ts(df, lookback=10, initial_cash=initial_cash)
    m = mom["metrics"]
    print(f"  Momentum10d: ann={m['annualised_return']*100:+.1f}%  sharpe={m['sharpe']:.3f}  maxdd={m['max_drawdown']*100:.1f}%  n_trades={mom['n_trades']}  cost_drag={m['cost_drag_bps']:.2f}bps  turnover={m['turnover_pct']:.1f}%")

    mr = mean_reversion_bb(df, window=5, num_std=1.5, initial_cash=initial_cash)
    m = mr["metrics"]
    print(f"  MeanRev5d  : ann={m['annualised_return']*100:+.1f}%  sharpe={m['sharpe']:.3f}  maxdd={m['max_drawdown']*100:.1f}%  n_trades={mr['n_trades']}  cost_drag={m['cost_drag_bps']:.2f}bps  turnover={m['turnover_pct']:.1f}%")
    print()
