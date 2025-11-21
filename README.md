# Dynamic Diagonal Call Income Strategy Backtester

A Python backtesting tool for the dynamic diagonal call income strategy that generates consistent monthly income while maintaining long-term bullish exposure.

## Strategy Overview

The strategy combines:
1. **Long-dated LEAP** (deep ITM call) for long-term exposure
2. **Monthly short calls** with dynamic strike selection for income
3. **Share purchases** when price approaches short strike to lock in gains
4. **Automatic hedging** through adaptive strike selection

## Features

- ✅ Black-Scholes option pricing model
- ✅ Dynamic strike selection based on market movement
- ✅ Automatic share purchase when approaching short call strike
- ✅ Monthly rolling of short calls
- ✅ Comprehensive P&L tracking
- ✅ Visual performance reports
- ✅ Detailed trade logs
- ✅ **Two versions**: Yahoo Finance (free) and Bloomberg API (real data)

## Installation

### Version 1: Yahoo Finance (No Bloomberg Required)

**On Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Version 2: Bloomberg API (Requires Bloomberg Terminal)

**On Windows with Bloomberg Terminal:**
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements_bloomberg.txt

# Install Bloomberg API
pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

**Requirements:**
- Bloomberg Terminal must be installed and running
- You must be logged into Bloomberg Terminal
- Terminal runs on `localhost:8194`

## Usage

### Version 1: Yahoo Finance (Free Data)

**Run the example:**
```bash
python example.py
```

**Custom usage:**
```python
from diagonal_call_backtest import DiagonalCallStrategy

strategy = DiagonalCallStrategy(
    ticker='AAPL',  # Simple ticker format
    start_date='2023-01-01',
    end_date='2024-11-01',
)

results_df, trades_df = strategy.run_backtest()
```

### Version 2: Bloomberg API (Real Bloomberg Data)

**⚠️ Make sure Bloomberg Terminal is running first!**

**Run the example:**
```bash
python example_bloomberg.py
```

**Custom usage:**
```python
from diagonal_call_backtest_bloomberg import DiagonalCallStrategy

strategy = DiagonalCallStrategy(
    ticker='AAPL US Equity',  # Bloomberg ticker format
    start_date='2023-01-01',
    end_date='2024-11-01',
)

results_df, trades_df = strategy.run_backtest()
```

**Bloomberg Ticker Format:**
- US Stocks: `AAPL US Equity`, `MSFT US Equity`
- ETFs: `SPY US Equity`, `QQQ US Equity`

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ticker` | Stock ticker symbol | Required |
| `start_date` | Backtest start date (YYYY-MM-DD) | Required |
| `end_date` | Backtest end date (YYYY-MM-DD) | Required |
| `leap_dte` | Days to expiration for LEAP | 730 (2 years) |
| `leap_delta` | Target delta for LEAP (deep ITM) | 0.80 |
| `short_call_dte` | Days to expiration for short calls | 30 (monthly) |
| `volatility` | Implied volatility estimate | 0.30 (30%) |
| `risk_free_rate` | Risk-free interest rate | 0.04 (4%) |
| `otm_pct_up` | OTM % when market is up | 0.08 (8%) |
| `otm_pct_down` | OTM % when market is down | 0.03 (3%) |
| `approach_threshold` | Distance to trigger share purchase | 0.02 (2%) |

## Output Files

- **backtest_results.csv** - Daily P&L and position data
- **backtest_trades.csv** - Detailed trade log
- **diagonal_call_backtest_results.png** - Visual performance charts

## Strategy Logic

### Dynamic Strike Selection

```python
# When market is UP from entry
short_call_strike = current_price * (1 + 8%)  # Sell further OTM

# When market is DOWN from entry
short_call_strike = current_price * (1 + 3%)  # Sell closer to money
```

### Share Purchase Trigger

When stock price approaches within 2% of short call strike:
- Buy 100 shares at market price
- If price > strike at expiration → shares called away at strike
- Realize gains while keeping LEAP intact

### Monthly Rolling

Every ~21 trading days:
- Close existing short call
- Sell new short call with dynamically adjusted strike
- Collect premium

## Example Output

```
================================================================================
BACKTEST SUMMARY
================================================================================

📈 Ticker: AAPL
📅 Period: 2023-01-01 to 2024-11-01
💰 Initial LEAP Cost: $4,523.50
💵 Total Premium Collected: $3,245.00
📊 Final P&L: $8,456.23
📈 Return: 186.94%

🔵 Stock Return: 45.23%

📝 Short Calls Sold: 22
💵 Average Premium per Call: $147.50

📉 Max Drawdown: $-234.56
```

## Customization

### Test Different Tickers

```python
# Test on SPY
strategy = DiagonalCallStrategy(ticker='SPY', ...)

# Test on NVDA
strategy = DiagonalCallStrategy(ticker='NVDA', ...)
```

### Adjust Strike Selection

```python
# More aggressive (further OTM when up)
strategy = DiagonalCallStrategy(
    otm_pct_up=0.12,      # 12% OTM
    otm_pct_down=0.05,    # 5% OTM
    ...
)
```

### Different Timeframes

```python
# Weekly short calls
strategy = DiagonalCallStrategy(short_call_dte=7, ...)

# Longer LEAP
strategy = DiagonalCallStrategy(leap_dte=1095, ...)  # 3 years
```

## Limitations

- Uses Black-Scholes pricing (simplification of real market)
- Assumes constant implied volatility
- Does not account for:
  - Bid-ask spreads
  - Transaction costs
  - Early assignment risk
  - Earnings events
  - Dividend adjustments
  - Liquidity constraints

## Future Enhancements

- [ ] Integration with Bloomberg API for real market data
- [ ] Real-time implied volatility (IV) calculations
- [ ] Earnings calendar integration
- [ ] Multiple LEAP contracts
- [ ] Transaction cost modeling
- [ ] Greeks tracking (delta, theta, vega)
- [ ] Risk metrics (Sharpe ratio, Sortino ratio)

## GitHub Workflow

```bash
# Initialize git repo
git init
git add .
git commit -m "Initial commit - diagonal call backtester"

# Create GitHub repo and push
git remote add origin https://github.com/yourusername/bloomberg-backtesting.git
git push -u origin main

# On Windows machine
git clone https://github.com/yourusername/bloomberg-backtesting.git
cd bloomberg-backtesting
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python diagonal_call_backtest.py
```

## License

MIT License - feel free to modify and use for your own trading research.

## Disclaimer

This is for educational and research purposes only. Past performance does not guarantee future results. Options trading involves substantial risk and is not suitable for all investors.
