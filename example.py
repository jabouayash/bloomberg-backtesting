"""
Simple example of running the diagonal call backtest
"""

from diagonal_call_backtest import DiagonalCallStrategy, plot_results, print_summary


def run_simple_backtest():
    """Run a simple backtest on Apple"""

    print("Running Simple Diagonal Call Backtest on AAPL...\n")

    # Create strategy with default parameters
    strategy = DiagonalCallStrategy(
        ticker='AAPL',
        start_date='2023-01-01',
        end_date='2024-11-01',
    )

    # Run backtest
    results_df, trades_df = strategy.run_backtest()

    # Show results
    print_summary(results_df, trades_df, 'AAPL')
    plot_results(results_df, 'AAPL')

    return results_df, trades_df


def run_custom_backtest():
    """Run backtest with custom parameters"""

    print("Running Custom Diagonal Call Backtest on SPY...\n")

    # More aggressive parameters
    strategy = DiagonalCallStrategy(
        ticker='SPY',
        start_date='2023-01-01',
        end_date='2024-11-01',
        leap_dte=730,           # 2 years
        short_call_dte=30,      # Monthly
        volatility=0.25,        # 25% IV (SPY typically lower than AAPL)
        otm_pct_up=0.10,        # 10% OTM when market up
        otm_pct_down=0.04,      # 4% OTM when market down
        approach_threshold=0.015  # 1.5% threshold for buying shares
    )

    # Run backtest
    results_df, trades_df = strategy.run_backtest()

    # Show results
    print_summary(results_df, trades_df, 'SPY')
    plot_results(results_df, 'SPY')

    return results_df, trades_df


def compare_multiple_tickers():
    """Compare strategy performance across different stocks"""

    tickers = ['AAPL', 'MSFT', 'NVDA']
    results = {}

    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"Running backtest for {ticker}")
        print('='*80)

        strategy = DiagonalCallStrategy(
            ticker=ticker,
            start_date='2023-01-01',
            end_date='2024-11-01',
        )

        results_df, trades_df = strategy.run_backtest()
        results[ticker] = {
            'results': results_df,
            'trades': trades_df,
            'final_pnl': results_df['PnL'].iloc[-1],
            'premium_collected': results_df['Cumulative_Premium'].iloc[-1]
        }

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON ACROSS TICKERS")
    print("="*80)

    for ticker, data in results.items():
        print(f"\n{ticker}:")
        print(f"  Final P&L: ${data['final_pnl']:.2f}")
        print(f"  Premium Collected: ${data['premium_collected']:.2f}")

    return results


if __name__ == "__main__":
    # Choose which example to run:

    # Example 1: Simple backtest with defaults
    results_df, trades_df = run_simple_backtest()

    # Example 2: Custom parameters (uncomment to run)
    # results_df, trades_df = run_custom_backtest()

    # Example 3: Compare multiple tickers (uncomment to run)
    # results = compare_multiple_tickers()
