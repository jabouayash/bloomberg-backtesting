"""
Simple example of running the diagonal call backtest with Bloomberg data

REQUIREMENTS:
1. Bloomberg Terminal must be running
2. You must be logged into Bloomberg Terminal
3. blpapi must be installed
"""

from diagonal_call_backtest_bloomberg import DiagonalCallStrategy, plot_results, print_summary


def run_aapl_backtest():
    """Run a simple backtest on Apple using Bloomberg data"""

    print("="*80)
    print("BLOOMBERG DIAGONAL CALL STRATEGY BACKTESTER")
    print("="*80)
    print("\n⚠️  REQUIREMENTS:")
    print("  1. Bloomberg Terminal must be running")
    print("  2. You must be logged into Bloomberg Terminal")
    print("  3. Terminal will connect to localhost:8194")
    print("\nReady to proceed?")
    input("Press Enter to continue or Ctrl+C to cancel...\n")

    # Create strategy with Bloomberg data
    strategy = DiagonalCallStrategy(
        ticker='AAPL US Equity',  # Bloomberg ticker format
        start_date='2023-01-01',
        end_date='2024-11-01',
    )

    # Run backtest
    results_df, trades_df = strategy.run_backtest()

    # Show results
    print_summary(results_df, trades_df, 'AAPL')
    plot_results(results_df, 'AAPL')

    # Save results to CSV
    results_df.to_csv('backtest_results_bloomberg.csv', index=False)
    trades_df.to_csv('backtest_trades_bloomberg.csv', index=False)
    print("\n💾 Results saved to 'backtest_results_bloomberg.csv' and 'backtest_trades_bloomberg.csv'")

    return results_df, trades_df


def run_custom_ticker():
    """Run backtest on a custom ticker"""

    print("\n" + "="*80)
    print("Custom Ticker Backtest")
    print("="*80)

    # Get ticker from user
    ticker = input("\nEnter Bloomberg ticker (e.g., 'MSFT US Equity', 'SPY US Equity'): ")

    strategy = DiagonalCallStrategy(
        ticker=ticker,
        start_date='2023-01-01',
        end_date='2024-11-01',
        otm_pct_up=0.08,
        otm_pct_down=0.03,
    )

    results_df, trades_df = strategy.run_backtest()
    print_summary(results_df, trades_df, ticker)
    plot_results(results_df, ticker)

    # Save results to CSV
    results_df.to_csv('backtest_results_bloomberg.csv', index=False)
    trades_df.to_csv('backtest_trades_bloomberg.csv', index=False)
    print("\n💾 Results saved to 'backtest_results_bloomberg.csv' and 'backtest_trades_bloomberg.csv'")

    return results_df, trades_df


if __name__ == "__main__":
    # Run AAPL backtest
    results_df, trades_df = run_aapl_backtest()

    # Uncomment to run custom ticker
    # results_df, trades_df = run_custom_ticker()
