"""
Dynamic Diagonal Call Income Strategy Backtester - Bloomberg Version

This script backtests a strategy using Bloomberg API data that:
1. Buys a long-dated deep ITM call (LEAP)
2. Sells monthly OTM calls with dynamic strike selection
3. Buys shares when price approaches short call strike
4. Adjusts to market movements to generate income while maintaining upside

Requirements:
- Bloomberg Terminal must be running
- blpapi installed
"""

import numpy as np
import pandas as pd
import blpapi
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Tuple, Dict


class BloombergDataFetcher:
    """Fetch historical data from Bloomberg Terminal"""

    def __init__(self):
        self.session = None

    def connect(self):
        """Connect to Bloomberg Terminal"""
        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost('localhost')
        sessionOptions.setServerPort(8194)

        print("Connecting to Bloomberg Terminal...")
        self.session = blpapi.Session(sessionOptions)

        if not self.session.start():
            raise Exception("Failed to start Bloomberg session. Is Bloomberg Terminal running?")

        if not self.session.openService("//blp/refdata"):
            raise Exception("Failed to open //blp/refdata service")

        print("✓ Connected to Bloomberg Terminal")

    def disconnect(self):
        """Disconnect from Bloomberg"""
        if self.session:
            self.session.stop()

    def get_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical price data from Bloomberg

        Args:
            ticker: Bloomberg ticker (e.g., 'AAPL US Equity')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume
        """
        print(f"Fetching historical data for {ticker}...")

        # Convert dates to Bloomberg format (YYYYMMDD)
        start = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
        end = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')

        # Create request
        refDataService = self.session.getService("//blp/refdata")
        request = refDataService.createRequest("HistoricalDataRequest")

        request.append("securities", ticker)
        request.append("fields", "PX_OPEN")
        request.append("fields", "PX_HIGH")
        request.append("fields", "PX_LOW")
        request.append("fields", "PX_LAST")
        request.append("fields", "PX_VOLUME")

        request.set("startDate", start)
        request.set("endDate", end)
        request.set("periodicitySelection", "DAILY")

        # Send request
        self.session.sendRequest(request)

        # Process response
        data = []
        while True:
            ev = self.session.nextEvent(500)

            for msg in ev:
                if msg.hasElement("securityData"):
                    securityData = msg.getElement("securityData")
                    fieldData = securityData.getElement("fieldData")

                    for i in range(fieldData.numValues()):
                        fields = fieldData.getValueAsElement(i)
                        date = fields.getElementAsDatetime("date")

                        try:
                            data.append({
                                'Date': pd.Timestamp(date.year, date.month, date.day),
                                'Open': fields.getElementAsFloat("PX_OPEN") if fields.hasElement("PX_OPEN") else np.nan,
                                'High': fields.getElementAsFloat("PX_HIGH") if fields.hasElement("PX_HIGH") else np.nan,
                                'Low': fields.getElementAsFloat("PX_LOW") if fields.hasElement("PX_LOW") else np.nan,
                                'Close': fields.getElementAsFloat("PX_LAST"),
                                'Volume': fields.getElementAsFloat("PX_VOLUME") if fields.hasElement("PX_VOLUME") else 0,
                            })
                        except Exception as e:
                            print(f"Warning: Error processing field data: {e}")
                            continue

            if ev.eventType() == blpapi.Event.RESPONSE:
                break

        if not data:
            raise ValueError(f"No data retrieved for {ticker}")

        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        print(f"✓ Retrieved {len(df)} days of data")
        return df


class BlackScholes:
    """Simple Black-Scholes option pricing model"""

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Black-Scholes call option price

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)

        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option delta"""
        if T <= 0:
            return 1.0 if S > K else 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)


class DiagonalCallStrategy:
    """Backtest the dynamic diagonal call income strategy using Bloomberg data"""

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        leap_dte: int = 730,
        leap_delta: float = 0.80,
        short_call_dte: int = 30,
        volatility: float = 0.30,
        risk_free_rate: float = 0.04,
        otm_pct_up: float = 0.08,
        otm_pct_down: float = 0.03,
        approach_threshold: float = 0.02,
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.leap_dte = leap_dte
        self.leap_delta = leap_delta
        self.short_call_dte = short_call_dte
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.otm_pct_up = otm_pct_up
        self.otm_pct_down = otm_pct_down
        self.approach_threshold = approach_threshold

        # Download historical data from Bloomberg
        self.fetcher = BloombergDataFetcher()
        self.fetcher.connect()
        self.data = self._download_data()
        self.fetcher.disconnect()

        # Strategy state
        self.leap_strike = None
        self.leap_entry_price = None
        self.leap_entry_date = None
        self.short_call_strike = None
        self.short_call_premium = None
        self.short_call_entry_date = None
        self.shares_owned = 0
        self.share_cost_basis = 0

        # Results tracking
        self.results = []
        self.trades = []

    def _download_data(self) -> pd.DataFrame:
        """Download historical stock data from Bloomberg"""
        # Add ' US Equity' suffix if not present
        bloomberg_ticker = self.ticker if ' ' in self.ticker else f"{self.ticker} US Equity"

        df = self.fetcher.get_historical_data(
            bloomberg_ticker,
            self.start_date,
            self.end_date
        )

        if df.empty:
            raise ValueError(f"No data found for {bloomberg_ticker}")

        return df

    def _calculate_leap_strike(self, current_price: float) -> float:
        """Calculate LEAP strike based on target delta"""
        strike = current_price * 0.80
        return round(strike / 5) * 5

    def _calculate_short_call_strike(self, current_price: float, initial_price: float) -> float:
        """Calculate dynamic short call strike based on market movement"""
        price_change_pct = (current_price - initial_price) / initial_price

        if price_change_pct >= 0:
            otm_pct = self.otm_pct_up
        else:
            otm_pct = self.otm_pct_down

        strike = current_price * (1 + otm_pct)
        return round(strike / 5) * 5

    def _days_to_years(self, days: int) -> float:
        """Convert days to years for Black-Scholes"""
        return days / 365.0

    def run_backtest(self) -> pd.DataFrame:
        """Run the backtest"""
        print("\n" + "="*80)
        print("Starting Backtest")
        print("="*80)

        initial_price = self.data['Close'].iloc[0]
        monthly_counter = 0

        for i, (date, row) in enumerate(self.data.iterrows()):
            current_price = row['Close']

            # Initialize LEAP on first day
            if self.leap_strike is None:
                self._enter_leap(date, current_price)
                self._enter_short_call(date, current_price, initial_price)
                monthly_counter = 0

            # Roll short call monthly (approximately every 21 trading days)
            monthly_counter += 1
            if monthly_counter >= 21:
                self._exit_short_call(date, current_price)
                self._enter_short_call(date, current_price, initial_price)
                monthly_counter = 0

            # Check if we should buy shares
            if self.shares_owned == 0 and self.short_call_strike:
                distance_to_strike = (self.short_call_strike - current_price) / current_price
                if 0 < distance_to_strike < self.approach_threshold:
                    self._buy_shares(date, current_price)

            # Check if shares get called away
            if self.shares_owned > 0 and monthly_counter == 0:
                if current_price >= self.short_call_strike:
                    self._shares_called_away(date, current_price)

            # Calculate daily P&L
            pnl = self._calculate_pnl(date, current_price)

            self.results.append({
                'Date': date,
                'Price': current_price,
                'LEAP_Strike': self.leap_strike,
                'Short_Call_Strike': self.short_call_strike,
                'Shares_Owned': self.shares_owned,
                'PnL': pnl['total'],
                'LEAP_PnL': pnl['leap'],
                'Short_Call_PnL': pnl['short_call'],
                'Share_PnL': pnl['shares'],
                'Cumulative_Premium': pnl['cumulative_premium']
            })

        results_df = pd.DataFrame(self.results)
        trades_df = pd.DataFrame(self.trades)

        return results_df, trades_df

    def _enter_leap(self, date: datetime, price: float):
        """Enter LEAP position"""
        self.leap_strike = self._calculate_leap_strike(price)
        leap_dte_years = self._days_to_years(self.leap_dte)
        self.leap_entry_price = BlackScholes.call_price(
            price, self.leap_strike, leap_dte_years,
            self.risk_free_rate, self.volatility
        )
        self.leap_entry_date = date

        self.trades.append({
            'Date': date,
            'Action': 'BUY_LEAP',
            'Strike': self.leap_strike,
            'Price': self.leap_entry_price,
            'Stock_Price': price,
            'DTE': self.leap_dte
        })

        print(f"\n{date.date()} - ENTERED LEAP")
        print(f"  Stock: ${price:.2f}")
        print(f"  Strike: ${self.leap_strike:.2f}")
        print(f"  Premium Paid: ${self.leap_entry_price:.2f}")

    def _enter_short_call(self, date: datetime, price: float, initial_price: float):
        """Sell short-term OTM call"""
        self.short_call_strike = self._calculate_short_call_strike(price, initial_price)
        short_dte_years = self._days_to_years(self.short_call_dte)
        self.short_call_premium = BlackScholes.call_price(
            price, self.short_call_strike, short_dte_years,
            self.risk_free_rate, self.volatility
        )
        self.short_call_entry_date = date

        self.trades.append({
            'Date': date,
            'Action': 'SELL_SHORT_CALL',
            'Strike': self.short_call_strike,
            'Price': self.short_call_premium,
            'Stock_Price': price,
            'DTE': self.short_call_dte
        })

        otm_pct = ((self.short_call_strike - price) / price) * 100
        print(f"\n{date.date()} - SOLD SHORT CALL")
        print(f"  Strike: ${self.short_call_strike:.2f} ({otm_pct:.1f}% OTM)")
        print(f"  Premium: ${self.short_call_premium:.2f}")

    def _exit_short_call(self, date: datetime, price: float):
        """Buy back short call"""
        if self.short_call_strike is None:
            return

        days_held = (date - self.short_call_entry_date).days
        remaining_dte = max(self.short_call_dte - days_held, 0)
        remaining_dte_years = self._days_to_years(remaining_dte)

        buyback_price = BlackScholes.call_price(
            price, self.short_call_strike, remaining_dte_years,
            self.risk_free_rate, self.volatility
        )

        profit = self.short_call_premium - buyback_price

        self.trades.append({
            'Date': date,
            'Action': 'BUY_BACK_SHORT_CALL',
            'Strike': self.short_call_strike,
            'Price': buyback_price,
            'Stock_Price': price,
            'Profit': profit
        })

    def _buy_shares(self, date: datetime, price: float):
        """Buy 100 shares when approaching short strike"""
        self.shares_owned = 100
        self.share_cost_basis = price

        self.trades.append({
            'Date': date,
            'Action': 'BUY_SHARES',
            'Shares': 100,
            'Price': price,
            'Cost': price * 100
        })

        print(f"\n{date.date()} - BOUGHT 100 SHARES @ ${price:.2f}")

    def _shares_called_away(self, date: datetime, price: float):
        """Shares are called away at strike price"""
        if self.shares_owned == 0:
            return

        profit = (self.short_call_strike - self.share_cost_basis) * self.shares_owned

        self.trades.append({
            'Date': date,
            'Action': 'SHARES_CALLED_AWAY',
            'Shares': self.shares_owned,
            'Strike': self.short_call_strike,
            'Cost_Basis': self.share_cost_basis,
            'Profit': profit
        })

        print(f"\n{date.date()} - SHARES CALLED AWAY")
        print(f"  Profit: ${profit:.2f}")

        self.shares_owned = 0
        self.share_cost_basis = 0

    def _calculate_pnl(self, date: datetime, price: float) -> Dict[str, float]:
        """Calculate current P&L"""
        # LEAP P&L
        if self.leap_strike and self.leap_entry_date:
            days_held = (date - self.leap_entry_date).days
            remaining_dte = max(self.leap_dte - days_held, 1)
            remaining_dte_years = self._days_to_years(remaining_dte)

            current_leap_value = BlackScholes.call_price(
                price, self.leap_strike, remaining_dte_years,
                self.risk_free_rate, self.volatility
            )
            leap_pnl = current_leap_value - self.leap_entry_price
        else:
            leap_pnl = 0

        # Short call P&L
        if self.short_call_strike and self.short_call_entry_date:
            days_held = (date - self.short_call_entry_date).days
            remaining_dte = max(self.short_call_dte - days_held, 1)
            remaining_dte_years = self._days_to_years(remaining_dte)

            current_short_value = BlackScholes.call_price(
                price, self.short_call_strike, remaining_dte_years,
                self.risk_free_rate, self.volatility
            )
            short_call_pnl = self.short_call_premium - current_short_value
        else:
            short_call_pnl = 0

        # Share P&L
        if self.shares_owned > 0:
            share_pnl = (price - self.share_cost_basis) * self.shares_owned
        else:
            share_pnl = 0

        # Cumulative premium
        cumulative_premium = sum(
            trade['Price'] for trade in self.trades
            if trade['Action'] == 'SELL_SHORT_CALL'
        )

        return {
            'leap': leap_pnl,
            'short_call': short_call_pnl,
            'shares': share_pnl,
            'cumulative_premium': cumulative_premium,
            'total': leap_pnl + short_call_pnl + share_pnl
        }


def plot_results(results_df: pd.DataFrame, ticker: str):
    """Create visualization of backtest results"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Stock price and strikes
    ax1 = axes[0]
    ax1.plot(results_df['Date'], results_df['Price'], label='Stock Price', linewidth=2)
    ax1.plot(results_df['Date'], results_df['LEAP_Strike'],
             label='LEAP Strike', linestyle='--', alpha=0.7)
    ax1.plot(results_df['Date'], results_df['Short_Call_Strike'],
             label='Short Call Strike', linestyle='--', alpha=0.7)
    ax1.set_title(f'{ticker} Price and Strikes', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: P&L breakdown
    ax2 = axes[1]
    ax2.plot(results_df['Date'], results_df['LEAP_PnL'], label='LEAP P&L', alpha=0.7)
    ax2.plot(results_df['Date'], results_df['Short_Call_PnL'], label='Short Call P&L', alpha=0.7)
    ax2.plot(results_df['Date'], results_df['Share_PnL'], label='Share P&L', alpha=0.7)
    ax2.plot(results_df['Date'], results_df['PnL'], label='Total P&L',
             linewidth=2, color='black')
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.set_title('P&L Breakdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('P&L ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative premium
    ax3 = axes[2]
    ax3.fill_between(results_df['Date'], results_df['Cumulative_Premium'],
                      alpha=0.3, label='Cumulative Premium Collected')
    ax3.plot(results_df['Date'], results_df['Cumulative_Premium'],
             linewidth=2, color='green')
    ax3.set_title('Cumulative Premium Collected', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Premium ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('diagonal_call_backtest_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Chart saved as 'diagonal_call_backtest_results.png'")
    plt.show()


def print_summary(results_df: pd.DataFrame, trades_df: pd.DataFrame, ticker: str):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)

    initial_investment = abs(results_df['LEAP_PnL'].iloc[0])
    final_pnl = results_df['PnL'].iloc[-1]
    total_premium = results_df['Cumulative_Premium'].iloc[-1]

    print(f"\n📈 Ticker: {ticker}")
    print(f"📅 Period: {results_df['Date'].iloc[0].date()} to {results_df['Date'].iloc[-1].date()}")
    print(f"💰 Initial LEAP Cost: ${initial_investment:.2f}")
    print(f"💵 Total Premium Collected: ${total_premium:.2f}")
    print(f"📊 Final P&L: ${final_pnl:.2f}")
    print(f"📈 Return: {(final_pnl / initial_investment) * 100:.2f}%")

    # Stock performance
    stock_return = ((results_df['Price'].iloc[-1] - results_df['Price'].iloc[0]) /
                    results_df['Price'].iloc[0]) * 100
    print(f"\n🔵 Stock Return: {stock_return:.2f}%")

    # Trade statistics
    short_calls_sold = len(trades_df[trades_df['Action'] == 'SELL_SHORT_CALL'])
    avg_premium = total_premium / short_calls_sold if short_calls_sold > 0 else 0

    print(f"\n📝 Short Calls Sold: {short_calls_sold}")
    print(f"💵 Average Premium per Call: ${avg_premium:.2f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Example: Backtest AAPL diagonal call strategy using Bloomberg data
    print("="*80)
    print("BLOOMBERG DIAGONAL CALL STRATEGY BACKTESTER")
    print("="*80)
    print("\n⚠️  Make sure Bloomberg Terminal is running before proceeding!")
    input("Press Enter to continue...")

    strategy = DiagonalCallStrategy(
        ticker='AAPL US Equity',  # Bloomberg format
        start_date='2023-01-01',
        end_date='2024-11-01',
        leap_dte=730,
        short_call_dte=30,
        volatility=0.30,
        otm_pct_up=0.08,
        otm_pct_down=0.03,
        approach_threshold=0.02
    )

    # Run backtest
    results_df, trades_df = strategy.run_backtest()

    # Print summary
    print_summary(results_df, trades_df, 'AAPL')

    # Plot results
    plot_results(results_df, 'AAPL')

    # Save results
    results_df.to_csv('backtest_results_bloomberg.csv', index=False)
    trades_df.to_csv('backtest_trades_bloomberg.csv', index=False)
    print("\n💾 Results saved to CSV files")
