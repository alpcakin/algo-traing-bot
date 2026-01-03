"""
Backtesting Engine - CORRECTED VERSION
Simulates trading with position management, TP/SL execution, and P&L tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy import TrendFollowingStrategy, generate_signals
from config import *


class Position:
    """Represents a single trading position"""
    
    def __init__(self, entry_time, entry_price, direction, lot_size, 
                 sl_price, tp_pips, symbol, position_id):
        self.position_id = position_id
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction  # 'LONG' or 'SHORT'
        self.lot_size = lot_size
        self.sl_price = sl_price
        self.symbol = symbol
        
        # Calculate TP price
        pip_size = 0.01 if 'JPY' in symbol else 0.0001
        tp_distance = tp_pips * pip_size
        
        if direction == 'LONG':
            self.tp_price = entry_price + tp_distance
        else:
            self.tp_price = entry_price - tp_distance
        
        # Status
        self.status = 'OPEN'
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0
        
    
    def check_tp_hit(self, candle):
        """Check if TP was hit during this candle"""
        if self.direction == 'LONG':
            return candle['high'] >= self.tp_price
        else:
            return candle['low'] <= self.tp_price
    
    
    def check_sl_hit(self, candle, mitigation_high, mitigation_low):
        """Check if SL was hit (body close outside mitigation)"""
        if self.direction == 'LONG':
            # LONG SL = close below mitigation LOW
            return candle['close'] < mitigation_low
        else:
            # SHORT SL = close above mitigation HIGH
            return candle['close'] > mitigation_high
    
    
    def close_position(self, exit_time, exit_price, exit_reason):
        """Close the position and calculate P&L"""
        self.status = 'CLOSED'
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        
        # Calculate P&L
        pip_size = 0.01 if 'JPY' in self.symbol else 0.0001
        pip_value_per_lot = 10  # Standard lot
        
        if self.direction == 'LONG':
            pips = (exit_price - self.entry_price) / pip_size
        else:
            pips = (self.entry_price - exit_price) / pip_size
        
        # P&L = pips * lot_size * pip_value
        self.pnl = pips * self.lot_size * pip_value_per_lot
        
        return self.pnl


class Backtester:
    """Backtest engine with position management"""
    
    def __init__(self, initial_balance, symbol):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.symbol = symbol
        
        self.positions = []  # All positions (open + closed)
        self.open_positions = []  # Currently open positions
        self.closed_positions = []  # Closed positions
        
        self.position_counter = 0
        self.equity_curve = []
        
    
    def open_position(self, entry_time, entry_price, direction, lot_size, 
                      sl_price, tp_pips):
        """Open a new position"""
        self.position_counter += 1
        
        position = Position(
            entry_time=entry_time,
            entry_price=entry_price,
            direction=direction,
            lot_size=lot_size,
            sl_price=sl_price,
            tp_pips=tp_pips,
            symbol=self.symbol,
            position_id=self.position_counter
        )
        
        self.positions.append(position)
        self.open_positions.append(position)
        
        return position
    
    
    def close_position(self, position, exit_time, exit_price, exit_reason):
        """Close a position and update balance"""
        pnl = position.close_position(exit_time, exit_price, exit_reason)
        
        self.balance += pnl
        self.open_positions.remove(position)
        self.closed_positions.append(position)
        
        return pnl
    
    
    def close_all_positions(self, exit_time, exit_price, exit_reason):
        """Close all open positions"""
        total_pnl = 0
        
        for position in self.open_positions[:]:  # Copy list to avoid modification during iteration
            pnl = self.close_position(position, exit_time, exit_price, exit_reason)
            total_pnl += pnl
        
        return total_pnl
    
    
    def update_positions(self, candle, current_bias, mitigation_high, mitigation_low):
        """
        Update all open positions: check TP, SL, bias change
        """
        for position in self.open_positions[:]:  # Copy to avoid modification issues

            # 1. Check individual TP
            if position.check_tp_hit(candle):
                self.close_position(position, candle['time'], position.tp_price, 'TP_HIT')
                continue

            # 2. Check if bias changed (opposite to position direction)
            if (position.direction == 'LONG' and current_bias == 'SHORT') or \
               (position.direction == 'SHORT' and current_bias == 'LONG'):
                self.close_position(position, candle['time'], candle['close'], 'BIAS_CHANGE')
                continue

            # 3. Check SL (mitigation break - body close)
            if position.check_sl_hit(candle, mitigation_high, mitigation_low):
                self.close_position(position, candle['time'], candle['close'], 'SL_HIT')
                continue
    
    
    def get_stats(self):
        """Calculate backtest statistics"""
        if not self.closed_positions:
            return None
        
        total_trades = len(self.closed_positions)
        winning_trades = [p for p in self.closed_positions if p.pnl > 0]
        losing_trades = [p for p in self.closed_positions if p.pnl < 0]
        
        total_pnl = sum(p.pnl for p in self.closed_positions)
        total_win = sum(p.pnl for p in winning_trades)
        total_loss = sum(p.pnl for p in losing_trades)
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = total_win / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        
        profit_factor = abs(total_win / total_loss) if total_loss != 0 else float('inf')
        
        # Max drawdown
        equity_curve = [self.initial_balance]
        balance = self.initial_balance
        for p in self.closed_positions:
            balance += p.pnl
            equity_curve.append(balance)
        
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_win': total_win,
            'total_loss': total_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_balance': self.balance,
            'return_pct': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'max_drawdown': max_dd
        }


def run_backtest(df, symbol, initial_balance, strategy_params):
    """
    Run full backtest on dataset
    """
    print(f"\n{'='*60}")
    print(f"RUNNING BACKTEST: {symbol}")
    print(f"{'='*60}")
    print(f"Period: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    print(f"Candles: {len(df):,}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    
    # Create strategy
    strategy = TrendFollowingStrategy(**strategy_params)
    
    # Create backtester
    backtester = Backtester(initial_balance, symbol)
    
    # Main backtest loop
    print("\nRunning backtest...")

    for i in range(0, len(df)):
        candle = df.iloc[i]

        # Check for daily reset
        if strategy.check_daily_reset(candle['time']):
            # Close all positions at start of new day
            if backtester.open_positions:
                backtester.close_all_positions(candle['time'], candle['close'], 'DAILY_RESET')

        # Check if we should close all positions (20:00)
        if strategy.should_close_all_positions(candle['time']):
            if backtester.open_positions:
                backtester.close_all_positions(candle['time'], candle['close'], 'END_OF_DAY')

        # Check if news time - close all positions
        if strategy.enable_news_filter and strategy.news_filter:
            is_news, event = strategy.news_filter.is_news_time(candle['time'])
            if is_news and backtester.open_positions:
                backtester.close_all_positions(candle['time'], candle['close'], f'NEWS_{event}')

        # Update swing levels
        strategy.update_swing_levels(df, i)

        # Check for bias change
        bias_changed = strategy.check_bias_change(df, i)

        # Update all open positions (TP, SL, bias change)
        backtester.update_positions(
            candle,
            strategy.bias,
            strategy.mitigation_high,
            strategy.mitigation_low
        )

        # Entry logic: during trading hours, if mitigation is set, and NOT high volatility
        if strategy.is_trading_hours(candle['time']) and strategy.should_enter(candle):
            # Check volatility filter
            if strategy.is_high_volatility(df, i):
                continue  # Skip entry during high volatility

            if strategy.mitigation_high is not None and strategy.mitigation_low is not None:
                entry_price = strategy.get_entry_price(candle)
                lot_size = strategy.calculate_position_size(backtester.balance, entry_price, symbol)

                if lot_size > 0:
                    backtester.open_position(
                        entry_time=candle['time'],
                        entry_price=entry_price,
                        direction=strategy.bias,
                        lot_size=lot_size,
                        sl_price=strategy.get_sl_price(),
                        tp_pips=strategy.individual_tp_pips
                    )
        
        # Track equity
        if i % 1000 == 0:
            pip_size = 0.01 if 'JPY' in symbol else 0.0001
            open_pnl = sum(
                (p.lot_size * ((candle['close'] - p.entry_price) / pip_size) * 10)
                if p.direction == 'LONG' else
                (p.lot_size * ((p.entry_price - candle['close']) / pip_size) * 10)
                for p in backtester.open_positions
            )
            current_equity = backtester.balance + open_pnl
            print(f"  Progress: {i}/{len(df)} | Balance: ${backtester.balance:,.2f} | Open: {len(backtester.open_positions)}")
    
    # Close any remaining positions
    if backtester.open_positions:
        last_candle = df.iloc[-1]
        backtester.close_all_positions(last_candle['time'], last_candle['close'], 'END_OF_DATA')
    
    # Get statistics
    stats = backtester.get_stats()
    
    return backtester, stats


def print_stats(stats):
    """Print backtest statistics"""
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Winning Trades: {stats['winning_trades']} ({stats['win_rate']:.2f}%)")
    print(f"Losing Trades: {stats['losing_trades']}")
    print(f"\nP&L:")
    print(f"  Total: ${stats['total_pnl']:,.2f}")
    print(f"  Total Win: ${stats['total_win']:,.2f}")
    print(f"  Total Loss: ${stats['total_loss']:,.2f}")
    print(f"  Avg Win: ${stats['avg_win']:,.2f}")
    print(f"  Avg Loss: ${stats['avg_loss']:,.2f}")
    print(f"\nMetrics:")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    print(f"  Final Balance: ${stats['final_balance']:,.2f}")
    print(f"  Return: {stats['return_pct']:,.2f}%")
    print(f"  Max Drawdown: {stats['max_drawdown']:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Load data
    symbol = "EURUSD"
    df = pd.read_csv(f"data/raw/{symbol}_M1_{START_DATE}_{END_DATE}.csv")
    df['time'] = pd.to_datetime(df['time'])
    
    # Strategy parameters
    strategy_params = {
        'individual_tp_pips': INDIVIDUAL_TP_PIPS,
        'risk_per_trade': RISK_PER_TRADE,
        'emergency_sl_percent': EMERGENCY_SL_PERCENT,
        'trading_hours': TRADING_HOURS,
        'analysis_hours': ANALYSIS_HOURS,
        'min_mitigation_distance_pips': MIN_MITIGATION_DISTANCE_PIPS,
        'swing_lookback': SWING_LOOKBACK,
        'enable_news_filter': ENABLE_NEWS_FILTER,
        'news_buffer_before': NEWS_BEFORE_MINUTES,
        'news_buffer_after': NEWS_AFTER_MINUTES,
        'enable_volatility_filter': ENABLE_VOLATILITY_FILTER,
        'atr_period': ATR_PERIOD,
        'atr_multiplier': ATR_MULTIPLIER
    }
    
    # Run backtest
    backtester, stats = run_backtest(df, symbol, INITIAL_BALANCE, strategy_params)
    
    # Print results
    if stats:
        print_stats(stats)
    else:
        print("No trades executed!")