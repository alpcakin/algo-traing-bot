"""
Strategy Visualization Module
Displays candlestick charts with swing levels, bias changes, and trades
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy import TrendFollowingStrategy
from backtester import run_backtest
from config import *


def visualize_strategy(df, strategy, backtester, start_idx=0, end_idx=None, day=None):
    """
    Visualize strategy execution on a candlestick chart

    Args:
        df: Price data
        strategy: Strategy instance
        backtester: Backtester instance with executed trades
        start_idx: Start index for visualization
        end_idx: End index for visualization
        day: Specific day to visualize (e.g., '2025-12-04')
    """

    # Filter by day if specified
    if day:
        day_data = df[df['time'].dt.date == pd.to_datetime(day).date()]
        if len(day_data) == 0:
            print(f"No data found for day {day}")
            return
        start_idx = day_data.index[0]
        end_idx = day_data.index[-1] + 1

    if end_idx is None:
        end_idx = min(start_idx + 500, len(df))  # Max 500 candles

    df_subset = df.iloc[start_idx:end_idx].copy()

    # Create figure
    fig = go.Figure()

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=df_subset['time'],
        open=df_subset['open'],
        high=df_subset['high'],
        low=df_subset['low'],
        close=df_subset['close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Track swing highs/lows, reference levels, mitigation levels
    swing_highs_x = []
    swing_highs_y = []
    swing_lows_x = []
    swing_lows_y = []

    reference_high_lines = []
    reference_low_lines = []
    mitigation_lines = []

    bias_changes = []

    # Re-run strategy to capture events
    temp_strategy = TrendFollowingStrategy(
        individual_tp_pips=INDIVIDUAL_TP_PIPS,
        risk_per_trade=RISK_PER_TRADE,
        emergency_sl_percent=EMERGENCY_SL_PERCENT,
        trading_hours=TRADING_HOURS,
        analysis_hours=ANALYSIS_HOURS,
        min_mitigation_distance_pips=MIN_MITIGATION_DISTANCE_PIPS,
        swing_lookback=SWING_LOOKBACK
    )

    current_ref_high = None
    current_ref_low = None
    current_mit_high = None
    current_mit_low = None

    for i in range(start_idx, end_idx):
        candle = df.iloc[i]

        # Check daily reset
        temp_strategy.check_daily_reset(candle['time'])

        # Update swing levels
        temp_strategy.update_swing_levels(df, i)

        # Track swing highs (from pending list)
        if temp_strategy.pending_swing_highs:
            for swing_price, swing_idx in temp_strategy.pending_swing_highs:
                if swing_idx >= start_idx and swing_idx < end_idx:
                    if swing_idx not in [df.iloc[j].name for j in range(len(swing_highs_x))]:
                        swing_highs_x.append(df.iloc[swing_idx]['time'])
                        swing_highs_y.append(swing_price)

        # Track swing lows (from pending list)
        if temp_strategy.pending_swing_lows:
            for swing_price, swing_idx in temp_strategy.pending_swing_lows:
                if swing_idx >= start_idx and swing_idx < end_idx:
                    if swing_idx not in [df.iloc[j].name for j in range(len(swing_lows_x))]:
                        swing_lows_x.append(df.iloc[swing_idx]['time'])
                        swing_lows_y.append(swing_price)

        # Track reference level changes
        if temp_strategy.reference_high != current_ref_high:
            current_ref_high = temp_strategy.reference_high
            if current_ref_high is not None:
                reference_high_lines.append({
                    'time': candle['time'],
                    'price': current_ref_high,
                    'type': 'ref_high'
                })

        if temp_strategy.reference_low != current_ref_low:
            current_ref_low = temp_strategy.reference_low
            if current_ref_low is not None:
                reference_low_lines.append({
                    'time': candle['time'],
                    'price': current_ref_low,
                    'type': 'ref_low'
                })

        # Track mitigation level changes
        if temp_strategy.mitigation_high != current_mit_high or temp_strategy.mitigation_low != current_mit_low:
            current_mit_high = temp_strategy.mitigation_high
            current_mit_low = temp_strategy.mitigation_low
            if current_mit_high is not None and current_mit_low is not None:
                mitigation_lines.append({
                    'time': candle['time'],
                    'high': current_mit_high,
                    'low': current_mit_low
                })

        # Check for bias change
        prev_bias = temp_strategy.bias
        temp_strategy.check_bias_change(df, i)
        if temp_strategy.bias != prev_bias:
            bias_changes.append({
                'time': candle['time'],
                'new_bias': temp_strategy.bias,
                'price': candle['close']
            })

    # Add swing highs
    if swing_highs_x:
        fig.add_trace(go.Scatter(
            x=swing_highs_x,
            y=swing_highs_y,
            mode='markers',
            name='Swing High',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                line=dict(width=1, color='darkred')
            )
        ))

    # Add swing lows
    if swing_lows_x:
        fig.add_trace(go.Scatter(
            x=swing_lows_x,
            y=swing_lows_y,
            mode='markers',
            name='Swing Low',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                line=dict(width=1, color='darkgreen')
            )
        ))

    # Add reference high lines
    for ref in reference_high_lines:
        fig.add_hline(
            y=ref['price'],
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Ref High: {ref['price']:.5f}",
            annotation_position="right"
        )

    # Add reference low lines
    for ref in reference_low_lines:
        fig.add_hline(
            y=ref['price'],
            line_dash="dot",
            line_color="cyan",
            annotation_text=f"Ref Low: {ref['price']:.5f}",
            annotation_position="right"
        )

    # Add mitigation zones (rectangles)
    for mit in mitigation_lines:
        fig.add_hrect(
            y0=mit['low'],
            y1=mit['high'],
            fillcolor="yellow",
            opacity=0.2,
            line_width=0,
        )

    # Add bias changes (as markers, not vlines - plotly timestamp issue)
    if bias_changes:
        bc_times = [bc['time'] for bc in bias_changes]
        bc_prices = [bc['price'] for bc in bias_changes]
        bc_colors = ['green' if bc['new_bias'] == 'LONG' else 'red' for bc in bias_changes]

        fig.add_trace(go.Scatter(
            x=bc_times,
            y=bc_prices,
            mode='markers+text',
            name='Bias Change',
            marker=dict(symbol='diamond', size=15, color=bc_colors, line=dict(width=2, color='black')),
            text=[bc['new_bias'] for bc in bias_changes],
            textposition="top center"
        ))

    # Add trades from backtester
    if backtester and backtester.closed_positions:
        long_entries_x = []
        long_entries_y = []
        short_entries_x = []
        short_entries_y = []

        tp_exits_x = []
        tp_exits_y = []
        bias_exits_x = []
        bias_exits_y = []
        sl_exits_x = []
        sl_exits_y = []

        for pos in backtester.closed_positions:
            # Filter positions in our time range
            entry_time = pos.entry_time
            if entry_time < df_subset['time'].iloc[0] or entry_time > df_subset['time'].iloc[-1]:
                continue

            # Entry points
            if pos.direction == 'LONG':
                long_entries_x.append(pos.entry_time)
                long_entries_y.append(pos.entry_price)
            else:
                short_entries_x.append(pos.entry_time)
                short_entries_y.append(pos.entry_price)

            # Exit points
            if pos.exit_reason == 'TP_HIT':
                tp_exits_x.append(pos.exit_time)
                tp_exits_y.append(pos.exit_price)
            elif pos.exit_reason == 'BIAS_CHANGE':
                bias_exits_x.append(pos.exit_time)
                bias_exits_y.append(pos.exit_price)
            elif pos.exit_reason == 'SL_HIT':
                sl_exits_x.append(pos.exit_time)
                sl_exits_y.append(pos.exit_price)

        # Add LONG entries
        if long_entries_x:
            fig.add_trace(go.Scatter(
                x=long_entries_x,
                y=long_entries_y,
                mode='markers',
                name='LONG Entry',
                marker=dict(
                    symbol='arrow-up',
                    size=10,
                    color='lightgreen',
                    line=dict(width=2, color='green')
                )
            ))

        # Add SHORT entries
        if short_entries_x:
            fig.add_trace(go.Scatter(
                x=short_entries_x,
                y=short_entries_y,
                mode='markers',
                name='SHORT Entry',
                marker=dict(
                    symbol='arrow-down',
                    size=10,
                    color='lightcoral',
                    line=dict(width=2, color='red')
                )
            ))

        # Add TP exits
        if tp_exits_x:
            fig.add_trace(go.Scatter(
                x=tp_exits_x,
                y=tp_exits_y,
                mode='markers',
                name='TP Exit',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='gold',
                    line=dict(width=2, color='orange')
                )
            ))

        # Add BIAS_CHANGE exits
        if bias_exits_x:
            fig.add_trace(go.Scatter(
                x=bias_exits_x,
                y=bias_exits_y,
                mode='markers',
                name='Bias Change Exit',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='purple',
                    line=dict(width=2, color='darkviolet')
                )
            ))

        # Add SL exits
        if sl_exits_x:
            fig.add_trace(go.Scatter(
                x=sl_exits_x,
                y=sl_exits_y,
                mode='markers',
                name='SL Exit',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='black',
                    line=dict(width=2, color='darkred')
                )
            ))

    # Update layout
    fig.update_layout(
        title=f'Strategy Visualization ({df_subset["time"].iloc[0].date()} to {df_subset["time"].iloc[-1].date()})',
        xaxis_title='Time',
        yaxis_title='Price',
        height=800,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    # Show figure
    fig.show()


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
        'swing_lookback': SWING_LOOKBACK
    }

    # Run backtest
    print("Running backtest for visualization...")
    backtester, stats = run_backtest(df, symbol, INITIAL_BALANCE, strategy_params)

    # Create strategy instance for visualization
    strategy = TrendFollowingStrategy(**strategy_params)

    # Visualize first trading day (11:00-19:00 UTC = 12:00-20:00 Budapest)
    # Find first day with trading hours
    trading_data = df[(df['time'].dt.hour >= 11) & (df['time'].dt.hour < 19)]
    if len(trading_data) > 0:
        first_trading_day = trading_data.iloc[0]['time'].date()
        print(f"\nVisualizing day: {first_trading_day} (Trading hours: 11:00-19:00 UTC = 12:00-20:00 Budapest)")
        visualize_strategy(df, strategy, backtester, day=str(first_trading_day))
    else:
        print("No trading hours found in data!")
