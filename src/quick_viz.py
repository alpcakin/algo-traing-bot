"""
Quick visualization - shows a specific day without full backtest
"""

import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy import TrendFollowingStrategy
from config import *

# Load data
symbol = "EURUSD"
df = pd.read_csv(f"data/raw/{symbol}_M1_{START_DATE}_{END_DATE}.csv")
df['time'] = pd.to_datetime(df['time'])

# Pick a specific day to visualize (12:00-20:00 UTC)
DAY_TO_VISUALIZE = "2025-12-9"  # Change this to see different days

# Filter to specific day
day_mask = (df['time'].dt.date == pd.to_datetime(DAY_TO_VISUALIZE).date())
df_day = df[day_mask].copy()

print(f"Visualizing {DAY_TO_VISUALIZE}")
print(f"Total candles for day: {len(df_day)}")
print(f"Time range: {df_day['time'].min()} to {df_day['time'].max()}")

# Further filter to trading hours (12:00-20:00 UTC)
trading_mask = (df_day['time'].dt.hour >= 12) & (df_day['time'].dt.hour < 20)
df_trading = df_day[trading_mask].copy()

print(f"Trading hours candles: {len(df_trading)}")
print(f"Trading hours range: {df_trading['time'].min()} to {df_trading['time'].max()}")

# Create strategy
strategy = TrendFollowingStrategy(
    individual_tp_pips=INDIVIDUAL_TP_PIPS,
    risk_per_trade=RISK_PER_TRADE,
    emergency_sl_percent=EMERGENCY_SL_PERCENT,
    trading_hours=TRADING_HOURS,
    analysis_hours=ANALYSIS_HOURS,
    min_mitigation_distance_pips=MIN_MITIGATION_DISTANCE_PIPS,
    swing_lookback=SWING_LOOKBACK
)

# Track events
swing_highs = []
swing_lows = []
bias_changes = []
entries = []
ref_high_updates = []
ref_low_updates = []

prev_ref_high = None
prev_ref_low = None
prev_bias = strategy.bias

# Process day data
for i in range(len(df)):
    candle = df.iloc[i]

    # Check daily reset
    strategy.check_daily_reset(candle['time'])

    # Update swing levels
    strategy.update_swing_levels(df, i)

    # Track reference updates
    if strategy.reference_high != prev_ref_high and strategy.reference_high is not None:
        ref_high_updates.append({
            'time': candle['time'],
            'price': strategy.reference_high
        })
        prev_ref_high = strategy.reference_high

    if strategy.reference_low != prev_ref_low and strategy.reference_low is not None:
        ref_low_updates.append({
            'time': candle['time'],
            'price': strategy.reference_low
        })
        prev_ref_low = strategy.reference_low

    # Check bias change
    strategy.check_bias_change(df, i)
    if strategy.bias != prev_bias:
        bias_changes.append({
            'time': candle['time'],
            'bias': strategy.bias,
            'price': candle['close']
        })
        prev_bias = strategy.bias

    # Check entry
    if strategy.is_trading_hours(candle['time']) and strategy.should_enter(candle):
        if strategy.mitigation_high is not None and strategy.mitigation_low is not None:
            entries.append({
                'time': candle['time'],
                'price': candle['close'],
                'direction': strategy.bias,
                'mit_high': strategy.mitigation_high,
                'mit_low': strategy.mitigation_low
            })

# Get swing highs/lows from pending lists (after processing)
for price, idx in strategy.pending_swing_highs:
    if idx < len(df):
        swing_highs.append({
            'time': df.iloc[idx]['time'],
            'price': price
        })

for price, idx in strategy.pending_swing_lows:
    if idx < len(df):
        swing_lows.append({
            'time': df.iloc[idx]['time'],
            'price': price
        })

print(f"\nEvents found:")
print(f"  Swing Highs: {len(swing_highs)}")
print(f"  Swing Lows: {len(swing_lows)}")
print(f"  Bias Changes: {len(bias_changes)}")
print(f"  Entries: {len(entries)}")

# Create chart
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=df_day['time'],
    open=df_day['open'],
    high=df_day['high'],
    low=df_day['low'],
    close=df_day['close'],
    name='EURUSD'
))

# Swing highs
if swing_highs:
    sh_times = [s['time'] for s in swing_highs if s['time'] in df_day['time'].values]
    sh_prices = [s['price'] for s in swing_highs if s['time'] in df_day['time'].values]
    fig.add_trace(go.Scatter(
        x=sh_times, y=sh_prices,
        mode='markers',
        name='Swing High',
        marker=dict(symbol='triangle-down', size=15, color='red')
    ))

# Swing lows
if swing_lows:
    sl_times = [s['time'] for s in swing_lows if s['time'] in df_day['time'].values]
    sl_prices = [s['price'] for s in swing_lows if s['time'] in df_day['time'].values]
    fig.add_trace(go.Scatter(
        x=sl_times, y=sl_prices,
        mode='markers',
        name='Swing Low',
        marker=dict(symbol='triangle-up', size=15, color='green')
    ))

# Reference levels and mitigation zones REMOVED for performance
# (Was causing browser lag with too many updates)

# Bias changes (filter to day only)
bias_changes_day = [bc for bc in bias_changes if bc['time'] in df_day['time'].values]
if bias_changes_day:
    bc_times = [bc['time'] for bc in bias_changes_day]
    bc_prices = [bc['price'] for bc in bias_changes_day]
    bc_colors = ['green' if bc['bias'] == 'LONG' else 'red' for bc in bias_changes_day]

    fig.add_trace(go.Scatter(
        x=bc_times,
        y=bc_prices,
        mode='markers+text',
        name='Bias Change',
        marker=dict(symbol='diamond', size=15, color=bc_colors, line=dict(width=2, color='black')),
        text=[bc['bias'] for bc in bias_changes_day],
        textposition="top center"
    ))

# Entries
long_entries = [e for e in entries if e['direction'] == 'LONG' and e['time'] in df_day['time'].values]
short_entries = [e for e in entries if e['direction'] == 'SHORT' and e['time'] in df_day['time'].values]

if long_entries:
    fig.add_trace(go.Scatter(
        x=[e['time'] for e in long_entries],
        y=[e['price'] for e in long_entries],
        mode='markers',
        name='LONG Entry',
        marker=dict(symbol='arrow-up', size=12, color='lightgreen', line=dict(width=2, color='green'))
    ))

if short_entries:
    fig.add_trace(go.Scatter(
        x=[e['time'] for e in short_entries],
        y=[e['price'] for e in short_entries],
        mode='markers',
        name='SHORT Entry',
        marker=dict(symbol='arrow-down', size=12, color='lightcoral', line=dict(width=2, color='red'))
    ))

# Mark trading hours
fig.add_vrect(
    x0=df_trading['time'].min(),
    x1=df_trading['time'].max(),
    fillcolor="lightblue",
    opacity=0.1,
    annotation_text="Trading Hours (12:00-20:00 UTC)"
)

fig.update_layout(
    title=f'Strategy Analysis - {DAY_TO_VISUALIZE}',
    xaxis_title='Time (UTC)',
    yaxis_title='Price',
    height=900,
    xaxis_rangeslider_visible=False,
    hovermode='x unified'
)

# Save to HTML file instead of using temporary server
output_file = f"chart_{DAY_TO_VISUALIZE}.html"
fig.write_html(output_file)
print(f"\nChart saved to: {output_file}")
print("Opening in browser...")

# Open the file
import webbrowser
webbrowser.open(f'file://{os.path.abspath(output_file)}')
