"""
Price Action Trend Following Strategy - SWING-BASED VERSION

Key Concepts:
- Daily Reset: 12:00-13:00 analysis period, fresh start every day
- Swing High/Low: Left 2 Right 2 candle pattern for reference levels
- Reference levels require BREAK (not equal/touch)
- Mitigation: Last counter-trend candle BEFORE a new swing high/low
- LONG bias: RED candle before new HIGH becomes mitigation
- SHORT bias: GREEN candle before new LOW becomes mitigation
- Bias changes when price CLOSES outside mitigation candle
- Entry: ALL candles when bias is active (color independent)
- Exit: TP hit, bias change, or end of day
- News Filter: No trading 15min before, 30min after high-impact news
- Scalp parameters: 0.1 pip min distance, 5 pip TP
"""

import pandas as pd
import numpy as np
from news_filter import NewsFilter


class TrendFollowingStrategy:
    """
    Swing-based trend following strategy with daily reset
    """

    def __init__(self, individual_tp_pips=5, risk_per_trade=0.0001,
                 emergency_sl_percent=0.02, trading_hours=(13, 20),
                 analysis_hours=(12, 13), min_mitigation_distance_pips=0.1,
                 swing_lookback=2, enable_news_filter=True,
                 news_buffer_before=15, news_buffer_after=30,
                 enable_volatility_filter=True, atr_period=14,
                 atr_multiplier=2.0):
        """
        Args:
            individual_tp_pips: TP in pips for each individual position (scalp)
            risk_per_trade: Risk per trade as fraction of balance (0.0001 = 0.01%)
            emergency_sl_percent: Emergency SL as fraction of balance (0.02 = 2%)
            trading_hours: Tuple of (start_hour, end_hour) - entries until 19:55, close at 20:00
            analysis_hours: Tuple of (start_hour, end_hour) for daily analysis
            min_mitigation_distance_pips: Minimum distance for mitigation update
            swing_lookback: Number of candles on each side for swing detection (default: 2)
            enable_news_filter: Enable/disable news filter
            news_buffer_before: Minutes before news to stop trading
            news_buffer_after: Minutes after news to stop trading
            enable_volatility_filter: Enable/disable volatility filter
            atr_period: Period for ATR calculation (default: 14)
            atr_multiplier: Multiplier for high volatility threshold (default: 2.0)
        """
        self.individual_tp_pips = individual_tp_pips
        self.risk_per_trade = risk_per_trade
        self.emergency_sl_percent = emergency_sl_percent
        self.trading_hours = trading_hours
        self.analysis_hours = analysis_hours
        self.min_mitigation_distance_pips = min_mitigation_distance_pips
        self.swing_lookback = swing_lookback
        self.enable_news_filter = enable_news_filter
        self.enable_volatility_filter = enable_volatility_filter
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

        # Initialize news filter
        if self.enable_news_filter:
            self.news_filter = NewsFilter(news_buffer_before, news_buffer_after)
            # Load hardcoded news for 2025-2026
            self.news_filter.load_hardcoded_news(2025)
            self.news_filter.load_hardcoded_news(2026)
            from news_filter import KNOWN_NEWS_2025_2026
            self.news_filter.load_custom_news(KNOWN_NEWS_2025_2026)
        else:
            self.news_filter = None

        # State variables
        self.bias = 'LONG'  # Start with LONG, will adjust during analysis
        self.mitigation_high = None
        self.mitigation_low = None
        self.mitigation_candle_idx = None

        # Reference high/low (swing-based)
        self.reference_high = None
        self.reference_low = None
        self.reference_high_idx = None
        self.reference_low_idx = None

        # Pending swing highs/lows (not yet confirmed)
        self.pending_swing_highs = []  # [(price, index), ...]
        self.pending_swing_lows = []   # [(price, index), ...]

        # Last analysis day tracking
        self.last_analysis_day = None


    def is_bullish_candle(self, row):
        """Check if candle is bullish (green)"""
        return row['close'] > row['open']


    def is_bearish_candle(self, row):
        """Check if candle is bearish (red)"""
        return row['close'] < row['open']


    def is_swing_high(self, df, idx):
        """
        Check if candle at idx is a swing high (left 2 right 2)
        Returns True if the HIGH at idx is greater than all surrounding candles
        """
        if idx < self.swing_lookback or idx >= len(df) - self.swing_lookback:
            return False

        center_high = df.iloc[idx]['high']

        # Check left candles
        for i in range(idx - self.swing_lookback, idx):
            if df.iloc[i]['high'] >= center_high:
                return False

        # Check right candles
        for i in range(idx + 1, idx + self.swing_lookback + 1):
            if df.iloc[i]['high'] >= center_high:
                return False

        return True


    def is_swing_low(self, df, idx):
        """
        Check if candle at idx is a swing low (left 2 right 2)
        Returns True if the LOW at idx is lower than all surrounding candles
        """
        if idx < self.swing_lookback or idx >= len(df) - self.swing_lookback:
            return False

        center_low = df.iloc[idx]['low']

        # Check left candles
        for i in range(idx - self.swing_lookback, idx):
            if df.iloc[i]['low'] <= center_low:
                return False

        # Check right candles
        for i in range(idx + 1, idx + self.swing_lookback + 1):
            if df.iloc[i]['low'] <= center_low:
                return False

        return True


    def find_last_counter_candle_before_index(self, df, before_idx, candle_type):
        """
        Find the last counter-trend candle before a specific index

        Args:
            before_idx: Search before this index
            candle_type: 'red' or 'green'

        Returns:
            (candle_data, index) or (None, None)
        """
        search_start = max(0, before_idx - 100)  # Look back max 100 candles

        for i in range(before_idx - 1, search_start - 1, -1):
            candle = df.iloc[i]

            if candle_type == 'red' and self.is_bearish_candle(candle):
                return candle, i
            elif candle_type == 'green' and self.is_bullish_candle(candle):
                return candle, i

        return None, None


    def reset_daily_state(self):
        """Reset state for new trading day"""
        self.bias = 'LONG'
        self.mitigation_high = None
        self.mitigation_low = None
        self.mitigation_candle_idx = None
        self.reference_high = None
        self.reference_low = None
        self.reference_high_idx = None
        self.reference_low_idx = None
        self.pending_swing_highs = []
        self.pending_swing_lows = []


    def is_analysis_period(self, timestamp):
        """Check if current time is in analysis period (12:00-13:00)"""
        hour = timestamp.hour
        return self.analysis_hours[0] <= hour < self.analysis_hours[1]


    def calculate_atr(self, df, current_idx):
        """Calculate ATR (Average True Range) for volatility filtering"""
        if current_idx < self.atr_period:
            return None

        true_ranges = []
        for i in range(current_idx - self.atr_period + 1, current_idx + 1):
            candle = df.iloc[i]
            if i == 0:
                tr = candle['high'] - candle['low']
            else:
                prev_close = df.iloc[i-1]['close']
                tr = max(
                    candle['high'] - candle['low'],
                    abs(candle['high'] - prev_close),
                    abs(candle['low'] - prev_close)
                )
            true_ranges.append(tr)

        return sum(true_ranges) / len(true_ranges)


    def is_high_volatility(self, df, current_idx):
        """Check if current volatility is too high (avoid consolidation breakouts)"""
        if not self.enable_volatility_filter:
            return False

        current_atr = self.calculate_atr(df, current_idx)
        if current_atr is None:
            return False

        # Calculate average ATR over longer period (50 candles)
        if current_idx < 50:
            return False

        atr_values = []
        for i in range(current_idx - 50, current_idx):
            atr = self.calculate_atr(df, i)
            if atr is not None:
                atr_values.append(atr)

        if not atr_values:
            return False

        avg_atr = sum(atr_values) / len(atr_values)

        # High volatility if current ATR > multiplier * average ATR
        return current_atr > (self.atr_multiplier * avg_atr)


    def is_trading_hours(self, timestamp):
        """
        Check if current time is within trading hours (13:00-19:55)
        AND not during news blackout period
        """
        hour = timestamp.hour
        minute = timestamp.minute

        # After 19:55, no new entries
        if hour == 19 and minute >= 55:
            return False

        # Check basic trading hours
        if not (self.trading_hours[0] <= hour < self.trading_hours[1]):
            return False

        # Check news filter
        if self.enable_news_filter and self.news_filter:
            is_news, event = self.news_filter.is_news_time(timestamp)
            if is_news:
                return False  # Block trading during news

        return True


    def should_close_all_positions(self, timestamp):
        """Check if it's time to close all positions (20:00)"""
        return timestamp.hour >= self.trading_hours[1]


    def check_daily_reset(self, timestamp):
        """
        Check if we need to reset for a new day
        Returns True if reset happened
        """
        current_day = timestamp.date()

        # If it's a new day and we're in analysis period
        if self.last_analysis_day != current_day and self.is_analysis_period(timestamp):
            self.reset_daily_state()
            self.last_analysis_day = current_day
            return True

        return False


    def update_swing_levels(self, df, current_idx):
        """
        Update swing high/low tracking
        Check for new swings and confirm pending ones

        ONLY during analysis period (12:00-13:00) and trading hours (13:00-20:00)
        """
        candle = df.iloc[current_idx]

        # Only update during analysis or trading hours
        hour = candle['time'].hour
        if not (self.analysis_hours[0] <= hour < self.trading_hours[1]):
            return

        # If we don't have initial references yet, initialize them with first swing
        if self.reference_high is None and self.pending_swing_highs:
            highest_swing = max(self.pending_swing_highs, key=lambda x: x[0])
            self.reference_high = highest_swing[0]
            self.reference_high_idx = highest_swing[1]

        if self.reference_low is None and self.pending_swing_lows:
            lowest_swing = min(self.pending_swing_lows, key=lambda x: x[0])
            self.reference_low = lowest_swing[0]
            self.reference_low_idx = lowest_swing[1]

        # Detect new swing high (need to wait 2 candles to confirm)
        if current_idx >= self.swing_lookback + 2:
            swing_idx = current_idx - self.swing_lookback
            if self.is_swing_high(df, swing_idx):
                swing_price = df.iloc[swing_idx]['high']
                # Add to pending if not already there
                if not any(idx == swing_idx for _, idx in self.pending_swing_highs):
                    self.pending_swing_highs.append((swing_price, swing_idx))

        # Detect new swing low (need to wait 2 candles to confirm)
        if current_idx >= self.swing_lookback + 2:
            swing_idx = current_idx - self.swing_lookback
            if self.is_swing_low(df, swing_idx):
                swing_price = df.iloc[swing_idx]['low']
                # Add to pending if not already there
                if not any(idx == swing_idx for _, idx in self.pending_swing_lows):
                    self.pending_swing_lows.append((swing_price, swing_idx))

        # Check if pending swing highs are confirmed (price broke reference low)
        if self.reference_low is not None and self.pending_swing_highs:
            # BREAK required: low must be LESS than reference_low (not equal)
            if candle['low'] < self.reference_low:
                # Find the highest pending swing high
                highest_swing = max(self.pending_swing_highs, key=lambda x: x[0])
                self.reference_high = highest_swing[0]
                self.reference_high_idx = highest_swing[1]
                self.pending_swing_highs = []  # Clear all pending

                # Update mitigation if in LONG bias
                if self.bias == 'LONG':
                    self.update_mitigation_for_new_high(df, self.reference_high_idx)

        # Check if pending swing lows are confirmed (price broke reference high)
        if self.reference_high is not None and self.pending_swing_lows:
            # BREAK required: high must be GREATER than reference_high (not equal)
            if candle['high'] > self.reference_high:
                # Find the lowest pending swing low
                lowest_swing = min(self.pending_swing_lows, key=lambda x: x[0])
                self.reference_low = lowest_swing[0]
                self.reference_low_idx = lowest_swing[1]
                self.pending_swing_lows = []  # Clear all pending

                # Update mitigation if in SHORT bias
                if self.bias == 'SHORT':
                    self.update_mitigation_for_new_low(df, self.reference_low_idx)


    def update_mitigation_for_new_high(self, df, high_idx):
        """Update mitigation when new reference high is confirmed"""
        # Find last RED candle before this high
        last_red, last_red_idx = self.find_last_counter_candle_before_index(df, high_idx + 1, 'red')

        if last_red is not None:
            # Update mitigation (no minimum distance check)
            self.mitigation_high = last_red['high']
            self.mitigation_low = last_red['low']
            self.mitigation_candle_idx = last_red_idx


    def update_mitigation_for_new_low(self, df, low_idx):
        """Update mitigation when new reference low is confirmed"""
        # Find last GREEN candle before this low
        last_green, last_green_idx = self.find_last_counter_candle_before_index(df, low_idx + 1, 'green')

        if last_green is not None:
            # Update mitigation (no minimum distance check)
            self.mitigation_high = last_green['high']
            self.mitigation_low = last_green['low']
            self.mitigation_candle_idx = last_green_idx


    def check_bias_change(self, df, current_idx):
        """
        Check if bias should change based on mitigation break

        LONG bias changes if price CLOSES below mitigation LOW
        SHORT bias changes if price CLOSES above mitigation HIGH

        ONLY during analysis period (12:00-13:00) and trading hours (13:00-20:00)

        Returns: True if bias changed, False otherwise
        """
        candle = df.iloc[current_idx]

        if self.mitigation_high is None or self.mitigation_low is None:
            return False

        # Only allow bias change during analysis or trading hours
        hour = candle['time'].hour
        if not (self.analysis_hours[0] <= hour < self.trading_hours[1]):
            return False

        changed = False

        if self.bias == 'LONG':
            # LONG bias changes if price CLOSES below mitigation LOW (BREAK, not equal)
            if candle['close'] < self.mitigation_low:
                # Bias changes to SHORT
                self.bias = 'SHORT'

                # New mitigation = last green candle BEFORE the breaking candle (not including it)
                last_green, last_green_idx = self.find_last_counter_candle_before_index(df, current_idx, 'green')

                if last_green is not None:
                    self.mitigation_high = last_green['high']
                    self.mitigation_low = last_green['low']
                    self.mitigation_candle_idx = last_green_idx
                else:
                    # Fallback: use current candle
                    self.mitigation_high = candle['high']
                    self.mitigation_low = candle['low']
                    self.mitigation_candle_idx = current_idx

                changed = True

        elif self.bias == 'SHORT':
            # SHORT bias changes if price CLOSES above mitigation HIGH (BREAK, not equal)
            if candle['close'] > self.mitigation_high:
                # Bias changes to LONG
                self.bias = 'LONG'

                # New mitigation = last red candle BEFORE the breaking candle (not including it)
                last_red, last_red_idx = self.find_last_counter_candle_before_index(df, current_idx, 'red')

                if last_red is not None:
                    self.mitigation_high = last_red['high']
                    self.mitigation_low = last_red['low']
                    self.mitigation_candle_idx = last_red_idx
                else:
                    # Fallback: use current candle
                    self.mitigation_high = candle['high']
                    self.mitigation_low = candle['low']
                    self.mitigation_candle_idx = current_idx

                changed = True

        return changed


    def should_enter(self, candle):
        """
        Check if we should enter a trade on this candle
        Entry: ALL candles when bias is active (color independent)
        """
        if self.bias in ['LONG', 'SHORT']:
            return True
        return False


    def get_entry_price(self, candle):
        """Get entry price (close of the candle)"""
        return candle['close']


    def get_sl_price(self):
        """Get stop loss price (mitigation boundary)"""
        if self.bias == 'LONG':
            return self.mitigation_low
        else:
            return self.mitigation_high


    def calculate_position_size(self, balance, entry_price, symbol):
        """
        Calculate position size based on risk
        Risk = balance * risk_per_trade
        SL = mitigation boundary
        """
        sl_price = self.get_sl_price()

        if sl_price is None:
            return 0

        # Calculate SL distance in price
        sl_distance = abs(entry_price - sl_price)

        if sl_distance == 0:
            return 0

        # Risk amount in dollars
        risk_amount = balance * self.risk_per_trade

        # Pip size
        if 'JPY' in symbol:
            pip_size = 0.01
        else:
            pip_size = 0.0001

        pip_value_per_lot = 10  # Standard lot

        # SL distance in pips
        sl_pips = sl_distance / pip_size

        # Lot size
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        lot_size = round(lot_size, 2)

        # Minimum lot size
        if lot_size < 0.01:
            lot_size = 0.01

        return lot_size


def generate_signals(df, strategy):
    """Generate trading signals with daily reset"""
    signals = []

    print(f"Starting signal generation...")

    # Process each candle
    for i in range(0, len(df)):
        candle = df.iloc[i]

        # Check for daily reset
        if strategy.check_daily_reset(candle['time']):
            signals.append({
                'index': i,
                'time': candle['time'],
                'type': 'DAILY_RESET',
                'price': candle['close']
            })

        # Update swing levels
        strategy.update_swing_levels(df, i)

        # Check for bias change
        bias_changed = strategy.check_bias_change(df, i)

        if bias_changed:
            signals.append({
                'index': i,
                'time': candle['time'],
                'type': 'BIAS_CHANGE',
                'new_bias': strategy.bias,
                'mitigation_high': strategy.mitigation_high,
                'mitigation_low': strategy.mitigation_low,
                'price': candle['close']
            })

        # Check entry (only during trading hours, not analysis period)
        if strategy.is_trading_hours(candle['time']) and strategy.should_enter(candle):
            if strategy.mitigation_high is not None and strategy.mitigation_low is not None:
                entry_price = strategy.get_entry_price(candle)

                signal = {
                    'index': i,
                    'time': candle['time'],
                    'type': 'ENTRY',
                    'direction': strategy.bias,
                    'entry_price': entry_price,
                    'mitigation_high': strategy.mitigation_high,
                    'mitigation_low': strategy.mitigation_low,
                    'sl': strategy.get_sl_price()
                }

                signals.append(signal)

    return signals
