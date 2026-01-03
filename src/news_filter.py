"""
High-Impact News Filter
Blocks trading around major economic news events
"""

import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time


class NewsFilter:
    """Filter trades around high-impact news events"""

    def __init__(self, buffer_minutes_before=15, buffer_minutes_after=30):
        """
        Args:
            buffer_minutes_before: Minutes before news to stop trading
            buffer_minutes_after: Minutes after news to stop trading
        """
        self.buffer_before = buffer_minutes_before
        self.buffer_after = buffer_minutes_after
        self.news_events = []  # List of (datetime, event_name)

    def load_hardcoded_news(self, year=2025):
        """
        Load known recurring high-impact news times
        These are approximate - actual dates may vary
        """
        news = []

        # NFP (First Friday of each month at 13:30 UTC)
        for month in range(1, 13):
            # Find first Friday
            first_day = datetime(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            if days_until_friday == 0 and first_day.weekday() != 4:
                days_until_friday = 7
            nfp_date = first_day + timedelta(days=days_until_friday)
            news.append((nfp_date.replace(hour=13, minute=30), "NFP"))

        # CPI (Usually around 13th of month at 13:30 UTC - approximate)
        for month in range(1, 13):
            cpi_date = datetime(year, month, 13, 13, 30)
            news.append((cpi_date, "CPI"))

        # FOMC (8 times per year - approximate dates)
        # Typically: Jan, Mar, May, Jun, Jul, Sep, Nov, Dec
        fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
        for month in fomc_months:
            # Usually mid-month, 19:00 UTC
            fomc_date = datetime(year, month, 15, 19, 0)
            news.append((fomc_date, "FOMC"))

        self.news_events.extend(news)
        return len(news)

    def scrape_forexfactory(self, start_date, end_date):
        """
        Scrape Forex Factory calendar for high-impact news
        WARNING: This may be blocked or rate-limited

        Args:
            start_date: datetime object
            end_date: datetime object
        """
        # Note: This is a simplified example
        # Real implementation would need to handle pagination, rate limits, etc.

        print("Note: Forex Factory scraping not implemented for backtest")
        print("Using hardcoded news events instead")
        return 0

    def is_news_time(self, timestamp):
        """
        Check if timestamp is within news blackout period

        Args:
            timestamp: pandas Timestamp or datetime

        Returns:
            (is_blackout, event_name or None)
        """
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        for news_time, event_name in self.news_events:
            # Check if within buffer zone
            time_diff = (timestamp - news_time).total_seconds() / 60  # minutes

            # Before news: -buffer_before to 0
            # After news: 0 to +buffer_after
            if -self.buffer_before <= time_diff <= self.buffer_after:
                return True, event_name

        return False, None

    def get_next_news(self, current_time):
        """Get next upcoming news event"""
        if isinstance(current_time, pd.Timestamp):
            current_time = current_time.to_pydatetime()

        upcoming = [(t, name) for t, name in self.news_events if t > current_time]
        if upcoming:
            upcoming.sort(key=lambda x: x[0])
            return upcoming[0]
        return None, None

    def get_news_in_range(self, start_date, end_date):
        """Get all news events in date range"""
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.to_pydatetime()
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.to_pydatetime()

        events = [(t, name) for t, name in self.news_events
                  if start_date <= t <= end_date]
        events.sort(key=lambda x: x[0])
        return events

    def load_custom_news(self, news_list):
        """
        Load custom news events

        Args:
            news_list: List of tuples (datetime, event_name)
        """
        self.news_events.extend(news_list)

    def clear_news(self):
        """Clear all loaded news events"""
        self.news_events = []


# Specific high-impact news for December 2025
# Source: Forex Factory Calendar (provided by user)
# TIMES IN UTC (Forex Factory shows UTC, converted to match MT5 data timezone)
# Only HIGH IMPACT (red/3-star) events included
KNOWN_NEWS_2025_2026 = [
    # December 2025 - High Impact Events Only (all times +2 hours corrected)
    (datetime(2025, 12, 1, 17, 0), "ISM Manufacturing PMI"),  # 4:00pm UTC on ForexFactory
    (datetime(2025, 12, 3, 15, 15), "ADP Non-Farm Employment Change"),  # 2:15pm UTC
    (datetime(2025, 12, 3, 17, 0), "ISM Services PMI"),  # 4:00pm UTC
    (datetime(2025, 12, 4, 15, 30), "Unemployment Claims"),  # 2:30pm UTC
    (datetime(2025, 12, 5, 17, 0), "Core PCE Price Index"),  # 4:00pm UTC
    (datetime(2025, 12, 5, 17, 0), "Prelim UoM Consumer Sentiment"),  # 4:00pm UTC
    (datetime(2025, 12, 5, 17, 0), "Prelim UoM Inflation Expectations"),  # 4:00pm UTC
    (datetime(2025, 12, 9, 15, 15), "ADP Weekly Employment Change"),  # 2:15pm UTC
    (datetime(2025, 12, 9, 17, 0), "JOLTS Job Openings"),  # 4:00pm UTC
    (datetime(2025, 12, 10, 15, 30), "Employment Cost Index"),  # 2:30pm UTC
    (datetime(2025, 12, 10, 21, 0), "Federal Funds Rate"),  # 8:00pm UTC - MAJOR
    (datetime(2025, 12, 10, 21, 0), "FOMC Economic Projections"),  # 8:00pm UTC - MAJOR
    (datetime(2025, 12, 10, 21, 0), "FOMC Statement"),  # 8:00pm UTC - MAJOR
    (datetime(2025, 12, 10, 21, 30), "FOMC Press Conference"),  # 8:30pm UTC - MAJOR
    (datetime(2025, 12, 11, 15, 30), "Unemployment Claims"),  # 2:30pm UTC
    (datetime(2025, 12, 12, 9, 0), "UK GDP m/m"),  # 8:00am UTC
    (datetime(2025, 12, 16, 9, 0), "UK Claimant Count Change"),  # 8:00am UTC
    (datetime(2025, 12, 16, 10, 30), "German Flash Manufacturing PMI"),  # 9:30am UTC
    (datetime(2025, 12, 16, 10, 30), "German Flash Services PMI"),  # 9:30am UTC
    (datetime(2025, 12, 16, 11, 30), "UK Flash Manufacturing PMI"),  # 10:30am UTC
    (datetime(2025, 12, 16, 11, 30), "UK Flash Services PMI"),  # 10:30am UTC
    (datetime(2025, 12, 16, 15, 15), "ADP Weekly Employment Change"),  # 2:15pm UTC
    (datetime(2025, 12, 16, 15, 30), "Average Hourly Earnings m/m"),  # 2:30pm UTC - MAJOR
    (datetime(2025, 12, 16, 15, 30), "Core Retail Sales m/m"),  # 2:30pm UTC
    (datetime(2025, 12, 16, 15, 30), "Non-Farm Employment Change (NFP)"),  # 2:30pm UTC - MAJOR
    (datetime(2025, 12, 16, 15, 30), "Retail Sales m/m"),  # 2:30pm UTC
    (datetime(2025, 12, 16, 15, 30), "Unemployment Rate"),  # 2:30pm UTC - MAJOR
    (datetime(2025, 12, 16, 16, 45), "Flash Manufacturing PMI"),  # 3:45pm UTC
    (datetime(2025, 12, 16, 16, 45), "Flash Services PMI"),  # 3:45pm UTC
    (datetime(2025, 12, 17, 9, 0), "UK CPI y/y"),  # 8:00am UTC
    (datetime(2025, 12, 18, 14, 0), "BOE Official Bank Rate"),  # 1:00pm UTC - MAJOR
    (datetime(2025, 12, 18, 14, 0), "MPC Official Bank Rate Votes"),  # 1:00pm UTC - MAJOR
    (datetime(2025, 12, 18, 14, 0), "Monetary Policy Summary"),  # 1:00pm UTC - MAJOR
    (datetime(2025, 12, 18, 15, 15), "ECB Main Refinancing Rate"),  # 2:15pm UTC - MAJOR
    (datetime(2025, 12, 18, 15, 15), "ECB Monetary Policy Statement"),  # 2:15pm UTC - MAJOR
    (datetime(2025, 12, 18, 15, 30), "US CPI y/y"),  # 2:30pm UTC - MAJOR
    (datetime(2025, 12, 18, 15, 30), "Unemployment Claims"),  # 2:30pm UTC
    (datetime(2025, 12, 18, 15, 45), "ECB Press Conference"),  # 2:45pm UTC - MAJOR
    (datetime(2025, 12, 19, 9, 0), "UK Retail Sales m/m"),  # 8:00am UTC
    (datetime(2025, 12, 23, 15, 30), "Prelim GDP q/q"),  # 2:30pm UTC
    (datetime(2025, 12, 24, 15, 30), "Unemployment Claims"),  # 2:30pm UTC
    (datetime(2025, 12, 30, 21, 0), "FOMC Meeting Minutes"),  # 8:00pm UTC
    (datetime(2025, 12, 31, 15, 30), "Unemployment Claims"),  # 2:30pm UTC
]


if __name__ == "__main__":
    # Test the news filter
    news_filter = NewsFilter(buffer_minutes_before=15, buffer_minutes_after=30)

    # Load hardcoded events
    count = news_filter.load_hardcoded_news(2025)
    print(f"Loaded {count} hardcoded news events for 2025")

    # Load specific known events
    news_filter.load_custom_news(KNOWN_NEWS_2025_2026)
    print(f"Loaded {len(KNOWN_NEWS_2025_2026)} specific news events")

    # Test a specific time
    test_time = datetime(2025, 12, 6, 13, 20)  # 10 min before NFP
    is_blackout, event = news_filter.is_news_time(test_time)
    print(f"\nTest time: {test_time}")
    print(f"Is blackout: {is_blackout}")
    if event:
        print(f"Event: {event}")

    # Get news in date range
    start = datetime(2025, 12, 1)
    end = datetime(2025, 12, 31)
    events = news_filter.get_news_in_range(start, end)
    print(f"\nNews events in December 2025:")
    for event_time, event_name in events:
        print(f"  {event_time}: {event_name}")
