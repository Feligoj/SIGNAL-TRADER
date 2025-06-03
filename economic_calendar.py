import requests
import datetime

def fetch_economic_events():
    """
    Fetches today's high-impact economic events from a public API (e.g., Forex Factory, or use a placeholder if no free API).
    Returns a list of dicts: {currency, impact, time_utc, title}
    """
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"  # Unofficial ForexFactory JSON
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        today = datetime.datetime.utcnow().strftime("%b %d, %Y")
        events = []
        for event in data:
            if event.get("impact") == "High" and event.get("date") == today:
                events.append({
                    "currency": event.get("currency"),
                    "impact": event.get("impact"),
                    "time_utc": event.get("time"),
                    "title": event.get("title")
                })
        return events
    except Exception:
        # Suppress all errors and do not log anything for economic news API
        return []

def is_high_impact_news(symbol, window_minutes=30):
    """
    Checks if there is a high-impact event for the given currency symbol within +/- window_minutes of now.
    Fails open: If any error occurs (API/network/parse), always returns (False, None) and does not log.
    """
    try:
        # Map Deriv symbol to currency code
        symbol_map = {
            "frxEURUSD": ["EUR", "USD"],
            "frxGBPUSD": ["GBP", "USD"],
            "frxUSDJPY": ["USD", "JPY"],
            "frxAUDUSD": ["AUD", "USD"],
            "frxUSDCHF": ["USD", "CHF"],
            "frxUSDCAD": ["USD", "CAD"],
            "frxEURGBP": ["EUR", "GBP"],
            "frxEURJPY": ["EUR", "JPY"],
            "frxGBPJPY": ["GBP", "JPY"],
            "frxAUDJPY": ["AUD", "JPY"],
            "frxEURAUD": ["EUR", "AUD"]
        }
        if symbol not in symbol_map:
            return False, None
        events = fetch_economic_events()
        if not events:
            # Fail open: No events fetched, allow signal (no log)
            return False, None
        now = datetime.datetime.utcnow()
        for event in events:
            if event["currency"] in symbol_map[symbol]:
                # Parse event time (e.g., '13:30')
                try:
                    event_time = datetime.datetime.strptime(event["time_utc"], "%H:%M")
                    event_time = now.replace(hour=event_time.hour, minute=event_time.minute, second=0, microsecond=0)
                    delta = abs((event_time - now).total_seconds()) / 60
                    if delta <= window_minutes:
                        return True, event
                except Exception:
                    continue
        return False, None
    except Exception:
        # Suppress all errors and do not log anything for economic news API
        return False, None

if __name__ == "__main__":
    # Example usage
    symbol = "frxEURUSD"
    is_news, event = is_high_impact_news(symbol)
    if is_news:
        print(f"High-impact news for {symbol}: {event}")
    else:
        print(f"No high-impact news for {symbol} in the next 30 minutes.")
