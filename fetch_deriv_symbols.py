import requests
import logging

def fetch_deriv_symbols():
    """
    Fetches all active symbols from Deriv API that support rise/fall contracts.
    Excludes Jump, Boom, Crash, Step indices.
    Returns a list of valid symbol names.
    """
    url = "https://api.deriv.com/api/v1/active_symbols"
    params = {
        "product_type": "basic",  # Only basic (rise/fall) contracts
        "landing_company": "svg"  # SVG is the most common for synthetic indices
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        symbols = []
        for symbol in data.get("active_symbols", []):
            name = symbol.get("symbol")
            market = symbol.get("market")
            display_name = symbol.get("display_name", "")
            # Exclude Jump, Boom, Crash, Step indices
            if any(x in display_name.upper() for x in ["JUMP", "BOOM", "CRASH", "STEP"]):
                continue
            # Only include rise/fall tradeable
            if symbol.get("allow_forward_starting") or symbol.get("exchange_is_open"):
                symbols.append(name)
        return sorted(set(symbols))
    except Exception as e:
        logging.error(f"Failed to fetch Deriv symbols: {e}")
        return []

if __name__ == "__main__":
    valid_symbols = fetch_deriv_symbols()
    print("# Auto-generated Deriv rise/fall tradeable symbols (excluding Jump, Boom, Crash, Step):")
    print("SYMBOLS = [")
    for s in valid_symbols:
        print(f'    "{s}",')
    print("]")
