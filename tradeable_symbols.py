# List of all Deriv tradeable rise/fall indices and major currencies, excluding Jump, Boom, Crash, Step indices
SYMBOLS = [
    # Volatility Indices (Rise/Fall tradeable)
    "R_10", "R_25", "R_50", "R_75",
    "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V",
    # Major Currencies (Rise/Fall tradeable)
    "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCHF", "frxUSDCAD",
    "frxEURGBP", "frxEURJPY", "frxGBPJPY", "frxAUDJPY", "frxEURAUD"
]

# Strict mode pairs
STRICT_MODE_PAIRS = []

# Confirmation settings
REQUIRED_CONFIRMATIONS = 1  # Default confirmations for all pairs (can be changed)

# ...existing code...
