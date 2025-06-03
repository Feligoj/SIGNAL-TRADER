# config.py — centralized configuration for signal bot

# Deriv API credentials
APP_ID = "1089"
API_TOKEN = "YG9Nou7uTuC0HWo"

# Symbols to analyze
from tradeable_symbols import SYMBOLS as ALL_SYMBOLS

# Toggle analysis of currency pairs (forex)
ANALYZE_FOREX_PAIRS = False  # Set to True to enable forex analysis

if ANALYZE_FOREX_PAIRS:
    SYMBOLS = ALL_SYMBOLS
else:
    SYMBOLS = [s for s in ALL_SYMBOLS if not s.startswith("frx")]

# Strict mode pairs
STRICT_MODE_PAIRS = []

# Cooldown durations (in minutes)
DEFAULT_COOLDOWN = 15       # Used for normal mode pairs
STRICT_COOLDOWN = 20        # Used for strict mode pairs

# --- Minimum score enforcement ---
# Only use these for signal filtering! Do NOT use MIN_SCORE, MIN_SCORE_TO_SEND, or STRICT_MIN_SCORE for signal logic.
MIN_SCORE_FOREX = 75      # Minimum score for forex pairs (stricter)
MIN_SCORE_NON_FOREX = 55 # Minimum score for non-forex pairs (stricter)

# Enable test mode (simulates alerts, doesn't execute trades)
TEST_MODE = False

# Filter toggles
USE_ADX_FILTER = False       # Blocks signals if ADX is too strong
USE_FLAT_FILTER = False      # Blocks signals in low-volatility ranges  

# Thresholds for filters
ADX_MIN = 0
ADX_MAX = 30
ADX_STRONG_TREND = 31
FLAT_VOL_RATIO = 0.7        # Recent vol < 70% of average = flat market

# --- Diagnostic and Tuning Options ---
# 1. Enable detailed logging of blocked signals
LOG_BLOCKED_SIGNALS = True
# 2. Volatility threshold for trade filter (default 0.7)
VOLATILITY_THRESHOLD = 0.7
# 3. Number of confirmations required (default 2)
REQUIRED_CONFIRMATIONS = 0  # Set to zero for now, user will adjust later
REQUIRED_CONFIRMATIONS_MAP = {
    # Per-symbol or group overrides
    # Volatility indices (VIX-like)
    "R_10": 0,
    "R_25": 0,
    "R_75": 0,
    "1HZ10V": 0,
    "1HZ25V": 0,
    "1HZ50V": 0,
    "1HZ75V": 0,
    "1HZ100V": 0,
    # Forex pairs removed
    # Strict mode pairs (if any)
    # "SOME_STRICT_PAIR": 0,
}
# 4. Allow single strong reversal pattern to override confirmations (default False)
ALLOW_STRONG_PATTERN_OVERRIDE = True  # Allow strong pattern to override confirmations
# 5. Enable/disable secondary (lower confidence) signal tier (default False)
ENABLE_SECONDARY_SIGNAL = False
# 6. Enable/disable volatility expansion filter (default False)
USE_VOLATILITY_FILTER = False
# 7. Enable trend-following logic per symbol/group (default True for trending pairs, False for mean-reverting)
TREND_FOLLOWING_ENABLED_MAP = {
    # Volatility indices (VIX-like)
    "R_10": False,
    "R_25": False,
    "R_75": True,   # R_75 can trend, enable trend-following
    "1HZ10V": False,
    "1HZ25V": False,
    "1HZ50V": True,   # 1HZ50V can trend
    "1HZ75V": True,   # 1HZ75V can trend
    "1HZ100V": True,  # 1HZ100V can trend
    # Forex pairs removed
    # Strict mode pairs (if any)
    # "SOME_STRICT_PAIR": True,
}

# --- Multi-Timeframe Confirmation (MTF) ---
ENABLE_MTF_CONFIRMATION = True  # Enabled by user request
MTF_CONFIRMATION_PAIRS = [
    "R_75", "1HZ50V", "1HZ75V"  # Whipsaw-prone indices
    # Forex pairs removed
]

# --- Adaptive Duration Settings ---
ADAPTIVE_MIN_DURATION = 15
ADAPTIVE_MAX_DURATION = 20
ADAPTIVE_DURATION_WEIGHTS = {
    'score': 1/12,
    'confirmations': 0.7,
    'range': 1.0,
    'distance': 1.0,
    'counter_trend': 3.0
}

# --- Machine Learning Signal Scoring ---
USE_ML_SCORING = False  # Set to True to enable ML-based scoring (requires model and data)