import pandas as pd
import numpy as np
import logging
import datetime
from config import (
    USE_ADX_FILTER, USE_FLAT_FILTER,
    ADX_STRONG_TREND, FLAT_VOL_RATIO,
    ADX_MIN, ADX_MAX, LOG_BLOCKED_SIGNALS, VOLATILITY_THRESHOLD,
    REQUIRED_CONFIRMATIONS, ALLOW_STRONG_PATTERN_OVERRIDE, ENABLE_SECONDARY_SIGNAL,
    SYMBOLS, STRICT_MODE_PAIRS,
    ENABLE_MTF_CONFIRMATION, MTF_CONFIRMATION_PAIRS,
    MIN_SCORE_FOREX, MIN_SCORE_NON_FOREX  # <-- Add these
)
from economic_calendar import is_high_impact_news

logger = logging.getLogger(__name__)

RSI_OVERSOLD = 20    # Only buy when extremely oversold
RSI_OVERBOUGHT = 80  # Only sell when extremely overbought

class TechnicalAnalysis:
    # --- Class-level shared variables ---
    forex = [
        "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCHF", "frxUSDCAD",
        "frxEURGBP", "frxEURJPY", "frxGBPJPY", "frxAUDJPY", "frxEURAUD"
    ]
    # Economic news cache (3 hour refresh)
    _news_cache = {'timestamp': None, 'data': None}

    @staticmethod
    def get_required_confirmations(symbol):
        """
        Returns the required number of confirmations for a given symbol.
        Uses REQUIRED_CONFIRMATIONS_MAP from config if available, else falls back to REQUIRED_CONFIRMATIONS.
        """
        from config import REQUIRED_CONFIRMATIONS_MAP, REQUIRED_CONFIRMATIONS
        return REQUIRED_CONFIRMATIONS_MAP.get(symbol, REQUIRED_CONFIRMATIONS)

    @staticmethod
    def get_pair_settings(symbol):
        """
        Returns a dictionary of indicator settings and weights for the given symbol.
        Handles symbol-specific tweaks, vix-like, forex, strict mode, and non-forex logic.
        """
        vix_like = [
            "R_10", "R_25", "R_50", "R_75",
            "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V"
        ]
        forex = TechnicalAnalysis.forex
        symbol_tweaks = {
            "R_50": {
                'rsi_period': 7,
                'rsi_oversold': 12,
                'rsi_overbought': 88,
                'ema_fast': 3,
                'ema_slow': 10,
                'macd_fast': 8,
                'macd_slow': 16,
                'macd_signal': 5,
                'stoch_k': 8,
                'stoch_d': 2,
                'adx_period': 7,
                'required_confirmations': TechnicalAnalysis.get_required_confirmations("R_50"),
                'score_weights': {
                    'macd': 0.30,  # Increased
                    'rsi': 0.22,   # Increased
                    'stoch': 0.15, # Increased
                    'candlestick': 0.12, # Increased
                    'support_resistance': 0.12, # Increased
                    'bollinger': 0.22, # Increased
                    'adx': 0.12, # Increased
                    'divergence': 0.12 # Increased
                }
            },
            "frxUSDJPY": {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_fast': 14,
                'ema_slow': 40,
                'macd_fast': 16,
                'macd_slow': 32,
                'macd_signal': 12,
                'stoch_k': 14,
                'stoch_d': 3,
                'adx_period': 20,
                'required_confirmations': TechnicalAnalysis.get_required_confirmations("frxUSDJPY"),
                'score_weights': {
                    'macd': 0.30,
                    'rsi': 0.22,
                    'stoch': 0.15,
                    'candlestick': 0.12,
                    'support_resistance': 0.12,
                    'bollinger': 0.22,
                    'adx': 0.12,
                    'divergence': 0.12
                }
            }
        }
        strict_mode = STRICT_MODE_PAIRS if STRICT_MODE_PAIRS else []
        if symbol in symbol_tweaks:
            tweaks = symbol_tweaks[symbol]
            tweaks['required_confirmations'] = TechnicalAnalysis.get_required_confirmations(symbol)
            return tweaks
        if symbol in vix_like:
            return {
                'rsi_period': 7,
                'rsi_oversold': 15,
                'rsi_overbought': 85,
                'ema_fast': 3,
                'ema_slow': 10,
                'macd_fast': 8,
                'macd_slow': 16,
                'macd_signal': 5,
                'stoch_k': 8,
                'stoch_d': 2,
                'adx_period': 7,
                'required_confirmations': TechnicalAnalysis.get_required_confirmations(symbol),
                'score_weights': {
                    'macd': 0.30,
                    'rsi': 0.22,
                    'stoch': 0.15,
                    'candlestick': 0.12,
                    'support_resistance': 0.12,
                    'bollinger': 0.22,
                    'adx': 0.12,
                    'divergence': 0.12
                }
            }
        elif symbol in forex:
            return {
                'rsi_period': 14,
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'ema_fast': 10,
                'ema_slow': 30,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'stoch_k': 14,
                'stoch_d': 3,
                'adx_period': 14,
                'required_confirmations': TechnicalAnalysis.get_required_confirmations(symbol),
                'score_weights': {
                    'macd': 0.30,
                    'rsi': 0.22,
                    'stoch': 0.15,
                    'candlestick': 0.12,
                    'support_resistance': 0.12,
                    'bollinger': 0.22,
                    'adx': 0.12,
                    'divergence': 0.12
                }
            }
        elif symbol in strict_mode:
            return {
                'rsi_period': 14,
                'rsi_oversold': 20,
                'rsi_overbought': 80,
                'ema_fast': 5,
                'ema_slow': 20,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'stoch_k': 14,
                'stoch_d': 3,
                'adx_period': 14,
                'required_confirmations': TechnicalAnalysis.get_required_confirmations(symbol),
                'score_weights': {
                    'macd': 0.30,
                    'rsi': 0.22,
                    'stoch': 0.15,
                    'candlestick': 0.12,
                    'support_resistance': 0.12,
                    'bollinger': 0.22,
                    'adx': 0.12,
                    'divergence': 0.12
                },
                'min_score': STRICT_MIN_SCORE
            }
        elif symbol not in forex:
            # Looser settings for non-forex (reversal signals)
            return {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_fast': 7,
                'ema_slow': 21,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'stoch_k': 14,
                'stoch_d': 3,
                'adx_period': 14,
                'required_confirmations': 1,
                'score_weights': {
                    'macd': 0.30,
                    'rsi': 0.22,
                    'stoch': 0.15,
                    'candlestick': 0.12,
                    'support_resistance': 0.12,
                    'bollinger': 0.22,
                    'adx': 0.12,
                    'divergence': 0.12
                }
            }
        else:
            # Default: balanced
            return {
                'rsi_period': 14,
                'rsi_oversold': 20,
                'rsi_overbought': 80,
                'ema_fast': 5,
                'ema_slow': 20,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'stoch_k': 14,
                'stoch_d': 3,
                'adx_period': 14,
                'required_confirmations': TechnicalAnalysis.get_required_confirmations(symbol),
                'score_weights': {
                    'macd': 0.30,
                    'rsi': 0.22,
                    'stoch': 0.15,
                    'candlestick': 0.12,
                    'support_resistance': 0.12,
                    'bollinger': 0.22,
                    'adx': 0.12,
                    'divergence': 0.12
                }
            }

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Calculate all required indicators for the given DataFrame.
        Returns the DataFrame with new indicator columns.
        Handles edge cases and returns df unchanged if invalid.
        """
        required_cols = {'open', 'high', 'low', 'close'}
        if df is None or df.empty or not required_cols.issubset(df.columns):
            logger.warning("calculate_indicators: DataFrame is empty or missing OHLC columns.")
            return df
        try:
            settings = TechnicalAnalysis.get_pair_settings(symbol or "")
            df['rsi'] = TechnicalAnalysis.compute_rsi(df['close'], period=settings['rsi_period'])
            df['ema_fast'] = df['close'].ewm(span=settings['ema_fast']).mean()
            df['ema_slow'] = df['close'].ewm(span=settings['ema_slow']).mean()
            df['macd_line'] = df['close'].ewm(span=settings['macd_fast']).mean() - df['close'].ewm(span=settings['macd_slow']).mean()
            df['macd_signal'] = df['macd_line'].ewm(span=settings['macd_signal']).mean()
            df['macd_hist'] = df['macd_line'] - df['macd_signal']
            df['upper_bb'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
            df['lower_bb'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()
            df['stoch_k'], df['stoch_d'] = TechnicalAnalysis.compute_stochastic(df, k_period=settings['stoch_k'], d_period=settings['stoch_d'])
            df['adx'] = TechnicalAnalysis.compute_adx(df, period=settings['adx_period'])
            df['tr'] = TechnicalAnalysis.compute_true_range(df)
            df['volatility'] = df['tr']
        except Exception as e:
            logger.warning(f"calculate_indicators: Exception during indicator calculation: {e}")
        return df

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI).
        Args:
            series: Price series (close).
            period: Lookback period.
        Returns:
            RSI values as pd.Series.
        Raises:
            ValueError if input is invalid.
        """
        if series is None or series.empty:
            raise ValueError("RSI: Series cannot be empty")
        if len(series) < period:
            raise ValueError(f"RSI: Series length ({len(series)}) < period ({period})")
        if series.isna().any():
            raise ValueError("RSI: Series contains NaN values")
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
        """
        Calculate Stochastic Oscillator (K, D).
        Args:
            df: DataFrame with 'high', 'low', 'close'.
            k_period: Lookback for K.
            d_period: Lookback for D.
        Returns:
            Tuple of (stoch_k, stoch_d) as pd.Series.
        Raises:
            ValueError if input is invalid.
        """
        if df is None or df.empty:
            raise ValueError("Stochastic: DataFrame cannot be empty")
        for col in ['high', 'low', 'close']:
            if col not in df.columns:
                raise ValueError(f"Stochastic: Missing column {col}")
        if len(df) < k_period:
            raise ValueError(f"Stochastic: DataFrame length ({len(df)}) < k_period ({k_period})")
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d

    @staticmethod
    def compute_true_range(df: pd.DataFrame) -> pd.Series:
        """
        Calculate True Range (TR).
        Args:
            df: DataFrame with 'high', 'low', 'close'.
        Returns:
            pd.Series of true range values.
        Raises:
            ValueError if input is invalid.
        """
        if df is None or df.empty:
            raise ValueError("TrueRange: DataFrame cannot be empty")
        for col in ['high', 'low', 'close']:
            if col not in df.columns:
                raise ValueError(f"TrueRange: Missing column {col}")
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    @staticmethod
    def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        Args:
            df: DataFrame with 'high', 'low', 'close'.
            period: Lookback window.
        Returns:
            pd.Series of ADX values.
        Raises:
            ValueError if input is invalid.
        """
        if df is None or df.empty:
            raise ValueError("ADX: DataFrame cannot be empty")
        for col in ['high', 'low', 'close']:
            if col not in df.columns:
                raise ValueError(f"ADX: Missing column {col}")
        high = df['high']
        low = df['low']
        close = df['close']
        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        tr = TechnicalAnalysis.compute_true_range(df)
        atr = tr.ewm(span=period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
        adx = dx.ewm(span=period, min_periods=period).mean()
        return adx

    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, lookback: int = 100) -> tuple:
        """
        Detect support and resistance levels using rolling min/max over a lookback window.
        Args:
            df: DataFrame with 'high', 'low'.
            lookback: Number of candles to look back.
        Returns:
            Tuple (support, resistance).
        """
        # Use 100 candles for stronger support/resistance zones
        if len(df) < lookback:
            # Not enough candles, fallback to min/max of available data
            support = df['low'].min()
            resistance = df['high'].max()
        else:
            recent_lows = df['low'].rolling(lookback).min()
            recent_highs = df['high'].rolling(lookback).max()
            support = recent_lows.iloc[-1]
            resistance = recent_highs.iloc[-1]
        # Handle NaN just in case
        if pd.isna(support):
            support = df['low'].min()
        if pd.isna(resistance):
            resistance = df['high'].max()
        return support, resistance

    @staticmethod
    def detect_candlestick_pattern(df):
        if len(df) < 3:
            return None
        c1 = df.iloc[-3]
        c2 = df.iloc[-2]

        body_c1 = abs(c1['close'] - c1['open'])
        body_c2 = abs(c2['close'] - c2['open'])

        if c1['close'] < c1['open'] and c2['close'] > c2['open'] and c2['close'] > c1['open'] and c2['open'] < c1['close']:
            return 'bullish_engulfing'

        if c1['close'] > c1['open'] and c2['close'] < c2['open'] and c2['close'] < c1['open'] and c2['open'] > c1['close']:
            return 'bearish_engulfing'

        lower_wick = c2['open'] - c2['low'] if c2['close'] > c2['open'] else c2['close'] - c2['low']
        upper_wick = c2['high'] - c2['close'] if c2['close'] > c2['open'] else c2['high'] - c2['open']
        if body_c2 < lower_wick and lower_wick > upper_wick * 2:
            return 'hammer'

        if body_c2 < upper_wick and upper_wick > lower_wick * 2:
            return 'shooting_star'

        return None

    @staticmethod
    def detect_divergence(df):
        low1, low2 = df['close'].iloc[-5], df['close'].iloc[-1]
        rsi1, rsi2 = df['rsi'].iloc[-5], df['rsi'].iloc[-1]
        macd1, macd2 = df['macd_line'].iloc[-5], df['macd_line'].iloc[-1]

        bullish_div = low2 < low1 and (rsi2 > rsi1 or macd2 > macd1)

        high1, high2 = df['close'].iloc[-5], df['close'].iloc[-1]
        rsi_b1, rsi_b2 = df['rsi'].iloc[-5], df['rsi'].iloc[-1]
        macd_b1, macd_b2 = df['macd_line'].iloc[-5], df['macd_line'].iloc[-1]

        bearish_div = high2 > high1 and (rsi_b2 < rsi_b1 or macd_b2 < macd_b1)

        return bullish_div, bearish_div

    @staticmethod
    def is_trend_following_enabled(symbol):
        """
        Returns True if trend-following logic is enabled for the symbol.
        Uses TREND_FOLLOWING_ENABLED_MAP from config if available, else defaults to True for forex, False otherwise.
        """
        from config import TREND_FOLLOWING_ENABLED_MAP
        forex = TechnicalAnalysis.forex
        if symbol in TREND_FOLLOWING_ENABLED_MAP:
            return TREND_FOLLOWING_ENABLED_MAP[symbol]
        if symbol in forex:
            return True
        return False

    @staticmethod
    def score_trend_signal(df, symbol=None):
        """
        Trend-following scoring system for currency pairs (forex and high-volatility non-forex).
        Scores only strong trend entries using high-quality indicators.
        """
        settings = TechnicalAnalysis.get_pair_settings(symbol or "")
        score = 0.0
        breakdown = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        # EMA cross (trend start)
        bullish_ema_cross = latest['ema_fast'] > latest['ema_slow'] and prev['ema_fast'] <= prev['ema_slow']
        bearish_ema_cross = latest['ema_fast'] < latest['ema_slow'] and prev['ema_fast'] >= prev['ema_slow']
        if bullish_ema_cross or bearish_ema_cross:
            score += 25
            breakdown.append('EMA cross')
        # ADX strong trend
        if latest['adx'] > 25:
            score += 20
            breakdown.append('ADX strong')
        # MACD trend confirmation
        bullish_macd = latest['macd_line'] > latest['macd_signal']
        bearish_macd = latest['macd_line'] < latest['macd_signal']
        if (bullish_ema_cross and bullish_macd) or (bearish_ema_cross and bearish_macd):
            score += 20
            breakdown.append('MACD trend confirm')
        # RSI in trend zone (not overbought/oversold, e.g. 40-60)
        if 40 < latest['rsi'] < 60:
            score += 10
            breakdown.append('RSI trend zone')
        # Bollinger Band expansion (volatility breakout)
        bb_width = df['upper_bb'].iloc[-1] - df['lower_bb'].iloc[-1]
        bb_width_prev = df['upper_bb'].iloc[-2] - df['lower_bb'].iloc[-2]
        if bb_width > bb_width_prev * 1.1:
            score += 10
            breakdown.append('BB expansion')
        # Price action: price above upper BB (bullish) or below lower BB (bearish)
        if bullish_ema_cross and latest['close'] > df['upper_bb'].iloc[-1]:
            score += 10
            breakdown.append('Price breakout (bull)')
        if bearish_ema_cross and latest['close'] < df['lower_bb'].iloc[-1]:
            score += 10
            breakdown.append('Price breakout (bear)')
        # Cap score at 100
        score = round(min(score, 100), 1)
        logger.info(f"Weighted Score: {score}%")
        return score, breakdown

    @staticmethod
    def score_signal(df, symbol=None):
        required_cols = {'open', 'high', 'low', 'close'}
        if df is None or df.empty or not required_cols.issubset(df.columns):
            logger.warning(f"score_signal: DataFrame is empty or missing OHLC columns for {symbol}. Returning NEUTRAL.")
            return None, ["Invalid or missing data"]

        min_score = MIN_SCORE_NON_FOREX

        settings = TechnicalAnalysis.get_pair_settings(symbol or "")
        confirmations = getattr(settings, 'required_confirmations', 0) if hasattr(settings, 'required_confirmations') else 0
        score = 0.0
        blocked = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        adx = latest['adx']
        debug_steps = []
        # Ensure support, resistance, avg_range, and distance are defined before use
        support, resistance = TechnicalAnalysis.detect_support_resistance(df)
        avg_range = df['tr'].iloc[-20:].mean() if 'tr' in df.columns else 0
        distance = abs(latest['close'] - support if latest['close'] < support else latest['close'] - resistance)
        pattern = None
        try:
            pattern = TechnicalAnalysis.detect_candlestick_pattern(df)
        except Exception as e:
            logger.warning(f"Error detecting candlestick pattern: {e}")
        # --- Relaxed ADX reversal filter ---
        if USE_ADX_FILTER:
            if 5 <= adx <= 50:  # Relaxed range
                debug_steps.append(f"ADX in relaxed range: {adx:.2f} (allowed)")
            else:
                blocked.append(f"❌ ADX not in 5-50: reversal signals blocked")
                debug_steps.append(f"ADX out of relaxed range: {adx:.2f} (blocked)")
        if USE_ADX_FILTER and latest['adx'] > 50 and latest['adx'] > prev['adx']:
            blocked.append(f"❌ ADX rising above 50")
            debug_steps.append(f"ADX rising: {latest['adx']:.2f} > {prev['adx']:.2f}")
        # --- Relaxed FLAT filter ---
        if USE_FLAT_FILTER:
            recent_vol = df['volatility'].iloc[-5:].mean()
            avg_vol = df['volatility'].rolling(20).mean().iloc[-1]
            if recent_vol < avg_vol * 0.5:  # Lowered threshold
                blocked.append(f"❌ Flat market < 50% avg vol")
                debug_steps.append(f"Flat market: recent_vol={recent_vol:.4f}, avg_vol={avg_vol:.4f}")
            else:
                debug_steps.append(f"Volatility OK: recent_vol={recent_vol:.4f}, avg_vol={avg_vol:.4f}")
        # --- Require volatility expansion for reversal signals ---
        volatility_ratio = latest['tr'] / df['tr'].rolling(20).mean().iloc[-1]
        debug_steps.append(f"Volatility ratio: {volatility_ratio:.2f}")

        # --- Indicator logic: make triggers less strict ---
        # MACD cross (reversal, less strict: allow any cross, not just strict)
        macd_cross_loose = df['macd_hist'].iloc[-2] < 0 and df['macd_hist'].iloc[-1] > 0
        if macd_cross_loose:
            score += settings['score_weights']['macd'] * 100
            confirmations += 1  # Allow loose logic to increment confirmations
            debug_steps.append(f"MACD loose cross: True (+{settings['score_weights']['macd']*100:.1f}, +1 conf)")
        else:
            debug_steps.append("MACD loose cross: False")
        # RSI less strict: allow near oversold/overbought (within 5 points)
        rsi_near_extreme = (
            latest['rsi'] < settings['rsi_oversold'] + 5 or latest['rsi'] > settings['rsi_overbought'] - 5
        )
        if rsi_near_extreme:
            score += settings['score_weights']['rsi'] * 100
            confirmations += 1
            debug_steps.append(f"RSI near extreme: True (+{settings['score_weights']['rsi']*100:.1f}, +1 conf)")
        else:
            debug_steps.append("RSI near extreme: False")
        # Stoch less strict: allow any cross above 20/under 80
        stoch_cross_loose = (
            (df['stoch_k'].iloc[-2] < df['stoch_d'].iloc[-2] and df['stoch_k'].iloc[-1] > df['stoch_d'].iloc[-1]) or
            (df['stoch_k'].iloc[-2] > df['stoch_d'].iloc[-2] and df['stoch_k'].iloc[-1] < df['stoch_d'].iloc[-1])
        )
        if stoch_cross_loose:
            score += settings['score_weights']['stoch'] * 100
            confirmations += 1
            debug_steps.append(f"Stoch loose cross: True (+{settings['score_weights']['stoch']*100:.1f}, +1 conf)")
        else:
            debug_steps.append("Stoch loose cross: False")
        # Candle at level: less strict, allow if within 2.5x avg range
        candle_at_level_loose = pattern in ['bullish_engulfing', 'bearish_engulfing', 'hammer', 'shooting_star'] and distance < avg_range * 2.5
        if candle_at_level_loose:
            score += settings['score_weights']['candlestick'] * 100
            confirmations += 1
            debug_steps.append(f"Candle at level (loose): True (+{settings['score_weights']['candlestick']*100:.1f}, +1 conf)")
        else:
            debug_steps.append("Candle at level (loose): False")
        # Near S/R: less strict, within 2.5x avg range
        if distance < avg_range * 2.5:
            score += settings['score_weights']['support_resistance'] * 100
            confirmations += 1
            debug_steps.append(f"Near S/R (loose): True (+{settings['score_weights']['support_resistance']*100:.1f}, +1 conf)")
        else:
            debug_steps.append("Near S/R (loose): False")
        # BB reversal: less strict, allow within 0.3*bb_width
        bb_reversal_loose = False
        bb_width = df['upper_bb'].iloc[-1] - df['lower_bb'].iloc[-1]
        if latest['close'] < df['lower_bb'].iloc[-1] - 0.3 * bb_width:
            bb_reversal_loose = True
        elif latest['close'] > df['upper_bb'].iloc[-1] + 0.3 * bb_width:
            bb_reversal_loose = True
        if bb_reversal_loose:
            score += settings['score_weights']['bollinger'] * 100
            confirmations += 1
            debug_steps.append(f"BB loose reversal: True (+{settings['score_weights']['bollinger']*100:.1f}, +1 conf)")
        else:
            debug_steps.append("BB loose reversal: False")
        # ADX: less strict, allow <30
        adx_score_weight = settings['score_weights'].get('adx', 0.08)
        if latest['adx'] < 30:
            score += adx_score_weight * 100
            confirmations += 1
            debug_steps.append(f"ADX < 30 (loose): True (+{adx_score_weight*100:.1f}, +1 conf)")
        else:
            debug_steps.append("ADX < 30 (loose): False")
        # Divergence: keep as is
        div_score_weight = settings['score_weights'].get('divergence', 0.08)
        bullish_div, bearish_div = TechnicalAnalysis.detect_divergence(df)
        if bullish_div:
            score += div_score_weight * 100
            confirmations += 1
            debug_steps.append(f"Bullish divergence: True (+{div_score_weight*100:.1f}, +1 conf)")
        else:
            debug_steps.append("Bullish divergence: False")
        score = round(min(score, 100), 1)
        # --- Blocked/return check after all scoring ---
        if ALLOW_STRONG_PATTERN_OVERRIDE and pattern in ['bullish_engulfing', 'bearish_engulfing', 'hammer', 'shooting_star']:
            confirmations = settings['required_confirmations']
        if (confirmations < settings['required_confirmations'] or score < min_score):
            blocked.append(f"❌ Not enough confirmations (need {settings['required_confirmations']}) or score too low (min {min_score})")
            if LOG_BLOCKED_SIGNALS:
                logger.info("Blocked: " + " | ".join(blocked))
            logger.info(f"Weighted Score: {score}%")
            logger.debug("Score debug steps: " + " | ".join(debug_steps))
            return None, {
                "reason": "Blocked: not enough confirmations or score too low",
                "score": score,
                "confirmations": confirmations,
                "details": blocked,
                "debug": debug_steps
            }
        logger.info(f"Weighted Score: {score}%")
        logger.debug("Score debug steps: " + " | ".join(debug_steps))
        return score, [pattern, support, resistance, avg_range, distance, latest['adx'], debug_steps]

    @staticmethod
    def calculate_adaptive_duration(score, confirmations, avg_range, volatility, distance, counter_trend, symbol=None):
        """
        Calculate adaptive signal duration based on score, confirmations, volatility, and other factors.
        Returns an integer duration (in minutes or candles).
        """
        # Base duration: higher score/confirmations = shorter duration, more volatility = shorter duration
        base = 5
        if score >= 90:
            base = 2
        elif score >= 80:
            base = 3
        elif score >= 70:
            base = 4
        # Adjust for volatility: higher volatility = shorter duration
        if volatility > avg_range * 1.5:
            base = max(1, base - 1)
        # Counter-trend signals: increase duration
        if counter_trend:
            base += 1
        # Clamp duration between 1 and 10
        return max(1, min(base, 10))

    # --- Multi-timeframe confirmation logic ---
    def analyze(self, df_3m, symbol):
        # Main analysis on 3m candles
        df = self.calculate_indicators(df_3m, symbol)
        score, breakdown = self.score_signal(df, symbol)
        mtf_penalty = 0
        session_penalty = 0
        macd_5m_boost = 0
        macd_5m_confirmed = False
        # if session_flag:
        #     if not breakdown:
        #         breakdown = []
        #     breakdown.append(f"⚠️ {session_name}: Signal generated during major market open/close. Increased volatility/noise possible.")
        #     session_penalty = 5
        # --- Multi-Timeframe Confirmation (MTF) ---
        mtf_applied = False
        is_trend_signal = False
        if breakdown and isinstance(breakdown, list):
            is_trend_signal = any(
                x in breakdown for x in ['EMA cross', 'MACD trend confirm', 'ADX strong', 'RSI trend zone', 'BB expansion', 'Price breakout (bull)', 'Price breakout (bear)']
            )
        if ENABLE_MTF_CONFIRMATION and symbol in MTF_CONFIRMATION_PAIRS and score is not None:
            try:
                from deriv_api import fetch_ohlcv
                # Fetch 5m candles for confirmation
                df_5m = fetch_ohlcv(symbol, timeframe='M5')
                df_5m = self.calculate_indicators(df_5m, symbol)
                # 5m MACD reversal confirmation
                macd_hist_5m_prev = df_5m['macd_hist'].iloc[-2]
                macd_hist_5m_now = df_5m['macd_hist'].iloc[-1]
                macd_line_5m_prev = df_5m['macd_line'].iloc[-2]
                macd_signal_5m_prev = df_5m['macd_signal'].iloc[-2]
                macd_line_5m_now = df_5m['macd_line'].iloc[-1]
                macd_signal_5m_now = df_5m['macd_signal'].iloc[-1]
                # MACD histogram cross or MACD line cross
                macd_hist_cross = (macd_hist_5m_prev < 0 and macd_hist_5m_now > 0) or (macd_hist_5m_prev > 0 and macd_hist_5m_now < 0)
                macd_line_cross = (macd_line_5m_prev < macd_signal_5m_prev and macd_line_5m_now > macd_signal_5m_now) or (macd_line_5m_prev > macd_signal_5m_prev and macd_line_5m_now < macd_signal_5m_now)
                if macd_hist_cross or macd_line_cross:
                    macd_5m_boost = 15
                    macd_5m_confirmed = True
                    breakdown.append(f"5m MACD reversal confirm: +{macd_5m_boost}")
                else:
                    breakdown.append("5m MACD no reversal confirm")
                # Optionally, reduce duration if 5m MACD confirms
                mtf_applied = True
            except Exception as e:
                logger.warning(f"MTF MACD confirmation failed: {e}")
        if score is None:
            return {
                'signal': 'NEUTRAL',
                'strength': 0,
                'details': {
                    'score': 0,
                    'duration': 5,
                    'win_rate': 50.0,
                    'adx_value': df['adx'].iloc[-1]
                }
            }
        latest = df.iloc[-1]
        support, resistance = TechnicalAnalysis.detect_support_resistance(df)
        avg_range = df['tr'].iloc[-20:].mean()
        distance = abs(latest['close'] - support if latest['close'] < support else latest['close'] - resistance)
        adx = latest['adx']
        volatility = df['volatility'].iloc[-1]
        pattern = TechnicalAnalysis.detect_candlestick_pattern(df)
        settings = TechnicalAnalysis.get_pair_settings(symbol)
        signal_direction = 'BUY' if df['close'].iloc[-1] < df['open'].iloc[-1] else 'SELL'
        trend_direction = 'UP' if adx > 25 and df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1] else (
            'DOWN' if adx > 25 and df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1] else 'FLAT')
        counter_trend = (
            (signal_direction == 'BUY' and trend_direction == 'DOWN') or
            (signal_direction == 'SELL' and trend_direction == 'UP')
        )
        confirmations = 0
        if breakdown:
            confirmations = sum(1 for x in breakdown if x not in [None, 0, '', False])
        # Add 5m MACD boost to score if confirmed
        final_score = max(0, score + macd_5m_boost - mtf_penalty - session_penalty)
        # Reduce duration by 20% if 5m MACD confirms
        duration = TechnicalAnalysis.calculate_adaptive_duration(
            final_score, confirmations, avg_range, volatility, distance, counter_trend, symbol=symbol
        )
        if macd_5m_confirmed:
            duration = max(1, int(duration * 0.8))
        logger.info(f"Adaptive duration: {duration}")
        signal = signal_direction
        strength = 3 if final_score >= 85 else 2
        win_rate = min(95.0, max(50.0, final_score))
        return {
            'signal': signal,
            'strength': strength,
            'details': {
                'score': final_score,
                'breakdown': breakdown,
                'duration': duration,
                'win_rate': win_rate,
                'adx_value': adx
            }
        }
