# signal_analyzer.py — uses config-driven thresholds for cooldown + score

import logging
from datetime import datetime, timedelta
from technical_analysis import TechnicalAnalysis
from telegram_alerts import send_telegram_message
from config import (
    SYMBOLS, TEST_MODE, STRICT_MODE_PAIRS, 
    DEFAULT_COOLDOWN, STRICT_COOLDOWN,
    MIN_SCORE_FOREX, MIN_SCORE_NON_FOREX
)

class SignalAnalyzer:
    def __init__(self, deriv_api):
        self.deriv_api = deriv_api
        self.ta = TechnicalAnalysis()
        self.last_signal_time = {}

        try:
            logging.info("Loading previous signal timestamps...")
            for symbol in SYMBOLS:
                self.last_signal_time[symbol] = datetime.utcnow() - timedelta(minutes=DEFAULT_COOLDOWN)
        except Exception as e:
            logging.warning(f"Could not load previous timestamps: {e}")

    def scan_once(self):
        return self.scan()

    def scan(self):
        signals = []
        total_score = 0
        total_scanned = 0
        total_candles_received = 0
        now = datetime.utcnow()
        best_signals = {}

        for symbol in SYMBOLS:
            if symbol == "R_50":
                continue  # Skip R_50 as requested
            try:
                logging.info(f"Fetching market data for {symbol}")
                df = self.deriv_api.get_candles(symbol, timeframe=3, count=50)  # Use 3m candles for main analysis
                if df is None or df.empty:
                    continue

                total_candles_received += len(df)
                analysis = self.ta.analyze(df, symbol)
                total_score += analysis['details']['score']
                total_scanned += 1

                signal = analysis['signal']
                score = analysis['details']['score']
                strength = analysis['strength']
                duration = analysis['details']['duration'] if 'duration' in analysis['details'] else None
                win_rate = analysis['details']['win_rate'] if 'win_rate' in analysis['details'] else None
                # Only log duration if a signal is actually sent (i.e., not blocked)
                if signal != "NEUTRAL" and duration is not None:
                    if isinstance(duration, (int, float)) and duration > 0:
                        logging.info(f"DEBUG: analyze() returned duration={duration} for {symbol}")
                    else:
                        logging.warning(f"DEBUG: analyze() returned invalid duration={duration} for {symbol}")

                # Use MIN_SCORE_FOREX or MIN_SCORE_NON_FOREX from config.py for all signals
                cooldown_minutes = STRICT_COOLDOWN if symbol in STRICT_MODE_PAIRS else DEFAULT_COOLDOWN
                # Set required_score based on symbol type
                if symbol.startswith('frx'):
                    required_score = MIN_SCORE_FOREX
                else:
                    required_score = MIN_SCORE_NON_FOREX

                if signal != "NEUTRAL" and score >= required_score:
                    last_time = self.last_signal_time.get(symbol)
                    cooldown = timedelta(minutes=cooldown_minutes)

                    if last_time and (now - last_time) < cooldown:
                        minutes_left = ((last_time + cooldown) - now).total_seconds() // 60
                        logging.info(f"Skipping {symbol} signal - cooldown active for {minutes_left:.1f} more minutes")
                        continue

                    if symbol not in best_signals or score > best_signals[symbol]['score']:
                        best_signals[symbol] = {
                            'analysis': analysis,
                            'score': score,
                            'signal': signal,
                            'strength': strength,
                            'duration': duration,
                            'win_rate': win_rate
                        }

            except Exception as e:
                logging.warning(f"⚠️ Error processing {symbol}: {e}")

        if best_signals:
            for symbol, signal_data in best_signals.items():
                signal = signal_data['signal']
                score = signal_data['score']
                strength = signal_data['strength']
                # Use the correct adaptive duration for the signal type
                analysis = signal_data['analysis']
                details = analysis['details']
                # Determine if this is a trend or reversal signal
                is_trend = False
                if signal == 'BUY' or signal == 'SELL':
                    # If trend_score is higher and used, or if trend logic is used, use trend_duration
                    if details.get('trend_score', 0) == score:
                        is_trend = True
                # Use the correct duration
                if is_trend and 'trend_duration' in details:
                    duration = details['trend_duration']
                elif not is_trend and 'reversal_duration' in details:
                    duration = details['reversal_duration']
                else:
                    duration = signal_data['duration']
                win_rate = signal_data['win_rate']

                cooldown_minutes = STRICT_COOLDOWN if symbol in STRICT_MODE_PAIRS else DEFAULT_COOLDOWN
                last_signal_time = self.last_signal_time.get(symbol)
                min_cooldown = timedelta(minutes=cooldown_minutes)

                if last_signal_time and (now - last_signal_time) < min_cooldown:
                    remaining = (min_cooldown - (now - last_signal_time)).total_seconds() // 60
                    logging.info(f"❌ Skipping {symbol} {signal} - Cooldown active for {remaining:.0f} more minutes")
                    continue

                logging.info(f"✅ Cooldown check passed for {symbol}. Sending signal with duration {duration//60 if duration >= 60 else duration}m ...")
                message = self.format_message(symbol, signal, strength, score, duration, win_rate)
                send_telegram_message(message)
                logging.info(f"📤 Signal sent: {symbol} {signal} ({score:.1f}%) [Duration: {duration//60 if duration >= 60 else duration}m]")
                signals.append((symbol, signal, score))
                self.last_signal_time[symbol] = now

        if not signals:
            avg_score = round(total_score / total_scanned, 1) if total_scanned else 0
            logging.info(f"❌ 0 signals generated | Avg Score: {avg_score}% | Total Candles: {total_candles_received}")

        return signals

    def format_message(self, symbol, signal, strength, score, duration, win_rate):
        stars = "★" * strength
        price = 0
        try:
            df = self.deriv_api.get_candles(symbol, timeframe=5, count=1)
            if df is not None and not df.empty:
                price = df['close'].iloc[-1]
        except:
            price = 1000 + hash(symbol) % 5000

        emoji = "🟢" if signal == "BUY" else "🔴"
        # Duration in minutes, as integer (no division, duration is already in minutes)
        duration_minutes = int(round(duration))

        message = (
            f"{emoji} Ligoo Signals\n\n"
            f"Symbol: {symbol}\n"
            f"Signal: {signal}\n"
            f"Strength: {stars}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Price: ${price:.2f}\n"
            f"Duration: {duration_minutes} minutes"
        )

        if TEST_MODE:
            message = "🧪 TEST MODE ENABLED\n\n" + message

        return message

    def run_periodic_scan(self):
        import time
        while True:
            now = datetime.utcnow()
            # Wait until the next 3rd minute of the clock (e.g., 1:00, 1:03, 1:06, ...)
            seconds_until_next_3rd = ((3 - (now.minute % 3)) % 3) * 60 - now.second
            if seconds_until_next_3rd <= 0:
                seconds_until_next_3rd += 180  # Always wait forward
            minutes, seconds = divmod(seconds_until_next_3rd, 60)
            logging.info(f"⏳ Waiting {int(minutes)}m {int(seconds)}s until next scan at the next 3rd minute...")
            time.sleep(seconds_until_next_3rd)
            self.scan()
