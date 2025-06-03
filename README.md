# Deriv Signal Bot

This is a trading signal bot for Deriv VIX pairs that analyzes market data using multiple technical indicators and sends high-quality trading signals to a Telegram channel.

## Features

- Analyzes 5-minute timeframe for volatility index pairs
- Uses multiple weighted technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Detects trend reversals with high accuracy
- Enforces 15-minute cooldown between signals for the same pair
- Calculates dynamic win rates and trade durations
- Sends formatted signals to Telegram

## Setup Instructions

1. **Install dependencies**:
   ```
   pip install pandas numpy websockets httpx
   ```

2. **Configure API tokens**:
   - Edit the `config.py` file to add your Deriv API token
   - Update the Telegram bot token in `telegram_alerts.py`

3. **Configure trading pairs**:
   - The bot is configured to analyze 1HZ10V, 1HZ25V, 1HZ75V, R_10, and R_25
   - You can modify the list in `config.py`

4. **Run the bot**:
   ```
   python main.py
   ```

## Sending Signals to Telegram

The bot will send signals in this format:
```
🟢 New Signal Alert
Symbol: 1HZ10V
Signal: BUY
Strength: ★★★
Win Rate: 85.0%
Price: $5153.09
Duration: 15 minutes
```

## Custom Settings

- Edit the thresholds in `config.py` to adjust signal quality
- The bot sends signals with 85%+ win probability by default
- Adjust the cooldown time (15 minutes by default) in `signal_analyzer.py`

## Files Description

- `main.py`: Entry point and main execution loop
- `deriv_api.py`: Handles WebSocket connection to Deriv API
- `signal_analyzer.py`: Coordinates the analysis process
- `technical_analysis.py`: Implements trading indicators and signal logic
- `telegram_alerts.py`: Sends notifications to Telegram channel
- `config.py`: Centralizes configuration parameters