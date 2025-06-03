# How to Run Your Deriv Signal Bot Locally

## Files You'll Need
1. `main.py` - The main entry point
2. `config.py` - Configuration settings
3. `deriv_api.py` - API connection handling
4. `signal_analyzer.py` - Signal analysis logic
5. `technical_analysis.py` - Technical indicators
6. `telegram_alerts.py` - Telegram notification system

## Setup Instructions

1. **Create a project folder** on your local machine and save all the above files there.

2. **Install the required packages** using pip:
   ```
   pip install pandas numpy websockets httpx
   ```

3. **Configure your API tokens**:
   - In `config.py`: Add your Deriv API token
   - In `telegram_alerts.py`: Add your Telegram bot token and channel ID

4. **Run the bot**:
   ```
   python main.py
   ```

## Getting API Tokens

1. **Deriv API Token**:
   - Log into your Deriv account
   - Go to Dashboard > API Token
   - Create a token with "Read" scope
   - Copy and paste into config.py

2. **Telegram Bot Token**:
   - Message @BotFather on Telegram
   - Create a new bot with /newbot
   - Get your token and add to telegram_alerts.py
   - Create a channel and add your bot as admin
   - Get the channel ID and add to telegram_alerts.py

## Customization

Feel free to modify these settings in config.py:
- Trading pairs (SYMBOLS list)
- Technical indicator parameters
- Score and win rate thresholds
- Signal cooldown period