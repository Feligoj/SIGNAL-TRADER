# telegram_alerts.py

import logging
import httpx

# ✅ Your Telegram credentials
TELEGRAM_BOT_TOKEN = "7454382536:AAEUniRIuluu1rZxrBG94XH2sVnYe9ZMYzM"
TELEGRAM_CHANNEL_ID = "-1002655769973"

logger = logging.getLogger(__name__)

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": message,
        "disable_web_page_preview": True
    }

    try:
        logger.info("Sending message to Telegram...")
        with httpx.Client(timeout=15) as client:
            response = client.post(url, json=payload)
        if response.status_code == 200:
            logger.info("✅ Message sent to Telegram successfully")
            return True
        else:
            logger.warning(f"❌ Telegram error {response.status_code}: {response.text}")
            with httpx.Client(timeout=15) as client:
                response = client.post(url, data=payload)
            if response.status_code == 200:
                logger.info("✅ Message sent to Telegram with retry")
                return True
            else:
                logger.warning(f"❌ Telegram retry error {response.status_code}: {response.text}")
                return False
    except Exception as e:
        logger.error(f"❌ Telegram send failed: {e}")
        return False
