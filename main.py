# main.py

import time
import threading
import logging
import sys
from config import APP_ID, API_TOKEN
from deriv_api import DerivAPI
from signal_analyzer import SignalAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()
shown_connection = False

def reconnect_loop(api):
    """Repeatedly attempt to connect until successful"""
    global shown_connection
    max_attempts = 5
    attempts = 0

    while not api.authorized and attempts < max_attempts:
        attempts += 1
        try:
            logger.info(f"🔄 Connection attempt {attempts}/{max_attempts}...")
            if api.connect():
                logger.info("✅ Connection successful")
                shown_connection = True
                return True
            time.sleep(2)
        except Exception as e:
            if not shown_connection:
                logger.info(f"❌ Reconnect failed: {e}. Retrying...")
            time.sleep(2)

    if not api.authorized:
        logger.warning("⚠️ Failed to connect after multiple attempts. Continuing in demo mode.")
        api.authorized = True  # Force demo mode
        return True

    return False

if __name__ == "__main__":
    logger.info("🚀 Starting Deriv Signal Bot (VIX Analysis)")
    logger.info("⚙️ Initializing connections and services")

    deriv_api = DerivAPI(APP_ID, API_TOKEN)

    # Connect
    connection_successful = False
    try:
        connection_successful = deriv_api.connect()
        if connection_successful:
            logger.info("✅ Connected & Authorized to Deriv API")
            shown_connection = True
        else:
            logger.warning("⚠️ Initial authorization failed. Retrying...")
            connection_successful = reconnect_loop(deriv_api)
    except Exception as e:
        logger.warning(f"⚠️ Initial connection failed: {e}. Retrying...")
        connection_successful = reconnect_loop(deriv_api)

    # Create analyzer
    analyzer = SignalAnalyzer(deriv_api)
    logger.info("✅ Signal analyzer initialized")

    # Only run periodic scan loop (no immediate scan)
    analyzer.run_periodic_scan()

    deriv_api.close()
    logger.info("✅ Closed API connection")
