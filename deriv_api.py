# deriv_api.py

import logging
import json
import pandas as pd
import httpx
import time
import os
import numpy as np
import websockets
import asyncio
from datetime import datetime, timedelta
import random
import uuid
import threading
from concurrent.futures import Future

class DerivAPI:
    def __init__(self, app_id, api_token=None):
        self.app_id = app_id
        self.api_token = api_token
        self.api_url = "https://api.deriv.com"
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        self.authorized = False
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False  # Prevent duplicate logs if handlers are attached elsewhere
        self.client = httpx.Client(timeout=30.0)
        self.client.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        self.symbols_data = {}
        self.last_update = {}
        self.websocket = None
        self.loop = None
        self.req_id = 0
        self.pending_requests = {}

    def connect(self):
        try:
            self.logger.info("Connecting to Deriv API...")
            if not self.api_token:
                self.logger.warning("No API token provided. Will try to connect without authentication.")
            else:
                self.logger.info("Using API token for authentication")
            # Windows event loop policy fix
            if os.name == 'nt':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            connection_successful = self.loop.run_until_complete(self._connect_websocket())
            if connection_successful:
                thread = threading.Thread(target=self._start_websocket_listener, daemon=True)
                thread.start()
                # Start keepalive ping thread
                ping_thread = threading.Thread(target=self._start_keepalive_ping, daemon=True)
                ping_thread.start()
                self.logger.info("Connected successfully to Deriv API")
                return True
            else:
                self.logger.error("Failed to connect to Deriv API")
                return False
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False

    def _start_keepalive_ping(self):
        # Send ping every 30 seconds to keep connection alive
        while True:
            try:
                if self.websocket:
                    coro = self.websocket.ping()
                    if self.loop and not self.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(coro, self.loop)
                time.sleep(30)
            except Exception as e:
                self.logger.warning(f"Keepalive ping failed: {e}")
                time.sleep(10)

    async def _connect_websocket(self):
        try:
            self.websocket = await websockets.connect(self.ws_url)
            if self.api_token:
                auth_req = {
                    "authorize": self.api_token,
                    "req_id": self._get_req_id()
                }
                await self.websocket.send(json.dumps(auth_req))
                response = await self.websocket.recv()
                auth_response = json.loads(response)
                if "error" in auth_response:
                    self.logger.error(f"Authorization failed: {auth_response['error']['message']}")
                    self.authorized = False
                    return False
                self.authorized = True
                self.logger.info("Successfully authenticated with Deriv API")
            else:
                self.authorized = True
                self.logger.info("Connected without authentication - limited access")
            return True
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
            return False

    def _start_websocket_listener(self):
        self.loop.run_until_complete(self._listen_to_websocket())

    async def _listen_to_websocket(self):
        try:
            while True:
                if not self.websocket:
                    break
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=60)
                    response = json.loads(message)
                    await self._process_message(response)
                except asyncio.TimeoutError:
                    # Only log this to file, not terminal
                    self.logger.debug("WebSocket receive timeout, sending ping...")
                    try:
                        pong_waiter = await self.websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=10)
                    except Exception as e:
                        self.logger.error(f"Ping failed: {e}")
                        await self._reconnect()
                except websockets.exceptions.ConnectionClosed:
                    self.logger.error("WebSocket connection closed")
                    await self._reconnect()
        except Exception as e:
            self.logger.error(f"Error in WebSocket listener: {e}")

    async def _reconnect(self):
        try:
            self.logger.info("Attempting to reconnect to Deriv API...")
            # Wait a bit before reconnecting
            await asyncio.sleep(5)
            await self._connect_websocket()
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")

    async def _process_message(self, message):
        req_id = message.get('req_id')
        if req_id and req_id in self.pending_requests:
            self.pending_requests[req_id].set_result(message)

    def _get_req_id(self):
        self.req_id += 1
        return self.req_id

    async def _get_real_candles(self, symbol, timeframe=5, count=50):
        if not self.websocket or not self.loop:
            self.logger.error("WebSocket not connected")
            return None
        try:
            future = self.loop.create_future()
            req_id = self._get_req_id()
            self.pending_requests[req_id] = future
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "granularity": timeframe * 60,
                "start": 1,
                "style": "candles",
                "req_id": req_id
            }
            await self._send_request(request)
            response = await asyncio.wait_for(future, timeout=10)
            if "error" in response:
                self.logger.error(f"API error: {response['error']['message']}")
                return None
            if "candles" not in response or not response["candles"]:
                self.logger.error("No candle data received")
                return None
            candles = response["candles"]
            data = []
            for candle in candles:
                data.append({
                    "timestamp": pd.to_datetime(candle["epoch"] * 1000, unit="ms"),
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"])
                })
            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            df["tr"] = self._calculate_tr(df)
            return df
        except asyncio.TimeoutError:
            self.logger.error("Request timed out")
            return None
        except Exception as e:
            self.logger.error(f"Error getting candles: {e}")
            return None

    async def _send_request(self, request):
        if self.websocket:
            await self.websocket.send(json.dumps(request))

    def _calculate_tr(self, df):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        return np.max(ranges, axis=1)

    def get_candles(self, symbol, timeframe=5, count=50, callback=None):
        try:
            now = datetime.now()
            if (symbol in self.symbols_data and
                symbol in self.last_update and
                (now - self.last_update[symbol]).total_seconds() < 900):
                df = self.symbols_data[symbol]
                self.logger.info(f"Using cached data for {symbol}")
                return df

            self.logger.info(f"Fetching market data for {symbol}")

            if self.authorized and self.websocket:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._get_real_candles(symbol, timeframe, count),
                        self.loop
                    )
                    df = future.result(timeout=15)

                    if df is not None and not df.empty:
                        self.logger.info(f"Successfully fetched real market data for {symbol}")
                        self.symbols_data[symbol] = df
                        self.last_update[symbol] = now
                        return df
                except Exception as e:
                    self.logger.error(f"Error fetching real data for {symbol}: {e}")

            self.logger.warning(f"Using generated data for {symbol} - API connection failed")
            df = self._generate_realistic_data_with_reversals(symbol, count)
            self.symbols_data[symbol] = df
            self.last_update[symbol] = now

            if callback:
                callback(df)

            return df
        except Exception as e:
            self.logger.error(f"Error getting candles for {symbol}: {e}")
            return self._generate_realistic_data_with_reversals(symbol, count)

    def _generate_realistic_data_with_reversals(self, symbol, count):
        now = datetime.now()
        dates = [now - timedelta(minutes=i*5) for i in range(count-1, -1, -1)]
        try:
            vol_factor = int(''.join(filter(str.isdigit, symbol)))
            if vol_factor == 0:
                vol_factor = 10
        except:
            vol_factor = 10
        volatility = vol_factor / 100.0
        base_price = 1000.0 * (1 + volatility)
        np.random.seed(int(time.time()) % 1000 + hash(symbol) % 100)
        trend_periods = int(count / (8 + np.random.randint(0, 5)))
        trend_changes = np.zeros(count)
        for i in range(0, count, trend_periods):
            if i // trend_periods % 2 == 0:
                trend = np.linspace(0, volatility * 200, min(trend_periods, count - i))
            else:
                trend = np.linspace(volatility * 200, 0, min(trend_periods, count - i))
            trend_changes[i:i+len(trend)] = trend
        fluctuations = np.random.normal(0, base_price * 0.005 * volatility, count)
        changes = trend_changes + fluctuations
        for i in range(trend_periods, count, trend_periods):
            if i < count - 2:
                changes[i-2] *= 1.5
                changes[i-1] *= -1.8
                changes[i] *= 1.2
        closes = base_price + np.cumsum(changes)
        data = []
        for i, date in enumerate(dates):
            close = max(closes[i], 0.01)
            position_in_trend = i % trend_periods
            trend_start = position_in_trend < 3
            trend_end = position_in_trend > trend_periods - 3
            if trend_start or trend_end:
                high_low_factor = 2.0 + volatility
            else:
                high_low_factor = 1.2 + (volatility / 2)
            high_low_spread = abs(changes[i]) * high_low_factor
            if i == 0:
                open_price = close - changes[i]/2
            else:
                open_price = closes[i-1]
            is_bullish = close > open_price
            if is_bullish:
                if trend_end and i % 2 == 0:
                    high = close + high_low_spread * 0.7
                    low = open_price - high_low_spread * 0.3
                else:
                    high = close + high_low_spread * 0.4
                    low = open_price - high_low_spread * 0.2
            else:
                if trend_start and i % 2 == 1:
                    high = open_price + high_low_spread * 0.3
                    low = close - high_low_spread * 0.7
                else:
                    high = open_price + high_low_spread * 0.2
                    low = close - high_low_spread * 0.4
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close
            })
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def close(self):
        self.logger.info("Connection closed")
        try:
            if self.websocket:
                coro = self.websocket.close()
                if self.loop and not self.loop.is_closed():
                    asyncio.run_coroutine_threadsafe(coro, self.loop)
            if self.loop and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception as e:
            self.logger.error(f"Error closing websocket: {e}")

def fetch_ohlcv(symbol, timeframe='M5', count=100):
    """
    Fetch OHLCV data for a symbol and timeframe. Returns a DataFrame with columns:
    ['open', 'high', 'low', 'close', 'volume', 'epoch']
    timeframe: string, e.g. 'M5', 'H1', 'D1'
    count: number of candles to fetch
    """
    # Map timeframe string to Deriv granularity in seconds
    tf_map = {
        'M1': 60,
        'M5': 300,
        'M15': 900,
        'M30': 1800,
        'H1': 3600,
        'H4': 14400,
        'D1': 86400
    }
    granularity = tf_map.get(timeframe.upper(), 300)  # Default to M5 if not found
    # Try to use the main DerivAPI instance if available
    try:
        # If you have a global/main API instance, use it. Otherwise, create a new one (not recommended for prod)
        from config import APP_ID, API_TOKEN
        api = None
        if 'api' in globals():
            api = globals()['api']
        else:
            from deriv_api import DerivAPI
            api = DerivAPI(APP_ID, API_TOKEN)
            api.connect()
        df = api.get_candles(symbol, timeframe=granularity//60, count=count)
        if df is not None and not df.empty:
            # Add dummy volume and epoch columns if missing
            if 'volume' not in df.columns:
                df['volume'] = 0
            if 'epoch' not in df.columns:
                df['epoch'] = df.index.astype(np.int64) // 10**9
            return df.reset_index().rename(columns={'timestamp': 'datetime'})
        else:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'epoch'])
    except Exception as e:
        logging.getLogger(__name__).warning(f"fetch_ohlcv fallback: {e}")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'epoch'])
