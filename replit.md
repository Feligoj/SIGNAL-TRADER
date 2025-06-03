# Deriv Signal Bot Documentation

## Overview

This repository contains a trading signal bot for Deriv's volatility indices (VIX markets). The bot analyzes price action using multiple technical indicators (RSI, MACD, Bollinger Bands, ADX, etc.) to generate trading signals. When a signal meets certain criteria, it sends alerts via Telegram.

The system is designed to run continuously, scanning multiple volatility indices at regular intervals, applying technical analysis, and generating signals when predefined conditions are met.

## User Preferences

```
Preferred communication style: Simple, everyday language.
```

## System Architecture

The bot is architected as a Python application with several modular components:

1. **Core Components**:
   - `main.py`: Entry point and main execution loop
   - `deriv_api.py`: Handles WebSocket communication with Deriv's API
   - `signal_analyzer.py`: Coordinates the analysis process
   - `technical_analysis.py`: Implements trading indicators and signal logic
   - `telegram_alerts.py`: Sends notifications to Telegram channel
   - `config.py`: Centralizes configuration parameters

2. **Data Flow**:
   - Fetches candle data from Deriv API
   - Processes data through technical indicators
   - Evaluates signals based on configurable criteria
   - Sends alerts when valid signals are detected

3. **Execution Model**:
   - Continuous loop scanning each configured symbol
   - Implements cooldown periods between signals
   - Handles reconnection to API if connection is lost

## Key Components

### 1. Deriv API Interface (`deriv_api.py`)

This module manages the WebSocket connection to Deriv's API, handling authentication and data retrieval. The code shows a partially implemented connection mechanism, but the `get_candles` method referenced in the signal analyzer appears to be missing from the current implementation.

Key functionality:
- Establishing secure WebSocket connections
- API authorization using tokens
- Error handling and reconnection logic

### 2. Signal Analyzer (`signal_analyzer.py`)

Coordinates the analysis process by:
- Fetching candle data for each configured symbol
- Passing data to technical analyzer
- Applying cooldown logic to prevent signal spam
- Implementing different signal criteria based on symbol type

### 3. Technical Analysis (`technical_analysis.py`)

Implements various trading indicators and combines their signals:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ADX (Average Directional Index)
- Stochastic Oscillator (implied)
- Candlestick pattern analysis

The analyzer uses a scoring system (0-100) where different indicators contribute different weights to the final score.

### 4. Telegram Alerts (`telegram_alerts.py`)

Simple module to send formatted messages to a Telegram channel using a bot token.

### 5. Configuration (`config.py`)

Centralizes all settings for the bot:
- API credentials
- Trading symbols (Volatility indices)
- Technical indicator parameters
- Filtering and scoring thresholds
- Signal confirmation requirements

## Data Flow

1. The main loop initiates scanning at regular intervals
2. For each configured symbol:
   - Recent candle data is retrieved from Deriv API
   - Data is processed through technical indicators
   - A composite score is calculated based on multiple indicators
   - If the score exceeds thresholds and cooldown periods are respected, a signal is generated
   - Trading signals are sent to a Telegram channel

## External Dependencies

The system relies on the following key libraries:
- `websockets`: For WebSocket communication with Deriv API
- `pandas`: For data manipulation and analysis
- `numpy`: For numerical operations
- `httpx`: For HTTP requests to Telegram API

## Deployment Strategy

The bot is designed to run as a continuous service. The repository includes Replit configuration for deployment:

1. **Runtime Environment**:
   - Python 3.11 or higher
   - Required packages: pandas, numpy, websockets, httpx

2. **Execution**:
   - The main entry point is `main.py`
   - The bot runs in an infinite loop with automatic reconnection logic
   - Configuration can be adjusted through `config.py`

3. **Monitoring**:
   - The application logs status and events to standard output
   - Signal delivery can be confirmed through the Telegram channel

## Implementation Notes

1. **Missing Functionality**: The current code appears to have some incomplete functions. Specifically, the `get_candles` method is referenced but not fully implemented in the visible code.

2. **Authentication**: The bot uses a sample API token that should be replaced with a valid token for production use.

3. **Signal Logic**: The system uses a sophisticated scoring mechanism that combines multiple technical indicators with different weights. This approach helps reduce false signals.

4. **Different Symbol Requirements**: The code implements stricter criteria for higher volatility pairs (75V, 150V, 200V, 300V) including longer cooldown periods.

5. **Weekend Trading**: The system has higher thresholds for weekend trading when markets can behave differently.

6. **Filters**: ADX and "flat market" filters can be enabled to avoid signals in trending or sideways markets.