import pandas as pd
import numpy as np
import pytest
from technical_analysis import TechnicalAnalysis

# File: tests/test_technical_analysis_trend.py


@pytest.fixture
def bullish_trend_df():
    # Simulate a strong bullish trend
    data = {
        'open': np.linspace(1, 2, 50),
        'high': np.linspace(1.01, 2.01, 50),
        'low': np.linspace(0.99, 1.99, 50),
        'close': np.linspace(1, 2, 50) + np.linspace(0, 0.05, 50),
    }
    df = pd.DataFrame(data)
    return TechnicalAnalysis.calculate_indicators(df.copy(), symbol="frxEURUSD")

@pytest.fixture
def bearish_trend_df():
    # Simulate a strong bearish trend
    data = {
        'open': np.linspace(2, 1, 50),
        'high': np.linspace(2.01, 1.01, 50),
        'low': np.linspace(1.99, 0.99, 50),
        'close': np.linspace(2, 1, 50) - np.linspace(0, 0.05, 50),
    }
    df = pd.DataFrame(data)
    return TechnicalAnalysis.calculate_indicators(df.copy(), symbol="frxEURUSD")

@pytest.fixture
def flat_market_df():
    # Simulate a flat market
    data = {
        'open': np.full(50, 1.5),
        'high': np.full(50, 1.51),
        'low': np.full(50, 1.49),
        'close': np.full(50, 1.5),
    }
    df = pd.DataFrame(data)
    return TechnicalAnalysis.calculate_indicators(df.copy(), symbol="frxEURUSD")

def test_bullish_trend_signal(bullish_trend_df):
    score, breakdown = TechnicalAnalysis.score_trend_signal(bullish_trend_df, symbol="frxEURUSD")
    assert score > 70, f"Expected high score for strong bullish trend, got {score}"
    assert breakdown['trend'] == 'bullish'

def test_bearish_trend_signal(bearish_trend_df):
    score, breakdown = TechnicalAnalysis.score_trend_signal(bearish_trend_df, symbol="frxEURUSD")
    assert score > 70, f"Expected high score for strong bearish trend, got {score}"
    assert breakdown['trend'] == 'bearish'

def test_flat_market_signal(flat_market_df):
    score, breakdown = TechnicalAnalysis.score_trend_signal(flat_market_df, symbol="frxEURUSD")
    assert score < 30, f"Expected low score for flat market, got {score}"
    assert breakdown['trend'] == 'flat'

def test_trend_signal_edge_cases():
    # Short series, should not error
    df = pd.DataFrame({
        'open': [1, 1.01, 1.02],
        'high': [1.01, 1.02, 1.03],
        'low': [0.99, 1.00, 1.01],
        'close': [1, 1.01, 1.02],
    })
    df = TechnicalAnalysis.calculate_indicators(df.copy(), symbol="frxEURUSD")
    score, breakdown = TechnicalAnalysis.score_trend_signal(df, symbol="frxEURUSD")
    assert isinstance(score, float)
    assert 'trend' in breakdown