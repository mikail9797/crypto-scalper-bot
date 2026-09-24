"""
Расширенное распознавание паттернов: разворотные, продолжения, объёмы, дивергенции.
Используется как усилитель уверенности сигнала, а не как обязательный фильтр.
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Tuple, Optional


# ==================== СВЕЧНЫЕ ПАТТЕРНЫ ====================

def detect_hammer(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    body = abs(last['close'] - last['open'])
    range_ = last['high'] - last['low']
    if range_ == 0 or body == 0:
        return False
    lower_wick = min(last['open'], last['close']) - last['low']
    upper_wick = last['high'] - max(last['open'], last['close'])
    return (lower_wick > body * 2 and
            upper_wick < body * 0.5 and
            body / range_ < 0.35)


def detect_hanging_man(df: pd.DataFrame) -> bool:
    if not detect_hammer(df):
        return False
    if len(df) < 20:
        return False
    ema20 = df['close'].ewm(span=20).mean()
    return df['close'].iloc[-5:].min() > ema20.iloc[-1]


def detect_morning_star(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    body1 = abs(c1['close'] - c1['open'])
    body2 = abs(c2['close'] - c2['open'])
    if c1['close'] >= c1['open'] or c3['close'] <= c3['open']:
        return False
    if body2 > body1 * 0.3:
        return False
    if c3['close'] < (c1['open'] + c1['close']) / 2:
        return False
    return True


def detect_evening_star(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    body1 = abs(c1['close'] - c1['open'])
    body2 = abs(c2['close'] - c2['open'])
    if c1['close'] <= c1['open'] or c3['close'] >= c3['open']:
        return False
    if body2 > body1 * 0.3:
        return False
    if c3['close'] > (c1['open'] + c1['close']) / 2:
        return False
    return True


def detect_pin_bar(df: pd.DataFrame, direction: str = "bullish") -> bool:
    last = df.iloc[-1]
    body = abs(last['close'] - last['open'])
    range_ = last['high'] - last['low']
    if range_ == 0:
        return False
    if direction == "bullish":
        lower_wick = min(last['open'], last['close']) - last['low']
        return lower_wick / range_ > 0.66 and body / range_ < 0.2
    else:
        upper_wick = last['high'] - max(last['open'], last['close'])
        return upper_wick / range_ > 0.66 and body / range_ < 0.2


# ==================== ПАТТЕРНЫ ПРОДОЛЖЕНИЯ ====================

def detect_flag(df: pd.DataFrame, lookback: int = 20) -> Optional[Dict]:
    if len(df) < lookback + 10:
        return None
    impulse = df.iloc[-(lookback + 10):-lookback]
    impulse_move = impulse['close'].iloc[-1] - impulse['close'].iloc[0]
    impulse_pct = abs(impulse_move) / impulse['close'].iloc[0] * 100
    if impulse_pct < 2.0:
        return None
    consolidation = df.iloc[-lookback:]
    cons_range = (consolidation['high'].max() - consolidation['low'].min()) / consolidation['close'].mean() * 100
    if cons_range > impulse_pct * 0.4:
        return None
    return {
        'pattern': 'флаг',
        'direction': 'bullish' if impulse_move > 0 else 'bearish',
        'impulse_pct': impulse_pct
    }


def detect_pennant(df: pd.DataFrame, lookback: int = 15) -> Optional[Dict]:
    if len(df) < lookback + 10:
        return None
    impulse = df.iloc[-(lookback + 10):-lookback]
    impulse_move = impulse['close'].iloc[-1] - impulse['close'].iloc[0]
    impulse_pct = abs(impulse_move) / impulse['close'].iloc[0] * 100
    if impulse_pct < 2.0:
        return None
    cons = df.iloc[-lookback:]
    highs = cons['high'].rolling(3).max().dropna()
    lows = cons['low'].rolling(3).min().dropna()
    if len(highs) < 5 or len(lows) < 5:
        return None
    high_slope = np.polyfit(range(len(highs)), highs.values, 1)[0]
    low_slope = np.polyfit(range(len(lows)), lows.values, 1)[0]
    if high_slope < 0 and low_slope > 0:
        return {
            'pattern': 'вымпел',
            'direction': 'bullish' if impulse_move > 0 else 'bearish'
        }
    return None


def detect_triangle(df: pd.DataFrame, lookback: int = 20) -> Optional[Dict]:
    if len(df) < lookback:
        return None
    highs = df['high'].iloc[-lookback:].rolling(3).max().dropna()
    lows = df['low'].iloc[-lookback:].rolling(3).min().dropna()
    if len(highs) < 10 or len(lows) < 10:
        return None
    high_slope = np.polyfit(range(len(highs)), highs.values, 1)[0]
    low_slope = np.polyfit(range(len(lows)), lows.values, 1)[0]
    price_level = df['close'].iloc[-1]
    high_slope_norm = high_slope / price_level * 100
    low_slope_norm = low_slope / price_level * 100
    if abs(high_slope_norm) < 0.01 and abs(low_slope_norm) < 0.01:
        return {'pattern': 'симметричный треугольник', 'direction': 'neutral'}
    if abs(high_slope_norm) < 0.02 and low_slope_norm > 0.01:
        return {'pattern': 'восходящий треугольник', 'direction': 'bullish'}
    if abs(low_slope_norm) < 0.02 and high_slope_norm < -0.01:
        return {'pattern': 'нисходящий треугольник', 'direction': 'bearish'}
    return None


def detect_wedge(df: pd.DataFrame, lookback: int = 20) -> Optional[Dict]:
    if len(df) < lookback:
        return None
    highs = df['high'].iloc[-lookback:].rolling(3).max().dropna()
    lows = df['low'].iloc[-lookback:].rolling(3).min().dropna()
    if len(highs) < 10 or len(lows) < 10:
        return None
    high_slope = np.polyfit(range(len(highs)), highs.values, 1)[0]
    low_slope = np.polyfit(range(len(lows)), lows.values, 1)[0]
    price_level = df['close'].iloc[-1]
    high_slope_norm = high_slope / price_level * 100
    low_slope_norm = low_slope / price_level * 100
    if high_slope_norm > 0.01 and low_slope_norm > 0.01:
        if abs(high_slope_norm - low_slope_norm) < 0.03:
            return {'pattern': 'восходящий клин', 'direction': 'bearish'}
    if high_slope_norm < -0.01 and low_slope_norm < -0.01:
        if abs(high_slope_norm - low_slope_norm) < 0.03:
            return {'pattern': 'нисходящий клин', 'direction': 'bullish'}
    return None


def detect_diamond(df: pd.DataFrame, lookback: int = 30) -> Optional[Dict]:
    if len(df) < lookback:
        return None
    mid = lookback // 2
    first_half = df.iloc[-lookback:-mid]
    second_half = df.iloc[-mid:]
    range1 = first_half['high'].max() - first_half['low'].min()
    range2 = second_half['high'].max() - second_half['low'].min()
    if range1 > 0 and range2 / range1 < 0.6:
        trend = df['close'].iloc[-lookback] - df['close'].iloc[-lookback - 10] if len(df) > lookback + 10 else 0
        direction = 'bearish' if trend > 0 else 'bullish'
        return {'pattern': 'бриллиант', 'direction': direction}
    return None


# ==================== ОБЪЁМНЫЙ АНАЛИЗ ====================

def check_volume_confirmation(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback:
        return False
    avg_volume = df['volume'].iloc[-lookback:-1].mean()
    last_volume = df['volume'].iloc[-1]
    return last_volume > avg_volume * 1.5


def check_volume_divergence(df: pd.DataFrame, lookback: int = 30) -> Optional[str]:
    if len(df) < lookback:
        return None
    recent = df.iloc[-lookback:]
    price_slope = np.polyfit(range(len(recent)), recent['close'].values, 1)[0]
    volume_slope = np.polyfit(range(len(recent)), recent['volume'].values, 1)[0]
    if price_slope > 0 and volume_slope < 0:
        return 'bearish'
    if price_slope < 0 and volume_slope < 0:
        return 'bullish'
    return None


# ==================== ДИВЕРГЕНЦИИ RSI ====================

def check_rsi_divergence(df: pd.DataFrame, lookback: int = 30) -> Optional[str]:
    if len(df) < lookback + 14:
        return None
    rsi = df.ta.rsi(length=14)
    if rsi is None or len(rsi.dropna()) < lookback:
        return None
    recent_price = df['close'].iloc[-lookback:]
    recent_rsi = rsi.iloc[-lookback:]
    price_highs = recent_price.rolling(5).max()
    rsi_highs = recent_rsi.rolling(5).max()
    price_lows = recent_price.rolling(5).min()
    rsi_lows = recent_rsi.rolling(5).min()
    p1_idx = len(recent_price) - 5
    p2_idx = len(recent_price) - 1
    if price_highs.iloc[p2_idx] > price_highs.iloc[p1_idx]:
        if rsi_highs.iloc[p2_idx] < rsi_highs.iloc[p1_idx]:
            return 'bearish'
    if price_lows.iloc[p2_idx] < price_lows.iloc[p1_idx]:
        if rsi_lows.iloc[p2_idx] > rsi_lows.iloc[p1_idx]:
            return 'bullish'
    return None


# ==================== АГРЕГАТОР ====================

def analyze_advanced_patterns(df: pd.DataFrame, signal_dir: str) -> Dict:
    confirmations = []
    
    if signal_dir == 'LONG':
        if detect_hammer(df):
            confirmations.append('молот')
        if detect_morning_star(df):
            confirmations.append('утренняя звезда')
        if detect_pin_bar(df, 'bullish'):
            confirmations.append('бычий пин-бар')
    else:
        if detect_hanging_man(df):
            confirmations.append('повешенный')
        if detect_evening_star(df):
            confirmations.append('вечерняя звезда')
        if detect_pin_bar(df, 'bearish'):
            confirmations.append('медвежий пин-бар')
    
    flag = detect_flag(df)
    if flag and flag['direction'] == signal_dir.lower():
        confirmations.append(flag['pattern'])
    
    pennant = detect_pennant(df)
    if pennant and pennant['direction'] == signal_dir.lower():
        confirmations.append(pennant['pattern'])
    
    triangle = detect_triangle(df)
    if triangle and triangle['direction'] == signal_dir.lower():
        confirmations.append(triangle['pattern'])
    
    wedge = detect_wedge(df)
    if wedge and wedge['direction'] == signal_dir.lower():
        confirmations.append(wedge['pattern'])
    
    diamond = detect_diamond(df)
    if diamond and diamond['direction'] == signal_dir.lower():
        confirmations.append(diamond['pattern'])
    
    if check_volume_confirmation(df):
        confirmations.append('рост объёма')
    
    vol_div = check_volume_divergence(df)
    if vol_div and vol_div == signal_dir.lower():
        confirmations.append('дивергенция объёма')
    
    rsi_div = check_rsi_divergence(df)
    if rsi_div and rsi_div == signal_dir.lower():
        confirmations.append('RSI дивергенция')
    
    count = len(confirmations)
    if count >= 3:
        confidence = 'STRONG'
    elif count >= 1:
        confidence = 'MEDIUM'
    else:
        confidence = 'WEAK'
    
    return {
        'confirmations': confirmations,
        'count': count,
        'confidence': confidence
    }
