import os
import json
import time
import threading
import datetime
from collections import deque
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import pandas as pd
import pandas_ta as ta
import ccxt
import requests
import websocket


# ==================== КОНФИГУРАЦИЯ ====================

COINGLASS_BASE = "https://open-api-v4.coinglass.com"
COINGLASS_HEADERS = {
    "accept": "application/json",
    "CG-API-KEY": os.environ.get("COINGLASS_API_KEY", "")
}

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT"
]

TIMEFRAME_A = "5m"
TIMEFRAME_B = "3m"
SWING_ATR_MULT = 1.5
SWING_MIN_BARS = 3
CANDLES_LIMIT = 300
WS_WAIT_SECONDS = 10

STATS_FILE = Path("signal_stats.json")
DAILY_REPORT_HOUR_UTC = 0

# Через сколько часов считать сигнал «не сработавшим»
SIGNAL_TIMEOUT_HOURS = 6
# Минимальный возраст сигнала для проверки (часы)
SIGNAL_MIN_AGE_HOURS = 1


# ==================== СТАТИСТИКА ====================

class SignalStats:
    def __init__(self, path: Path = STATS_FILE):
        self.path = path
        self.data = self._load()
    
    def _load(self) -> Dict:
        if self.path.exists():
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Stats] Load error: {e}")
        
        return {
            "total_signals": 0,
            "a_signals": 0,
            "b_signals": 0,
            "long_count": 0,
            "short_count": 0,
            "by_symbol": {},
            "by_day": {},
            "active_signals": [],
            "completed_signals": [],
            "tp_stats": {
                "tp1_hit": 0, "tp2_hit": 0, "tp3_hit": 0,
                "sl_hit": 0, "no_hit": 0
            },
            "last_report_date": None,
            "started_at": datetime.datetime.utcnow().isoformat()
        }
    
    def save(self):
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Stats] Save error: {e}")
    
    def add_signal(self, signal: Dict, symbol: str):
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        sig_type = signal.get('type', 'A')
        
        self.data["total_signals"] += 1
        
        if sig_type == 'A':
            self.data["a_signals"] = self.data.get("a_signals", 0) + 1
        else:
            self.data["b_signals"] = self.data.get("b_signals", 0) + 1
        
        if signal['signal'] == 'LONG':
            self.data["long_count"] += 1
        else:
            self.data["short_count"] += 1
        
        if symbol not in self.data["by_symbol"]:
            self.data["by_symbol"][symbol] = {"LONG": 0, "SHORT": 0}
        self.data["by_symbol"][symbol][signal['signal']] += 1
        
        if today not in self.data["by_day"]:
            self.data["by_day"][today] = {
                "total": 0, "A": 0, "B": 0,
                "LONG": 0, "SHORT": 0, "symbols": {}
            }
        
        day = self.data["by_day"][today]
        day["total"] += 1
        day[sig_type] += 1
        day[signal['signal']] += 1
        
        if symbol not in day["symbols"]:
            day["symbols"][symbol] = 0
        day["symbols"][symbol] += 1
        
        # Сохраняем сигнал для последующей проверки TP/SL
        if "active_signals" not in self.data:
            self.data["active_signals"] = []
        
        self.data["active_signals"].append({
            "symbol": symbol,
            "signal": signal['signal'],
            "type": sig_type,
            "entry": signal['entry'],
            "stop_loss": signal['stop_loss'],
            "tp1": signal['tp1'],
            "tp2": signal['tp2'],
            "tp3": signal['tp3'],
            "atr": signal['atr'],
            "created_at": datetime.datetime.utcnow().isoformat(),
            "result": None
        })
    
    def should_send_daily_report(self) -> bool:
        now = datetime.datetime.utcnow()
        today = now.strftime("%Y-%m-%d")
        
        if now.hour >= DAILY_REPORT_HOUR_UTC:
            if self.data.get("last_report_date") != today:
                yesterday = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                if yesterday in self.data["by_day"]:
                    return True
        return False
    
    def build_daily_report(self) -> Optional[str]:
        now = datetime.datetime.utcnow()
        yesterday = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        day_data = self.data["by_day"].get(yesterday)
        if not day_data:
            return None
        
        text = f"📊 <b>Отчёт за {yesterday}</b>\n\n"
        text += f"Всего сигналов: <b>{day_data['total']}</b>\n"
        text += f"🚀 A-SIGNAL: {day_data.get('A', 0)}\n"
        text += f"⚡ B-SIGNAL: {day_data.get('B', 0)}\n"
        text += f"🟢 LONG: {day_data['LONG']} | 🔴 SHORT: {day_data['SHORT']}\n\n"
        
        if day_data.get('symbols'):
            text += "<b>По монетам:</b>\n"
            sorted_symbols = sorted(
                day_data['symbols'].items(), 
                key=lambda x: x[1], reverse=True
            )
            for sym, cnt in sorted_symbols:
                text += f"  • {sym}: {cnt}\n"
        
        text += f"\n📈 <b>Всего за всё время:</b>\n"
        text += f"Сигналов: {self.data['total_signals']}\n"
        text += f"🚀 A: {self.data.get('a_signals', 0)} | ⚡ B: {self.data.get('b_signals', 0)}\n"
        text += f"🟢 {self.data['long_count']} | 🔴 {self.data['short_count']}"
        
        # Статистика TP/SL
        tp_stats = self.data.get("tp_stats", {})
        total_checked = sum(tp_stats.values())
        
        if total_checked > 0:
            text += f"\n\n🎯 <b>Статистика TP/SL ({total_checked} проверено):</b>\n"
            text += f"  TP1: {tp_stats.get('tp1_hit', 0)}\n"
            text += f"  TP2: {tp_stats.get('tp2_hit', 0)}\n"
            text += f"  TP3: {tp_stats.get('tp3_hit', 0)}\n"
            text += f"  SL: {tp_stats.get('sl_hit', 0)}\n"
            
            if tp_stats.get('no_hit', 0) > 0:
                text += f"  Без движения: {tp_stats['no_hit']}\n"
            
            wins = (tp_stats.get('tp1_hit', 0) + 
                    tp_stats.get('tp2_hit', 0) + 
                    tp_stats.get('tp3_hit', 0))
            win_rate = wins / total_checked * 100
            text += f"\n  <b>Win rate: {win_rate:.1f}%</b>"
        
        return text
    
    def mark_report_sent(self):
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        self.data["last_report_date"] = today


# ==================== ПРОВЕРКА TP/SL ====================

def check_active_signals(stats: SignalStats):
    """
    Проверяет активные сигналы: достигли ли они TP или SL.
    Вызывается при каждом запуске бота перед поиском новых сигналов.
    """
    active = stats.data.get("active_signals", [])
    if not active:
        print("[Check] No active signals to check")
        return
    
    now = datetime.datetime.utcnow()
    still_active = []
    checked = 0
    
    for sig in active:
        try:
            signal_time = datetime.datetime.fromisoformat(sig["created_at"])
            hours_elapsed = (now - signal_time).total_seconds() / 3600
            
            # Проверяем только сигналы старше SIGNAL_MIN_AGE_HOURS
            if hours_elapsed < SIGNAL_MIN_AGE_HOURS:
                still_active.append(sig)
                continue
            
            # Загружаем свечи для проверки
            df = fetch_ohlcv(sig["symbol"], TIMEFRAME_A, limit=200)
            
            entry = sig["entry"]
            sl = sig["stop_loss"]
            tp1 = sig["tp1"]
            tp2 = sig["tp2"]
            tp3 = sig["tp3"]
            direction = sig["signal"]
            
            # Ограничиваем проверку свечами после сигнала
            df_after = df[df.index >= signal_time.replace(tzinfo=None)]
            
            if len(df_after) == 0:
                still_active.append(sig)
                continue
            
            if direction == "LONG":
                # SL ниже входа — если low свечей ушёл ниже SL
                if df_after["low"].min() <= sl:
                    sig["result"] = "SL"
                    sig["hit_price"] = sl
                    stats.data["tp_stats"]["sl_hit"] += 1
                # TP3 выше — если high ушёл выше TP3
                elif df_after["high"].max() >= tp3:
                    sig["result"] = "TP3"
                    sig["hit_price"] = tp3
                    stats.data["tp_stats"]["tp3_hit"] += 1
                elif df_after["high"].max() >= tp2:
                    sig["result"] = "TP2"
                    sig["hit_price"] = tp2
                    stats.data["tp_stats"]["tp2_hit"] += 1
                elif df_after["high"].max() >= tp1:
                    sig["result"] = "TP1"
                    sig["hit_price"] = tp1
                    stats.data["tp_stats"]["tp1_hit"] += 1
                elif hours_elapsed > SIGNAL_TIMEOUT_HOURS:
                    sig["result"] = "NO_HIT"
                    stats.data["tp_stats"]["no_hit"] += 1
                else:
                    still_active.append(sig)
                    continue
            else:  # SHORT
                # SL выше входа — если high ушёл выше SL
                if df_after["high"].max() >= sl:
                    sig["result"] = "SL"
                    sig["hit_price"] = sl
                    stats.data["tp_stats"]["sl_hit"] += 1
                # TP3 ниже — если low ушёл ниже TP3
                elif df_after["low"].min() <= tp3:
                    sig["result"] = "TP3"
                    sig["hit_price"] = tp3
                    stats.data["tp_stats"]["tp3_hit"] += 1
                elif df_after["low"].min() <= tp2:
                    sig["result"] = "TP2"
                    sig["hit_price"] = tp2
                    stats.data["tp_stats"]["tp2_hit"] += 1
                elif df_after["low"].min() <= tp1:
                    sig["result"] = "TP1"
                    sig["hit_price"] = tp1
                    stats.data["tp_stats"]["tp1_hit"] += 1
                elif hours_elapsed > SIGNAL_TIMEOUT_HOURS:
                    sig["result"] = "NO_HIT"
                    stats.data["tp_stats"]["no_hit"] += 1
                else:
                    still_active.append(sig)
                    continue
            
            sig["checked_at"] = now.isoformat()
            sig["hours_to_result"] = round(hours_elapsed, 1)
            checked += 1
            
            # Перемещаем в завершённые
            if "completed_signals" not in stats.data:
                stats.data["completed_signals"] = []
            stats.data["completed_signals"].append(sig)
            
            print(f"  [Check] {sig['symbol']} {sig['signal']}: {sig['result']}")
            
        except Exception as e:
            print(f"  [Check] Error for {sig.get('symbol', '?')}: {e}")
            still_active.append(sig)
    
    stats.data["active_signals"] = still_active
    if checked > 0:
        print(f"[Stats] Checked {checked} signals, {len(still_active)} still active")


# ==================== COINGLASS REST ====================

class CoinGlassClient:
    def __init__(self, cache_ttl: int = 90):
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.last_request = 0
        self.min_interval = 1.5

    def _throttle(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()

    def _get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        cache_key = f"{endpoint}:{str(sorted(params.items()))}"
        if cache_key in self.cache:
            ts, data = self.cache[cache_key]
            if time.time() - ts < self.cache_ttl:
                return data

        self._throttle()
        try:
            resp = requests.get(
                f"{COINGLASS_BASE}{endpoint}",
                headers=COINGLASS_HEADERS,
                params=params, timeout=10
            )
            if resp.status_code == 429:
                time.sleep(30)
                return self._get(endpoint, params)
            if resp.status_code != 200:
                return None
            data = resp.json()
            self.cache[cache_key] = (time.time(), data)
            return data
        except Exception as e:
            print(f"[CoinGlass] {e}")
            return None

    def get_funding_rate(self, symbol: str):
        return self._get("/api/futures/funding-rate/history", {
            "symbol": symbol, "interval": "5m", "limit": 10
        })

    def get_open_interest(self, symbol: str):
        return self._get("/api/futures/open-interest/history", {
            "symbol": symbol, "interval": "5m", "limit": 20
        })

    def get_long_short_ratio(self, symbol: str):
        return self._get("/api/futures/global-long-short-account-ratio/history", {
            "symbol": symbol, "interval": "5m", "limit": 10
        })


# ==================== WEBSOCKET ЛИКВИДАЦИЙ ====================

class QuickLiquidationStream:
    def __init__(self, api_key: str):
        self.ws_url = f"wss://open-ws.coinglass.com/ws-api?cg-api-key={api_key}"
        self.recent = deque(maxlen=200)
        self._lock = threading.Lock()
        self.should_run = True
        self.ws = None
        self.connected = False

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            channel = data.get("channel") or data.get("c")
            
            if channel in ("liquidation_orders", "liquidationOrders"):
                orders = data.get("data") or data.get("d") or []
                with self._lock:
                    for order in orders:
                        self.recent.append({
                            "side": order.get("side") or order.get("s"),
                            "volume_usd": float(order.get("volume_usd") or order.get("v") or 0),
                            "timestamp": time.time()
                        })
        except Exception:
            pass

    def _on_open(self, ws):
        self.connected = True
        ws.send(json.dumps({
            "method": "subscribe",
            "channels": ["liquidationOrders"]
        }))

    def _on_error(self, ws, error):
        self.connected = False

    def _on_close(self, ws, code, msg):
        self.connected = False

    def _run(self):
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            self.ws.run_forever(ping_interval=20, ping_timeout=5)
        except Exception:
            pass

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.should_run = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

    def get_pressure(self, window_seconds: int = 300) -> Dict:
        cutoff = time.time() - window_seconds
        long_liq = 0.0
        short_liq = 0.0
        with self._lock:
            for liq in self.recent:
                if liq["timestamp"] > cutoff:
                    if liq["side"] in ("long", "LONG"):
                        long_liq += liq["volume_usd"]
                    else:
                        short_liq += liq["volume_usd"]
        return {"long_liquidated_usd": long_liq, "short_liquidated_usd": short_liq}


# ==================== ИНДИКАТОРЫ ====================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    st = df.ta.supertrend(length=10, multiplier=3.0)
    st_dir_col = [c for c in st.columns if c.startswith('SUPERTd')][0]
    df['st_direction'] = st[st_dir_col]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, unit='ms')

    df['vwap'] = df.ta.vwap()

    bb = df.ta.bbands(length=20, std=2)
    bb_upper_col = [c for c in bb.columns if c.startswith('BBU')][0]
    bb_lower_col = [c for c in bb.columns if c.startswith('BBL')][0]
    df['bb_upper'] = bb[bb_upper_col]
    df['bb_lower'] = bb[bb_lower_col]

    stochrsi = df.ta.stochrsi(length=14, rsi_length=7, k=3, d=3)
    stoch_k_col = [c for c in stochrsi.columns if c.startswith('STOCHRSIk')][0]
    stoch_d_col = [c for c in stochrsi.columns if c.startswith('STOCHRSId')][0]
    df['stoch_k'] = stochrsi[stoch_k_col]
    df['stoch_d'] = stochrsi[stoch_d_col]

    df['atr'] = df.ta.atr(length=14)
    df['ema_200'] = df.ta.ema(length=200)

    return df


def apply_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    ha_open = [df['open'].iloc[0]]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + df['ha_close'].iloc[i-1]) / 2)
    df['ha_open'] = ha_open
    df['ha_direction'] = (df['ha_close'] > df['ha_open']).astype(int) * 2 - 1
    
    return df


# ==================== ПАТТЕРНЫ ====================

def detect_candlestick_patterns(df: pd.DataFrame) -> Dict:
    patterns = {}
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body_last = abs(last['close'] - last['open'])
    range_last = last['high'] - last['low']

    if (prev['close'] < prev['open'] and
        last['close'] > last['open'] and
        last['close'] > prev['open'] and
        last['open'] < prev['close']):
        patterns['engulfing_bullish'] = True

    if (prev['close'] > prev['open'] and
        last['close'] < last['open'] and
        last['close'] < prev['open'] and
        last['open'] > prev['close']):
        patterns['engulfing_bearish'] = True

    if range_last > 0 and body_last / range_last < 0.33:
        lower_wick = min(last['open'], last['close']) - last['low']
        upper_wick = last['high'] - max(last['open'], last['close'])
        if lower_wick > body_last * 2 and upper_wick < body_last * 0.5:
            patterns['hammer'] = True
            patterns['pin_bar_bullish'] = True
        if upper_wick > body_last * 2 and lower_wick < body_last * 0.5:
            patterns['shooting_star'] = True
            patterns['pin_bar_bearish'] = True

    return patterns


def find_swings_atr(df: pd.DataFrame, atr_mult: float = 1.5,
                    min_bars: int = 3) -> List[Tuple]:
    if 'atr' not in df.columns:
        df['atr'] = df.ta.atr(length=14)

    swings = []
    last_idx = 0
    last_price = df['close'].iloc[0]
    direction = None

    for i in range(1, len(df)):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        threshold = df['atr'].iloc[i] * atr_mult

        if direction is None:
            if high - last_price > threshold:
                direction = 'up'
            elif last_price - low > threshold:
                direction = 'down'
            continue

        if direction == 'up':
            if high > last_price:
                last_price = high
                last_idx = i
            elif last_price - low > threshold:
                if i - last_idx >= min_bars:
                    swings.append((last_idx, last_price, 'H'))
                direction = 'down'
                last_price = low
                last_idx = i
        else:
            if low < last_price:
                last_price = low
                last_idx = i
            elif high - last_price > threshold:
                if i - last_idx >= min_bars:
                    swings.append((last_idx, last_price, 'L'))
                direction = 'up'
                last_price = high
                last_idx = i

    return swings


def detect_head_and_shoulders(swings: List[Tuple]) -> Optional[Dict]:
    highs = [s for s in swings if s[2] == 'H']
    if len(highs) < 3:
        return None
    left, head, right = highs[-3], highs[-2], highs[-1]
    tol = 0.02
    if (head[1] > left[1] * (1 + tol) and
        head[1] > right[1] * (1 + tol) and
        abs(left[1] - right[1]) / left[1] < tol):
        return {'pattern': 'голова и плечи', 'signal': 'SHORT'}
    return None


def detect_double_top_bottom(swings: List[Tuple]) -> Optional[Dict]:
    highs = [s for s in swings if s[2] == 'H']
    lows = [s for s in swings if s[2] == 'L']
    tol = 0.02

    if len(highs) >= 2:
        f, s = highs[-2], highs[-1]
        if abs(f[1] - s[1]) / f[1] < tol:
            return {'pattern': 'двойная вершина', 'signal': 'SHORT'}
    if len(lows) >= 2:
        f, s = lows[-2], lows[-1]
        if abs(f[1] - s[1]) / f[1] < tol:
            return {'pattern': 'двойное дно', 'signal': 'LONG'}
    return None


# ==================== A-СИГНАЛ (5m) ====================

def generate_signal_a(df: pd.DataFrame) -> Optional[Dict]:
    if len(df) < 210:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if (last['st_direction'] == 1 and
        last['close'] > last['vwap'] and
        last['close'] > last['ema_200'] and
        prev['stoch_k'] < 20 and last['stoch_k'] > last['stoch_d'] and
        prev['low'] <= prev['bb_lower'] and last['close'] > last['bb_lower']):
        return {
            "signal": "LONG", "type": "A",
            "entry": float(last['close']),
            "stop_loss": float(last['close'] - last['atr'] * 1.5),
            "tp1": float(last['close'] + last['atr'] * 1.0),
            "tp2": float(last['close'] + last['atr'] * 2.0),
            "tp3": float(last['close'] + last['atr'] * 3.0),
            "atr": float(last['atr'])
        }

    if (last['st_direction'] == -1 and
        last['close'] < last['vwap'] and
        last['close'] < last['ema_200'] and
        prev['stoch_k'] > 80 and last['stoch_k'] < last['stoch_d'] and
        prev['high'] >= prev['bb_upper'] and last['close'] < last['bb_upper']):
        return {
            "signal": "SHORT", "type": "A",
            "entry": float(last['close']),
            "stop_loss": float(last['close'] + last['atr'] * 1.5),
            "tp1": float(last['close'] - last['atr'] * 1.0),
            "tp2": float(last['close'] - last['atr'] * 2.0),
            "tp3": float(last['close'] - last['atr'] * 3.0),
            "atr": float(last['atr'])
        }

    return None


# ==================== B-СИГНАЛ (3m) ====================

def generate_signal_b(df_3m: pd.DataFrame) -> Optional[Dict]:
    if len(df_3m) < 100:
        return None

    df = calculate_indicators(df_3m)
    df = apply_heikin_ashi(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    atr_pct = last['atr'] / last['close'] * 100
    if atr_pct < 0.1:
        return None

    patterns = detect_candlestick_patterns(df)

    ha_flip_bull = (prev['ha_direction'] == -1 and last['ha_direction'] == 1)
    has_bull_pattern = (
        patterns.get('engulfing_bullish') or
        patterns.get('hammer') or
        patterns.get('pin_bar_bullish')
    )

    if (last['st_direction'] == 1 and
        last['close'] > last['vwap'] and
        (ha_flip_bull or last['ha_direction'] == 1) and
        has_bull_pattern):
        return {
            "signal": "LONG", "type": "B",
            "entry": float(last['close']),
            "stop_loss": float(last['close'] - last['atr'] * 1.2),
            "tp1": float(last['close'] + last['atr'] * 0.8),
            "tp2": float(last['close'] + last['atr'] * 1.5),
            "tp3": float(last['close'] + last['atr'] * 2.2),
            "atr": float(last['atr']),
            "recommended_size": "30-50%"
        }

    ha_flip_bear = (prev['ha_direction'] == 1 and last['ha_direction'] == -1)
    has_bear_pattern = (
        patterns.get('engulfing_bearish') or
        patterns.get('shooting_star') or
        patterns.get('pin_bar_bearish')
    )

    if (last['st_direction'] == -1 and
        last['close'] < last['vwap'] and
        (ha_flip_bear or last['ha_direction'] == -1) and
        has_bear_pattern):
        return {
            "signal": "SHORT", "type": "B",
            "entry": float(last['close']),
            "stop_loss": float(last['close'] + last['atr'] * 1.2),
            "tp1": float(last['close'] - last['atr'] * 0.8),
            "tp2": float(last['close'] - last['atr'] * 1.5),
            "tp3": float(last['close'] - last['atr'] * 2.2),
            "atr": float(last['atr']),
            "recommended_size": "30-50%"
        }

    return None


# ==================== ФИЛЬТР COINGLASS ====================

def apply_coinglass_filter(signal: Dict, cg: CoinGlassClient, symbol: str) -> Dict:
    if not signal:
        return None

    warnings = []
    confirmations = []

    funding = cg.get_funding_rate(symbol)
    if funding and funding.get('data'):
        rate = float(funding['data'][0].get('close', 0))
        if signal['signal'] == 'LONG' and rate > 0.001:
            warnings.append(f"Funding {rate*100:.3f}%")
        if signal['signal'] == 'SHORT' and rate < -0.001:
            warnings.append(f"Funding {rate*100:.3f}%")

    oi = cg.get_open_interest(symbol)
    if oi and oi.get('data') and len(oi['data']) >= 2:
        curr = float(oi['data'][0].get('close', 0))
        prev = float(oi['data'][1].get('close', 0))
        change = (curr - prev) / prev * 100 if prev else 0
        if abs(change) > 1.0:
            confirmations.append(f"OI {change:+.2f}%")

    ls = cg.get_long_short_ratio(symbol)
    if ls and ls.get('data'):
        ratio = float(ls['data'][0].get('longShortRatio', 1))
        if signal['signal'] == 'LONG' and ratio > 2.5:
            warnings.append(f"L/S {ratio:.2f}")
        if signal['signal'] == 'SHORT' and ratio < 0.5:
            warnings.append(f"L/S {ratio:.2f}")

    signal['warnings'] = warnings
    signal['confirmations'] = confirmations
    return signal


# ==================== TELEGRAM ====================

def _send_telegram_raw(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[Telegram] Missing credentials")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": chat_id, "text": text, "parse_mode": "HTML"
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[Telegram] {e}")
        return False


def send_signal(signal: Dict, symbol: str):
    if not signal:
        return

    sig_type = signal.get('type', 'A')

    if sig_type == 'B':
        header = f"⚡ <b>B-SIGNAL {signal['signal']} {symbol}</b> [3m]\n"
        header += f"<i>Рекомендуемый размер: {signal.get('recommended_size', '30-50%')}</i>\n"
    else:
        header = f"🚀 <b>A-SIGNAL {signal['signal']} {symbol}</b> [5m]\n"

    text = header + "\n"
    text += f"💰 Вход: {signal['entry']:.4f}\n"
    text += f"🛑 Стоп: {signal['stop_loss']:.4f}\n"
    text += f"📊 ATR: {signal['atr']:.4f}\n\n"
    text += f"🎯 TP1: {signal['tp1']:.4f}\n"
    text += f"🎯 TP2: {signal['tp2']:.4f}\n"
    text += f"🎯 TP3: {signal['tp3']:.4f}\n"

    if signal.get('confirmations'):
        text += "\n✅ " + ", ".join(signal['confirmations'])
    if signal.get('warnings'):
        text += "\n⚠️ " + ", ".join(signal['warnings'])

    if _send_telegram_raw(text):
        print(f"  [Telegram] {sig_type}-signal sent for {symbol}")


# ==================== БИРЖА (OKX) ====================

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    okx_symbol = symbol.replace("USDT", "-USDT")
    
    exchange = ccxt.okx({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(okx_symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low',
                                       'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


# ==================== MAIN ====================

def main():
    print("=" * 50)
    print("Crypto Scalper Bot — A(5m) + B(3m) [OKX]")
    print(f"Time: {datetime.datetime.utcnow().isoformat()} UTC")
    print("=" * 50)

    api_key = os.environ.get("COINGLASS_API_KEY", "")
    if not api_key:
        print("[ERROR] COINGLASS_API_KEY not set")
        return

    stats = SignalStats()
    print(f"[Stats] Total signals so far: {stats.data['total_signals']}")
    print(f"[Stats] Active signals: {len(stats.data.get('active_signals', []))}")

    # Проверяем активные сигналы ПЕРЕД поиском новых
    check_active_signals(stats)

    cg = CoinGlassClient(cache_ttl=90)

    liq_stream = QuickLiquidationStream(api_key)
    liq_stream.start()
    print(f"[WS] Collecting liquidations for {WS_WAIT_SECONDS}s...")
    time.sleep(WS_WAIT_SECONDS)

    new_signals = 0

    for symbol in SYMBOLS:
        try:
            # ===== A-СИГНАЛ (5m) =====
            print(f"\n--- {symbol} [A-SIGNAL 5m] ---")
            df_a = fetch_ohlcv(symbol, TIMEFRAME_A, limit=CANDLES_LIMIT)
            df_a = calculate_indicators(df_a)

            signal_a = generate_signal_a(df_a)
            if signal_a:
                patterns = detect_candlestick_patterns(df_a)
                swings = find_swings_atr(df_a, SWING_ATR_MULT, SWING_MIN_BARS)
                hs = detect_head_and_shoulders(swings)
                dt = detect_double_top_bottom(swings)

                confirmations = []
                if signal_a['signal'] == 'LONG' and patterns.get('engulfing_bullish'):
                    confirmations.append('бычье поглощение')
                if signal_a['signal'] == 'SHORT' and patterns.get('engulfing_bearish'):
                    confirmations.append('медвежье поглощение')
                if hs and hs['signal'] == signal_a['signal']:
                    confirmations.append(hs['pattern'])
                if dt and dt['signal'] == signal_a['signal']:
                    confirmations.append(dt['pattern'])

                pressure = liq_stream.get_pressure(window_seconds=300)
                if signal_a['signal'] == 'LONG' and pressure['long_liquidated_usd'] > 1_000_000:
                    confirmations.append(f"Long squeeze ${pressure['long_liquidated_usd']/1e6:.1f}M")
                if signal_a['signal'] == 'SHORT' and pressure['short_liquidated_usd'] > 1_000_000:
                    confirmations.append(f"Short squeeze ${pressure['short_liquidated_usd']/1e6:.1f}M")

                signal_a = apply_coinglass_filter(signal_a, cg, symbol)
                if confirmations:
                    signal_a['confirmations'] = signal_a.get('confirmations', []) + confirmations

                print(f"  A: {signal_a['signal']} @ {signal_a['entry']:.4f}")
                send_signal(signal_a, symbol)
                stats.add_signal(signal_a, symbol)
                new_signals += 1
            else:
                print(f"  No A-signal")

            # ===== B-СИГНАЛ (3m) =====
            print(f"--- {symbol} [B-SIGNAL 3m] ---")
            df_b = fetch_ohlcv(symbol, TIMEFRAME_B, limit=CANDLES_LIMIT)
            signal_b = generate_signal_b(df_b)

            if signal_b:
                signal_b = apply_coinglass_filter(signal_b, cg, symbol)
                print(f"  B: {signal_b['signal']} @ {signal_b['entry']:.4f}")
                send_signal(signal_b, symbol)
                stats.add_signal(signal_b, symbol)
                new_signals += 1
            else:
                print(f"  No B-signal")

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")

    liq_stream.stop()
    stats.save()
    print(f"\n[Stats] New signals this run: {new_signals}")

    if stats.should_send_daily_report():
        report = stats.build_daily_report()
        if report:
            print("[Report] Sending daily report...")
            if _send_telegram_raw(report):
                stats.mark_report_sent()
                stats.save()
                print("[Report] Sent successfully")

    print("\nDone.")


if __name__ == "__main__":
    main()
