# trading_bot/utils/indicators.py

import pandas as pd # Ensure pandas is installed (e.g., pip install pandas)
import numpy as np # For NaN handling in RSI

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    # ... (content of this function is unchanged but kept for completeness) ...
    if not isinstance(data, pd.Series): raise TypeError("Input 'data' must be a pandas Series.")
    if not isinstance(period, int) or period <= 0: raise ValueError("Input 'period' must be a positive integer.")
    if len(data) < period: return pd.Series([np.nan] * len(data), index=data.index, dtype='float64')
    return data.ewm(span=period, adjust=False, min_periods=period).mean()

def get_ema_signals(prices: pd.Series, fast_period: int, mid_period: int, slow_period: int) -> dict:
    # ... (content of this function is unchanged but kept for completeness) ...
    if not isinstance(prices, pd.Series): raise TypeError("Input 'prices' must be a pandas Series.")
    if not all(isinstance(p, int) and p > 0 for p in [fast_period, mid_period, slow_period]): raise ValueError("EMA periods must be positive integers.")
    if not (fast_period < mid_period < slow_period): raise ValueError("EMA periods should be ordered: fast < mid < slow.")
    ema_fast = calculate_ema(prices, fast_period); ema_mid = calculate_ema(prices, mid_period); ema_slow = calculate_ema(prices, slow_period)
    results = {"fast_ema": np.nan, "mid_ema": np.nan, "slow_ema": np.nan, "trendUp": False, "trendDown": False, "emaFastCrossAboveMid": False}
    if len(prices) < max(slow_period, 2) or ema_fast.isna().all() or ema_mid.isna().all() or ema_slow.isna().all(): return results
    latest_fast_ema = ema_fast.iloc[-1]; latest_mid_ema = ema_mid.iloc[-1]; latest_slow_ema = ema_slow.iloc[-1]
    results["fast_ema"] = latest_fast_ema if pd.notna(latest_fast_ema) else np.nan
    results["mid_ema"] = latest_mid_ema if pd.notna(latest_mid_ema) else np.nan
    results["slow_ema"] = latest_slow_ema if pd.notna(latest_slow_ema) else np.nan
    if pd.isna(latest_fast_ema) or pd.isna(latest_mid_ema) or pd.isna(latest_slow_ema): return results
    results["trendUp"] = latest_fast_ema > latest_mid_ema and latest_mid_ema > latest_slow_ema
    results["trendDown"] = latest_fast_ema < latest_mid_ema and latest_mid_ema < latest_slow_ema
    if len(ema_fast.dropna()) >= 2 and len(ema_mid.dropna()) >= 2:
        prev_fast_ema = ema_fast.iloc[-2]; prev_mid_ema = ema_mid.iloc[-2]
        if pd.notna(prev_fast_ema) and pd.notna(prev_mid_ema): results["emaFastCrossAboveMid"] = (prev_fast_ema <= prev_mid_ema) and (latest_fast_ema > latest_mid_ema)
    return results

def get_higher_timeframe_trend(prices_df: pd.DataFrame, htf_period: str = '1h', htf_ema_period: int = 50) -> dict:
    # ... (content of this function is unchanged but kept for completeness) ...
    if not isinstance(prices_df, pd.DataFrame): raise TypeError("Input 'prices_df' must be a pandas DataFrame.")
    if not isinstance(prices_df.index, pd.DatetimeIndex): raise ValueError("DataFrame 'prices_df' must have a DatetimeIndex.")
    if 'close' not in prices_df.columns: raise ValueError("DataFrame 'prices_df' must contain a 'close' column.")
    required_cols_for_ohlc_resample = ['open', 'high', 'low', 'close']
    if not all(col in prices_df.columns for col in required_cols_for_ohlc_resample):
        print(f"Warning: Input DataFrame is missing some of {required_cols_for_ohlc_resample}. Resampling 'close' price directly.")
        ohlc_agg = {'close': 'last'};_o=prices_df.columns;
        if 'open' in _o: ohlc_agg['open']='first';
        if 'high' in _o: ohlc_agg['high']='max';
        if 'low' in _o: ohlc_agg['low']='min';
    else: ohlc_agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    results = {"htf_ema": np.nan, "htf_close": np.nan, "htfTrendUp": False, "htfTrendDown": False}
    try: htf_df = prices_df.resample(htf_period).agg(ohlc_agg); htf_df.dropna(subset=['close'], inplace=True)
    except Exception as e: print(f"Error during resampling: {e}"); return results
    if htf_df.empty or len(htf_df['close']) < htf_ema_period:
        if not htf_df.empty: results['htf_close'] = htf_df['close'].iloc[-1] if pd.notna(htf_df['close'].iloc[-1]) else np.nan
        return results
    htf_ema = calculate_ema(htf_df['close'], htf_ema_period); htf_df['ema'] = htf_ema
    latest_htf_data = htf_df.iloc[-1]; latest_htf_close = latest_htf_data['close']; latest_htf_ema = latest_htf_data['ema']
    results["htf_close"] = latest_htf_close if pd.notna(latest_htf_close) else np.nan
    results["htf_ema"] = latest_htf_ema if pd.notna(latest_htf_ema) else np.nan
    if pd.isna(latest_htf_close) or pd.isna(latest_htf_ema): return results
    results["htfTrendUp"] = latest_htf_close > latest_htf_ema; results["htfTrendDown"] = latest_htf_close < latest_htf_ema
    return results

def get_volume_indicators(df: pd.DataFrame, volume_ma_period: int = 20, spike_multiplier: float = 2.0, min_big_order_value: float = 10000) -> dict:
    # ... (content of this function is unchanged but kept for completeness) ...
    if not isinstance(df, pd.DataFrame): raise TypeError("Input 'df' must be a pandas DataFrame.")
    if 'close' not in df.columns or 'volume' not in df.columns: raise ValueError("DataFrame 'df' must contain 'close' and 'volume' columns.")
    if not (isinstance(volume_ma_period, int) and volume_ma_period > 0): raise ValueError("'volume_ma_period' must be a positive integer.")
    if not (isinstance(spike_multiplier, (int, float)) and spike_multiplier > 0): raise ValueError("'spike_multiplier' must be a positive number.")
    if not (isinstance(min_big_order_value, (int, float)) and min_big_order_value >= 0): raise ValueError("'min_big_order_value' must be a non-negative number.")
    results = {"current_volume": np.nan, "volume_ma": np.nan, "isVolumeSpike": False, "isBigOrder": False, "estimated_trade_value": np.nan}
    if df.empty or 'volume' not in df or 'close' not in df: return results
    latest_volume = df['volume'].iloc[-1] if pd.notna(df['volume'].iloc[-1]) else np.nan
    latest_close = df['close'].iloc[-1] if pd.notna(df['close'].iloc[-1]) else np.nan
    results["current_volume"] = latest_volume
    if pd.isna(latest_volume) or pd.isna(latest_close): return results
    results["estimated_trade_value"] = latest_volume * latest_close
    if results["estimated_trade_value"] > min_big_order_value: results["isBigOrder"] = True
    if len(df['volume'].dropna()) < volume_ma_period: return results
    volume_ma = df['volume'].rolling(window=volume_ma_period, min_periods=volume_ma_period).mean()
    latest_volume_ma = volume_ma.iloc[-1] if pd.notna(volume_ma.iloc[-1]) else np.nan
    results["volume_ma"] = latest_volume_ma
    if pd.notna(latest_volume) and pd.notna(latest_volume_ma) and latest_volume_ma > 0:
        if latest_volume > spike_multiplier * latest_volume_ma: results["isVolumeSpike"] = True
    return results

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    # ... (content of this function is unchanged but kept for completeness) ...
    if not isinstance(prices, pd.Series): raise TypeError("Input 'prices' must be a pandas Series.")
    if not (isinstance(period, int) and period > 0): raise ValueError("RSI 'period' must be a positive integer.")
    if len(prices) < period + 1: return pd.Series([np.nan] * len(prices), index=prices.index, name='RSI')
    delta = prices.diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=True).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=True).mean()
    rs = avg_gain / avg_loss; rsi = 100 - (100 / (1 + rs))
    if len(prices) >= period +1: rsi.iloc[0:period] = np.nan
    else: return pd.Series([np.nan] * len(prices), index=prices.index, name='RSI')
    return rsi.rename('RSI')

def calculate_macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    # ... (content of this function is unchanged but kept for completeness) ...
    if not isinstance(prices, pd.Series): raise TypeError("Input 'prices' must be a pandas Series.")
    if not all(isinstance(p, int) and p > 0 for p in [fast_period, slow_period, signal_period]): raise ValueError("All MACD periods must be positive integers.")
    if fast_period >= slow_period: raise ValueError("MACD fast_period must be less than slow_period.")
    min_len_required = slow_period + signal_period -1
    ema_fast = calculate_ema(prices, fast_period); ema_slow = calculate_ema(prices, slow_period)
    macd_line = ema_fast - ema_slow; signal_line = calculate_ema(macd_line, signal_period); macd_histogram = macd_line - signal_line
    df_macd = pd.DataFrame({'MACD_line': macd_line, 'Signal_line': signal_line, 'MACD_histogram': macd_histogram})
    if len(prices) < min_len_required: return pd.DataFrame(np.nan, index=prices.index, columns=['MACD_line', 'Signal_line', 'MACD_histogram'])
    return df_macd

def get_oscillator_signals(prices: pd.Series, rsi_period: int = 14, rsi_overbought: int = 70, rsi_oversold: int = 30, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> dict:
    # ... (content of this function is unchanged but kept for completeness) ...
    results = {"rsi_value": np.nan, "isRsiOverbought": False, "isRsiOversold": False, "macd_line": np.nan, "signal_line": np.nan, "macd_histogram": np.nan, "isMacdBullish": False, "isMacdBearish": False, "macdGoldenCross": False, "macdDeadCross": False}
    if not isinstance(prices, pd.Series) or prices.empty: return results
    rsi_series = calculate_rsi(prices, rsi_period)
    if not rsi_series.empty and pd.notna(rsi_series.iloc[-1]):
        results["rsi_value"] = rsi_series.iloc[-1]
        if results["rsi_value"] >= rsi_overbought: results["isRsiOverbought"] = True
        if results["rsi_value"] <= rsi_oversold: results["isRsiOversold"] = True
    df_macd = calculate_macd(prices, macd_fast, macd_slow, macd_signal)
    if not df_macd.empty:
        latest_macd_line = df_macd['MACD_line'].iloc[-1]; latest_signal_line = df_macd['Signal_line'].iloc[-1]; latest_macd_hist = df_macd['MACD_histogram'].iloc[-1]
        if pd.notna(latest_macd_line): results["macd_line"] = latest_macd_line
        if pd.notna(latest_signal_line): results["signal_line"] = latest_signal_line
        if pd.notna(latest_macd_hist): results["macd_histogram"] = latest_macd_hist
        if pd.notna(latest_macd_line) and pd.notna(latest_signal_line):
            results["isMacdBullish"] = latest_macd_line > latest_signal_line; results["isMacdBearish"] = latest_macd_line < latest_signal_line
            if len(df_macd['MACD_line'].dropna()) >= 2 and len(df_macd['Signal_line'].dropna()) >= 2:
                prev_macd_line = df_macd['MACD_line'].iloc[-2]; prev_signal_line = df_macd['Signal_line'].iloc[-2]
                if pd.notna(prev_macd_line) and pd.notna(prev_signal_line):
                    results["macdGoldenCross"] = (prev_macd_line <= prev_signal_line) and (latest_macd_line > latest_signal_line)
                    results["macdDeadCross"] = (prev_macd_line >= prev_signal_line) and (latest_macd_line < latest_signal_line)
    return results

def get_pivot_points(df: pd.DataFrame, lookback_left: int = 5, lookback_right: int = 5) -> dict:
    """Identifies the most recent confirmed pivot high and low."""
    if not isinstance(df, pd.DataFrame): raise TypeError("Input 'df' must be a pandas DataFrame.")
    if 'high' not in df.columns or 'low' not in df.columns:
        raise ValueError("DataFrame 'df' must contain 'high' and 'low' columns.")
    if not (isinstance(lookback_left, int) and lookback_left >= 0): raise ValueError("lookback_left must be a non-negative integer.")
    if not (isinstance(lookback_right, int) and lookback_right >= 0): raise ValueError("lookback_right must be a non-negative integer.")

    n = len(df)
    current_resistance = np.nan
    current_support = np.nan

    # Iterate backwards from the end of the DataFrame, ensuring enough data for lookbacks
    # Pivot confirmation requires (index - lookback_left) >= 0 and (index + lookback_right) < n
    # For the *most recent* confirmed pivot, we look at indices that allow full right lookback.
    # The latest possible bar that could be a pivot is n - 1 - lookback_right.

    # Find most recent Pivot High
    # Start from an index that allows for right lookback, search backwards
    for i in range(n - 1 - lookback_right, lookback_left - 1, -1):
        if i < 0: continue # Should not happen with loop range but safeguard
        is_pivot_high = True
        current_high = df['high'].iloc[i]
        # Left lookback
        for j in range(1, lookback_left + 1):
            if i - j < 0 or df['high'].iloc[i-j] > current_high:
                is_pivot_high = False; break
        if not is_pivot_high: continue
        # Right lookback
        for j in range(1, lookback_right + 1):
            if i + j >= n or df['high'].iloc[i+j] > current_high: # Strict inequality for uniqueness
                is_pivot_high = False; break
        if is_pivot_high:
            current_resistance = current_high
            break # Found the most recent one

    # Find most recent Pivot Low
    for i in range(n - 1 - lookback_right, lookback_left - 1, -1):
        if i < 0: continue
        is_pivot_low = True
        current_low = df['low'].iloc[i]
        # Left lookback
        for j in range(1, lookback_left + 1):
            if i - j < 0 or df['low'].iloc[i-j] < current_low:
                is_pivot_low = False; break
        if not is_pivot_low: continue
        # Right lookback
        for j in range(1, lookback_right + 1):
            if i + j >= n or df['low'].iloc[i+j] < current_low: # Strict inequality
                is_pivot_low = False; break
        if is_pivot_low:
            current_support = current_low
            break # Found the most recent one

    return {"currentResistance": current_resistance, "currentSupport": current_support}

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    if not isinstance(df, pd.DataFrame): raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        raise ValueError("DataFrame 'df' must contain 'high', 'low', and 'close' columns.")
    if not (isinstance(period, int) and period > 0): raise ValueError("ATR 'period' must be a positive integer.")

    if len(df) < period + 1: # Need at least period+1 for prev_close and EMA
        return pd.Series([np.nan] * len(df), index=df.index, name='ATR')

    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))

    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    tr.iloc[0] = np.nan # First TR is NaN because of prev_close shift

    # ATR is typically an EMA of TR. Using Wilder's smoothing (alpha = 1/period)
    # which is equivalent to ewm with com = period -1 and adjust=True
    atr = tr.ewm(com=period - 1, min_periods=period, adjust=True).mean()

    return atr.rename('ATR')

def get_price_volatility_signals(df: pd.DataFrame, pivot_left: int = 5, pivot_right: int = 5, atr_period: int = 14) -> dict:
    """Generates signals based on price structure (pivots) and volatility (ATR)."""
    results = {
        "currentResistance": np.nan, "currentSupport": np.nan,
        "latest_atr": np.nan, "breakoutUp": False, "volatilityRatio": np.nan
    }
    if not isinstance(df, pd.DataFrame): raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        raise ValueError("DataFrame 'df' must contain 'high', 'low', and 'close' columns.")
    if df.empty: return results

    pivots = get_pivot_points(df, pivot_left, pivot_right)
    results.update(pivots)

    atr_series = calculate_atr(df, atr_period)
    if not atr_series.empty and pd.notna(atr_series.iloc[-1]):
        results["latest_atr"] = atr_series.iloc[-1]

    latest_close = df['close'].iloc[-1]
    if pd.notna(latest_close) and pd.notna(results["currentResistance"]):
        if latest_close > results["currentResistance"]:
            results["breakoutUp"] = True

    if pd.notna(results["latest_atr"]) and pd.notna(latest_close) and latest_close != 0:
        results["volatilityRatio"] = results["latest_atr"] / latest_close

    return results

if __name__ == '__main__':
    # ... (previous tests truncated for brevity) ...
    print("--- Testing calculate_ema ---"); print("EMA 10:\n", calculate_ema(pd.Series([10,11,12,13,14,15,16,17,18,19,20]),10).tail(3))
    print("\n--- Testing get_ema_signals ---"); signals_crossover_ema = get_ema_signals(pd.Series([30,29,28,27,26,25,24,23,22,22.5,24,26,27,28]),3,6,9); print("\nSignals (Crossover Data):"); [print(f"  {k}: {v}") for k,v in signals_crossover_ema.items()]
    base_time = pd.to_datetime('2023-01-01 00:00:00'); num_points_for_htf = 10*12+(12*2); prices_for_htf = [100+(i*0.01)+(i//24*1) for i in range(num_points_for_htf)]; index_for_htf = [base_time + pd.Timedelta(minutes=5*i) for i in range(num_points_for_htf)]; df_for_htf = pd.DataFrame({'open':[p-0.01 for p in prices_for_htf],'high':[p+0.02 for p in prices_for_htf],'low':[p-0.02 for p in prices_for_htf],'close':prices_for_htf,'volume':[10+i%5 for i in range(num_points_for_htf)]}, index=pd.DatetimeIndex(index_for_htf)); htf_trend_up_test = get_higher_timeframe_trend(df_for_htf,htf_period='1h',htf_ema_period=10); print("\n--- Testing get_higher_timeframe_trend ---"); print("\nHTF Trend (Uptrend Data, 1h EMA 10):"); [print(f"  {k}: {v}") for k,v in htf_trend_up_test.items()]
    if pd.notna(htf_trend_up_test['htf_ema']) and htf_trend_up_test['htfTrendUp']: assert htf_trend_up_test['htf_close'] > htf_trend_up_test['htf_ema']
    print("\n--- Testing get_volume_indicators ---"); vol_ind_sb_test = get_volume_indicators(df_for_htf.tail(30),5,2.0,500); print("\nVolume Indicators (Sample):"); [print(f"  {k}: {v}") for k,v in vol_ind_sb_test.items()]
    rsi_test_prices = pd.Series([44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,45.84,46.08,45.89,46.03,45.61,46.28,46.28,46.00,46.03,46.41,46.22,45.64,46.23,46.26,46.00,46.00,45.69,45.00,44.00,43.00,43.50,44.50]); print("\n--- Testing calculate_rsi ---"); print("RSI (14) sample (last 5 values):\n", calculate_rsi(rsi_test_prices,14).tail()); print("RSI (5) with 6 data points (1 RSI value expected):\n", calculate_rsi(rsi_test_prices.head(6),5))
    macd_df=calculate_macd(rsi_test_prices,12,26,9);print("\n--- Testing calculate_macd ---");print("MACD DataFrame sample (last 5 values from 30 input points):\n", macd_df.tail());assert macd_df['MACD_line'].iloc[-1] is not np.nan; assert pd.isna(macd_df['Signal_line'].iloc[-1]); macd_df_sufficient=calculate_macd(pd.Series(np.arange(1,40)),12,26,9); print("MACD DataFrame with sufficient data (39 points):\n", macd_df_sufficient.tail(3)); assert macd_df_sufficient['Signal_line'].iloc[-1] is not np.nan
    print("\n--- Testing get_oscillator_signals ---"); base_s1=np.linspace(100,120,35); surge_s1=np.array([base_s1[-1]+(i*i*0.1) for i in range(1,21)]); prices_ob_bull=pd.Series(np.concatenate([base_s1,surge_s1])); osc_signals_ob_bull=get_oscillator_signals(prices_ob_bull,14,70); print("\nOscillator Signals (RSI OB, MACD Bullish):"); [print(f"  {k}: {v:.4f}" if isinstance(v,float) else f"  {k}: {v}") for k,v in osc_signals_ob_bull.items()]; assert osc_signals_ob_bull['isRsiOverbought']; assert osc_signals_ob_bull['isMacdBullish']
    base_s2=np.linspace(150,130,35); decline_s2=np.array([base_s2[-1]-(i*i*0.1) for i in range(1,21)]); prices_os_bear=pd.Series(np.concatenate([base_s2,decline_s2])); osc_signals_os_bear=get_oscillator_signals(prices_os_bear,14,30); print("\nOscillator Signals (RSI OS, MACD Bearish):"); [print(f"  {k}: {v:.4f}" if isinstance(v,float) else f"  {k}: {v}") for k,v in osc_signals_os_bear.items()]; assert osc_signals_os_bear['isRsiOversold']; assert osc_signals_os_bear['isMacdBearish']
    gc_phase1=np.linspace(50,40,20).tolist();gc_phase2=np.linspace(39,30,15).tolist();gc_phase3=np.linspace(30.5,38,8).tolist();prices_golden_cross=pd.Series(gc_phase1+gc_phase2+gc_phase3);osc_signals_gc=get_oscillator_signals(prices_golden_cross,14);print("\nOscillator Signals (MACD Golden Cross):");[print(f"  {k}: {v:.4f}" if isinstance(v,float) else f"  {k}: {v}") for k,v in osc_signals_gc.items()];assert osc_signals_gc['isMacdBullish']
    dc_phase1=np.linspace(30,40,20).tolist();dc_phase2=np.linspace(41,50,15).tolist();dc_phase3=np.linspace(49.5,42,8).tolist();prices_dead_cross=pd.Series(dc_phase1+dc_phase2+dc_phase3);osc_signals_dc=get_oscillator_signals(prices_dead_cross,14);print("\nOscillator Signals (MACD Dead Cross):");[print(f"  {k}: {v:.4f}" if isinstance(v,float) else f"  {k}: {v}") for k,v in osc_signals_dc.items()];assert osc_signals_dc['isMacdBearish']
    prices_insufficient_osc=pd.Series([10,11,12,13,14]);osc_signals_ins=get_oscillator_signals(prices_insufficient_osc);print("\nOscillator Signals (Insufficient Data):");[print(f"  {k}: {v:.4f}" if isinstance(v,float) else f"  {k}: {v}") for k,v in osc_signals_ins.items()];assert pd.isna(osc_signals_ins['rsi_value']);assert pd.isna(osc_signals_ins['macd_line'])

    # --- New tests for Pivots, ATR, Price/Volatility Signals ---
    print("\n--- Testing get_pivot_points ---")
    # Data: Low at index 2 (20), High at index 7 (130) with lookback 2,2
    # Index: 0  1   2   3   4   5   6    7    8   9  10  11  12
    highs = [50,60,70,60,80,120,125,130,110,100,90,80,70]
    lows =  [40,50,20,30,70,110,100,105,90, 80,70,60,50]
    closes= [45,55,65,55,75,115,120,125,105,95,85,75,65]
    pivot_df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})

    pivots_2_2 = get_pivot_points(pivot_df, lookback_left=2, lookback_right=2)
    print("Pivots (2,2):", pivots_2_2) # Expected: Res: 130 (idx 7), Sup: 20 (idx 2)
    assert pivots_2_2['currentResistance'] == 130.0
    assert pivots_2_2['currentSupport'] == 20.0

    pivots_1_1 = get_pivot_points(pivot_df.tail(7), lookback_left=1, lookback_right=1) # Use last 7 points: [125,130,110,100,90,80,70], lows [100,105,90,80,70,60,50]
    # Relative to tail(7): Highs: H(idx 1)=130, Lows: L(idx 6)=50
    # Expected: Res: 130 (original idx 7, relative idx 1 for tail(7)). For Sup: no pivot low is found with lookback_right=1 in this segment.
    print("Pivots (1,1 on tail(7)):", pivots_1_1)
    assert pivots_1_1['currentResistance'] == 130.0
    assert pd.isna(pivots_1_1['currentSupport']) # Correct for lookback_right=1 as no low satisfies the condition

    pivots_insufficient = get_pivot_points(pivot_df.head(3), 2, 2) # Not enough data for lookbacks
    print("Pivots (insufficient data):", pivots_insufficient)
    assert pd.isna(pivots_insufficient['currentResistance'])
    assert pd.isna(pivots_insufficient['currentSupport'])

    print("\n--- Testing calculate_atr ---")
    atr_df = pivot_df.copy() # Re-use the pivot_df which has H,L,C
    atr_14 = calculate_atr(atr_df, period=14) # Needs 15 points for first ATR
    print("ATR (14) with 13 points (expect all NaNs):\n", atr_14)
    assert atr_14.isna().all()

    # Need more data for ATR test
    atr_highs = highs * 3 # 39 points
    atr_lows = lows * 3
    atr_closes = closes * 3
    atr_test_df = pd.DataFrame({'high': atr_highs, 'low': atr_lows, 'close': atr_closes})
    atr_5 = calculate_atr(atr_test_df, period=5)
    print("ATR (5) with 39 points (last 5):\n", atr_5.tail())
    assert not atr_5.tail(1).isna().iloc[0] # Last value should not be NaN

    print("\n--- Testing get_price_volatility_signals ---")
    # Use atr_test_df, which is long enough. Lookback 2,2. ATR period 5.
    # Most recent pivot high in atr_test_df.tail(2+2+1=5 -> no, full series)
    # Last confirmed pivot high in atr_test_df with (2,2) is at original index 7 (130), repeated.
    # So, original index 7+13 = 20 (val 130), 7+13+13 = 33 (val 130)
    # Most recent is index 33 (value 130)
    # Most recent pivot low is index 2+13+13 = 28 (value 20)

    pv_signals = get_price_volatility_signals(atr_test_df, pivot_left=2, pivot_right=2, atr_period=5)
    print("Price/Volatility Signals (atr_test_df):", pv_signals)
    assert pv_signals['currentResistance'] == 130.0 # from original index 33
    assert pv_signals['currentSupport'] == 20.0   # from original index 28
    assert pd.notna(pv_signals['latest_atr'])
    assert pd.notna(pv_signals['volatilityRatio'])

    # Test breakout: make last close higher than resistance
    breakout_df = atr_test_df.copy()
    breakout_df.loc[len(breakout_df)-1, 'close'] = 135
    pv_signals_breakout = get_price_volatility_signals(breakout_df, 2, 2, 5)
    print("Price/Volatility Signals (Breakout Up):", pv_signals_breakout)
    assert pv_signals_breakout['breakoutUp'] == True

    # Test insufficient data for signals
    pv_signals_insufficient = get_price_volatility_signals(atr_test_df.head(5)) # Too short for ATR and Pivots
    print("Price/Volatility Signals (Insufficient):", pv_signals_insufficient)
    assert pd.isna(pv_signals_insufficient['currentResistance'])
    assert pd.isna(pv_signals_insufficient['latest_atr'])
    assert pv_signals_insufficient['breakoutUp'] == False

    print("\nAll price/volatility indicator tests seem to have logical assertions pass if run directly.")
