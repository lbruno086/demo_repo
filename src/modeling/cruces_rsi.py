# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:54:55 2024

@author: Bruno
"""

import numpy as np
import pandas as pd
import talib
from talib import abstract
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
import os
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import plotly.io as pio

import warnings
import importlib
import sys

import pandas_ta as ta  # Importante para usar .ta.nvi() y .ta.pvi()


########################################################################
# 1) Función para descargar datos desde Yahoo Finance
########################################################################
def get_data(symbol: str, history_months_price: int) -> pd.DataFrame:
    """
    Obtiene los precios diarios del último {history_months_price} meses para el símbolo proporcionado,
    usando yfinance. Devuelve un DataFrame con columnas [close, high, low, open, volume].
    """
    end = datetime.now()
    start = end - relativedelta(months=history_months_price)

    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        print(
            f"No se obtuvieron datos para {symbol} entre {start.date()} y {end.date()}"
        )
        return pd.DataFrame()

    # Reordenar y renombrar columnas
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    # Nota: En tu código original reasignas a [close, high, low, open, volume],
    # pero luego usas df.columns = ['Close','High','Low','Open','Volume'].
    # Ajustemos para que sea coherente con TA-Lib:
    df.columns = ["close", "high", "low", "open", "volume"]

    return df


########################################################################
# 2) Función de Koncorde (adaptación manboto / Blai5)
########################################################################
def get_konkorde_params(df_stocks: pd.DataFrame):
    """
    Calcula las 4 líneas base del Koncorde: (val_blue, val_brown, val_green, val_avg).
    df_stocks requiere al menos columnas: [Open, High, Low, Close, Volume].

    Necesita:
      - pandas_ta (df_stocks.ta.nvi, df_stocks.ta.pvi)
      - talib
    """

    # Asegúrate de que las columnas sean las esperadas por talib (mayúsculas).
    # Para talib, solemos usar df['Close'], etc. Pero tu DF usa minúsculas.
    # A continuación, creamos un df temporal con mayúsculas (o ajustamos llamadas):
    df_tmp = pd.DataFrame(
        {
            "Open": df_stocks["open"],
            "High": df_stocks["high"],
            "Low": df_stocks["low"],
            "Close": df_stocks["close"],
            "Volume": df_stocks["volume"],
        }
    )

    # tprice = (Open + High + Low + Close) / 4
    tprice = (df_tmp["Open"] + df_tmp["High"] + df_tmp["Low"] + df_tmp["Close"]) / 4.0

    # PVI y su EMA(15)
    pvi = df_tmp.ta.pvi(cumulative=True, append=False)
    m = 15
    pvim = talib.EMA(pvi, timeperiod=m)
    pvimax = pvim.rolling(window=90).max()
    pvimin = pvim.rolling(window=90).min()
    oscp = (pvi - pvim) * 100.0 / (pvimax - pvimin)

    # NVI y su EMA(15)
    nvi = df_tmp.ta.nvi(cumulative=True, append=False)
    nvim = talib.EMA(nvi, timeperiod=m)
    nvimax = nvim.rolling(window=90).max()
    nvimin = nvim.rolling(window=90).min()
    val_blue = (nvi - nvim) * 100.0 / (nvimax - nvimin)  # línea azul

    # MFI(14)
    xmf = talib.MFI(
        df_tmp["High"], df_tmp["Low"], df_tmp["Close"], df_tmp["Volume"], timeperiod=14
    )

    # Bollinger con SMA(25) y 2*desv.std
    basis = talib.SMA(tprice, 25)
    dev = 2.0 * talib.STDDEV(tprice, 25)
    upper = basis + dev
    lower = basis - dev
    OB1 = (upper + lower) / 2.0
    OB2 = upper - lower
    BollOsc = ((tprice - OB1) / OB2) * 100.0

    # RSI(14) sobre tprice
    xrsi = talib.RSI(tprice, 14)

    # Stoch(21,3)
    def calc_stoch(src, length, smooth_fastd):
        ll = df_tmp["Low"].rolling(window=length).min()
        hh = df_tmp["High"].rolling(window=length).max()
        k = 100.0 * (src - ll) / (hh - ll)
        return talib.SMA(k, smooth_fastd)

    stoc = calc_stoch(tprice, 21, 3)

    # Línea marrón
    val_brown = (xrsi + xmf + BollOsc + (stoc / 3.0)) / 2.0
    # Línea verde
    val_green = val_brown + oscp
    # EMA de la línea marrón
    val_avg = talib.EMA(val_brown, timeperiod=m)

    return val_blue, val_brown, val_green, val_avg


########################################################################
# 3) Helper para detectar cruce alcista en últimos n días
########################################################################
def check_bullish_cross_in_last_n(values1, values2, n=3):
    """
    Devuelve True si 'values1' cruza por encima 'values2'
    en alguno de los últimos n días.

    Cruce al alza (abajo->arriba) implica:
       values1[i-1] < values2[i-1]  AND  values1[i] > values2[i].

    Asumimos que both arrays tienen la misma longitud >= 2.
    """
    length = len(values1)
    if length < 2:
        return False

    start_idx = max(1, length - n - 1)  # para buscar cruces en los últimos n días
    for i in range(start_idx, length):
        if i == 0:
            continue
        prev_diff = values1[i - 1] - values2[i - 1]
        curr_diff = values1[i] - values2[i]
        if prev_diff < 0 and curr_diff > 0:
            return True
    return False


########################################################################
# 4) Función principal: obtención de indicadores + cálculo de señales
########################################################################
def get_indicators_df(symbol: str, history_days_indicators: int = 7) -> pd.DataFrame:
    """
    Obtiene los indicadores de los últimos {history_days_indicators} días
    para la acción {symbol}, retornando un DataFrame de 1 sola fila con:
      - symbol
      - Para cada columna calculada: col_today y col_last_7
      - 3 señales: signal_1, signal_2, signal_3 (en la misma fila).

    Señal 1:
        (Close cruza SMA30 [ayer < SMA30, hoy > SMA30])
        AND (RSI cruza RSI_EMA21 en últimos 3 días)
    Señal 2:
        Señal 1 + (MACD_line cruza MACD_signal en últimos 3 días)
    Señal 3:
        Señal 2 + (Koncorde_blue_today > 0)  [Manos grandes comprando]
    """
    # 1) Descargar ~120 días de histórico
    df = get_data(symbol, 120)
    if df.empty:
        return pd.DataFrame({"symbol": [symbol]})

    # Renombrar a mayúsculas por compatibilidad con TA-Lib
    df.columns = ["Close", "High", "Low", "Open", "Volume"]

    # 2) Calcular indicadores básicos
    # --------------------------------
    df["SMA30"] = talib.SMA(df["Close"], timeperiod=30)
    df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
    df["RSI_EMA21"] = talib.EMA(df["RSI"], timeperiod=21)

    macd_line, macd_signal, macd_hist = talib.MACD(
        df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["MACD_line"] = macd_line
    df["MACD_signal"] = macd_signal

    # EMA(10,20,50,100,200) (opcionales)
    for period in [10, 20, 50, 100, 200]:
        df[f"ema_{period}"] = talib.EMA(df["Close"], timeperiod=period)

    df["ATR"] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["ATR_Standardized"] = df["ATR"] / df["Close"]

    # 3) Agregar Koncorde
    # --------------------------------
    # Re-convertimos a minúsculas temporalmente para la función
    df_konc = df.copy()
    df_konc.columns = df_konc.columns.str.lower()
    val_blue, val_brown, val_green, val_avg = get_konkorde_params(df_konc)

    df["Koncorde_blue"] = val_blue
    df["Koncorde_brown"] = val_brown
    df["Koncorde_green"] = val_green
    df["Koncorde_avgBrn"] = val_avg

    # 4) Recortar a los últimos (history_days_indicators + 1) registros
    #    para luego armar la fila final
    df = df.iloc[-(history_days_indicators + 1) :]

    # 5) Calcular las 3 señales en base a esos datos (8 filas = hoy + 7 días previos)
    # --------------------------------------------------------------------------------
    # a) Convertir a arrays
    arr_close = df["Close"].values
    arr_sma30 = df["SMA30"].values
    arr_rsi = df["RSI"].values
    arr_rsi_ema21 = df["RSI_EMA21"].values
    arr_macd_line = df["MACD_line"].values
    arr_macd_signal = df["MACD_signal"].values
    arr_konc_blue = df["Koncorde_blue"].values

    # b) Check del cruce Close-SMA30 en el día de ayer->hoy
    #    (ayer < sma30_ayer) AND (hoy > sma30_hoy)
    signal_close_sma = False
    if len(arr_close) >= 2 and len(arr_sma30) >= 2:
        if (arr_close[-2] < arr_sma30[-2]) and (arr_close[-1] > arr_sma30[-1]):
            signal_close_sma = True

    # c) Check RSI cruza RSI_EMA21 en últimos 3 días
    signal_rsi_cross = check_bullish_cross_in_last_n(arr_rsi, arr_rsi_ema21, n=3)

    # d) Señal 1
    signal_1 = signal_close_sma and signal_rsi_cross

    # e) Cruce MACD en últimos 3 días
    signal_macd_cross = check_bullish_cross_in_last_n(
        arr_macd_line, arr_macd_signal, n=3
    )

    # f) Señal 2 = Señal 1 + cruce MACD
    signal_2 = signal_1 and signal_macd_cross

    # g) Manos grandes comprando => Koncorde_blue hoy > 0
    signal_konc = False
    if len(arr_konc_blue) > 0:
        if arr_konc_blue[-1] > 0:
            signal_konc = True

    # h) Señal 3 = Señal 2 + Koncorde
    signal_3 = signal_2 and signal_konc

    # 6) Convertir DF -> 1 sola fila con sufijos _today, _last_7
    # -----------------------------------------------------------
    row_dict = {"symbol": symbol}

    for col in df.columns:
        vals = df[col].values
        if len(vals) == 0:
            continue
        # Valor 'hoy'
        row_dict[f"{col}_today"] = vals[-1]
        # Últimos N (p. ej. 7) días
        row_dict[f"{col}_last_{history_days_indicators}"] = list(vals[:-1])

    # Agregar señales a ese row_dict
    # (Guardamos como booleanos; se puede guardar como 0/1 si prefieres)
    row_dict["signal_1"] = signal_1
    row_dict["signal_2"] = signal_2
    row_dict["signal_3"] = signal_3

    # Devolver DataFrame de 1 sola fila
    indicators_df = pd.DataFrame([row_dict])
    return indicators_df


########################################################################
# 5) Obtener datos fundamentales y unificar
########################################################################
def get_stock_info_columns(symbol: str) -> pd.DataFrame:
    """
    Usa config.info['info names'] para filtrar campos yfinance.
    Retorna DataFrame con 1 fila y columna 'SYMBOL' (luego renombrada a 'symbol').
    """
    try:
        required_columns = config.info["info names"]
        info = yf.Ticker(symbol).info
        if not info:
            raise ValueError(f"No se pudo obtener información para: {symbol}")
        filtered_info = {col: info.get(col, None) for col in required_columns}
        filtered_info["SYMBOL"] = symbol
        df = pd.DataFrame([filtered_info])
        return df
    except Exception as e:
        print(f"Error obteniendo info financiera para {symbol}: {e}")
        return pd.DataFrame()


def unify_data(symbols: list, history_days_indicators: int = 7) -> pd.DataFrame:
    """
    Unifica la información fundamental con los indicadores TA-Lib en un solo DataFrame (fila por símbolo).
    """
    final_df = pd.DataFrame()
    for sym in symbols:
        try:
            # Fundamentales
            df_fund = get_stock_info_columns(sym)
            if df_fund.empty:
                continue
            df_fund.rename(columns={"SYMBOL": "symbol"}, inplace=True)

            # Indicadores
            df_ind = get_indicators_df(
                sym, history_days_indicators=history_days_indicators
            )
            if df_ind.empty:
                continue

            # Unir en una sola fila
            merged = pd.merge(df_ind, df_fund, on="symbol", how="inner")
            final_df = pd.concat([final_df, merged], axis=0, ignore_index=True)
        except:
            pass

    return final_df


########################################################################
# 6) Ejemplo de uso
########################################################################
if __name__ == "__main__":
    # Ejemplo: cargar lista de símbolos desde tu Excel
    lst_mkt_cap = list(
        set(
            pd.read_excel("C:/Users/Bruno/Desktop/Tickers/all_tickers.xlsx")[
                "Ticker"
            ].tolist()
        )
    )
    print("Número de símbolos:", len(lst_mkt_cap))

    df_unif = unify_data(lst_mkt_cap, history_days_indicators=7)

    # Observa las columnas resultantes:
    print(df_unif.columns)
    print(df_unif[["symbol", "signal_1", "signal_2", "signal_3"]].head(10))

    # Por ejemplo, filtrar donde la señal_3 sea True
    df_signal3 = df_unif[df_unif["signal_3"] == True]
    print("Tickers con Señal 3 activa:")
    print(df_signal3[["symbol", "signal_1", "signal_2", "signal_3"]])
