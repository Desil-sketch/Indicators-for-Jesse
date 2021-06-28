from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

#jesse backtest  '2021-01-03' '2021-03-02'
VWMACD = namedtuple('VWMACD',['vwmacd', 'signal', 'histogram'])

'''
https://www.tradingview.com/script/PHRFdHIH-VW-MACD/#chart-view-comments
''' 

def vwmacd(candles: np.ndarray, slowperiod: int= 26, fastperiod: int=12, signalperiod: int = 9, source_type: str = "close", sequential: bool = False ) -> VWMACD:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    fastma = talib.SMA(candles[:,5] * source, fastperiod)/talib.SMA(candles[:,5],fastperiod)
    slowma = talib.SMA(candles[:,5] * source, slowperiod)/talib.SMA(candles[:,5],slowperiod)
    vwmacd = fastma - slowma 
    signal = talib.SMA(vwmacd,signalperiod)
    histogram = vwmacd - signal 
    res = histogram
    if sequential:
        return VWMACD(vwmacd,signal,histogram)
    else:
        return VWMACD(vwmacd[-1],signal[-1],histogram[-1])
