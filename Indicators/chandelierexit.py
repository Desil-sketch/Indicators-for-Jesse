from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d
from collections import namedtuple

CHANDELIEREXIT = namedtuple('CHANDELIEREXIT',['longStop', 'shortStop', 'dir1'])
"""
https://www.tradingview.com/script/AqXxNS7j-Chandelier-Exit/#chart-view-comments
"""
def chandelierexit(candles: np.ndarray, length:int=22, mult:float=3.0, useClose: bool = True, source_type: str = "close", sequential: bool = False) -> CHANDELIEREXIT:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    atr = talib.ATR(candles[:, 3], candles[:, 4], candles[:, 2], timeperiod=length) * mult
    longStop,shortStop,dir1 = fast_chandelier(source,candles,length,atr,useClose)
    if sequential:
        return CHANDELIEREXIT(longStop,shortStop,dir1)
    else:
        return CHANDELIEREXIT(longStop[-1],shortStop[-1],dir1[-1])
    
@njit
def fast_chandelier(source,candles,length,atr,useClose):    
    dir1 = np.full_like(source,0)
    longStop = np.full_like(source,0)
    shortStop = np.full_like(source,0)
    for i in range(source.shape[0]):
        longStop[i] = np.amax(candles[i-length+1:,3]) - atr[i] if useClose == False else np.amax(candles[i-length+1:,2]) - atr[i]
        longStop[i] = np.maximum(longStop[i], longStop[i-1]) if candles[:,2][i-1] > longStop[i-1] else longStop[i]
        shortStop[i] = np.amin(candles[i-length+1:,4]) + atr[i] if useClose == False else np.amin(candles[i-length+1:,2]) + atr[i]
        shortStop[i] = np.minimum(shortStop[i], shortStop[i-1]) if candles[:,2][i-1] < shortStop[i-1] else shortStop[i] 
        if source[i] > shortStop[i-1]:
            dir1[i] = 1 
        elif source[i] < longStop[i-1]:
            dir1[i] = -1
        else:
            dir1[i] = dir1[i-1]  
    return longStop, shortStop, dir1
