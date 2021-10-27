from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
import tulipy as ti
import math 

"""
https://www.tradingview.com/v/WZDH2Dxt/
so smoothing not working unless 4 or higher
"""
#jesse backtest  '2021-01-03' '2021-03-02'

def madh(candles: np.ndarray, len1:int= 20, so:int=4,source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential) 
    source = get_candle_source(candles, source_type=source_type) 
    prevwma = ti.vwma(np.ascontiguousarray(source), np.ascontiguousarray(candles[:, 5]), period=len1)
    vwma = same_length(candles, prevwma) 
    fast = f_nwma(source,(source*candles[:,5]),len1) / (f_nwma(source,candles[:,5],len1))
    slow = vwma
    MAD1 = 100 * (fast - slow) / slow 
    MAD = MAD1 if so == 0 else f_nwma(source,MAD1,so)
    if sequential:
        return MAD
    else:
        return MAD[-1]
        
        
def f_nwma(source1,source,period):
    fast = period/2 
    lambda1 = period/fast
    alpha = lambda1 * (period -1)/(period - lambda1)
    average1 = talib.WMA(source,period)
    average2 = talib.WMA(average1,fast)
    nwma = (1 + alpha)*average1 - alpha*average2 
    return nwma
