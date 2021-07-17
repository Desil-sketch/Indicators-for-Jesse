from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import guvectorize,njit, prange, jit
import numba
import talib 
from typing import Union
from jesse.helpers import get_config, same_length
from collections import namedtuple
from numpy.lib.stride_tricks import sliding_window_view

TVA = namedtuple('TVA',['rising_bear', 'rising_bull', 'declining_bear', 'declining_bull'])

def tva(candles: np.ndarray, length:float=15, smo:int=3, source_type: str = "close", sequential: bool = False ) -> TVA:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    os = talib.WMA(source,length) - talib.SMA(source,length)
    rising_bear, rising_bull, declining_bear, declining_bull = fast_tva(os, source, candles, smo)
    if sequential:
        return TVA(rising_bear, rising_bull, declining_bear, declining_bull)
    else:
        return TVA(rising_bear[-1], rising_bull[-1], declining_bear[-1], declining_bull[-1])
"""
https://www.tradingview.com/script/bpfxYC3a-LUX-Trend-Volume-Accumulations/ 
""" 
       
@njit       
def fast_tva(os, source, candles, smo):
    rising_bear = np.full_like(source,0)
    rising_bull = np.full_like(source,0)
    declining_bull = np.full_like(source,0)
    declining_bear = np.full_like(source,0)
    pretv = np.full_like(source,0)
    prerv = np.full_like(source,np.nan)
    predv = np.full_like(source,np.nan)
    zeros = np.full_like(source,0)
    rv = np.full_like(source,np.nan)
    dv = np.full_like(source,np.nan)
    for i in range(smo,source.shape[0]):
        pretv[i] = candles[:,5][i] - candles[:,5][i-1]
        if pretv[i] > 0:
            prerv[i] = candles[:,5][i]
        else:
            prerv[i] = 0 
        rv[i] = (np.mean(prerv[i-(smo - 1) :i+1]))
        if pretv[i] < 0:
            predv[i] = candles[:,5][i]
        else:
            predv[i] = 0 
        dv[i] = (np.mean(predv[i-(smo - 1) :i+1]))
        if os[i] > 0:
            rising_bull[i] = rising_bull[i-1] + rv[i]
            declining_bull[i] = declining_bull[i-1] - dv[i] 
        else: 
            declining_bull[i] = 0
            rising_bull[i] = 0
        if os[i] < 0:
            declining_bear[i] = declining_bear[i-1] - dv[i]
            rising_bear[i] = rising_bear[i-1] + rv[i] 
        else:
            declining_bear[i] = 0
            rising_bear[i] = 0
    return rising_bear, rising_bull, declining_bear, declining_bull
 

