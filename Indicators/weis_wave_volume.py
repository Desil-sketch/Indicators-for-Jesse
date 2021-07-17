from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

#jesse backtest  '2021-01-03' '2021-03-02'

WEIS = namedtuple('WEIS',['up','dn'])

'''
https://www.tradingview.com/script/XttzkWc0-Weis-Wave-Volume-Pinescript-4/#chart-view-comments
''' 
  
def weis(candles: np.ndarray, trendDetectionLength:int=3, source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    up,dn = fast_weis(source,candles,trendDetectionLength)
    if sequential:
        return WEIS(up,dn)
    else:
        return WEIS(up[-1],dn[-1])

@njit
def fast_weis(source,candles,trendDetectionLength):
    mov = np.full_like(source,0)
    trend = np.full_like(source,0)
    isTrending = np.full_like(source,0)
    wave = np.full_like(source,0)
    vol = np.full_like(source,0)
    up = np.full_like(source,0)
    dn = np.full_like(source,0)
    for i in range(source.shape[0]):    
        if candles[:,2][i] > candles[:,2][i-1]:
            mov[i] = 1 
        elif candles[:,2][i] < candles[:,2][i-1]:   
            mov[i] = -1 
        else:
            mov[i] = 0 
        if mov[i] != 0 and mov[i] != mov[i-1]:
            trend[i] = mov[i] 
        else:
            trend[i] = trend[i-1]
        trending = 0.0
        othertrending = 0.0
        for j in range(0,trendDetectionLength):
            if candles[:,2][i-j] > candles[:,2][i-j-1]:
                trending = 1 + trending 
            else:
                trending = 0
                break 
        for j in range(0,trendDetectionLength):       
            if candles[:,2][i-j] < candles[:,2][i-j-1]:
                othertrending = 1 + othertrending 
            else:
                othertrending = 0 
                break 
        isTrending[i] = 1 if trending > 0 or othertrending > 0 else 0 
        wave[i] = trend[i] if trend[i] != wave[i-1] and isTrending[i] == 1 else wave[i-1] 
        vol[i] = vol[i-1] + candles[:,5][i] if wave[i] == wave[i-1] else candles[:,5][i]
        up[i] = vol[i] if wave[i] == 1 else 0 
        dn[i] = 0 if wave[i] == 1 else vol[i] 
    return up,dn

        
