from collections import namedtuple
import numpy as np
from numba import njit
import talib 
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
import jesse.indicators as ta
from typing import Union
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d

def lelec(candles: np.ndarray, period: int = 40, bars: int = 15, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    candles_high = candles[:,3]
    candles_low = candles[:,4]
    highest = talib.MAX(candles_high,period)
    lowest = talib.MIN(candles_low,period)
    res = fast_lelec(candles, source, bars, period,highest,lowest)
    if sequential: 
        return res
    else:    
        return res[-1]
"""
https://www.tradingview.com/script/jB2a9GAV-Leledc-levels-IS/
Only resistance works
"""        
#jesse backtest  '2021-01-03' '2021-03-02'
@njit
def fast_lelec(candles, source, bars, period,highest,lowest):
    sindex = np.full_like(source, 0)
    return1 = np.full_like(source,np.nan)
    resistance = np.full_like(source,0)
    support = np.full_like(source,0)
    bindex = np.full_like(source,0)
    highlel = np.full_like(source,0)
    lowlel = np.full_like(source,0)
    test = np.full_like(source,0)
    for i in range(period+1,source.shape[0]):
        bindex[i] = (bindex[i-1]) + 1 if candles[:,2][i] > candles[:,2][i-4] else bindex[i-1]
        sindex[i] = (sindex[i-1]) + 1 if candles[:,2][i] < candles[:,2][i-4] else sindex[i-1] 
        if bindex[i] > bars and candles[:,2][i] < candles[:,1][i] and (candles[:,3][i]) >= np.amax(candles[i-(period-1):i+1,3]):
            bindex[i] = 0 
            return1[i] = -1
        elif sindex[i] > bars and candles[:,2][i] > candles[:,1][i] and (candles[:,4][i]) <= np.amin(candles[i-(period-1):i+1,4]):
            sindex[i] = 0 
            return1[i] = 1
        if return1[i] == -1:
            highlel[i] = candles[:,3][i]
        else:
            highlel[i] = np.nan
        if return1[i] == 1:
            lowlel[i] = candles[:,4][i]
        else:
            lowlel[i] = np.nan
        if candles[:,2][i] < candles[:,1][i] and (return1[i] == -1 or return1[i] == 1) :
            resistance[i] = candles[:,3][i] 
        else:
            resistance[i] = resistance[i-1]
        if candles[:,2][i] > candles[:,1][i] and (return1[i] == 1 or return1[i] == -1) : 
            support[i] = candles[:,4][i]
        else:
            support[i] = support[i-1]
    return support
    

    
