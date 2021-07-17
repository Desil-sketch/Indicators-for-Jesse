from typing import Union
import talib
import numpy as np
from jesse.helpers import get_candle_source, same_length, slice_candles
from numba import njit

"""
https://www.tradingview.com/script/8MEEEGWl-Dickinson-Moving-Average-DMA/
"""

def dima(candles: np.ndarray, hullperiod: int = 7, emaperiod: int = 20, emagainlimit: int = 50, hull_matype: str = "WMA", source_type: str = "close", sequential: bool = False) -> Union[
    float, np.ndarray]:

    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)

    f_hma = talib.WMA((2 * talib.WMA(source, hullperiod / 2)) - talib.WMA(source, hullperiod), np.round(np.sqrt(hullperiod)))
    f_ehma = talib.EMA((2 * talib.EMA(source, hullperiod / 2)) - talib.EMA(source, hullperiod), np.round(np.sqrt(hullperiod)))

    ec = fast_dima(source, emaperiod, emagainlimit)
    hull = f_hma
    ehull = f_ehma
    dma =  np.empty_like(source)
    if hull_matype == "WMA": 
        dma = (ec+hull)/2
    elif hull_matype == "EMA":
        dma = (ec+ehull)/2
    fdma = dma
    if sequential: 
        return fdma
    else: 
        return fdma[-1]
        
@njit
def fast_dima(source, emaperiod, emagainlimit):
    alpha = 2 / (emaperiod + 1)
    e0 = np.copy(source)
    for i in range(source.shape[0]):
        e0[i] = alpha * source[i] + (1 - alpha) * e0[i - 1]
    ec = np.copy(source)
    gain = 0.0
    gain = emagainlimit / 10
    leasterror = np.full_like(source,1000000.0)
    bestgain = np.copy(source)
    error = np.copy(source)
    for i in  range(0,source.shape[0]):
        ec[i] = alpha * (e0[i] + gain * (source[i] - ec[i-1])) + ( 1 - alpha) * ec[i-1]
        error[i] = source[i] - ec[i]
        if np.abs(error[i]) < leasterror[i]: 
            leasterror[i] = np.abs(error[i])
            bestgain[i] = gain 
    for i in range(source.shape[0]):
        ec[i] = alpha * (e0[i] + bestgain[i] * (source[i] - ec[i-1])) + (1 - alpha) * ec[i-1]
    return ec 


