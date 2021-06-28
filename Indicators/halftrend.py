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

HALFTREND = namedtuple('Halftrend',['atrhigh', 'atrlow', 'halftrend', 'trend'])

"""
https://www.tradingview.com/script/U1SJ8ubc-HalfTrend/
"""
def halftrend(candles: np.ndarray, amplitude: float = 2, channeldev: float = 2 , source_type: str = "close", sequential: bool = False) -> HALFTREND:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    atr2 = (talib.ATR(candles[:, 3], candles[:, 4], candles[:, 2], timeperiod = 100))/2
    atrHigh, atrLow, ht, trend = fast_halftrend(candles,source,amplitude,channeldev,atr2)
    if sequential:
        return HALFTREND(atrHigh,atrLow,ht,trend)
    else:
        return HALFTREND(atrHigh[-1], atrLow[-1], ht[-1], trend[-1])

@njit
def fast_halftrend(candles, source, amplitude, channeldev, atr2):
    maxLowPrice = np.full_like(source,0.0)
    minHighPrice = np.full_like(source,0.0)
    highPrice = np.full_like(source,0.0)
    lowPrice = np.full_like(source,0.0)
    highma = np.full_like(source,0.0)
    lowma = np.full_like(source,0.0)
    dev = np.full_like(source,0.0)
    nextTrend = np.full_like(source,0.0)
    up = np.full_like(source,0.0)
    down = np.full_like(source,0.0)
    trend = np.full_like(source,0.0)
    atrHigh = np.full_like(source,0.0)
    atrLow = np.full_like(source,0.0)
    ht = np.full_like(source,0.0)
    test = np.full_like(source,0.0)
    sellSignal = np.full_like(source,0)
    buySignal = np.full_like(source,0)
    test1 = np.full_like(source,0)
    for i in range(amplitude,source.shape[0]):  
        dev[i] = channeldev * atr2[i]
        maxLowPrice[i] = (candles[:,4][i-1]) if np.isnan(maxLowPrice[i]) else maxLowPrice[i]
        minHighPrice[i] = (candles[:,3][i-1]) if np.isnan(minHighPrice[i]) else minHighPrice[i]
        highestbar = 0
        highindex = 0
        lowindex = 0
        lowestvalue = 10**10
        highestvalue = 0.0
        for j in range(0,amplitude):
            if highestvalue <= (candles[i-j,3]):
                highestvalue = (candles[i-j,3])
                highindex = -j
            if lowestvalue >= (candles[i-j,4]):
                lowestvalue = (candles[i-j,4])
                lowindex = -j

        highPrice[i] = candles[i-(np.abs(highindex)),3]
        lowPrice[i] = candles[i-(np.abs(lowindex)),4]
        highma[i] = np.mean(candles[i-amplitude+1:i+1,3])
        lowma[i] = np.mean(candles[i-amplitude+1:i+1,4])
        nextTrend[i] = nextTrend[i-1]
        trend[i] = trend[i-1]
        if nextTrend[i] == 1:
            maxLowPrice[i] = np.maximum(lowPrice[i], maxLowPrice[i-1])
            if highma[i] < maxLowPrice[i] and candles[:,2][i] < candles[:,4][i-1]:
                trend[i] = 1 
                nextTrend[i] = 0 
                minHighPrice[i] = highPrice[i] 
            else:
                minHighPrice[i] = minHighPrice[i-1]
        else:   
            minHighPrice[i] = np.minimum(highPrice[i], minHighPrice[i-1])
            if lowma[i] > minHighPrice[i] and candles[:,2][i] > candles[:,3][i-1]:
                trend[i] = 0
                nextTrend[i] = 1 
                maxLowPrice[i] = lowPrice[i]
            else:
                maxLowPrice[i] = maxLowPrice[i-1]
        if trend[i] == 0:
            if not np.isnan(trend[i-1]) and trend[i-1] != 0:
                up[i] = down[i] if np.isnan(up[i-1]) else down[i-1] 
            else:
                up[i] = maxLowPrice[i] if np.isnan(up[i-1]) else np.maximum(maxLowPrice[i], up[i-1])
            down[i] = down[i-1]
            atrHigh[i] = up[i] + dev[i] 
            atrLow[i] = up[i] - dev[i] 
        else:
            if not np.isnan(trend[i-1]) and trend[i-1] != 1:
                down[i] = up[i] if np.isnan(up[i-1]) else up[i-1] 
            else:
                down[i] = minHighPrice[i] if np.isnan(down[i-1]) else np.minimum(minHighPrice[i], down[i-1])
            up[i] = up[i-1]
            atrHigh[i] = down[i] + dev[i] 
            atrLow[i] = down[i] - dev[i] 
        
        ht[i] = up[i] if trend[i] == 0 else down[i] 
        # if trend[i] == 0 and trend[i-1] == 1:
            # buySignal[i] = 1 
            # sellSignal[i] = 0
        # elif trend[i] == 1 and trend[i-1] == 0:
            # buySignal[i] = 0 
            # sellSignal[i] = 1 
        # else:
            # buySignal[i] = buySignal[i-1]
            # sellSignal[i] = sellSignal[i-1] 
        """ 
        if Trend == 0 : buysignal elif Trend == 1 : sellSignal
        """
    return atrHigh, atrLow, ht, trend
    
