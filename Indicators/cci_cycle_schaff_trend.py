from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

'''
https://www.tradingview.com/script/SYXp2cAq-CCI-Cycle-Modified-Schaff-Trend-Cycle/#chart-view-comments
''' 
  
def cst(candles: np.ndarray, length:int=10,factor:float=0.5, source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)   
    ma = talib.SMA(source,length)
    cci = pine_cci(source,length,ma)
    ccis = pine_rma(source,2*pine_ema(source,cci,length/2)-pine_ema(source,cci,length),np.round(np.sqrt(length)))
    cci_cycle = ccic(source,candles,ccis,length,factor)
    if sequential:
        return cci_cycle
    else:
        return cci_cycle[-1]
        
@njit    
def ccic(source,candles,ccics,length,factor):
    m = np.full_like(source,0)
    v1 = np.full_like(source,0)
    v2 = np.full_like(source,0)
    f1 = np.full_like(source,0)
    pf = np.full_like(source,0)
    v3 = np.full_like(source,0)
    v4 = np.full_like(source,0)
    f2 = np.full_like(source,0)
    pff = np.full_like(source,0)
    for i in range((length+1),source.shape[0]):
        m[i] = ccics[i]
        v1[i] = np.amin(ccics[i-(length-1):i+1])
        v2[i] = np.amax(ccics[i-(length-1):i+1]) - v1[i]
        f1[i] = ((m[i] - v1[i]) /v2[i]) * 100 if (v2[i] > 0) else f1[i-1] 
        pf[i] = f1[i] if pf[i-1] == 0 else pf[i-1] + (factor * (f1[i] - pf[i-1]))
        v3[i] = np.amin(pf[i-(length-1):i+1])
        v4[i] = np.amax(pf[i-(length-1):i+1]) - v3[i] 
        f2[i] = ((pf[i] - v3[i])/v4[i]) * 100 if (v4[i] > 0) else f2[i-1] 
        pff[i] = f2[i] if pff[i-1] == 0 else pff[i-1] + (factor*(f2[i] - pff[i-1]))
    return pff 
@njit
def pine_cci(source,per,rollwin):
    mamean = np.full_like(source,0)
    cci = np.full_like(source,0)
    dev = np.full_like(source,0)
    for i in range(source.shape[0]):
        mamean = (rollwin)
        sum1 = 0.0
        val = 0.0
        for j in range(per):
            val = source[i-j]
            sum1 = sum1 + np.abs(val - mamean[i])
        dev[i] = sum1/per 
        cci[i] = (source[i] - mamean[i]) / (0.015 * dev[i])
    return cci    
    
@njit 
def pine_rma(source1, source2, length):
    alpha = 1/length
    sum1 = np.full_like(source1,0)
    for i in range(20,source1.shape[0]):
        sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1]
    return sum1 
      
@njit 
def pine_ema(source1, source2, length):
    alpha = 2 / (length + 1)
    sum1 = np.full_like(source1,0)
    for i in range(20,source1.shape[0]):
        sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1] 
    return sum1 
