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
https://www.tradingview.com/script/aDwNin08-Ehlers-Moving-Average-Difference-Hann-Indicator-CC/
"""
#jesse backtest  '2021-01-03' '2021-03-02'

def hann(candles: np.ndarray, shortLength:int=8,longLength:int=20,domCycle:int=27, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential) 
    source = get_candle_source(candles, source_type=source_type) 
    hann = fast_hann(source,candles,shortLength,longLength,domCycle)
    if sequential:
        return hann[-1]
    else:
        return hann[-1]

@njit        
def fast_hann(source,candles,shortLength,longLength,domCycle):
    longLength2 = np.ceil((shortLength + domCycle) / 2)
    filt11 = np.full_like(source,0)
    filt22 = np.full_like(source,0)
    madh = np.full_like(source,0)
    for i in range(source.shape[0]):
        cosine1 = 0.0 
        filt1 = 0.0
        coef1 = 0.0
        cosine2 = 0.0
        filt2 = 0.0
        coef2 = 0.0
        for j in range(1,shortLength+1):
            cosine1 = 1 - np.cos(2 * np.pi * (j) / (shortLength + 1))
            filt1 = filt1 + (cosine1 * source[(i-(j-1))])
            coef1 = coef1 + cosine1
        for j in range(1,longLength2+1):
            cosine2 = 1 - np.cos(2 * np.pi * (j) / (longLength2 + 1))
            filt2 = filt2 + (cosine2 * source[(i-(j-1))])
            coef2 = coef2 + cosine2 
        filt11[i] = filt1 / coef1 if coef1 != 0 else 0 
        filt22[i] = filt2 / coef2 if coef2 != 0 else 0 
        madh[i] = 100 * (filt11[i] - filt22[i]) / filt22[i] if filt22[i] != 0 else 0 
    return madh
