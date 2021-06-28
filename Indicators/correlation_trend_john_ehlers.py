from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

'''
https://www.tradingview.com/script/azjhI4tC-Correlation-Trend-Indicator-Dr-John-Ehlers/#chart-view-comments
Upper threshold = 0.5
Lower threshold = - 0.5 
''' 

def cti(candles: np.ndarray, period: int= 20, source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    correlation = fast_cti(source,period)
    if sequential:
        return correlation 
    else:
        return correlation[-1] 

@njit        
def fast_cti(source,period):
    denominator = np.full_like(source,0)
    for i in range(source.shape[0]):
        period = np.maximum(2, np.int(period))
        ex = 0.0 
        ey = 0.0
        ex2 = 0.0
        ey2 = 0.0
        exy = 0.0
        x = 0.0
        y = 0.0
        for j in range(period):
            x = source[i-j]
            y = -j 
            ex = ex + x 
            ex2 = ex2 + x * x 
            exy = exy + x * y 
            ey2 = ey2 + y * y 
            ey = ey + y 
        denominator[i] = (period * ex2 - ex * ex) * (period * ey2 - ey * ey)
        denominator[i] = 0 if denominator[i] == 0.0 else (period * exy - ex *ey)/np.sqrt(denominator[i])
    return denominator 
