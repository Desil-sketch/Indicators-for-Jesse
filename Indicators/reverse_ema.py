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
https://www.tradingview.com/script/umIHbltg/#chart-view-comment-form
"""
#jesse backtest  '2021-01-03' '2021-03-02'

def reverseema(candles: np.ndarray,source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential) 
    source = get_candle_source(candles, source_type=source_type) 
    rema = fast_reverseema(source,candles)
    if sequential:
        return rema
    else:
        return rema[-1]

@njit    
def fast_reverseema(source,candles):
    aa = 0.1
    cc = 1 - aa 
    ma = np.full_like(source,0)
    r1 = np.full_like(source,0)
    r2 = np.full_like(source,0)
    r3 = np.full_like(source,0)
    r4 = np.full_like(source,0)
    r5 = np.full_like(source,0)
    r6 = np.full_like(source,0)
    r7 = np.full_like(source,0)
    r8 = np.full_like(source,0)
    wa = np.full_like(source,0)
    for i in range(source.shape[0]):
        ma[i] = aa * source[i] + cc*ma[i-1]
        r1[i] = cc * ma[i] + ma[i-1]
        r2[i] = np.power(cc,2)*r1[i] + r1[i-1] 
        r3[i] = np.power(cc,4)*r2[i] + r2[i-1]
        r4[i] = np.power(cc,8)*r3[i] + r3[i-1] 
        r5[i] = np.power(cc,16)*r4[i] + r4[i-1]
        r6[i] = np.power(cc,32)*r5[i] + r5[i-1] 
        r7[i] = np.power(cc,64)*r6[i] + r6[i-1]
        r8[i] = np.power(cc,128)*r7[i] + r7[i-1] 
        wa[i] = ma[i] - aa * r8[i] 
    return wa 
