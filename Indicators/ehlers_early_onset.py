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
https://www.tradingview.com/script/Pk5PUfJT-blackcat-L2-Ehlers-Early-Onset-Trend/
"""
#jesse backtest  '2021-01-03' '2021-03-02'

def earlyonset(candles: np.ndarray, lperiod:int= 30, k1:float=0.4,source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential) 
    source = get_candle_source(candles, source_type=source_type) 
    quotient1 = fast_onset(source,candles,lperiod,k1)
    if sequential:
        return quotient1
    else:
        return quotient1[-1]
        

def fast_onset(source,candles,lperiod,k1):
    alpha1 = np.full_like(source,0)
    HP = np.full_like(source,0)
    a1 = 0.0
    b1 = 0.0
    c1 = 0.0
    c2 = 0.0
    c3 = 0.0
    Filt = np.full_like(source,0)
    Peak = np.full_like(source,0)
    X = np.full_like(source,0)
    quotient1 = np.full_like(source,0)
    for i in range(source.shape[0]):
        alpha1[i] = (np.cos( 0.707 * 2 * np.pi/100) + np.sin(0.707 * 2 * np.pi / 100) - 1) / np.cos( 0.707 * 2 * np.pi / 100)
        HP[i] = ( 1 - alpha1[i] / 2) * (1 - alpha1[i] / 2) * (source[i] - 2 * source[i-1] + source[i-2]) + 2 * (1 - alpha1[i]) * HP[i-1] - (1 - alpha1[i]) * (1 - alpha1[i]) * HP[i-2]
        a1 = np.exp( -1.414 * np.pi / lperiod)
        b1 = 2 * a1 * np.cos( 1.414*np.pi/lperiod)
        c2 = b1 
        c3 = -a1 * a1 
        c1 = 1 - c2 - c3
        Filt[i] = c1 * ( HP[i] + HP[i-1]) / 2 + c2 * Filt[i-1] + c3 * Filt[i-2]
        Peak[i] = 0.991 * Peak[i-1]
        if np.abs(Filt[i]) > Peak[i]:
            Peak[i] = np.abs(Filt[i])
        if Peak[i] != 0:
            X[i] = Filt[i] / Peak[i] 
        quotient1[i] = (X[i] + k1) / (k1 * X[i] + 1)
    return quotient1 
    
