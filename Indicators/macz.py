from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length, get_config
import numpy as np
from collections import namedtuple
from numba import njit
import talib 
from typing import Union
from numpy.lib.stride_tricks import sliding_window_view

MACZ = namedtuple('macz', ['hist', 'signal','maczt'])

"""
https://www.tradingview.com/script/HNrsbpJ2-MAC-Z-VWAP-Indicator-LazyBear/
"""

def macz(candles: np.ndarray, fastperiod: int= 24, slowperiod: int=32, signalperiod:int = 8,lengthZ:int=20,lengthSTD:int=25, source_type: str = "close", sequential: bool = False) -> MACZ:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    zscore = calc_zvwap(candles,source,lengthZ)
    fastMA = talib.SMA(source,fastperiod)
    slowMA = talib.SMA(source,slowperiod)
    macd = fastMA - slowMA
    S1 = source
    nrows1 = S1.size - lengthSTD + 1
    n1 = S1.strides[0]
    preS1 = np.lib.stride_tricks.as_strided(S1,shape=(nrows1,lengthSTD),strides=(n1,n1))
    S12 = same_length(source,np.std(preS1, axis=1))
    macz_t = zscore + macd / S12
    signal = talib.SMA(macz_t, signalperiod)
    hist = macz_t - signal
    
    if sequential: 
        return MACZ(hist,signal,macz_t)
    else:    
        return MACZ(hist[-1],signal[-1],macz_t[-1])
    
    
def calc_zvwap(candles,source,period):
    f1 = candles[:,5] * source
    f2 = candles[:,5]
    nrows1 = f1.size - period + 1 
    n1 = f1.strides[0] 
    presm1 = np.lib.stride_tricks.as_strided(f1,shape=(nrows1,period),strides=(n1,n1))
    sm1 = same_length(source,np.sum(presm1, axis=1))
    nrows2 = f2.size - period + 1 
    n2 = f2.strides[0] 
    presm2 = np.lib.stride_tricks.as_strided(f2,shape=(nrows2,period),strides=(n2,n2))
    sm2 = same_length(source,np.sum(presm2, axis=1))
    mean = sm1/sm2 
    vwapsd = np.sqrt(talib.SMA(np.power(source-mean,2),period))
    output = (source-mean)/(vwapsd)
    return output
   

    
   
