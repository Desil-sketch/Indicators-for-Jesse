from collections import namedtuple
import numpy as np
from numba import njit, jit
import talib 
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
import jesse.indicators as ta
from typing import Union
from collections import namedtuple

HVP = namedtuple('HVP', ['hvp', 'hvpsma', 'sig' ])

def hvpsma(candles: np.ndarray, length: int = 21, annuallength: int = 200, source_type: str = "close", sequential: bool = False) -> HVP: 
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    hvp = fast_hvpsma(source,length,annuallength)
    hvpsma = talib.SMA(hvp, length)
    srcema = talib.EMA(source,length)
    ones = np.ones_like(source)
    zeros = np.zeros_like(source)
    negativeone = np.full_like(source,-1)
    sig = np.full_like(source,0)
    if hvp[-1] >= hvpsma[-1] and source[-1] > srcema[-1]:
        sig = ones
    elif hvp[-1] >= hvpsma[-1] and source[-1] < srcema[-1]:
        sig = negativeone
    else:
        sig = zeros
    if sequential: 
        return HVP(hvp, hvpsma, sig)
    else:    
        return HVP(hvp[-1], hvpsma[-1],sig[-1])
"""
count only works at annuallength 200 or below on 4h chart
HVP not 100% accuracte 

"""	
	
	
@jit( error_model="numpy")
def fast_hvpsma(source, length, annuallength):
    r = np.full_like(source,0)
    rAvg = np.full_like(source,0)
    preSum1 = np.full_like(source,0)
    hv = np.full_like(source,0)
    hvp = np.full_like(source,0)
    hvpSMA = np.full_like(source,0)
    count1 = np.full_like(source,0)
    for i in range(length,source.shape[0]):
        r[i] = np.log(source[i] / source[i-1])
        rAvg[i] = np.mean(r[i-length+1:i+1])
        preSum1[i] = (np.power(r[i] - rAvg[i],2))
        hv[i] = np.sqrt(np.sum(preSum1[i-length+1:i+1]) / (length - 1))*np.sqrt(annuallength)
        count = 0
        hvj = 0
        for j in range(annuallength):
            if np.isnan(hv[i-j]):
                hv[i-j] = 0
            if hv[i-j] < hv[i]:
                hvj = 1
            else:
                hvj = 0
            count = count + hvj
        count1[i] = count
        hvp[i] = (count1[i] / annuallength) * 100
    return hvp
