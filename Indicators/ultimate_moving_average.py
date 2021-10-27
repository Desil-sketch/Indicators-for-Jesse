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
https://www.tradingview.com/script/hYVS9kuu-Ultimate-Moving-Average-CC-RedK/
RSI instead of MFI yet still not 100% accurate. 
"""
#jesse backtest  '2021-01-03' '2021-03-02'

def ultimatemovingaverage(candles: np.ndarray, acc : float = 1.0, minLength: int= 5, maxLength: int = 50, smoothLength:int=4,source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential) 
    source = get_candle_source(candles, source_type=source_type) 
    preuma = fast_uma_p1(source,candles,maxLength,minLength,acc)
    uma = talib.WMA(preuma,smoothLength) if smoothLength > 1 else preuma 
    slo = source - uma 
    if sequential:
        return uma
    else:
        return uma[-1]

@njit
def fast_uma_p1(source,candles, maxLength, minLength,acc):
    std1 = 0.0
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    length = np.full_like(source,0)
    len1 = 0
    mean = np.full_like(source,0)
    mf = np.full_like(source,0)
    mfScaled = 0.0
    p = 0.0
    preuma = np.full_like(source,0)
    rsi1 = np.full_like(source,0)
    mfi1 = np.full_like(source,0)
    u = np.full_like(source, 0)
    e = np.full_like(source, 0)
    rs = np.full_like(source, 0)
    res = np.full_like(source, 0)
    sumation1 = np.full_like(source, 1)
    sumation2 = np.full_like(source, 1)
    alpha = np.full_like(source,0)
    for i in range(maxLength,source.shape[0]):
        mean[i] = np.mean(source[(i-maxLength+1):i+1])
        sum1 = 0.0
        sum2 = 0.0
        for j in range(maxLength):
            sum1 = (source[i-j] + -mean[i])
            sum2 = sum2 + sum1 * sum1 
        std1 = np.sqrt(sum2 / maxLength)
        a = mean[i] - (1.75 * std1)
        b = mean[i] - (0.25 * std1)
        c = mean[i] + (0.25 * std1)
        d = mean[i] + (1.75 * std1)
        if source[i] >= b and source[i] <= c:
            length[i] = length[i-1] + 1
        elif source[i] < a or source[i] > d:
            length[i] = length[i-1] - 1 
        else:
            length[i] = length[i-1] 
        length[i] = np.maximum(np.minimum(length[i],maxLength),minLength)
        len1 = int(np.round(length[i]))
        alpha[i] = 1 / len1
        u[i] = np.maximum((source[i] - source[i-1]), 0)
        e[i] = np.maximum((source[i-1] - source[i]), 0) 
        sumation1[i] = alpha[i] * u[i] + (1 - alpha[i]) * (sumation1[i-1])
        sumation2[i] = alpha[i] * e[i] + (1 - alpha[i]) * (sumation2[i-1]) 
        rs[i] = sumation1[i]/sumation2[i] 
        rsi1[i] = 100 - 100 / ( 1 + rs[i])
        if candles[:,5][i] == 0:
            mfi1[i] = fast_mfi(source,candles,len1)[i]
        mf[i] = rsi1[i] #rsi1[i] if candles[:,5][i] == 0 else mfi1[i]
        mfScaled = (mf[i] * 2) - 100
        p = acc + (np.abs(mfScaled)/25)
        sum1 = 0.0
        weight = 0.0
        weightSum = 0.0
        for j in range(int(len1)):   
            weight = np.power(len1 - j,p)
            sum1 = sum1 + (source[i-j] * weight)
            weightSum = weightSum + weight
        preuma[i] = sum1 / weightSum
    return preuma
 
@njit
def fast_rsi(source,length):
    u = np.full_like(source, 0)
    d = np.full_like(source, 0)
    rs = np.full_like(source, 0)
    res = np.full_like(source, 0)
    sumation1 = np.full_like(source, 1)
    sumation2 = np.full_like(source, 1)
    alpha = np.full_like(source,0)
    for i in range(source.shape[0]):
        alpha[i] = 1 / length 
        u[i] = np.maximum((source[i] - source[i-1]),0)
        d[i] = np.maximum((source[i-1] - source[i]), 0) 
        sumation1[i] = alpha[i] * u[i] + (1 - alpha[i]) * (sumation1[i-1])
        sumation2[i] = alpha[i] * d[i] + (1 - alpha[i]) * (sumation2[i-1]) 
        rs[i] = sumation1[i]/sumation2[i]
        res[i] = 100 - 100 / ( 1 + rs[i])
    return res 
    
    
# @njit    
# def other_rsi(source1,source,length):
    # preup = np.full_like(source1,0)
    # predown = np.full_like(source1,0)
    # up = np.full_like(source1,0)
    # down = np.full_like(source1,0)
    # rsi = np.full_like(source1,0)
    # alpha = 0.0
    # for i in range(source1.shape[0]):
        # preup[i] = np.maximum((source[i] - source[i-1]),0)
        # predown[i] = -np.minimum((source[i] - source[i-1]),0)
        # alpha = 1 / length
        # up[i] =  alpha * preup[i] + (1 - alpha) * up[i-1] 
        # down[i] = alpha * predown[i] + (1 - alpha) * down[i-1] 
        # if down[i] == 0:
            # rsi[i] = 100 
        # elif up[i] == 0:
            # rsi[i] = 100 
        # else:
            # rsi[i] = 100 - (100 / (1 + up[i] / down[i]))
    # return rsi 

@njit  
def fast_mfi(source,candles,length):
    prefloat_lower = np.full_like(source,0)
    prefloat_upper = np.full_like(source,0)
    float_upper = np.full_like(source,0)
    float_lower = np.full_like(source,0)
    res = np.full_like(source,0)
    u = np.full_like(source, 0)
    d = np.full_like(source, 0)
    rs = np.full_like(source, 0)
    res = np.full_like(source, 0)
    sumation1 = np.full_like(source, 1)
    sumation2 = np.full_like(source, 1)
    alpha = np.full_like(source,0)
    length1 = 0
    for i in range(0,source.shape[0]):
        length1 = (length)
        prefloat_upper[i] = 0.0 * candles[:,5][i] if (source[i] - source[i-1]) <= 0.0 else source[i] * candles[:,5][i]
        prefloat_lower[i] = 0.0 * candles[:,5][i] if (source[i] - source[i-1]) >= 0.0 else source[i] * candles[:,5][i] 
        float_upper[i] = np.sum(prefloat_upper[(i-(length1+1)):i+1])
        float_lower[i] = np.sum(prefloat_lower[(i-(length1+1)):i+1]) 
        alpha[i] = 1 / float_lower[i] if float_lower[i] > 0 else 0 
        u[i] = np.maximum((float_upper[i] - float_upper[i-1]),0)
        d[i] = np.maximum((float_upper[i-1] - float_upper[i]), 0) 
        sumation1[i-1] = 0 if np.isnan(sumation1[i-1]) else sumation1[i-1]
        sumation2[i-1] = 0 if np.isnan(sumation2[i-1]) else sumation2[i-1] 
        sumation1[i] = alpha[i] * u[i] + (1 - alpha[i]) * (sumation1[i-1])
        sumation2[i] = alpha[i] * d[i] + (1 - alpha[i]) * (sumation2[i-1]) 
        # sumation1[i] = u[i] + (alpha[i] - 1) * sumation1[i-1] / alpha[i] 
        # sumation2[i] = d[i] + (alpha[i] - 1) * sumation2[i-1] / alpha[i] 
        rs[i] = sumation1[i]/sumation2[i]
        res[i] = 100 - 100 / ( 1 + rs[i])
    return res
    
