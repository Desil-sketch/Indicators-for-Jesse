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
https://www.tradingview.com/script/0Kp5Q6j2-Papercuts-Dynamic-EMA-Relative-Parameter-Function/#chart-view-comment-form
DyanmicEMA direction used to determine bull or bear.
"""
#jesse backtest  '2021-01-03' '2021-03-02'

def emarelativevolume(candles: np.ndarray, rvolMin:float=0.75, rvolMax:float=1.5,rvolLen:int=50,lengthMult:int= 5, stdEMA:int=50, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential) 
    source = get_candle_source(candles, source_type=source_type) 
    averageVolume = hma(source,candles[:,5], rvolLen)
    relativeVolume2input = candles[:,5]/averageVolume
    relativeVolume = hma(source,relativeVolume2input,rvolLen)
    theEMASlow = Ema_f(source,source,candles,stdEMA)
    theEMAFast = Ema_f(source,source,candles,(stdEMA/lengthMult))
    fastrev = fast_relativeVal(source,relativeVolume,candles,rvolMin,rvolMax,stdEMA,stdEMA/lengthMult)
    theEMADynamic = Ema_f2(source,source,candles,fastrev)
    
    if sequential:
        return theEMADynamic
    else:
        return theEMADynamic[-1]

@njit    
def fast_relativeVal(source2,source,candles,in_bot,in_top,out_bot,out_top):
    clampsource = np.full_like(source2,0)
    inDiffIncrement = np.full_like(source2,0)
    outDiffIncrement = np.full_like(source2,0)
    output = np.full_like(source2,0)
    for i in range(source2.shape[0]):
        if (source[i] > in_top):
            clampsource[i] = in_top
        else:
            if source[i] < in_bot:
                clampsource[i] = in_bot 
            else:
                clampsource[i] = source[i] 
        inDiffIncrement[i] = (in_top - in_bot)
        outDiffIncrement[i] = (out_top - out_bot)
        output[i] = out_bot + (clampsource[i] - in_bot) * outDiffIncrement[i] / inDiffIncrement[i] 
    return output

@njit 
def Ema_f2(source1, source2,candles, length):
    alpha = np.full_like(source1,0)
    sum1 = np.full_like(source1,0)
    for i in range(0,source1.shape[0]):
        alpha[i] =  2 / (length[i] + 1) 
        sum1[i-1] = 0 if np.isnan(sum1[i-1]) else sum1[i-1]
        sum1[i] = alpha[i] * source2[i] + (1 - alpha[i]) * sum1[i-1]
    return sum1
 
@njit 
def Ema_f(source1,source2,candles,length):
    alpha = 2 / (length+1)
    sum1 = np.full_like(source1,0)
    for i in range(0,source1.shape[0]):
        sum1[i-1] = 0 if np.isnan(sum1[i-1]) else sum1[i-1]
        sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1]
    return sum1  
 
@njit
def hma(source1,source,p1):
    final1 = np.full_like(source1,0)
    final2 = np.full_like(source1,0)
    final3 = np.full_like(source1,0)
    for i in range(p1,source1.shape[0]):
        norm1 = 0.0
        sum1 = 0.0
        weight1 = 0.0
        p0 = p1/2
        for j in range(int(p0)):
            weight1 = (p0 - (j)) * p0
            norm1 = norm1 + weight1
            sum1 = sum1 + source[i-j] * weight1
        final1[i] = 2 * (sum1 / norm1)
        weight2 = 0.0
        norm2 = 0.0
        sum2 = 0.0
        for j in range(int(p1)):
            weight2 = (p1 - (j)) * p1
            norm2 = norm2 + weight2
            sum2 = sum2 + source[i-j] * weight2
        final2[i] = (-(sum2/norm2)) + final1[i]
        p3 = np.floor(np.sqrt(p1))
        weight3 = 0.0
        norm3 = 0.0
        sum3 = 0.0
        for j in range(int(p3)):
            weight3 = (p3 - (j)) * p3
            norm3 = norm3 + weight3
            sum3 = sum3 + final2[i-j] * weight3
        final3[i] = (sum3 / norm3)
    return final3
        
