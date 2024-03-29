"""
List full of helpful functions
""" 
from numba import njit 
from numpy.lib.stride_tricks import as_strided
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d
import numpy as np





def max_rolling1(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.max(rolling,axis=axis)
        
def min_rolling1(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.min(rolling,axis=axis)        
        
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return as_strided(a, shape=shape, strides=strides)    

@njit
def std2(source,avg,per):
    std1 = np.full_like(source,0)
    for i in range(source.shape[0]):
        sum1 = 0.0
        sum2 = 0.0
        for j in range(per):
            sum1 = (source[i-j] + -avg[i])
            sum2 = sum2 + sum1 * sum1 
        std1[i] = np.sqrt(sum2 / per)
    return std1 
        
@njit 
def pine_wma(source1,source2,length):
    res = np.full_like(source1,length)
    for i in range(source1.shape[0]):
        weight = 0.0
        norm = 0.0 
        sum1 = 0.0
        for j in range(length):
            weight = (length - j)*length
            norm = norm + weight 
            sum1 = sum1 + source2[i-j] * weight
        res[i] = sum1/norm 
    return res 
    
@njit 
def pine_ema(source1, source2, length):
    alpha = 2 / (length + 1)
    sum1 = np.full_like(source1,0)
    for i in range(10,source1.shape[0]):
        sum1[i-1] = 0 if np.isnan(sum1[i-1]) else sum1[i-1]
        sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1]
    return sum1 

"""
Min/Max scaling of dynamic length
"""

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
    
@njit
def pine_percentrank(source,source2,length):
    pr = np.full_like(source,0)
    for i in range(length,source.shape[0]):
        count = 0.0
        for j in range(1,length+1):
            count = count + 1 if source2[i] >= source2[i-j] else count + 0 
        pr[i] = 100 * (count / length)
    return pr 
    
@njit 
def pine_rma(source1, source2, length):
    alpha = 1/length
    sum1 = np.full_like(source1,0)
    for i in range(source1.shape[0]):
        sum1 = alpha * source2[i] + (1 - alpha) * sum1[i-1]
    return sum1
   
@njit
def sma_numpy_acc(a, p):
    acc = np.empty_like(a)
    acc[0] = a[0]
    n = len(a)
    for i in range(1, n):
        acc[i] = acc[i-1] + a[i]
    for i in range(n-1, p-1, -1):
        acc[i] = (acc[i] - acc[i-p]) / p
    acc[p-1] /= p
    for i in range(p-1):
        acc[i] = np.nan
    return acc
"""
sma_numpy_acc is slightly faster as a moving average 
""" 
def highestlowestbarsback(source,candles,amplitude):
    for j in range(0,amplitude):
        if highestvalue <= (candles[i-j,3]):
            highestvalue = (candles[i-j,3])
            highindex = -j
        if lowestvalue >= (candles[i-j,4]):
            lowestvalue = (candles[i-j,4])
            lowindex = -j   
@njit
def pine_sma(source1,source2,length):
    sum1 = np.full_like(source1,0)
    for i in range(source1.shape[0]):
        sum2 = 0.0
        for j in range(length):
            sum2 = sum2 + source2[i-j]/length
        sum1[i] = sum2 
    return sum1 
    
@njit
def fast_rsi(source,length):
    u = np.full_like(source, 0)
    d = np.full_like(source, 0)
    rs = np.full_like(source, 0)
    res = np.full_like(source, 0)
    alpha = 1 / length 
    sumation1 = np.full_like(source, 1)
    sumation2 = np.full_like(source, 1)
    for i in range(source.shape[0]):
        u[i] = np.maximum((source[i] - source[i-1]),0)
        d[i] = np.maximum((source[i-1] - source[i]), 0) 
        sumation1[i] = alpha * u[i] + (1 - alpha) * (sumation1[i-1])
        sumation2[i] = alpha * d[i] + (1 - alpha) * (sumation2[i-1]) 
        rs[i] = sumation1[i]/sumation2[i]
        res[i] = 100 - 100 / ( 1 + rs[i])
    return res 
