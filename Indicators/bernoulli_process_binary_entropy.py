from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d
import scipy 
from numpy.lib.stride_tricks import as_strided

Bernoulli = namedtuple('Bernoulli',['info2','hvp','redtriangle','greentriangle'])

'''
https://www.tradingview.com/script/bvYZ1CdF-Bernoulli-Process-Binary-Entropy-Function/
''' 
  
def bpbe(candles: np.ndarray, len1: int= 22, range1:float=0.67, avg:int=88, vPR:int=5, bc:bool=True, vc:bool=True,source_type: str = "close", sequential: bool = False ) -> Bernoulli:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    info2,hvp,redtriangle,greentriangle = fast_bpbe(source,candles,len1,avg,bc,vc,vPR,range1)
    if sequential:
        return Bernoulli(info2,hvp,redtriangle,greentriangle)
    else:
        return Bernoulli(info2[-1],hvp[-1],redtriangle[-1],greentriangle[-1])

@njit
def fast_bpbe(source,candles,len1, avg, bc, vc,vPR,range1):
    cr = np.full_like(source,0)
    prevr = np.full_like(source,0)
    vr = np.full_like(source,0)
    percentrank1 = np.full_like(source,0)
    percentrank2 = np.full_like(source,0)
    vr2 = np.full_like(source,0)
    cr2 = np.full_like(source,0)
    preinfoc = np.full_like(source,0)
    infoc = np.full_like(source,0)
    preinfov = np.full_like(source,0)
    infov = np.full_like(source,0)
    info2 = np.full_like(source,0)
    hvp = np.full_like(source,0)
    redtriangle = np.full_like(source,0)
    greentriangle = np.full_like(source,0)
    histcolor = np.full_like(source,0)
    for i in range(avg,source.shape[0]):
        cr[i] = source[i] / (np.sum(source[i-(len1-1):i+1]))
        prevr[i] = np.log(candles[:,5][i])
        vr[i] = np.log(candles[:,5][i])/(np.sum(prevr[i-(len1-1):i+1]))
        count1 = 0.0
        count2 = 0.0 
        for j in range(1,avg+1):
            count1 = count1 + 1 if vr[i] >= vr[i-j] else count1 + 0 
            count2 = count2 + 1 if cr[i] >= cr[i-j] else count2 + 0 
        percentrank1[i] = 100 * (count1/avg) 
        percentrank2[i] = 100 * (count2/avg) 
        vr2[i] = np.minimum(np.maximum(percentrank1[i]/100,0.001),0.999)
        cr2[i] = np.minimum(np.maximum(percentrank2[i]/100,0.001),0.999)
        preinfoc[i] = (cr2[i]*np.log10(cr2[i])/np.log10(2)) - (1 - cr2[i]) * np.log10(1-cr2[i])/np.log10(2) if bc else 0 
        infoc[i] = np.sum(preinfoc[i-(len1-1):i+1])
        preinfov[i] = (vr2[i]*np.log10(vr2[i])/np.log10(2)) - (1 - vr2[i]) * np.log10(1-vr2[i])/np.log10(2) if vc else 0 
        infov[i] = np.sum(preinfov[i-(len1-1):i+1])
        info2[i] = infoc[i] - infov[i] 
        count3 = 0.0
        for j in range(1,avg+1):
            count3 = count3 + 1 if info2[i] >= info2[i-j] else count3 + 0
        hvp[i] = 100 * (count3/avg)
        redtriangle[i] = -1 if hvp[i] > (100-vPR) else np.nan
        greentriangle[i] = 1 if hvp[i] < vPR else np.nan 
        if info2[i] > range1:
            histcolor[i] = 1 
        elif info2[i] < -range1:
            histcolor[i] = -1 
        else:
            histcolor[i] = 0
    return info2, hvp, redtriangle, greentriangle 
        
    
@njit
def pine_percentrank(source,source2,length):
    pr = np.full_like(source,0)
    for i in range(length,source.shape[0]):
        count = 0.0
        for j in range(1,length+1):
            count = count + 1 if source2[i] >= source2[i-j] else count + 0 
        pr[i] = 100 * (count / length)
    return pr 
	

