from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d
import scipy 
from collections import namedtuple

CRSI = namedtuple('CRSI',['smaK','smaD'])

"""
https://www.tradingview.com/script/vWAPUAl9-Stochastic-Connors-RSI/#chart-view-comments
slightly inaccurate percentrank because percentile of score returns a scalar
 percentrank = np.floor(scipy.stats.percentileofscore(prepercentrank[i-roclength:],prepercentrank[i],kind='rank'))
""" 
def stochcrsi(candles: np.ndarray, stochlength: int= 3, smoothK: int=3, smoothD: int=3, rsilength:int=3,updownlength:int=2,roclength:int=100, source_type: str = "close", sequential: bool = False ) -> CRSI: 
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)  
    newupdown = fast_crsi(source,candles,updownlength)
    updownrsi = fast_rsi(newupdown,updownlength)
    rsi = fast_rsi(source,rsilength)
    prepercentrank = talib.ROC(source,1)
    percentrank = (scipy.stats.rankdata(prepercentrank[-roclength:],method='average'))
    finalpercentrank = same_length(source,percentrank)
    crsi = (rsi + updownrsi + (finalpercentrank))/3
    ll = min_filter1d_same(crsi,stochlength)
    hh = max_filter1d_same(crsi,stochlength)
    stoch = 100 * (crsi - ll)/(hh-ll)
    smaK = talib.SMA(stoch, smoothK)
    smaD = talib.SMA(smaK, smoothD)
    if sequential:
        return CRSI(smaK,smaD)
    else:
        return CRSI(smaK[-1],smaD[-1])
    
@njit    
def fast_crsi(source,candles,updownlength):
    updown = np.full_like(source,0)
    newupdown = np.full_like(source,0)
    updownrsi = np.full_like(source,0)
    for i in range(source.shape[0]):
        if source[i] > source[i-1]:
            newupdown[i] = updown[i-1] + 1 if updown[i-1] >= 0 else 1 
        else:
            if source[i] < source[i-1]: 
                newupdown[i] = updown[i-1] - 1 if updown[i-1] <= 0 else -1 
            else:
                newupdown[i] = 0 
    return newupdown
  
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
    
def max_filter1d_same(a, W, fillna=np.nan):
    out_dtype = np.full(0,fillna).dtype
    hW = (W-1)//2 # Half window size
    out = maximum_filter1d(a,size=W, origin=hW)
    if out.dtype is out_dtype:
        out[:W-1] = fillna
    else:
        out = np.concatenate((np.full(W-1,fillna), out[W-1:]))
    return out    

def min_filter1d_same(a, W, fillna=np.nan):
    out_dtype = np.full(0,fillna).dtype
    hW = (W-1)//2 # Half window size
    out = minimum_filter1d(a,size=W, origin=hW)
    if out.dtype is out_dtype:
        out[:W-1] = fillna
    else:
        out = np.concatenate((np.full(W-1,fillna), out[W-1:]))
    return out   
