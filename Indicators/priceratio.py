from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d
from collections import namedtuple

"""
https://www.tradingview.com/script/W5lBL0MV-John-Ehlers-The-Price-Radio/#chart-view-comments
"""

PriceRatio = namedtuple('PriceRatio',['deriv','AMplus', 'AMnegative', 'fm'])

def priceratio(candles: np.ndarray, period: int= 14, dev:float=1, source_type: str = "close", sequential: bool = False ) -> PriceRatio:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)  
    deriv = candles[:,2] - np_shift(candles[:,2],dev,np.nan)
    h = max_filter1d_same(deriv,period)
    l = min_filter1d_same(deriv,period)
    envelope = max_filter1d_same(np.abs(deriv),4)
    AMplus = talib.SMA(envelope,period)
    AMnegative = -(AMplus)
    t2 = fast_fm(source,candles,period,dev,h,l,deriv)
    fm = talib.SMA(t2,period)
    if sequential:
        return PriceRatio(deriv,AMplus,AMnegative,fm)
    else:
        return PriceRatio(deriv[-1],AMplus[-1],AMnegative[-1],fm[-1])

def fast_fm(source,candles, period, dev,h,l,deriv):
    t = np.full_like(source,0)
    t2 = np.full_like(source,0)
    for i in range(source.shape[0]):
        t[i] = l[i] if (10*deriv[i]) < l[i] else (10*deriv[i])
        t2[i] = h[i] if t[i] > h[i] else t[i] 
    return t2


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
