from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d

RSVV = namedtuple('RSVV',['m', 'ind'])


'''
https://www.tradingview.com/script/1IyvlXWs-Relative-Strength-Volatility-Variable-Bands-DW/#chart-view-comments
optional outputs of deviation bands 
''' 

def rsvv(candles: np.ndarray, vper: int = 7, per: int= 7, source_type: str = "close", sequential: bool = False ) -> RSVV:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    hiband = revrsi(source,source,vper,70)
    loband = revrsi(source,source,vper,30)
    vol = hiband - loband 
    vi = (vol - min_filter1d_same(vol,vper))/(max_filter1d_same(vol,vper) - min_filter1d_same(vol,vper))
    ma = vma(source,source,per,vi)
    srch = np.full_like(source,0)
    srcl = np.full_like(source,0)
    for i in range(source.shape[0]):
        srch[i] = source[i] if source[i] > ma[i] else ma[i] 
        srcl[i] = source[i] if source[i] < ma[i] else ma[i]    
    mah = vma(source,srch,per,vi)
    mal = vma(source,srcl,per,vi)
    m = (mah + mal)/2
    dist = (mah - mal)/2
    h5 = m + dist*5
    h4 = m + dist*4
    h3 = m + dist*3
    h2 = m + dist*2
    h1 = m + dist*1
    l1 = m - dist*1
    l2 = m - dist*2
    l3 = m - dist*3
    l4 = m - dist*4
    l5 = m - dist*5
    mup = np.full_like(source,0)
    mdn = np.full_like(source,0)
    for i in range(source.shape[0]):
        if m[i] > m[i-1]:
            mup[i] = 1 
            mdn[i] = 0 
        elif m[i] < m[i-1]:
            mup[i] = 0
            mdn[i] = 1 
        else: 
            mup[i] = mup[i-1]
            mdn[i] = mdn[i-1] 
    ind = np.full_like(source,0)
    for i in range(source.shape[0]):
        if (mup[i] > 0) and (source[i] >= h1[i]) and (source[i] > source[i-1]):
            ind[i] = 2 
        elif (mup[i] > 0) and (source[i] >= h1[i]) and (source[i] <= source[i-1]):
            ind[i] = 1 
        elif (mdn[i] > 0) and (source[i] <= l1[i]) and (source[i] < source[i-1]):
            ind[i] = -2 
        elif (mdn[i] > 0) and (source[i] <= l1[i]) and (source[i] >= source[i-1]):
            ind[i] = -1 
        else:
            ind[i] = 0 
    if sequential:
        return RSVV(m,ind)
    else:
        return RSVV(m[-1],ind[-1])

@njit    
def revrsi(source,x, t, v):
    wper = np.full_like(source,0)
    k = np.full_like(source,0)
    uc = np.full_like(source,0)
    dc = np.full_like(source,0)
    val = np.full_like(source,0)
    revrsi = np.full_like(source,0)
    for i in range(source.shape[0]):
        wper[i] = 2 * t - 1 
        k[i] = 2/(wper[i] + 1)
        uc[i] = k[i]*(x[i]-x[i-1]) + (1 - k[i]) * (uc[i-1]) if x[i] > x[i-1] else (1-k[i])*uc[i-1]
        dc[i] = (1 - k[i]) * (dc[i-1]) if x[i] > x[i-1] else k[i]*(x[i-1] - x[i]) + (1 - k[i]) * dc[i-1] 
        val[i] = (t - 1) * (dc[i] * v/(100-v) - uc[i])
        revrsi[i] = x[i] + val[i] if val[i] >= 0 else x[i] + val[i]*(100 - v)/v
    return revrsi
 
@njit 
def vma(source,x,t,o):
    vma = np.full_like(source,0)
    for i in range(source.shape[0]):
        vma[i] = x[i] 
        vma[i] = x[i] if np.isnan(vma[i-1]) else (1-(1/t)*o[i])*(vma[i-1])+(1/t)*o[i]*x[i]
    return vma 
    
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
