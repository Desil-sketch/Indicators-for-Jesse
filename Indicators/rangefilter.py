from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
import math 

RangeFilter = namedtuple('RangeFilter',['topdev', 'bottomdev', 'filt'])

def rangefilter(candles: np.ndarray, av_n:bool=False,av_samples:int=2, movement_source:str='close', range_mult: float = 2.6, range_period: int = 14, scale: str = "Average Change", smooth: bool = False, smoothing_period: int = 9, f_type: str = "Type 1", source_type: str = "close", sequential: bool = False) -> RangeFilter:
    """"
    Type 2 is not accurate because np.floor of calc1 variable
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)  
    tr = talib.TRANGE(candles[:,3],candles[:,4],candles[:,2])
    if movement_source == "Wicks":
        h_val = candles[:, 3]  
        l_val = candles[:, 4] 
    else: 
        h_val = candles[:, 2] 
        l_val = candles[:, 2] 
    topdev, bottomdev, filt = rng_filt(candles,source, h_val, l_val, range_period, f_type, smooth, smoothing_period,scale,tr,range_mult,av_n,av_samples)  
    if sequential: 
        return RangeFilter(topdev, bottomdev, filt)
    else:    
        return RangeFilter(topdev[-1],bottomdev[-1],filt[-1])

#jesse backtest  '2021-01-03' '2021-03-02'
 
@njit
def rng_filt(candles,source, h, l, t, f_type, smooth, st,scale,tr,qty,av_n,av_samples):
    r = np.full_like(source,0)
    rng_size = np.full_like(source,0)
    rng_filt1 = np.full_like(source,0)
    topdev = np.full_like(source,0)
    bottomdev = np.full_like(source,0)
    closediff = np.full_like(source,0)
    calc1 = np.full_like(source,0.0)
    sma1 = np.full_like(source,0)
    sma2 = np.full_like(source,0)
    cond = np.full_like(source,0)
    rng_filt2 = np.full_like(source,0)
    ones = np.full_like(source,1)
    hi_band2 = np.full_like(source,0)
    lo_band2 = np.full_like(source,0)
    for i in range(source.shape[0]):
        if scale == "% of Price":
            rng_size[i] = candles[:, 2][i] * qty/100
        elif scale == "ATR": 
            rng_size[i] = qty*pine_ema(source,tr, t,ones)[i]
        elif scale == "Average Change": 
            if np.isnan(candles[:,2][i]): 
                rng_size[i] = qty*pine_ema(source,tr, t,ones)[i]
            else: 
                closediff[i] = np.abs(candles[:,2][i] - candles[:,2][i-1])
                rng_size[i] = qty*(pine_ema(source,closediff,t,ones)[i])
        elif scale == "Standard Deviation":
            sma1[i] = sma_numpy_acc(np.power(candles[:,2], 2),t)[i]
            sma2[i] = sma_numpy_acc(candles[:,2],t)[i]
            rng_size[i] = np.sqrt(sma1[i] - np.power(sma2[i],2))*qty
        else:
            rng_size[i] = qty
        if smooth == True: 
            r[i] = pine_ema(source,rng_size, st,ones)[i] 
        else: 	
            r[i] = rng_size[i] 
        if f_type == "Type 1": 
            if h[i] > rng_filt1[i-1]:
                if ((h[i]-r[i]) < (rng_filt1[i-1])): 
                    rng_filt1[i] = rng_filt1[i-1]
                else:  
                     rng_filt1[i] = (h[i]-r[i])
            elif ((l[i] + r[i]) > rng_filt1[i-1]): 
                rng_filt1[i] = rng_filt1[i-1]
            else:
                rng_filt1[i] = (l[i] + r[i])
        if f_type == "Type 2": 
            if np.isnan(rng_filt1[i-1]):
                rng_filt1[i-1] = (h[i]+l[i])/2
            elif (h[i] >= (rng_filt1[i-1] + r[i])): 
                calc1[i] = (np.abs(source[i] - rng_filt1[i-1]))/r[i] 
                rng_filt1[i] = (rng_filt1[i-1] + (calc1[i])*r[i]) 
            elif (l[i] <= (rng_filt1[i-1] - r[i])): 
                calc1[i] = (np.abs(source[i] - rng_filt1[i-1]))/r[i]
                rng_filt1[i] = (rng_filt1[i-1] - (calc1[i])*r[i])
            else:   
                rng_filt1[i] = rng_filt1[i-1]
        topdev[i] = rng_filt1[i] + r[i] 
        bottomdev[i] = rng_filt1[i] - r[i]
        cond[i] = 1 if rng_filt1[i] != rng_filt1[i-1] else 0 
        rng_filt2[i] = pine_ema(source,rng_filt1,av_samples,cond)[i] if av_n else rng_filt1[i] 
        hi_band2[i] = pine_ema(source,topdev,av_samples,cond)[i] if av_n else topdev[i] 
        lo_band2[i] = pine_ema(source,bottomdev,av_samples,cond)[i] if av_n else bottomdev[i] 
    return hi_band2,lo_band2,rng_filt2


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
    
@njit 
def pine_ema(source1, source2, length,cond):
    alpha = 2 / (length + 1)
    sum1 = np.full_like(source1,0)
    for i in range(source1.shape[0]):
        sum1[i-1] = 0 if np.isnan(sum1[i-1]) else sum1[i-1]
        sum1[i] = (alpha * source2[i] + (1 - alpha) * sum1[i-1]) if cond[i] == 1 else sum1[i-1] 
    return sum1 
