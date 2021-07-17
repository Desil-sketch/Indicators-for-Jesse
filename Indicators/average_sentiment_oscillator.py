import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple

AVO = namedtuple('AVO',['asobulls','asobears'])

'''
https://www.tradingview.com/script/hz1PKu3G/#chart-view-comments
''' 
  
def avo(candles: np.ndarray, length:int=10, mode:int=0,source_type: str = "close", sequential: bool = False ) -> AVO:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    high = candles[:,3]
    low = candles[:,4]
    grouplow = same_length(candles,min_rolling1(low,length))
    grouphigh = same_length(candles,max_rolling1(high,length))
    TempBufferBulls, TempBufferBears = fast_avo(source,candles,length,mode,grouplow,grouphigh)
    ASOBulls = talib.SMA(TempBufferBulls,length)
    ASOBears = talib.SMA(TempBufferBears,length)
    if sequential:
        return AVO(ASOBulls,ASOBears)
    else:
        return AVO(ASOBulls[-1],ASOBears[-1])

@njit
def fast_avo(source,candles,length,mode,grouplow,grouphigh):
    intrarange = np.full_like(source,0)
    groupopen = np.full_like(source,0)
    grouprange = np.full_like(source,0)
    K1 = np.full_like(source,0)
    K2 = np.full_like(source,0)
    intrabarbulls = np.full_like(source,0)
    groupbulls = np.full_like(source,0)
    intrabarbears = np.full_like(source,0)
    groupbears = np.full_like(source,0)
    TempBufferBears = np.full_like(source,0)
    TempBufferBulls = np.full_like(source,0)
    ASOBulls = np.full_like(source,0)
    ASOBears = np.full_like(source,0)
    for i in range(length+2,source.shape[0]):
        intrarange[i] = candles[:,3][i] - candles[:,4][i] 
        groupopen[i] = candles[:,1][i-(length-1)]
        grouprange[i] = grouphigh[i] - grouplow[i]
        K1[i] = 1 if intrarange[i] == 0 else intrarange[i] 
        K2[i] = 1 if grouprange[i] == 0 else grouprange[i] 
        intrabarbulls[i] = ((((candles[:,2][i] - candles[:,4][i])+(candles[:,3][i]-candles[:,1][i]))/2)*100)/K1[i] 
        groupbulls[i] = ((((candles[:,2][i] - grouplow[i])+(grouphigh[i] - groupopen[i]))/2)*100)/K2[i]
        intrabarbears[i] = ((((candles[:,3][i] - candles[:,2][i])+(candles[:,1][i] - candles[:,4][i]))/2)*100)/K1[i]
        groupbears[i] = ((((grouphigh[i]-candles[:,2][i])+(groupopen[i]-grouplow[i]))/2)*100)/K2[i]
        if mode == 0:   
           TempBufferBulls[i] = (intrabarbulls[i]+groupbulls[i])/2 
           TempBufferBears[i] = (intrabarbears[i] + groupbears[i])/2
        elif mode == 1:
           TempBufferBulls[i] = intrabarbulls[i] 
           TempBufferBears[i] = intrabarbears[i] 
        else:
           TempBufferBulls[i] = groupbulls[i] 
           TempBufferBears[i] = groupbears[i] 
    return TempBufferBulls, TempBufferBears
   
    
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
