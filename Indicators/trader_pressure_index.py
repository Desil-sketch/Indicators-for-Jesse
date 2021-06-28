from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

TPX = namedtuple('TPX',['tpx', 'avgbulls', 'avgbears'])

'''
https://www.tradingview.com/script/v8sBugsW-RedK-Trader-Pressure-Index-TPX-v1-0/#chart-view-comments
''' 

def tpx(candles: np.ndarray, length: int= 7, smooth: int=3, clevel:int=30, source_type: str = "close", sequential: bool = False ) -> TPX:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    tpx, avgbulls, avgbears = fast_tpx(source,candles,length,smooth,clevel)
    if sequential:
        return TPX(tpx,avgbulls,avgbears) 
    else:
        return TPX(tpx[-1], avgbulls[-1], avgbears[-1]) 
	
@njit
def fast_tpx(source,candles,length,smooth,clevel):
    R = np.full_like(source,0)
    changeHigh = np.full_like(source,0)
    changeLow = np.full_like(source,0)
    hiup = np.full_like(source,0)
    loup = np.full_like(source,0)
    bulls = np.full_like(source,0)
    bears = np.full_like(source,0)
    hidn = np.full_like(source,0)
    lodn = np.full_like(source,0)
    tpx = np.full_like(source,0)
    for i in range(source.shape[0]):
        R[i] = np.maximum(candles[:,3][i],candles[:,3][i-1]) - np.minimum(candles[:,4][i], candles[:,4][i-1])
        changeHigh[i] = candles[:,3][i] - candles[:,3][i-1]
        changeLow[i] = candles[:,4][i] - candles[:,4][i-1] 
        hiup[i] = np.maximum(changeHigh[i],0)
        loup[i] = np.maximum(changeLow[i],0)
        bulls[i] = (hiup[i] + loup[i]) / R[i] * 100 
        avgbulls = pine_wma(source,bulls,length)
        hidn[i] = np.minimum(changeHigh[i],0)
        lodn[i] = np.minimum(changeLow[i],0)
        bears[i] = -1 * (hidn[i] + lodn[i]) / R[i] * 100 
        avgbears = pine_wma(source,bears,length)
        net = avgbulls - avgbears 
        tpx = pine_wma(source,net, smooth)
    return tpx, avgbulls, avgbears


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
