from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

'''
PineScript PivotHigh and PivotLow 
''' 
PIVOTS = namedtuple('PIVOTS',['newpivothigh', 'newpivotlow', 'lastpivothighprice','lastpivotlowprice'])
  
def pivots(candles: np.ndarray, source_type: str = "close",leftbars:int=7,rightbars:int=7, sequential: bool = False ) -> PIVOTS:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    sourcehigh = candles[:,3]
    sourcelow = candles[:,4]
    newpivothigh,newpivotlow,lastpivothighprice,lastpivotlowprice = Pine_Pivot(source,candles,sourcehigh,sourcelow,leftbars,rightbars)
    if sequential:
        return PIVOTS(newpivothigh,newpivotlow,lastpivothighprice,lastpivotlowprice)
    else:
        return PIVOTS(newpivothigh[-1],newpivotlow[-1],lastpivothighprice[-1],lastpivotlowprice[-1])
    

@njit
def Pine_Pivot(source,candles,sourcehigh,sourcelow,l,r):
    highmiddlesource = np.full_like(source,0)
    lowmiddlesource = np.full_like(source,0)
    pivothigh = np.full_like(source,0)
    pivotlow = np.full_like(source,0)
    lastpivothighprice = np.full_like(source,0)
    lastpivotlowprice = np.full_like(source,0)
    for i in range(source.shape[0]):
        highmiddlesource[i] = sourcehigh[i-r]
        lowmiddlesource[i] = sourcelow[i-r]
        if (np.all(highmiddlesource[i] >= sourcehigh[i-(l+r):i-(r)]) and np.all(highmiddlesource[i] > sourcehigh[i-(r-1):i+1])):
            pivothigh[i] = 1  
            lastpivothighprice[i] = highmiddlesource[i]  
        else:
            pivothigh[i] = 0 
            lastpivothighprice[i] = lastpivothighprice[i-1] 
        if (np.all(lowmiddlesource[i] <= sourcelow[i-(l+r):i-(r)]) and np.all(lowmiddlesource[i] < sourcelow[i-(r-1):i+1])):    
            pivotlow[i] = 1  
            lastpivotlowprice[i] = lowmiddlesource[i] 
        else:
            pivotlow[i] = 0 
            lastpivotlowprice[i] = lastpivotlowprice[i-1] 
    return pivothigh,pivotlow,lastpivothighprice,lastpivotlowprice 
    
