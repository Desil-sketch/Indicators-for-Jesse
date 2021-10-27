from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
import tulipy as ti
import math 
import sys
np.set_printoptions(threshold=sys.maxsize)
"""
https://www.tradingview.com/script/HBuT6e1v-Pivot-High-Low-Analysis-Forecast-LUX/
highpivotpredict and lowpivotpredict predict future pivots based on avg barssince of pivots with length offset.
"""
#jesse backtest  '2021-01-03' '2021-03-02'

PIVOTSINFO = namedtuple('PIVOTINFO',['highpivotpredict','lowpivotpredict'])

def pivotinfo(candles: np.ndarray, source_type: str = "close",leftbars:int=50,rightbars:int=50,length:int=50, sequential: bool = True) -> PIVOTSINFO:
    candles = slice_candles(candles, sequential) 
    source = get_candle_source(candles, source_type=source_type) 
    newpivothigh,newpivotlow,lastpivothighprice,lastpivotlowprice,lowdelta,highdelta,lowtime,hightime,avgph,avgpl,avgtimeph,avgtimepl,highpivotpredict,lowpivotpredict = Pine_Pivot(source,candles,leftbars,rightbars,length)
    if sequential:
        return PIVOTSINFO(highpivotpredict[-1],lowpivotpredict[-1])
    else:
        return PIVOTSINFO(highpivotpredict[-1],lowpivotpredict[-1])



@jit(error_model='numpy')
def Pine_Pivot(source,candles,l,r,length):
    highmiddlesource = np.full_like(source,0)
    lowmiddlesource = np.full_like(source,0)
    pivothigh = np.full_like(source,0)
    pivotlow = np.full_like(source,0)
    lastpivothighprice = np.full_like(source,0)
    lastpivotlowprice = np.full_like(source,0)
    lowtime = np.full_like(source,0)
    hightime = np.full_like(source,0)
    ph_x = np.full_like(source,0)
    pl_x = np.full_like(source,0)
    lowdelta = np.full_like(source,0)
    highdelta = np.full_like(source,0)
    lowdelta1 = np.full_like(source,0)
    highdelta1 = np.full_like(source,0)
    avgph = np.full_like(source,0)
    avgpl = np.full_like(source,0)
    sourcehigh = candles[:,3]
    sourcelow = candles[:,4] 
    hpcount = np.full_like(source,0)
    lpcount = np.full_like(source,0)
    hightimechange = np.full_like(source,0)
    lowtimechange = np.full_like(source,0)
    avgtimeph = np.full_like(source,0)
    avgtimepl = np.full_like(source,0)
    x12_ph = np.full_like(source,0)
    x12_pl = np.full_like(source,0)
    highpivotpredict = np.full_like(source,0)
    lowpivotpredict = np.full_like(source,0)
    for i in range(length+1,source.shape[0]):
        highmiddlesource[i] = sourcehigh[i-r]
        lowmiddlesource[i] = sourcelow[i-r]
        if (np.all(highmiddlesource[i] >= sourcehigh[i-(l+r):i-(r)]) and np.all(highmiddlesource[i] > sourcehigh[i-(r-1):i+1])):
            pivothigh[i] = 1  
            lastpivothighprice[i] = highmiddlesource[i]  
            highdelta[i] = ((lastpivothighprice[i] - lastpivothighprice[i-1])/lastpivothighprice[i-1]*100) if np.abs(lastpivothighprice[i-1]) > 0 else 0  
            highdelta1[i] = highdelta[i] 
            ph_x[i] = i 
            hpcount[i] = 1 + hpcount[i-1] 
            avgph[i] = (np.sum(highdelta1[:i+1]))/hpcount[i] 
            hightimechange[i] = i - pl_x[i-1] #if np.abs(ph_x[i-1]) > 0 else 0 
            avgtimeph[i] = (np.sum(hightimechange[:i+1]))/hpcount[i]
            x12_ph[i] = i + np.int(avgtimeph[i] - length)
        else:
            pivothigh[i] = 0 
            lastpivothighprice[i] = lastpivothighprice[i-1] 
            highdelta[i] = highdelta[i-1] 
            highdelta1[i] = 0 
            ph_x[i] = ph_x[i-1] 
            hpcount[i] = hpcount[i-1] 
            avgph[i] = avgph[i-1] 
            hightimechange[i] = 0
            avgtimeph[i] = avgtimeph[i-1] 
            x12_ph[i] = x12_ph[i-1] 
        if (np.all(lowmiddlesource[i] <= sourcelow[i-(l+r):i-(r)]) and np.all(lowmiddlesource[i] < sourcelow[i-(r-1):i+1])):    
            pivotlow[i] = 1  
            lastpivotlowprice[i] = lowmiddlesource[i] 
            lowdelta[i] = (lastpivotlowprice[i] - lastpivotlowprice[i-1])/lastpivotlowprice[i-1]*100 if np.abs(lastpivotlowprice[i-1]) > 0 else 0 
            lowdelta1[i] = lowdelta[i] 
            pl_x[i] = i 
            lpcount[i] = 1 + lpcount[i-1] 
            avgpl[i] = (np.sum(lowdelta1[:i+1]))/lpcount[i]
            lowtimechange[i] = i - pl_x[i-1] #if np.abs(pl_x[i-1]) > 0 else 0  
            avgtimepl[i] = (np.sum(lowtimechange[:i+1]))/lpcount[i] 
            x12_pl[i] = i + np.int(avgtimepl[i] - length)
        else:
            pivotlow[i] = 0 
            lastpivotlowprice[i] = lastpivotlowprice[i-1]
            lowdelta[i] = lowdelta[i-1] 
            lowdelta1[i] = 0 
            pl_x[i] = pl_x[i-1] 
            lpcount[i] = lpcount[i-1] 
            avgpl[i] = avgpl[i-1] 
            lowtimechange[i] = 0 
            avgtimepl[i] = avgtimepl[i-1]
            x12_pl[i] = x12_pl[i-1] 
        lowtime[i] = i - pl_x[i-1] 
        hightime[i] = i - ph_x[i-1] 
        highpivotpredict[i] = (i - x12_ph[i])
        lowpivotpredict[i] = (i - x12_pl[i]) 
    return pivothigh,pivotlow,lastpivothighprice,lastpivotlowprice,lowdelta,highdelta,lowtime,hightime, avgph,avgpl,avgtimeph,avgtimepl,highpivotpredict,lowpivotpredict
 
