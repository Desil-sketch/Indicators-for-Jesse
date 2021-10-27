from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
import tulipy as ti
import math 

"""
https://www.tradingview.com/script/sxZRzQzQ-Divergence-Indicator-any-oscillator/#chart-view-comment-form
Possibly Accurate, needs more testing
"""
#jesse backtest  '2021-01-03' '2021-03-02'

DIVERGENCES = namedtuple('Divergences',['bearCond', 'bullCond', 'hiddenBullCond','hiddenBearCond'])

def divergence(candles: np.ndarray, lbR:int=2, lbL:int=2, rangeUpper:int=200, rangeLower:int=0,source_type: str = "close", sequential: bool = False) -> DIVERGENCES:
    candles = slice_candles(candles, sequential) 
    source1 = get_candle_source(candles, source_type=source_type) 
    bearCond, bullCond, hiddenBullCond, hiddenBearCond  = fast_div(source,source,candles,lbR,lbL,rangeUpper,rangeLower)
    if sequential:
        return DIVERGENCES(bearCond,bullCond,hiddenBearCond,hiddenBullCond)
    else:
        return DIVERGENCES(bearCond[-1],bullCond[-1],hiddenBearCond[-1],hiddenBullCond[-1])
        
    
def fast_div(source1,source,candles,r,l,rangeUpper,rangeLower): 
    highmiddlesource = np.full_like(source1,0)
    lowmiddlesource = np.full_like(source1,0)
    pivothigh = np.full_like(source1,0)
    pivotlow = np.full_like(source1,0)
    lastpivothighprice = np.full_like(source1,0)
    lastpivotlowprice = np.full_like(source1,0)
    priceslowest = np.full_like(source1,np.nan)
    priceshighest = np.full_like(source1,np.nan)
    priceshigh = np.full_like(source1,np.nan)
    priceslow = np.full_like(source1,np.nan)
    highindices = np.full_like(source1,np.nan)
    lowindices = np.full_like(source1,np.nan)
    ivar = np.full_like(source1,0)
    for i in range(source1.shape[0]):
        highmiddlesource[i] = source[i-r]
        lowmiddlesource[i] = source[i-l]
        if (np.all(highmiddlesource[i] >= source[i-(l+r):i-(r)]) and np.all(highmiddlesource[i] > source[i-(r-1):i+1])):
            pivothigh[i] = 1  
            lastpivothighprice[i] = highmiddlesource[i]  
        else:
            pivothigh[i] = 0 
            lastpivothighprice[i] = lastpivothighprice[i-1]
        if (np.all(lowmiddlesource[i] <= source[i-(l+r):i-(r)]) and np.all(lowmiddlesource[i] < source[i-(r-1):i+1])):    
            pivotlow[i] = 1  
            lastpivotlowprice[i] = lowmiddlesource[i] 
        else:
            pivotlow[i] = 0 
            lastpivotlowprice[i] = lastpivotlowprice[i-1]
        if pivothigh[i] == 1:
            priceshigh[i] = source[i-r] 
            priceshighest[i] = candles[:,3][i-r]
            highindices[i] = (i-r) 
        if pivotlow[i] == 1:
            priceslow[i] = source[i-l] 
            priceslowest[i] = candles[:,4][i-l]
            lowindices[i] = (i-l)
        ivar[i] = i
    ivar1 = int(ivar[-1])
    priceshigh =  priceshigh[~np.isnan(priceshigh)]
    priceshigh = np.concatenate((np.full((source.shape[0] - priceshigh.shape[0]), np.nan), priceshigh)) 
    priceshighest =  priceshighest[~np.isnan(priceshighest)]
    priceshighest = np.concatenate((np.full((source.shape[0] - priceshighest.shape[0]), np.nan), priceshighest)) 
    priceslow =  priceslow[~np.isnan(priceslow)]
    priceslow = np.concatenate((np.full((source.shape[0] - priceslow.shape[0]), np.nan), priceslow)) 
    priceslowest =  priceslowest[~np.isnan(priceslowest)]
    priceslowest = np.concatenate((np.full((source.shape[0] - priceslowest.shape[0]), np.nan), priceslowest)) 
    highindices =  highindices[~np.isnan(highindices)]
    highindices = np.concatenate((np.full((source.shape[0] - highindices.shape[0]), np.nan), highindices)) 
    lowindices =  lowindices[~np.isnan(lowindices)]
    lowindices = np.concatenate((np.full((source.shape[0] - lowindices.shape[0]), np.nan), lowindices)) 
    oscHL = 1 if source[-(r+1)] > priceslow[-2] and (np.abs(lowindices[-2]-ivar1) >= rangeLower and np.abs(lowindices[-2]-ivar1) <= rangeUpper) else 0 
    priceLL = 1 if candles[:,4][-(r+1)] < priceslowest[-2] else 0 
    bullCond = 1 if priceLL == 1 and oscHL == 1 and pivotlow[-1] == 1 else 0 
    oscLL = 1 if (source[-(r+1)] < priceslow[-2] and np.abs(lowindices[-2]-ivar1) >= rangeLower and np.abs(lowindices[-2]-ivar1) <= rangeUpper) else 0 
    priceHL = 1 if candles[:,4][-(r+1)] > priceslowest[-2] else 0 
    hiddenBullCond = 1 if priceHL == 1 and oscLL == 1 and pivotlow[-1] == 1  else 0 
    oscLH = 1 if source[-(r+1)] < priceshigh[-2] and (np.abs(highindices[-2]-ivar1) >= rangeLower and np.abs(highindices[-2]-ivar1) <= rangeUpper) else 0 
    priceHH = 1 if candles[:,3][-(r+1)] > priceshighest[-2] else 0 
    bearCond = 1 if priceHH == 1 and oscLH == 1 and pivothigh[-1] == 1 else 0 
    oscHH = 1 if source[-(r+1)] > priceshigh[-2] and (np.abs(highindices[-2]-ivar1) >= rangeLower and np.abs(highindices[-2]-ivar1) <= rangeUpper) else 0 
    priceLH = 1 if candles[:,3][-(r+1)] < priceshighest[-2] else 0 
    hiddenBearCond = 1 if priceLH == 1 and oscHH == 1 and pivothigh[-1] == 1 else 0 
    return bearCond, bullCond, hiddenBullCond, hiddenBearCond 
