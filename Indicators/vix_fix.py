from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple
from custom_indicators.helpful_functions import std2
#jesse backtest  '2021-01-03' '2021-03-02'

VIXFIX = namedtuple('VIXFIX',['vixfix','color'])

'''
https://www.tradingview.com/script/zwFb8K6B-CM-Williams-Vix-Fix-V3-Ultimate-Filtered-Alerts-With-Threshold/#chart-view-comments
uses  "helpful_functions" 
''' 
  
def vixfix(candles: np.ndarray, source_type: str = "close",ph:float=0.85,lb:int=50,pd: int=22,bbl:int=20,mult:float=2.0, sequential: bool = False ) -> VIXFIX:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    ones = np.full_like(source,1)
    zeros = np.full_like(source,0)
    wvf = ((talib.MAX(source,pd)-candles[:,4])/(talib.MAX(source,pd)))*100
    sDev = mult * std2(source,wvf,bbl)
    midLine = talib.SMA(wvf,bbl)
    lowerBand = midLine - sDev
    upperBand = midLine + sDev
    rangeHigh = (talib.MAX(wvf,lb))* ph 
    if wvf[-1] >= upperBand[-1] or wvf[-1] >= rangeHigh[-1]:
        col = ones 
    else:
        col = zeros 
    if sequential:
        return VIXFIX(wvf,col)
    else:
        return VIXFIX(wvf[-1],col[-1])
	
