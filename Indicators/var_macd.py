from jesse.helpers import get_candle_source, slice_candles, np_shift,same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple

'''
https://www.tradingview.com/script/PhtqdGBM-MACD-ReLoaded/
''' 
Vmacd = namedtuple('vmacd',['source2','MATR','hist'])

def varmacd(candles: np.ndarray,matype:str='VAR', period1: int= 12, period2:int=26,trigger:int=9, source_type: str = "close", sequential: bool = False ) -> Vmacd:    
    candles = slice_candles(candles, sequential) if sequential else candles[-480:]
    source = get_candle_source(candles, source_type=source_type) 
    if matype == 'HMA':
        MA1 = talib.WMA(2 * talib.WMA(source,period1/2) - talib.WMA(source,period1),np.round(np.sqrt(period1)))
        MA2 = talib.WMA(2 * talib.WMA(source,period2/2) - talib.WMA(source,period2),np.round(np.sqrt(period2)))
        source2 = MA1 - MA2 
        MATR = talib.WMA(2 * talib.WMA(source2,trigger/2) - talib.WMA(source2,trigger),np.round(np.sqrt(trigger)))
        hist = source2 - MATR 
    elif matype == 'WWMA':
        MA1 = wwma(source,source,period1)
        MA2 = wwma(source,source,period2)
        source2 = MA1 - MA2 
        MATR = wwma(source,source2,trigger)
        hist = source2 - MATR 
    elif matype == 'EMA':
        MA1,MA2,MATR,source2,hist = pine_ema(source,period1,period2,trigger)
    elif matype == 'VAR':
        MA1 = fast_vidya(source,source,period1,period2)
        MA2 = fast_vidya(source,source,period2,period2)
        source2 = MA1 - MA2 
        MATR = fast_vidya(source,source2, trigger,period2) 
        hist = source2 - MATR 
    else:
        MA1 = np.nan 
        MA2 = np.nan 
        source2 = np.nan
        MATR = np.nan 
        hist = np.nan 
        
    if sequential:
        return Vmacd(source2,MATR,hist) 
    else:
        return Vmacd(source2[-1],MATR[-1],hist[-1])
    
    
@njit
def pine_ema(source1, length1, length2,trigger):
    alpha1 = 2 / (length1 + 1)
    alpha2 = 2 / (length2 + 1)
    triggeralpha = 2 / (trigger + 1)
    sum1 = np.full_like(source1,0)
    sum2 = np.full_like(source1,0)
    sum3 = np.full_like(source1,0)
    hist = np.full_like(source1,0)
    source2 = np.full_like(source1,0)
    for i in range(source1.shape[0]):
        sum1[i-1] = 0 if np.isnan(sum1[i-1]) else sum1[i-1]
        sum1[i] = alpha1 * source1[i] + (1 - alpha1) * sum1[i-1]
        sum2[i-1] = 0 if np.isnan(sum2[i-1]) else sum2[i-1] 
        sum2[i] = alpha2 * source1[i] + (1 - alpha2) * sum2[i-1] 
        source2[i] = sum1[i] - sum2[i] 
        sum3[i-1] = 0 if np.isnan(sum3[i-1]) else sum3[i-1] 
        sum3[i] = triggeralpha * source2[i] + (1 - triggeralpha) * sum3[i-1] 
        hist[i] = source2[i] - sum3[i] 
    return sum1, sum2, sum3,source2, hist     

@njit    
def wwma(source1,source,period):
    WWMA = np.full_like(source1,0)
    wwalpha = 1 /period
    for i in range(source1.shape[0]):
        WWMA[i] = wwalpha * source[i] + (1-wwalpha) * WWMA[i-1]
    return WWMA 
    
    
@jit(error_model='numpy')
def fast_vidya(source1,source, period,maxperiod):
    mom = np.full_like(source1,0)
    vidya = np.full_like(source1,0)
    CMOalpha = 2/(period+1)
    predownSum = np.full_like(source1,0)
    downSum = np.full_like(source1,0)
    upSum = np.full_like(source1,0)
    preupSum = np.full_like(source1,0)
    out = np.full_like(source1,0)
    for i in range(maxperiod+2,source1.shape[0]):
        mom[i] = (source[i] - source[i-1])
        preupSum[i] = np.maximum(mom[i],0)
        upSum[i] = np.sum(preupSum[i-(period-1):i+1])
        predownSum[i] = -(np.minimum(mom[i],0))
        downSum[i] = np.sum(predownSum[i-(period-1):i+1])
        out[i] = np.abs((upSum[i] - downSum[i]) / (upSum[i] + downSum[i]))
        vidya[i] = source[i] * CMOalpha * out[i] + vidya[i-1] * (1 - CMOalpha * out[i])
    return vidya
    
    
