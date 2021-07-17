from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple

VVMA = namedtuple('VVMA',['vv', 'vvangle', 'volatility', 'volatilityangle'])

'''
https://www.tradingview.com/script/pFYbl3e5-V-V-weighted-ma-JD/
with additional angles of moving average 
https://www.tradingview.com/script/LIEBG3UR-ma-angles-JD/#chart-view-comments
''' 
  
def vv_ma_angle(candles: np.ndarray, period: int= 14, source_type: str = "close", sequential: bool = False ) -> VVMA:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    tr = talib.TRANGE(candles[:,3],candles[:,4],candles[:,2])
    atr = talib.ATR(candles[:,3],candles[:,4],candles[:,2],timeperiod=14)
    vv, vvangle = vv_weighted_ma(source,period,candles,tr,atr)
    volatility, volatilityangle = volatility_weighted_ma(source,period,candles,tr,atr)
    if sequential:
        return VVMA(vv,vvangle,volatility,volatilityangle)
    else:
        return VVMA(vv[-1], vvangle[-1], volatility[-1], volatilityangle[-1])

@njit        
def vv_weighted_ma(source,period,candles,tr,atr):
    volwma = np.full_like(source,0)
    rad2degree = 180 / np.pi
    vvangle = np.full_like(source,0)
    for i in range(source.shape[0]):
        volatility_sum = 0.0
        tr_sum = 0.0
        for j in range(0,period+1):
            volatility_sum = source[i-j] * candles[:,5][i-j] * np.abs(tr[i-j]) * (period + 1 - j) + volatility_sum
            tr_sum = candles[:,5][i-j] * np.abs(tr[i-j]) * (period + 1 -j) + tr_sum 
        volwma[i] = volatility_sum / tr_sum 
        vvangle[i] = rad2degree * np.arctan((volwma[i] - volwma[i-1])/atr[i])
    return volwma, vvangle 

@njit
def volatility_weighted_ma(source,period,candles,tr,atr):
    volwma = np.full_like(source,0)
    rad2degree = 180 / np.pi
    volatilityangle = np.full_like(source,0)
    for i in range(source.shape[0]):
        vol_sum = 0.0
        tr_sum = 0.0 
        for j in range(0,period+1):
            vol_sum = source[i-j] * np.abs(tr[i-j]) * (period + 1 - j) + vol_sum
            tr_sum = np.abs(tr[i-j]) * (period + 1 - j) + tr_sum 
        volwma[i] = vol_sum/tr_sum 
        volatilityangle[i] = rad2degree * np.arctan((volwma[i] - volwma[i-1])/atr[i])
    return volwma, volatilityangle 
   
