from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

#jesse backtest  '2021-01-03' '2021-03-02'

'''
https://www.tradingview.com/script/NgLjvBWA-RedK-Compound-Ratio-Moving-Average-CoRa-Wave/#chart-view-comments

''' 

def compma(candles: np.ndarray, length: int= 20, ratio: float=2.0,smooth: bool = True,man_smooth:int=1, source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    cora_raw = fast_comp(source,length,0.01,ratio)
    s = np.maximum(np.round(np.sqrt(length)),1) if smooth else man_smooth
    cora_wave = pine_wma(cora_raw,s)
    if sequential:
        return cora_wave
    else:
        return cora_wave[-1]
    
@njit    
def fast_comp(source,length,start_wt,ratio):
    r = np.full_like(source,0)
    base = np.full_like(source,0)
    res = np.full_like(source,0)
    for i in range(source.shape[0]):
        r[i] = np.power((length/start_wt),(1/(length - 1)))-1
        base[i] = 1 + r[i] * ratio
        c_weight = 0.0
        numerator = 0.0
        denom = 0.0
        for j in range(length):
            c_weight = start_wt * np.power(base[i-j],(length - j))
            numerator = numerator + source[i-j] * c_weight
            denom = denom + c_weight
        res[i] = numerator/denom
    return res 

@njit    
def pine_wma(source,length):
    res = np.full_like(source,length)
    for i in range(source.shape[0]):
        weight = 0.0
        norm = 0.0 
        sum1 = 0.0
        for j in range(length):
            weight = (length - j)*length
            norm = norm + weight 
            sum1 = sum1 + source[i-j] * weight
        res[i] = sum1/norm 
    return res 
