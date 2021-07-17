from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

'''
https://www.tradingview.com/script/AKsKg1ih-McGinley-Dynamic-Improved-John-R-McGinley-Jr/#chart-view-comments
''' 
  
def mgd(candles: np.ndarray, period: int= 14,exponent:float=4.0,k:float=0.6, source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    McGinleyDynamic = imd(source,period,k,exponent)
    if sequential:
        return McGinleyDynamic
    else:
        return McGinleyDynamic[-1]

	
def imd(source,fperiod,fk,fexponent):
    md = np.full_like(source,0)
    for i in range(source.shape[0]):
        md[i] = md[i-1] + (source[i] - md[i-1]) / np.minimum(fperiod,np.maximum(1.0, fk*fperiod*np.power(source[i]/md[i-1],fexponent)))
    return md 
