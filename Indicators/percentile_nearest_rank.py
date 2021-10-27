from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
import tulipy as ti
import math 
from scipy.stats import rankdata
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d
from numpy.lib.stride_tricks import as_strided
"""
https://www.tradingview.com/script/YDpNmgQV-Percentile-Nearest-Rank-Using-Arrays-LUX/#chart-view-comments
"""
#jesse backtest  '2021-01-03' '2021-03-02'

def PNR(candles: np.ndarray,length:int = 15, p:int=50,source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential) 
    source = get_candle_source(candles, source_type=source_type) 
    rank = midrank = np.percentile(rolling_window(source,length),p,1)
    if sequential:
        return rank
    else:
        return rank[-1]
        
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return as_strided(a, shape=shape, strides=strides)  
