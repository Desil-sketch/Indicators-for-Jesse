from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d


EPMA = namedtuple('EPMA',['predict','trigger'])

"""
https://www.tradingview.com/script/kexjZZwL-Ehlers-Predictive-Moving-Average-CC/#chart-view-comments
"""

def epma(candles: np.ndarray, source_type: str = "hl2", sequential: bool = False ) -> EPMA:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    predict,trigger = fast_epma(source,candles)
    if sequential:
        return EPMA(predict,trigger)
    else:
        return EPMA(predict[-1],trigger[-1])
   
@njit   
def fast_epma(source,candles):
    wma1 = np.full_like(source,0)
    wma2 = np.full_like(source,0)
    predict = np.full_like(source,0)
    trigger = np.full_like(source,0)
    for i in range(source.shape[0]):
        wma1[i] = ((7 * source[i]) + (6 * source[i-1]) + (5 * source[i-2]) + (4 * source[i-3]) + (3 * source[i-4]) + (2 * source[i-5]) + source[i-6]) / 28
        wma2[i] = ((7 * wma1[i]) + (6 * wma1[i-1]) + (5 * wma1[i-2]) + (4 * wma1[i-3]) + (3 * wma1[i-4]) + (2 * wma1[i-5]) + wma1[i-6]) / 28 
        predict[i] = (2 * wma1[i]) - wma2[i] 
        trigger[i] = ((4 * predict[i]) + (3 * predict[i-1]) + (2 * predict[i-2]) + (predict[i-3]))/10
    return predict,trigger
