from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d
from collections import namedtuple


"""
https://www.tradingview.com/script/OGToVsFd-Ehlers-Modified-Optimum-Elliptic-Filter/
"""
def moef(candles: np.ndarray, source_type: str = "hl2", sequential: bool = False ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    moef = fast_moef(source,candles)
    if sequential:
        return moef
    else:
        return moef[-1]
    
@njit   
def fast_moef(source,candles):
    moef = np.full_like(source,0)
    for i in range(source.shape[0]):
        moef[i] = 0.13785 * (2 * source[i] - source[i-1]) + 0.0007 * (2 * source[i-1] - source[i-2]) + 0.13785 * (2 * source[i-2] - source[i-3]) + 1.2103 * (moef[i-1]) - 0.4867 * moef[i-2]
    return moef 
