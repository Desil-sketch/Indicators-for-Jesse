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

def jmarsx(candles: np.ndarray, length: int= 14, source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    rsx = fast_rsx(source,candles,length)
    if sequential:
        return rsx
    else:
        return rsx[-1]
@njit       
def fast_rsx(source,candles,length):  
    f8 = np.full_like(source,0)
    f10 = np.full_like(source,0)
    v8 = 0 
    f18 = 0 
    f20 = 0 
    f28 = 0
    f30 = 0 
    vC = 0 
    f38 = 0 
    f40 = 0
    f48 = 0
    f50 = 0
    v14 = 0
    f58 = 0
    f60 = 0
    v18 = 0
    f68 = 0
    f70 = 0
    v1C = 0
    f78 = 0
    f80 = 0
    v20 = 0
    v4 = 0
    rsx = np.full_like(source, 0)
    for i in range(source.shape[0]):
        f8[i] = 100*source[i] 
        f10[i] = f8[i-1]
        v8 = f8[i] - f10[i] 
        f18 = 3 / (length + 2)
        f20 = 1 - f18
        f28 = f20 * f28 + f18 * v8
        f30 = f18 * f28 + f20 * f30
        vC = f28 * 1.5 - f30 * 0.5 
        f38 = f20 * f38 + f18 * vC
        f40 = f18 * f38 + f20 * f40 
        v10 = f38 * 1.5 - f40 * 0.5
        f48 = f20 * f48 + f18 * v10 
        f50 = f18 * f48 + f20 * f50 
        v14 = f48 * 1.5 - f50 * 0.5 
        f58 = f20 * f58 + f18 * np.abs(v8) 
        f60 = f18 * f58 + f20 * f60 
        v18 = f58 * 1.5 - f60 * 0.5
        f68 = f20 * f68 + f18 * v18 
        f70 = f18 * f68 + f20 * f70 
        v1C = f68 * 1.5 - f70 * 0.5
        f78 = f20 * f78 + f18 * v1C
        f80 = f18 * f78 + f20 * f80 
        v20 = f78 * 1.5 - f80 * 0.5 
        v4 = (v14 / v20 + 1)* 50 if v20 > 0 else 50 
        if v4 > 100:
            rsx[i] = 100 
        elif v4 < 0: 
            rsx[i] = 0 
        else:
            rsx[i] = v4 
    return rsx 
        
