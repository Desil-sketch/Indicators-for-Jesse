from typing import Union
import numpy as np
import talib
from numba import njit,jit
from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
from numpy.lib.stride_tricks import sliding_window_view


"""
https://www.tradingview.com/script/hdrf0fXV-Variable-Index-Dynamic-Average-VIDYA/
"""

def vidyatv(candles: np.ndarray, period: int = 14, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    
    candles = slice_candles(candles, sequential) if sequential else candles[-480:]
    source = get_candle_source(candles, source_type=source_type)
    vidya = fast_vidya(source,period)
    if sequential: 
        return vidya
    else:    
        return vidya[-3:]


@jit(error_model='numpy')
def fast_vidya(source, period):
    mom = np.full_like(source,0)
    vidya = np.full_like(source,0)
    CMOalpha = 2/(period+1)
    predownSum = np.full_like(source,0)
    downSum = np.full_like(source,0)
    upSum = np.full_like(source,0)
    preupSum = np.full_like(source,0)
    out = np.full_like(source,0)
    for i in range(period+2,source.shape[0]):
        mom[i] = (source[i] - source[i-1])
        preupSum[i] = np.maximum(mom[i],0)
        upSum[i] = np.sum(preupSum[i-(period-1):i+1])
        predownSum[i] = -(np.minimum(mom[i],0))
        downSum[i] = np.sum(predownSum[i-(period-1):i+1])
        out[i] = np.abs((upSum[i] - downSum[i]) / (upSum[i] + downSum[i]))
        vidya[i] = source[i] * CMOalpha * out[i] + vidya[i-1] * (1 - CMOalpha * out[i])
    return vidya
