from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import jit, float64, int32, void, njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from numpy.lib.stride_tricks import sliding_window_view

"""
https://www.tradingview.com/script/QTvmm8XM-Mesa-Advanced-Triggers/#chart-view-comment-form
"""
def mesatrig(candles: np.ndarray, pivot_zone_lower: float = -0.5, pivot_zone_upper:float=0.5,  source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:   
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    candles_close = (candles[:,2])
    candles_low = (candles[:,3])
    candles_high = (candles[:,4])
    phase = fast_mesa(candles,source, pivot_zone_lower, pivot_zone_upper, candles_close, candles_high, candles_low)
    finalphase = np.array([phase])
    if sequential:
        return phase
    else:
        return finalphase[-1]
        
@njit
def fast_mesa(candles,source, pivot_zone_lower, pivot_zone_upper, candles_close, candles_high, candles_low):
    DomCycle = 15
    Phase = 0.0
    Phase2 = np.full_like(source,0)
    for i in range(0,source.shape[0]):
        RealPart = 0.0
        ImagPart = 0.0
        Weight = 0.0
        for j in range(0,DomCycle):
            Weight = ((candles_close[i-j] + candles_close[i-j] + candles_low[i-j] + candles_high[i-j]) * 10000)
            RealPart = RealPart + np.cos(90 * j / DomCycle) * Weight * 2
            ImagPart = ((ImagPart + np.sin(90 * j / DomCycle) * Weight) + (ImagPart + np.sin(180 * j/ DomCycle) * Weight))/2
        Phase = ((np.arctan(ImagPart / RealPart)) - 0.685) * 100 
        Phase2[i] = Phase
    return Phase
