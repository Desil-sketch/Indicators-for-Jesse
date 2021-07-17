
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple

WTO = namedtuple('WTO', ['wt1', 'wt2'])

def wto(candles: np.ndarray, channel_period: int = 10, average_period: int = 21, source_type: str = "hlc3", sequential: bool = False) -> WTO: 
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    esa = talib.EMA(source, timeperiod=channel_period)
    d = talib.EMA(abs(source - esa), timeperiod=channel_period)
    ci = (source - esa) / (0.015 * d)
    tci = talib.EMA(ci, timeperiod=average_period)
    wt1 = tci 
    wt2 = talib.SMA(wt1, timeperiod=4) 
    
    if sequential: 
        return WTO(wt1, wt2)
    else: 
        return WTO(wt1[-1], wt2[-1])
