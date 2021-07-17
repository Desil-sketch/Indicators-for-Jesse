
import numpy as np
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple
import tulipy as ti


def volatility(candles: np.ndarray, period: int = 14, sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = ti.volatility(np.ascontiguousarray(candles[:, 2]), period=period)
    return np.concatenate((np.full((candles.shape[0] - res.shape[0]), np.nan), res), axis=0) if sequential else res[-1]
