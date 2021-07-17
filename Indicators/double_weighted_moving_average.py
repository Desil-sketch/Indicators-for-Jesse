from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple

'''
https://www.tradingview.com/script/TZNHdMDL-Double-Weighted-Moving-Average/
''' 
  
def dwma(candles: np.ndarray, length: int= 14, source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    dwma = talib.WMA(talib.WMA(source,length),length)
    if sequential:
        return dwma 
    else:
        return dwma[-1] 
