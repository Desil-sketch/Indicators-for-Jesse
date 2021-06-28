from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d

"""
https://www.tradingview.com/script/WDGZQHGf-Double-Smoothed-Momenta/
"""
DOUBLEMOM = namedtuple('doublemom', ['mom','momEma'])

def doublemom(candles: np.ndarray, aperiod: int= 2, yperiod: int= 5, zperiod: int=25, source_type: str = "close", sequential: bool = False ) -> DOUBLEMOM:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    hc = max_filter1d_same(source,aperiod)
    lc = min_filter1d_same(source,aperiod)
    top = talib.EMA(talib.EMA(source - lc, yperiod), zperiod)
    bot = talib.EMA(talib.EMA(hc - lc, yperiod), zperiod)
    mom = np.full_like(source,0)
    if bot[-1] != 0:
        mom = 100 * top/bot
    else:
        mom = 0
    momEma = talib.EMA(mom, zperiod)
    if sequential:
        return DOUBLEMOM(mom,momEma)
    else:
        return DOUBLEMOM(mom[-1], momEma[-1])

def max_filter1d_same(a, W, fillna=np.nan):
    out_dtype = np.full(0,fillna).dtype
    hW = (W-1)//2 # Half window size
    out = maximum_filter1d(a,size=W, origin=hW)
    if out.dtype is out_dtype:
        out[:W-1] = fillna
    else:
        out = np.concatenate((np.full(W-1,fillna), out[W-1:]))
    return out    

def min_filter1d_same(a, W, fillna=np.nan):
    out_dtype = np.full(0,fillna).dtype
    hW = (W-1)//2 # Half window size
    out = minimum_filter1d(a,size=W, origin=hW)
    if out.dtype is out_dtype:
        out[:W-1] = fillna
    else:
        out = np.concatenate((np.full(W-1,fillna), out[W-1:]))
    return out    
