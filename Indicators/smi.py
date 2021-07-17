
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple
import jesse.indicators as ta

"""
https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/
"""
SMI = namedtuple('smi', ['sqzOn', 'sqzOff', 'noSqz', 'val'])

def smi(candles: np.ndarray, bbperiod: int = 20, lengthKC: int = 20, devbb: float = 2, devkc: float = 2, matype = 0, source_type: str = "close", sequential: bool = False) -> SMI: 
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    basis = talib.MA(source, timeperiod=lengthKC, matype=matype)
    rangetr = (talib.TRANGE(candles[:, 3], candles[:, 4], candles[:, 2]))
    rangema = talib.MA(rangetr, timeperiod=lengthKC, matype=matype)
    upperbands, middlebands, lowerbands = talib.BBANDS(source, timeperiod=bbperiod, nbdevup=devbb, nbdevdn=devbb, matype=matype)
    kcup = basis[-1] + rangema[-1] * devkc  
    kclow = basis[-1] - rangema[-1] * devkc
    sqzOn = (lowerbands[-1] > kclow) and (upperbands[-1] < kcup)
    sqzOff = (lowerbands[-1] < kclow) and (upperbands[-1] > kcup)
    noSqz = (sqzOn == False) and (sqzOff == False)
    previoushigh = talib.MAX(candles[:, 3], timeperiod=lengthKC)
    previouslow = talib.MIN(candles[:, 4], timeperiod=lengthKC)
    talibSMA = talib.SMA(candles[:, 2],lengthKC)
    array1 = (np.array(previoushigh) + np.array(previouslow)) / 2.0
    array2 = (np.array(array1) + np.array(talibSMA)) / 2.0
    Part1 = (candles[:, 2]  -  array2)
    val =  talib.LINEARREG((candles[:, 2]  -  array2), lengthKC)

    if sequential: 
        return SMI(sqzOn, sqzOff, noSqz, val,)
    else:    
        return SMI(sqzOn[-1], sqzOff[-1], noSqz[-1], val[-1])
