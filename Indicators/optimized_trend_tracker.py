from collections import namedtuple
import numpy as np
from numba import njit
import talib 
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
import jesse.indicators as ta
from typing import Union

OTT = namedtuple('ott', ['var', 'ott'])

def ott(candles: np.ndarray, period: int = 14, percent: float = 1.4, source_type: str = "close", sequential: bool = False) -> OTT:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    var, ott = VARMA(source,period, percent)
    if sequential: 
        return OTT(var,ott)
    else:    
        return OTT(var[-1], ott[-1])

#jesse backtest  '2021-01-03' '2021-03-02'

"""
https://www.tradingview.com/script/zVhoDQME/
Only VAR function used 
Same as TV except the two bar delay on OTT plot was removed
"""

@njit
def VARMA(source, length, percent):
    vud1 =  np.full_like(source, 0)
    vdd1 =  np.full_like(source, 0)
    vUD =  np.full_like(source, 0)
    vDD =  np.full_like(source, 0)
    VAR = np.full_like(source, 0)
    vCMO = np.full_like(source, 0)
    dir1 = np.full_like(source, 1)
    longstop = np.full_like(source, 0)
    longstopPrev = np.full_like(source,0)
    shortstop = np.full_like(source, 0)
    shortstopPrev = np.full_like(source, 0)
    fark = np.full_like(source,0)
    MT = np.full_like(source,0)
    OTT = np.full_like(source,0)
    valpha = 2 / (length+1)
    for i in range(source.shape[0]):    
        if (source[i] > source[i-1]):
            vud1[i] =  source[i] - source[i-1]
        else:
            vud1[i] = 0
        if (source[i] < source[i-1]):
            vdd1[i] = source[i-1] - source[i]
        else:
            vdd1[i] = 0 
        vUD[i] = (vud1[i-8] + vud1[i-7] + vud1[i-6] + vud1[i-5] + vud1[i-4] + vud1[i-3] + vud1[i-2] + vud1[i-1] + vud1[i])
        vDD[i] = (vdd1[i-8] + vdd1[i-7] + vdd1[i-6] + vdd1[i-5] + vdd1[i-4] + vdd1[i-3] + vdd1[i-2] + vdd1[i-1] + vdd1[i])
        vCMO[i] = (vUD[i] - vDD[i])/(vUD[i] + vDD[i])
        VAR[i] = ((valpha*np.abs(vCMO[i])*source[i]) + (1-valpha*np.abs(vCMO[i]))*(VAR[i-1]))
        fark[i] = VAR[i]*percent*0.01
        longstop[i] = VAR[i] - fark[i]
        longstopPrev[i] = longstop[i-1]
        if VAR[i] > longstopPrev[i]:
            longstop[i] = np.maximum(longstop[i], longstopPrev[i])
        else:
            longstop[i] = longstop[i]
        shortstop[i] = VAR[i] + fark[i] 
        shortstopPrev[i] = shortstop[i-1] 
        if VAR[i] < shortstopPrev[i]:
            shortstop[i] = np.minimum(shortstop[i], shortstopPrev[i])
        else:
            shortstop[i] = shortstop[i] 
        if dir1[i-1] == -1 and VAR[i] > shortstopPrev[i]:
            dir1[i] = 1 
        elif dir1[i-1] == 1 and VAR[i] < longstopPrev[i]:
            dir1[i] = -1 
        else:
            dir1[i] = dir1[i-1] 
        if dir1[i] == 1:
            MT[i] = longstop[i] 
        else:
            MT[i] = shortstop[i] 
        if VAR[i] > MT[i]:
            OTT[i] = MT[i]*(200 + percent)/200 
        else:
            OTT[i] = MT[i]*(200 - percent)/200    

    return VAR, OTT
