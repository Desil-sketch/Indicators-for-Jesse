from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from numpy.lib.stride_tricks import sliding_window_view
from collections import namedtuple

CON = namedtuple('con', ['highcond', 'lowcond', 'conscnt' ])

def consolidation(candles: np.ndarray, period: int= 10, conslen:int = 5,  source_type: str = "close", sequential: bool = False ) -> CON:  
    candles = slice_candles(candles, sequential)
    candles_high = candles[:, 3]
    candles_low = candles[:, 4]
    candles_close = candles[:,2]
    source = get_candle_source(candles, source_type=source_type)
    H1 = talib.MAX(candles_high, conslen)
    L1 = talib.MIN(candles_low, conslen)
    highcond, lowcond, conscnt = consolidation_fast(candles, period,conslen, candles_high, candles_low, source, H1,L1)
    if sequential:
        return CON(highcond,lowcond,conscnt)
    else:
        return CON(highcond[-1], lowcond[-1], conscnt[-1])
#jesse backtest  '2021-01-03' '2021-03-02'

"""
https://www.tradingview.com/script/xKSqR6P4-Consolidation-Zones-Live/
conscnt or conscntcounter = 1 when conscnt < conslen or not inside consolidation zone
"""

@njit
def consolidation_fast(candles, period, conslen, candles_high, candles_low, source,H1,L1):
    res = np.full_like(source, 0)
    hb1 = np.full_like(source,0)
    lb1 = np.full_like(source,0)
    dir1 = np.full_like(source,0)
    zz = np.full_like(source,np.NaN)
    pp1 = np.full_like(source,np.NaN)
    conscnt = np.full_like(source,0)
    condhigh = np.full_like(source,np.NaN)
    condlow = np.full_like(source,np.NaN)
    breakoutdown = np.full_like(source,0)
    breakoutup = np.full_like(source,0)
    zeros1 = np.full_like(source,0)
    conscntcounter = np.full_like(source,0)
    for i in range(source.shape[0]): 
        if np.all(candles_high[i] > candles_high[i-period:i]):
            hb1[i] = candles[:,3][i]
        else:
            hb1[i] = np.NaN
        if np.all(candles_low[i] < candles_low[i-period:i]):
            lb1[i] = candles[:,4][i] 
        else:
            lb1[i] = np.NaN   
        if not np.isnan(hb1[i]) and np.isnan(lb1[i]):
            dir1[i] = 1 
        elif not np.isnan(lb1[i]) and np.isnan(hb1[i]):
            dir1[i] = -1
        else:
            dir1[i] = dir1[i-1]
        if not np.isnan(hb1[i]) and not np.isnan(lb1[i]):
            if dir1[i] == 1 :
                zz[i] = hb1[i] 
            else:
                zz[i] = lb1[i] 
        else:
            if not np.isnan(hb1[i]):
                zz[i] = hb1[i] 
            elif not np.isnan(lb1[i]):
                zz[i] = lb1[i] 
            else:
                zz[i] = np.NaN

        pp = np.NaN
        for j in range(0,100):
            if np.isnan(source[i]) or dir1[i] != dir1[i-j]:
                break 
            if not np.isnan(zz[i-j]):
                if np.isnan(pp):
                    pp = zz[i-j]
                else:
                    if dir1[i-j] == 1 and zz[i-j] > pp: 
                        pp = zz[i-j] 
                    elif dir1[i-j] == -1 and zz[i-j] < pp:
                        pp = zz[i-j]
        pp1[i] = pp
        if pp1[i] != pp1[i-1]:
            if conscnt[i] > conslen:
                if pp1[i] > condhigh[i]:
                    breakoutup[i] = 1
                else:
                    breakoutup[i] == 0
                if pp1[i] < condlow[i]:
                    breakoutdown[i] = 1
                else:
                    breakoutdown[i] = 0
            if conscnt[i] > 0 and pp1[i] <= condhigh[i] and pp1[i] >= condlow[i]:
                conscnt[i] = conscnt[i-1] + 1
            else:
                conscnt[i] = 0 
        else:
            conscnt[i] = conscnt[i-1] + 1 
        
        if conscnt[i] >= conslen:
            if conscnt[i] == conslen:
                condhigh[i] = H1[i]
                condlow[i] = L1[i] 
            else:
                condhigh[i] = np.maximum(condhigh[i-1],candles[:,3][i])
                condlow[i] = np.minimum(condlow[i-1],candles[:,4][i])
        else:
            condhigh[i] = condhigh[i-1]
            condlow[i] = condlow[i-1]
            conscntcounter[i] = 1 

    return condhigh,condlow,conscntcounter

