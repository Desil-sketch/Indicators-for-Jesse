from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
import tulipy as ti
import math 

"""
tradingview.com/script/u64LZ5N3-MESA-Stochastic-Multi-Length/#chart-view-comments
need to add buy and sell signals 
"""
#jesse backtest  '2021-01-03' '2021-03-02'

MESAMULTI = namedtuple('MESAMULTI',['mesa1','mesa2','mesa3','mesa4','mesa5','trigger1','trigger2','trigger3','trigger4','trigger5','buysignal','sellsignal'])

def mesamulti(candles: np.ndarray, length1: int= 50, length2: int=21, length3: int=14, length4: int = 9, trig:int=2, source_type: str = "hl2", sequential: bool = False) -> MESAMULTI:
    candles = slice_candles(candles, sequential) 
    source = get_candle_source(candles, source_type=source_type) 
    mesa1 = fast_mesa(source,candles,length1)
    mesa2 = fast_mesa(source,candles,length2)
    mesa3 = fast_mesa(source,candles,length3)
    mesa4 = fast_mesa(source,candles,length4)
    mesa5 = (mesa1 + mesa2 + mesa3 + mesa4)/4
    trigger1 = talib.SMA(mesa1,trig)
    trigger2 = talib.SMA(mesa2,trig)
    trigger3 = talib.SMA(mesa3, trig)
    trigger4 = talib.SMA(mesa4, trig)
    trigger5 = talib.SMA(mesa5, trig)
    zeros = np.full_like(source,0)
    ones = np.full_like(source,1)
    mesa1count = ones if mesa1[-1] > mesa1[-2] else zeros
    mesa2count = ones if mesa2[-1] > mesa2[-2] else zeros
    mesa3count = ones if mesa3[-1] > mesa3[-2] else zeros
    mesa4count = ones if mesa4[-1] > mesa4[-2] else zeros
    mesa5count = ones if mesa5[-1] > mesa5[-2] else zeros
    if mesa1count[-1] + mesa2count[-1] + mesa3count[-1] + mesa4count[-1] + mesa5count[-1] >= 2:
        buysignal = ones 
        sellsignal = zeros  
    else:
        buysignal = zeros  
        sellsignal = ones 
        
    if sequential:
        return MESAMULTI(mesa1,mesa2,mesa3,mesa4,mesa5,trigger1,trigger2,trigger3,trigger4,trigger5,buysignal,sellsignal)
    else:
        return MESAMULTI(mesa1[-1], mesa2[-1], mesa3[-1],mesa4[-1],mesa5[-1],trigger1[-1],trigger2[-1],trigger3[-1],trigger4[-1],trigger5[-1],buysignal[-1],sellsignal[-1])

@njit
def fast_mesa(source,candles,length):
    alpha1 = np.full_like(source,0)
    HP = np.full_like(source,0)
    a1 = 0.0
    b1 = 0.0
    c2 = 0.0
    c3 = 0.0
    c1 = 0.0
    Filt = np.full_like(source,0)
    Stoc = np.full_like(source,0)
    MESAStochastic = np.full_like(source,0)
    for i in range(length,source.shape[0]):
        alpha1[i] = (np.cos(0.707 * 2*np.pi / 48) + np.sin(0.707 * 2*np.pi/48) - 1) / np.cos(0.707 * 2*np.pi / 48)
        HP[i] = (1 - alpha1[i] / 2) * (1 - alpha1[i] / 2) * (source[i] - 2 * source[i-1] + source[i-2]) + 2 * (1 - alpha1[i]) * HP[i-1] - (1 - alpha1[i]) * (1 - alpha1[i]) * HP[i-2] 
        a1 = np.exp(-1.414 * 3.14159 / 10)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / 10)
        c2 = b1 
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        Filt[i] = c1 * (HP[i] + HP[i-1])/2 + c2 * Filt[i-1] + c3 * Filt[i-2]
        HighestC = 0.0
        LowestC = 0.0
        for j in range(length):
            if j == 0:
                HighestC = Filt[i]
                LowestC = Filt[i] 
            if Filt[i-j] > HighestC:
                HighestC = Filt[i-j]
            if Filt[i-j] < LowestC:
                LowestC = Filt[i-j] 
        Stoc[i] = (Filt[i] - LowestC) / (HighestC - LowestC) 
        MESAStochastic[i] = c1 * (Stoc[i] + Stoc[i-1]) / 2 + c2 * MESAStochastic[i-1] + c3 * MESAStochastic[i-2] 
    return MESAStochastic
        
    
