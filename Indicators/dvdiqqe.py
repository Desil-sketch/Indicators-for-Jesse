from jesse.helpers import get_candle_source, slice_candles 
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length
from collections import namedtuple

DVDIQQE = namedtuple("dvdiqqe", ["dvdi", "st1", "ft1","center",'pdiv','ndiv']) 

def dvdiqqe(candles: np.ndarray, per: int = 13, smper: int = 6, dvdiper:int=50, dvdismper:int=2, vol_type: str = "Default", fmult: float = 2.618, smult: float = 4.236, center_type: str = "Dynamic", source_type: str = "close", sequential: bool = False) -> DVDIQQE:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type) 
    pdiv,ndiv = DVDI2(candles, source, dvdiper, dvdismper, vol_type)
    dvdi = DVDI1(candles, source, per, smper, vol_type)
    ft1 = t1(source, dvdi, per, fmult)
    st1 = t1(source, dvdi, per, smult)
    n = len(source)
    if center_type == "Dynamic" and sequential == True:
        center = np.nancumsum(dvdi)/(n)
    else: 
        center = np.full_like(source,0)
    if sequential:
        return DVDIQQE(dvdi, st1, ft1,center,pdiv,ndiv)
    else:
        return DVDIQQE(dvdi[-1], st1[-1], ft1[-1], center[-1],pdiv[-1],ndiv[-1])

"""
DVDI2 is more accurate with a higher dvdiper. period of 50 is very accurate
cmean only works with sequential and differs from tradingview 
""" 

def DVDI1(candles, source, t1, t2, v_type): 
    PVI = np.zeros_like(source)
    NVI = np.full_like(source, 0)
    vol = np.full_like(source, 0)
    tickvol = np.full_like(source, 0)
    tickrng = np.full_like(source, 0)
    rng = np.full_like(source, 0)
    for i in range(source.shape[0]):
        tick = 0.01
        rng = candles[:,2] - candles[:,1] 
        if (abs(rng[-1]) < tick):
            tickrng = tick
        else:
            tickrng = rng
        tickvol[i] = np.abs(tickrng[i])/tick
        if v_type == "Default":
            vol = candles[:,5]
        else:
            vol = tickvol  
        if (vol[i] > vol[i-1]):
            PVI[i] = (PVI[i-1] + (source[i] - source[i-1]))
        else:
            PVI[i] = PVI[i-1]
        if (vol[i] < vol[i-1]):
            NVI[i] = NVI[i-1] - (source[i] - source[i-1])
        else:
            NVI[i] = NVI[i-1]

    psig = talib.EMA(PVI, t1)
    pdiv = talib.EMA(PVI - psig, t2)
    nsig = talib.EMA(NVI, t1)
    ndiv = talib.EMA(NVI - nsig, t2)
    DVDI = pdiv - ndiv
    return DVDI

def DVDI2(candles, source, t1, t2, v_type): 
    PVI = np.zeros_like(source)
    NVI = np.full_like(source, 0)
    vol = np.full_like(source, 0)
    tickvol = np.full_like(source, 0)
    tickrng = np.full_like(source, 0)
    rng = np.full_like(source, 0)

    for i in range(source.shape[0]):
        tick = 0.01
        rng = candles[:,2] - candles[:,1] 
        if (abs(rng[-1]) < tick):
            tickrng = tick
        else:
            tickrng = rng
        tickvol[i] = np.abs(tickrng[i])/tick
        if v_type == "Default":
            vol = candles[:,5]
        else:
            vol = tickvol  
        if (vol[i] > vol[i-1]):
            PVI[i] = (PVI[i-1] + (source[i] - source[i-1]))
        else:
            PVI[i] = PVI[i-1]
        if (vol[i] < vol[i-1]):
            NVI[i] = NVI[i-1] - (source[i] - source[i-1])
        else:
            NVI[i] = NVI[i-1]
    psig = talib.EMA(PVI, t1)
    pdiv = talib.EMA(PVI - psig, t2)
    nsig = talib.EMA(NVI, t1)
    ndiv = talib.EMA(NVI - nsig, t2)
    DVDI = pdiv - ndiv
    return pdiv,ndiv
    
def t1(source2, source, t, m):
    t1 = np.zeros_like(source)
    rng = np.zeros_like(source)
    avgrng = np.zeros_like(source)
    for i in range(source2.shape[0]):
        wper = (t*2) - 1
        rng[i] = np.abs(source[i] - source[i-1])
    avgrng = talib.EMA(rng, wper)
    smgrng = (talib.EMA(avgrng, wper))*m 
    for i in range(source2.shape[0]):
        t1[i] = source[i] 
        if np.isnan(t1[i-1]):
            t1[i] = source[i] 
        else:
            if (source[i] > t1[i-1]):
                if ((source[i]-smgrng[i]) < t1[i-1]):
                    t1[i] = t1[i-1]
                else: 
                    t1[i] = (source[i] - smgrng[i])
            else: 
                if (source[i]+smgrng[i]) > t1[i-1]: 
                    t1[i] = t1[i-1]
                else: 
                    t1[i] = source[i]+smgrng[i] 
    return t1 

