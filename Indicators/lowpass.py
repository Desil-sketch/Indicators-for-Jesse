from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

"""
https://www.tradingview.com/script/SFGEoDmG-Low-Pass-Channel-DW/
"""

LowPass = namedtuple('LowPass',['hband', 'lband', 'filt'])

def lowpass(candles: np.ndarray, fixed_per: int = 20, max_cycle_limit: int = 60, cycle_mult: float = 1.0, tr_mult: float = 1.0, cutoff_type: str = "Adaptive",   source_type: str = "close", sequential: bool = False) -> LowPass:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type) 
    dc = dom_cycle(source, source, max_cycle_limit)
    if cutoff_type == "Fixed": 
        per = np.full_like(source,fixed_per)
    else: 
        per1 = np.round(dc*cycle_mult)
        per = np.where(per1 == 0, per1[-1], per1)
    if cutoff_type == "Fixed": 
        filt = lp(source, source, per)
    else:
        filt = lp(source,source,per)
    TR_sub0 = (candles[:,3] - candles[:,4])
    TR_sub1 = np.abs(candles[:,2] - (np_shift(candles[:,4],1,fill_value = np.nan)))
    TR_sub2 = np.abs(candles[:,2] - (np_shift(candles[:,3],1,fill_value = np.nan)))
    TR = np.maximum((np.maximum(TR_sub0,TR_sub1)), TR_sub2)
    ftr = lp(source, TR, per)*tr_mult
    hband = filt + ftr
    lband = filt - ftr 
    if sequential:
        return LowPass(hband, lband, filt)
    else:
        return LowPass(hband[-1], lband[-1], filt[-1])
    
@njit
def dom_cycle(source, x,lim):
    res = np.full_like(source, 0)
    val1 = np.full_like(source, 0)
    in_phase = np.full_like(source, 0)
    quadrature = np.full_like(source, 0)
    real = np.full_like(source, 0)
    real2 = np.full_like(source,0)
    imag = np.full_like(source, 0)
    deltaphase = np.full_like(source, 0)
    dom_cycle = np.full_like(source, 0)
    for i in range(source.shape[0]):
        val1[i] = x[i] - x[i-7]
        in_phase[i] = 1.25*((val1[i-4]) - (0.635*(val1[i-2]))) +  0.635*(in_phase[i-3])
        quadrature[i] = val1[i-2] - 0.338*val1[i] + 0.338*quadrature[i-2] 
        real[i] = 0.2*(in_phase[i]*in_phase[i-1] +  quadrature[i] * quadrature[i-1]) + 0.8 * real[i-1] 
        real2 = np.where(real[i] == 0, np.nan, real[i])
        imag[i] = 0.2*(in_phase[i]*quadrature[i-1] - in_phase[i-1] * quadrature[i]) + 0.8 * imag[i-1]
        deltaphase[i] = np.arctan(imag[i]/real2)
        val2 = 0.0
        inst_per = 0.0
        for j in range(0,lim + 1): 
            val2 = val2 + deltaphase[i-j]
            if val2 > (4*np.arcsin(1)) and (inst_per == 0.0):
                 inst_per = j 
        if (inst_per == 0.0):
            inst_per = inst_per
        dom_cycle[i] = 0.25*inst_per+0.75*(dom_cycle[i-1])
    return dom_cycle
        
@njit
def lp(source,x,t):
    omega = np.full_like(source,0)
    alpha = np.full_like(source,0)
    lp = np.full_like(source,0)
    for i in range(source.shape[0]):
        if not np.isnan(lp[i-1]):
            omega[i] = 4*np.arcsin(1)/t[i]
            alpha[i] = (1 - np.sin(omega[i]))/(np.cos(omega[i]))
            lp[i] = 0.5*(1 - alpha[i]) * (x[i] + x[i-1]) + alpha[i] * lp[i-1]
    return lp
