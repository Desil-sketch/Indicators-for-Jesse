from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

'''
https://www.tradingview.com/script/4bFCTkDv-Quadratic-Regression-Slope-DW/#chart-view-comments
lacks compression function
''' 

def qrs(candles: np.ndarray, period: int= 89, off: int = 0, smper : int = 2,norm:bool=False, source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    qrs1 = QRS(source,source,period,off)
    rms = RMS(source,qrs1,period)
    if qrs1[-1] > 0:
        norm_qrs = qrs1/rms 
    elif qrs1[-1] < 0:
        norm_qrs = -np.abs(qrs1)/rms 
    else:
        norm_qrs = 0 
    qrs2 = norm_qrs if norm else qrs1 
    if sequential:
        return qrs2[-1]
    else:
        return qrs2[-1] 

    
@jit(nopython=True, error_model='numpy')    
def RMS(source,source1,t):
    res1 = np.full_like(source,0)
    res = np.full_like(source,0)
    for i in range(t,source.shape[0]):
        res1[i] = np.power(source1[i],2)
        res[i] = np.sqrt(np.mean(res1[i-t:i]))
    return res 

@jit(nopython=True, error_model='numpy')
def QRS(source,y,t,o):
    x = np.full_like(source,1)
    b0 = np.full_like(source,0)
    preb1 = np.full_like(source,0)
    b1 = np.full_like(source,0)
    preb2 = np.full_like(source,0)
    b2 = np.full_like(source,0)
    preb3 = np.full_like(source,0)
    b3 = np.full_like(source,0)
    c0 = np.full_like(source,0)
    prec1 = np.full_like(source,0)
    c1 = np.full_like(source,0)
    prec2 = np.full_like(source,0)
    c2 = np.full_like(source,0)
    a0 = np.full_like(source,0)
    a1 = np.full_like(source,0)
    a2 = np.full_like(source,0)
    qr = np.full_like(source,0)
    qrs = np.full_like(source,0)
    for i in range(t,source.shape[0]):
        x[i] = x[i-1] + 1 
        b0[i] = np.sum(x[i-t+1:i+1])
        preb1[i] = np.power(x[i],2)
        b1[i] = np.sum(preb1[i-t+1:i+1])
        preb2[i] = np.power(x[i],3)
        b2[i] = np.sum(preb2[i-t+1:i+1])
        preb3[i] = np.power(x[i],4)
        b3[i] = np.sum(preb3[i-t+1:i+1])
        c0[i] = np.sum(y[i-t+1:i+1])
        prec1[i] = x[i]*y[i] 
        c1[i] = np.sum(prec1[i-t+1:i+1])
        prec2[i] = np.power(x[i],2)*y[i] 
        c2[i] = np.sum(prec2[i-t+1:i+1])
        a0[i] = (((b1[i]*b3[i] - np.power(b2[i],2))*(c0[i]*b2[i]-c1[i]*b1[i])) - ((b0[i]*b2[i] - np.power(b1[i],2))*(c1[i]*b3[i]-c2[i]*b2[i])))/(((b1[i] * b3[i] - np.power(b2[i],2))*(t*b2[i] - b0[i] *b1[i])) - ((b0[i]*b2[i] - np.power(b1[i],2))*((b0[i]*b3[i] - b1[i] * b2[i]))))
        a1[i] = ((c0[i] * b2[i] - c1[i] * b1[i]) - (t*b2[i] - b0[i]*b1[i])*a0[i])/(b0[i]*b2[i] - np.power(b1[i],2))
        a2[i] = (c0[i] - t*a0[i] - b0[i]*a1[i])/b1[i]
        qr[i] = a0[i] + a1[i]*(x[i]+o) + a2[i]*np.power((x[i]+o),2)
        qrs[i] = qr[i] - qr[i-1] 
    return qrs
