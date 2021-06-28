from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

ZSCORE = namedtuple('ZSCORE',['barcolor', 'z_average'])

'''
https://www.tradingview.com/script/lmHJLtdY-Multi-Z-Score-DW/#chart-view-comments
Modified so that Weights are equivalent to number of Z score. Ex: z_num:10 = 10 z scores and 10 weights. 
''' 

def zscore(candles: np.ndarray, z1per: int =3,z2per: int=5,z3per:int=8,z4per:int=13,z5per:int=21,z6per:int=34,z7per:int=55,z8per:int=89,z9per:int=144,z10per:int=233,z11per:int=377,z12per:int=610,z13per:int=987,z14per:int=1597,z15per:int=2584,z16per:int=4181,znum:int=10, w_method: str = 'Equal Weights',w_invert:bool=False,smper:int=2, source_type: str = "close", sequential: bool = False ) -> ZSCORE:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    avg = talib.SMA(source,3)
    zeros1 = np.full_like(source,0)
    z1 = z_score(source,source,z1per,avg) if znum >= 1 else zeros1 
    z2 = z_score(source,source,z2per,avg) if znum >= 2 else zeros1  
    z3 = z_score(source,source,z3per,avg) if znum >= 3 else zeros1  
    z4 = z_score(source,source,z4per,avg) if znum >= 4 else zeros1  
    z5 = z_score(source,source,z5per,avg) if znum >= 5 else zeros1  
    z6 = z_score(source,source,z6per,avg) if znum >= 6 else zeros1 
    z7 = z_score(source,source,z7per,avg) if znum >= 7 else zeros1  
    z8 = z_score(source,source,z8per,avg) if znum >= 8 else zeros1  
    z9 = z_score(source,source,z9per,avg) if znum >= 9 else zeros1 
    z10 = z_score(source,source,z10per,avg) if znum >= 10 else zeros1  
    z11 = z_score(source,source,z11per,avg) if znum >= 11 else zeros1  
    z12 = z_score(source,source,z12per,avg) if znum >= 12 else zeros1  
    z13 = z_score(source,source,z13per,avg) if znum >= 13 else zeros1  
    z14 = z_score(source,source,z14per,avg) if znum >= 14 else zeros1  
    z15 = z_score(source,source,z15per,avg) if znum >= 15 else zeros1  
    z16 = z_score(source,source,z16per,avg) if znum >= 16 else zeros1  
    w1 = z_wt(source,source,z1per,candles,w_method,w_invert ) if znum >= 1 else zeros1 
    w2 = z_wt(source,source,z2per,candles,w_method,w_invert ) if znum >= 2 else zeros1  
    w3 = z_wt(source,source,z3per,candles,w_method,w_invert ) if znum >= 3 else zeros1  
    w4 = z_wt(source,source,z4per,candles,w_method,w_invert ) if znum >= 4 else zeros1  
    w5 = z_wt(source,source,z5per,candles,w_method,w_invert ) if znum >= 5 else zeros1  
    w6 = z_wt(source,source,z6per,candles,w_method,w_invert ) if znum >= 6 else zeros1 
    w7 = z_wt(source,source,z7per,candles,w_method,w_invert ) if znum >= 7 else zeros1  
    w8 = z_wt(source,source,z8per,candles,w_method,w_invert ) if znum >= 8 else zeros1  
    w9 = z_wt(source,source,z9per,candles,w_method,w_invert ) if znum >= 9 else zeros1 
    w10 = z_wt(source,source,z10per,candles,w_method,w_invert ) if znum >= 10 else zeros1  
    w11 = z_wt(source,source,z11per,candles,w_method,w_invert ) if znum >= 11 else zeros1  
    w12 = z_wt(source,source,z12per,candles,w_method,w_invert ) if znum >= 12 else zeros1  
    w13 = z_wt(source,source,z13per,candles,w_method,w_invert ) if znum >= 13 else zeros1  
    w14 = z_wt(source,source,z14per,candles,w_method,w_invert ) if znum >= 14 else zeros1  
    w15 = z_wt(source,source,z15per,candles,w_method,w_invert ) if znum >= 15 else zeros1  
    w16 = z_wt(source,source,z16per,candles,w_method,w_invert ) if znum >= 16 else zeros1  
    w_composite = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + w10 + w11 + w12 + w13 + w14 + w15 + w16
    prez1 = (z1*w1 + z2*w2 + z3*w3 + z4*w4 + z5*w5 + z6*w6 + z7*w7 + z8*w8 + z9*w9 + z10*w10 + z11*w11 + z12*w12 + z13*w13 + z14*w14 + z15*w15 + z16*w16)/w_composite
    z_average = pine_ema(source,prez1,smper)
    barcolor = barcolor1(source,z_average)
    if sequential:
        return ZSCORE(barcolor,z_average)
    else:
        return ZSCORE(barcolor[-1],z_average[-1])


@njit
def barcolor1(source, z):
    barcolor = np.full_like(source,0)
    for i in range(source.shape[0]):    
        if (z[i] > 0) and (z[i] > z[i-1]):
            barcolor[i] = 2 
        elif (z[i] > 0) and (z[i] < z[i-1]):
            barcolor[i] = 1 
        elif (z[i] < 0) and (z[i] < z[i-1]):
            barcolor[i] = -2 
        elif (z[i] < 0) and (z[i] > z[i-1]):
            barcolor[i] = -1 
        else:
            barcolor[i] = 0 
    return barcolor
    
@njit 
def pine_ema(source1, source2, length):
    alpha = 2 / (length + 1)
    sum1 = np.full_like(source1,0)
    for i in range(10,source1.shape[0]):
        sum1[i-1] = 0 if np.isnan(sum1[i-1]) else sum1[i-1]
        sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1]
    return sum1 
    
def z_score(source,x,t,avg):
    avg = talib.SMA(x,t)
    res = (x - avg)/std2(x,avg,t)
    return res 

@njit
def std2(source,avg,per):
    std1 = np.full_like(source,0)
    for i in range(source.shape[0]):
        sum1 = 0.0
        sum2 = 0.0
        for j in range(per):
            sum1 = (source[i-j] + -avg[i])
            sum2 = sum2 + sum1 * sum1 
        std1[i] = np.sqrt(sum2 / per)
    return std1 

    
def z_wt(source,x,t,candles,method,invert):
    ones = np.full_like(source,1)
    avg = talib.SMA(x,t)
    if method=='Equal Weights':
        z_wt1 = ones 
    elif method == 'Volume Weights':
        z_wt1 = talib.SUM(candles[:,5],t)
    elif method == 'Length Weights':
        z_wt1 = t
    else:   
        z_wt1 = std2(x,avg,t)
    z_wt = 1/z_wt1 if invert else z_wt1 
    return z_wt 
