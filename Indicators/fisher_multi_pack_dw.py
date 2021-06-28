from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from numpy.lib.stride_tricks import as_strided

FISHMULT = namedtuple('FISHMULT',['osc', 'basiscolor', 'barcolor'])

'''
https://www.tradingview.com/script/hXp5irRI-Fisher-Multi-Pack-DW/#chart-view-comments
no compression, thresholds, or cmean 
''' 

def fishmulti(candles: np.ndarray, per: int= 13,smooth:bool=False,smper:int=1,otype:str="Fisher Transform",alpha:float=0.1, hth:float=0.95,lth:float=-0.95, source_type: str = "close", sequential: bool = False ) -> FISHMULT:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    if otype == "Fisher Transform":
        highval = same_length(candles,max_rolling1(source,per))
        lowval = same_length(candles,min_rolling1(source,per))
        fisher = fish(source,source,per,highval,lowval)
        osc1 = fisher 
    elif otype == "Inverse Fisher RSI":
        ifrsi = ifish_rsi(source,source,per,candles,alpha)
        osc1 = ifrsi 
    elif otype == "Inverse Fisher Stochastic":
        candles_high = candles[:, 3]
        candles_low = candles[:, 4]
        hh = talib.MAX(candles_high, per)
        ll = talib.MIN(candles_low, per)
        stoch1 = 100 * (source - ll) / (hh - ll)
        STOCH = same_length(candles,stoch1)
        ifstoch = ifish_stoch(source,source,per,candles,alpha,STOCH)
        osc1 = ifstoch 
    else:
        rollwin = talib.SMA(source,per)
        CCI = pine_cci(source,per,rollwin)
        ifcci = ifish_cci(source,source,per,candles,alpha,CCI)
        osc1 = ifcci
    osc2 = pine_ema(source,osc1,smper) if smooth else pine_ema(source,osc1,1)
    osc = osc2 
    basiscolor, barcolor = barcolor1(source,osc,hth,lth)
    if sequential:
        return FISHMULT(osc,basiscolor,barcolor) 
    else:
        return FISHMULT(osc[-1],basiscolor[-1],barcolor[-1])

@njit
def barcolor1(source,osc,hth,lth):
    barcolor = np.full_like(source,0)
    basiscolor = np.full_like(source,0)
    for i in range(source.shape[0]):
        if (osc[i] > 0) and (osc[i] >= osc[i-1]) and (osc[i] >= hth):
            barcolor[i] = 2
        elif (osc[i] >0) and (osc[i] > osc[i-1]) and (osc[i] < hth):
            barcolor[i] = 1.5
        elif (osc[i] > 0) and (osc[i] <osc[i-1]):
            barcolor[i] = 1 
        elif (osc[i] < 0) and (osc[i] <= osc[i-1]) and (osc[i] <= lth):
            barcolor[i] = -2
        elif (osc[i] <0) and (osc[i] < osc[i-1]) and (osc[i] > lth):
            barcolor[i] = -1.5
        elif (osc[i] <0) and (osc[i] > osc[i-1]):
            barcolor[i] = -1 
        else:
            barcolor[i] = 0 
        if osc[i] > 0:
            basiscolor[i] = 1 
        elif osc[i] < 0:
            basiscolor[i] = -1 
        else:
            basiscolor[i] = 0 
    return basiscolor, barcolor 
        
@njit
def pine_cci(source,per,rollwin):
    mamean = np.full_like(source,0)
    cci = np.full_like(source,0)
    dev = np.full_like(source,0)
    for i in range(source.shape[0]):
        sum1 = 0.0
        val = 0.0
        for j in range(per):
            val = source[i-j]
            sum1 = sum1 + np.abs(val - rollwin[i])
        dev[i] = sum1/per 
        cci[i] = (source[i] - rollwin[i]) / (0.015 * dev[i])
    return cci
 
@njit
def pine_rsi(source,length):
    u = np.full_like(source, 0)
    d = np.full_like(source, 0)
    rs = np.full_like(source, 0)
    res = np.full_like(source, 0)
    alpha = 1 / length 
    sumation1 = np.full_like(source, 1)
    sumation2 = np.full_like(source, 1)
    for i in range(source.shape[0]):
        u[i] = np.maximum((source[i] - source[i-1]),0)
        d[i] = np.maximum((source[i-1] - source[i]), 0)
        sumation1[i] = alpha * u[i] + (1 - alpha) * (sumation1[i-1])
        sumation2[i] = alpha * d[i] + (1 - alpha) * (sumation2[i-1]) 
        rs[i] = sumation1[i]/sumation2[i]
        res[i] = 100 - 100 / ( 1 + rs[i])
    return res  

@jit(nopython=True, error_model="numpy")
def fish(source,x,t,highval,lowval):
    val1 = np.full_like(source,0)
    val2 = np.full_like(source,0)
    fish = np.full_like(source,0)
    for i in range(t,source.shape[0]):
        # val1[i-1] = 0 if np.isnan(val1[i-1]) else val1[i-1]
        val1[i] = 0.66 * ((x[i] - lowval[i])/np.maximum((highval[i] - lowval[i]),0.001)-0.5) + 0.67*val1[i-1]
        if val1[i] > 0.99:
            val2[i] = 0.999 
        elif val1[i] < -0.99:
            val2[i] = -0.999 
        else:
            val2[i] = val1[i] 
        fish[i] = 0.5 * np.log((1+val2[i])/np.maximum(1-val2[i],0.001)) + 0.5*fish[i-1]
    return fish

@njit
def ifish_rsi(source,x,t,candles,alpha):
    val = np.full_like(source,0)
    rsi = np.full_like(source,0)
    wval1 = np.full_like(source,0.0)
    wval2 = np.full_like(source,0.0)
    ifish = np.full_like(source,0)
    for i in range(source.shape[0]):
        rsi = pine_rsi(x,t)
        val[i] = alpha*(rsi[i]-50)
        wval1 = pine_wma(source,val,t)
        wval2 = np.concatenate((np.full((candles.shape[0] - wval1.shape[0]), np.nan), wval1))
        ifish[i] = (np.exp(2*wval2[i]) - 1)/(np.exp(2*wval2[i])+1)
    return ifish 
    
@njit
def ifish_stoch(source,x,t,candles,alpha,stoch):
    val = np.full_like(source,0)
    wval1 = np.full_like(source,0.0)
    wval2 = np.full_like(source,0.0)
    ifish = np.full_like(source,0)
    for i in range(source.shape[0]):
        val[i] = alpha*(stoch[i]-50)
        wval1 = pine_wma(source,val,t)
        wval2 = np.concatenate((np.full((candles.shape[0] - wval1.shape[0]), np.nan), wval1))
        ifish[i] = (np.exp(2*wval2[i]) - 1)/(np.exp(2*wval2[i])+1)
    return ifish 
    
@njit
def ifish_cci(source,x,t,candles,alpha,CCI):
    val = np.full_like(source,0)
    wval1 = np.full_like(source,0.0)
    wval2 = np.full_like(source,0.0)
    ifish = np.full_like(source,0)
    for i in range(source.shape[0]):
        val[i] = alpha*(CCI[i])
        wval1 = pine_wma(source,val,t)
        wval2 = np.concatenate((np.full((candles.shape[0] - wval1.shape[0]), np.nan), wval1))
        ifish[i] = (np.exp(2*wval2[i]) - 1)/(np.exp(2*wval2[i])+1)
    return ifish 
    
@njit 
def pine_wma(source1,source2,length):
    res = np.full_like(source1,length)
    for i in range(source1.shape[0]):
        weight = 0.0
        norm = 0.0 
        sum1 = 0.0
        for j in range(length):
            weight = (length - j)*length
            norm = norm + weight 
            sum1 = sum1 + source2[i-j] * weight
        res[i] = sum1/norm 
    return res 
    
@njit 
def pine_ema(source1, source2, length):
    alpha = 2 / (length + 1)
    sum1 = np.full_like(source1,0)
    for i in range(10,source1.shape[0]):
        sum1[i-1] = 0 if np.isnan(sum1[i-1]) else sum1[i-1]
        sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1]
    return sum1 
    
def max_rolling1(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.max(rolling,axis=axis)
        
def min_rolling1(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.min(rolling,axis=axis)        
        
