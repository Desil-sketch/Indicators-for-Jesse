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
https://www.tradingview.com/script/GpvwlRQm/#chart-view-comments
AntiVolumeStopLoss modified to have max lenV of 200 
"""

def antivolumestoploss(candles: np.ndarray, lenF:int=12, lenS:int = 26, mult:float=2.0,source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential) if sequential else candles[-480:] 
    source = get_candle_source(candles, source_type=source_type)  
    prevwmaS = ti.vwma(np.ascontiguousarray(source), np.ascontiguousarray(candles[:, 5]), period=lenS)
    vwmaS = same_length(candles, prevwmaS) 
    prevwmaF = ti.vwma(np.ascontiguousarray(source), np.ascontiguousarray(candles[:, 5]), period=lenF)
    vwmaF = same_length(candles, prevwmaF) 
    AvgS = talib.SMA(source, lenS)
    AvgF = talib.SMA(source, lenF)
    VPC = vwmaS - AvgS
    VPR = vwmaF / AvgF
    VM = talib.SMA(candles[:,5],lenF)/talib.SMA(candles[:,5],lenS)
    VPCI = VPC * VPR * VM 
    DeV = mult * VPCI * VM 
    AVSL = Pricefun(candles,source,VPC,VPR,VM,candles[:,4],lenS,DeV)
    if sequential: 
        return AVSL
    else:    
        return AVSL[-1]

#jesse backtest  '2021-01-03' '2021-03-02'

@jit(error_model = 'numpy')	
def Pricefun(candles,source,VPC,VPR,VM,low,lenS,DeV):
    VPCI = np.full_like(source,0)
    lenV = 0
    PriceV = np.full_like(source,0)
    VPCc = np.full_like(source,0)
    preAVSL = np.full_like(source,0)
    AVSL = np.full_like(source,0)
    for i in range(lenS,source.shape[0]):
        VPCI[i] = VPC[i] * VPR[i] * VM[i]
        if VPC[i] < 0:
            lenV = np.int(np.round(np.abs(VPCI[i]-3)))
        elif VPC[i] >= 0:
            lenV = np.int(np.round(VPCI[i]+3))
        else:
            lenV = 1 
        if lenV >= 200:
            lenV = 200
        if (VPC[i] > -1 and VPC[i] < 0):
            VPCc[i] = -1 
        elif (VPC[i] < 1 and VPC[i] >= 0):
            VPCc[i] = 1 
        else:
            VPCc[i] = VPC[i] 
        Price = 0.0
        for j in range(0,lenV):
            Price = Price + (low[i-j] * 1 / VPCc[i-j] * 1 / VPR[i-j])
        PriceV[i] = ((Price / lenV)/100)
        preAVSL[i] = (candles[:,4][i] - PriceV[i] + DeV[i])
        AVSL[i] = np.mean(preAVSL[(i-lenS+1):i+1])
    return AVSL
	
	
