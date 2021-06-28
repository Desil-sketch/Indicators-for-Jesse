from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

SDV = namedtuple('SDV',['netvol', 'supply', 'demand','v'])

'''
https://www.tradingview.com/script/kfUHOMlX-RedK-Supply-Demand-Volume-Viewer-v1/#chart-view-comments
Buy or Sell with positive or negative volume respectively 
''' 

def sdv(candles: np.ndarray, l: int= 10,s:int=3, source_type: str = "close", sequential: bool = False ) -> SDV:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    netvol, supply,demand,v = fast_sdv(source,candles,s,l)
    if sequential:
        return SDV(netvol,supply,demand,v)
    else:
        return SDV(netvol[-1],supply[-1],demand[-1],v[-1]) 

@njit
def fast_sdv(source,candles,s,l):
    upday = np.full_like(source,0)
    v = np.full_like(source,0)
    body = np.full_like(source,0)
    barrange = np.full_like(source,0)
    wick = np.full_like(source,0)
    realrange = np.full_like(source,0)
    bscore = np.full_like(source,0)
    bullscore = np.full_like(source,0)
    demand = np.full_like(source,0)
    bearscore = np.full_like(source,0)
    supply = np.full_like(source,0)
    netvol = np.full_like(source,0) 
    for i in range(source.shape[0]):
        upday[i] = candles[:,2][i] > candles[:,1][i]
        v[i] = candles[:,5][i] 
        body[i] = np.abs(candles[:,2][i] - candles[:,1][i])
        barrange[i] = candles[:,3][i] - candles[:,4][i] 
        wick[i] = barrange[i] - body[i] 
        realrange[i] = barrange[i] + wick[i] 
        if barrange[i] > 0:
            if (candles[:,2][i] >= candles[:,1][i]):
                bscore[i] = barrange[i] / realrange[i]
            else:
                bscore[i] = wick[i] / realrange[i] 
        else:
            bscore[i] = 0.5 
        bullscore[i] = bscore[i] * v[i] 
        demand = pine_wma(source,pine_wma(source,bullscore,l),s)
        bearscore[i] = v[i] - bullscore[i]
        supply = pine_wma(source,pine_wma(source,bearscore,l),s)
        netvol[i] = demand[i] - supply[i]
    return netvol, supply, demand, v
    
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
