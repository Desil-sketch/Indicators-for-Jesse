from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple

EKC = namedtuple('EKC',['filt1','filt2'])

'''
https://www.tradingview.com/script/LvaRO3Un-powerful-moving-average-crossover/
''' 
  
def ekc(candles: np.ndarray, period: int= 14, source_type: str = "close", trackinrat_01:float= 2.0, trackinrat_02:float=2.0,kgain_01:float=0.7,kgain_02:float=0.8,sequential: bool = False ) -> EKC:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    filt1 = fast_ekc(source,candles,period,trackinrat_01,kgain_01)
    filt2 = fast_ekc(source,candles,period,trackinrat_02,kgain_02)
    if sequential:
        return EKC(filt1,filt2)
    else:
        return EKC(filt1[-1],filt2[-1]) 


@njit
def fast_ekc(source,candles,period,trackinrat_func,kgain_func):
    movement_uncertainty = np.full_like(source,0)
    measurement_uncertainty_mcbw = np.full_like(source,0)
    lamba1 = np.full_like(source,0)
    alpha = np.full_like(source,0)
    ehlers_optimal_price = np.full_like(source,0)
    for i in range(source.shape[0]):
        movement_uncertainty[i] = trackinrat_func*(source[i] - source[i-1]) + kgain_func*(movement_uncertainty[i-1]) 
        measurement_uncertainty_mcbw[i] = 1*(candles[:,3][i-1] - candles[:,4][i-1]) + kgain_func*(measurement_uncertainty_mcbw[i-1])
        lamba1[i] = np.abs(movement_uncertainty[i]/measurement_uncertainty_mcbw[i])
        alpha[i] = (-np.power(lamba1[i],2) + np.power((np.power(lamba1[i],4)+16*np.power(lamba1[i],2)) , 0.5))/8
        ehlers_optimal_price[i] = alpha[i] * source[i] + (1-alpha[i]) * ehlers_optimal_price[i-1] 
    return ehlers_optimal_price 
