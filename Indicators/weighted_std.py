from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple

#jesse backtest  '2021-01-03' '2021-03-02'

'''
https://www.tradingview.com/script/F6fK5IMa-Function-Weighted-Standard-Deviation/#chart-view-comments
''' 
  
def wstd(candles: np.ndarray, source_type: str = "close", mult:float=2.0,length: int=20, sequential: bool = False ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    weight = candles[:,5] 
    mid,upper,lower = weighted_std(source,weight,length,mult)
    if sequential: 
        return mid
    else:   
        return mid[-1] 
	
	
@njit	
def weighted_std(source,weight,length,mult):
    _xw = np.full_like(source,0)
    _sum_weight = np.full_like(source,0)
    _mean = np.full_like(source,0)
    _variance = np.full_like(source,0)
    _dev = np.full_like(source,0)
    _mse = np.full_like(source,0)
    _rmse = np.full_like(source,0)
    mid = np.full_like(source,0)
    upper = np.full_like(source,0)
    lower = np.full_like(source,0)
    for i in range(length,source.shape[0]):
        _xw[i] = source[i] * weight[i] 
        _sum_weight[i] = np.sum(weight[i-(length-1):i+1])
        _mean[i] = np.sum(_xw[i-(length-1):i+1]) / _sum_weight[i] 
        _sqerror_sum = 0
        _nonzero_n = 0 
        for j in range(length):
            _sqerror_sum = _sqerror_sum + np.power(_mean[i] - source[i], 2) * weight[i]
            _nonzero_n = _nonzero_n + 1 if weight[i] != 0 else _nonzero_n 
        _variance[i] = _sqerror_sum / ((_nonzero_n - 1) * _sum_weight[i] / _nonzero_n)
        _dev[i] = np.sqrt(_variance[i])
        _mse[i] = _sqerror_sum / _sum_weight[i] 
        _rmse[i] = np.sqrt(_mse[i])
        mid[i] = _mean[i] 
        upper[i] = _mean[i] + _dev[i] * mult
        lower[i] = _mean[i] - _dev[i] * mult 
    return mid, upper, lower

	
