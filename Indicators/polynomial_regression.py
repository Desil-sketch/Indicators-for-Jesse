from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple
import jsonpickle
from tempfile import TemporaryFile
outfile1 = TemporaryFile()
outfile1 = TemporaryFile()

#jesse backtest  '2018-01-01' '2021-10-01'
POLYREG = namedtuple('polyreg',['minval', 'maxval', 'buysignal','sellsignal'])

'''
https://www.tradingview.com/script/5q6K1Suu-Function-Polynomial-Regression-Strategy/
''' 
  
def polyreg(candles: np.ndarray, length: int = 100,source_type: str = "close", sequential: bool = False ) -> POLYREG:    
    candles = candles[-800:] if not sequential else slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    indices,prices = pivots(source,candles,length)
    P, Pmax,Pmin,Pstdev = polyreg1(source,candles,indices,prices,length)
    minval = P + Pmin
    maxval = P + Pmax
    minval2 = P - Pstdev
    maxval2 = P + Pstdev
    buysignal = 1 if (source[-2] < minval[-2] and source[-1] > minval[-1]) and not (source[-2] > maxval[-2] and source[-1] < maxval[-1]) else 0
    sellsignal = 1 if (source[-2] > maxval[-2] and source[-1] < maxval[-1]) and not (source[-2] < minval[-2] and source[-1] > minval[-1]) else 0
    if sequential:
        return POLYREG(minval,maxval,buysignal,sellsignal)
    else:
        return POLYREG(minval[-1],maxval[-1],buysignal,sellsignal)

@jit(error_model="numpy")
def polyreg1(source,candles,X,Y,length):
    _meanX = np.full_like(source,0)
    _meanY = np.full_like(source,0)
    _sXX = np.full_like(source,0)
    _sXY = np.full_like(source,0)
    _sXX2 = np.full_like(source,0)
    _sX2X2 = np.full_like(source,0)
    _b = np.full_like(source,0)
    _c = np.full_like(source,0)
    _a = np.full_like(source,0)
    _meanXYi = np.full_like(source,0)
    _meanY2i = np.full_like(source,0)
    _meanX2i = np.full_like(source,0) 
    _meanX3i = np.full_like(source,0) 
    _meanX4i = np.full_like(source,0) 
    _meanX2Yi = np.full_like(source,0) 
    Pstdev = np.full_like(source,0)
    _predictions1 = np.full_like(source,0)
    _max_dev1 = np.full_like(source,0)
    _min_dev1 = np.full_like(source,0)
    Pstdev = np.full_like(source,0)
    _sX2Y = np.full_like(source,0)
    sourcehigh = candles[:,3]
    sourcelow = candles[:,4]
    Xmod = np.full_like(source,0)
    _sizeY = length# Y.size - np.count_nonzero(np.isnan(Y))
    _sizeX = length# X.size - np.count_nonzero(np.isnan(X))
    # _nanY = _sizeY + np.count_nonzero(np.isnan(Y))
    # _nanX = _sizeX + np.count_nonzero(np.isnan(X))
    for i in range(source.shape[0]-(length+1),source.shape[0]):
        Xmod[i] = X[i] #+ 14946
        _meanX[i] = (np.sum(Xmod[i-length+1:i+1]))  / _sizeX 
        _meanY[i] = (np.sum(Y[i-length+1:i+1]))  / _sizeX
        _meanXY = 0
        _meanY2 = 0
        _meanX2 = 0 
        _meanX3 = 0 
        _meanX4 = 0 
        _meanX2Y = 0 
        _Xi = 0.0
        _Yi = 0.0
        for j in range(0,length):
            _Xi = Xmod[i-j]
            _Yi = Y[i-j] 
            _meanXY = _meanXY + (_Xi * _Yi)
            _meanY2 = _meanY2 + np.power(_Yi,2)
            _meanX2 = _meanX2 + np.power(_Xi,2)
            _meanX3 = _meanX3 + np.power(_Xi,3)
            _meanX4 = _meanX4 + np.power(_Xi,4)
            _meanX2Y = _meanX2Y + np.power(_Xi,2) * _Yi 
        _meanXYi[i] = _meanXY / _sizeX 
        _meanY2i[i] = _meanY2 / _sizeX 
        _meanX2i[i] = _meanX2 / _sizeX 
        _meanX3i[i] = _meanX3 / _sizeX 
        _meanX4i[i] = _meanX4 / _sizeX 
        _meanX2Yi[i] = _meanX2Y / _sizeX 
        _sXX[i] = _meanX2i[i] - _meanX[i] * _meanX[i] 
        _sXY[i] = _meanXYi[i] - _meanX[i] * _meanY[i] 
        _sXX2[i] = _meanX3i[i] - _meanX[i] * _meanX2i[i]
        _sX2X2[i] = _meanX4i[i] - _meanX2i[i] * _meanX2i[i]
        _sX2Y[i] = _meanX2Yi[i] - _meanX2i[i] * _meanY[i] 
        _b[i] = (_sXY[i] * _sX2X2[i] - _sX2Y[i] * _sXX2[i]) / (_sXX[i] * _sX2X2[i] - _sXX2[i] * _sXX2[i])
        _c[i] = (_sX2Y[i] * _sXX[i] - _sXY[i] * _sXX2[i]) / (_sXX[i] * _sX2X2[i] - _sXX2[i] * _sXX2[i])
        _a[i] = _meanY[i] - _b[i] * _meanX[i] - _c[i] * _meanX2i[i]
        _vector = 0.0
        _predictions = 0.0
        _diff = 0.0
        _max_dev = 0.0
        _min_dev = 0.0
        _stdev = 0.0
        for j in range(0,length):
            _Xi = Xmod[i-j]
            _vector = _a[i] + _b[i] * _Xi + _c[i] * _Xi * _Xi
            if j == 0:
                _predictions = _vector
            _Yi = Y[i-j]
            _diff = _Yi - _vector 
            if (_diff > _max_dev):
                _max_dev = _diff
            if (_diff < _min_dev):
                _min_dev = _diff 
            _stdev = _stdev + np.abs(_diff)
        _predictions1[i] = _predictions 
        _max_dev1[i] = _max_dev
        _min_dev1[i] = _min_dev 
        Pstdev[i] = _stdev / _sizeX 
    return _predictions1, _max_dev1, _min_dev1, Pstdev


@jit(error_model="numpy")
def pivots(source,candles,length):
    prices = np.full_like(candles[:,1],np.nan)
    indices = np.full_like(candles[:,1],np.nan)
    highmiddlesource = np.full_like(candles[:,1],0)
    lowmiddlesource = np.full_like(candles[:,1],0)
    pivothigh = np.full_like(candles[:,1],0)
    pivotlow = np.full_like(candles[:,1],0)
    lastpivothighprice = np.full_like(candles[:,1],0)
    lastpivotlowprice = np.full_like(candles[:,1],0)
    sourcehigh = candles[:,3]
    sourcelow = candles[:,4]
    r = 2 
    l = 2
    for i in range(5,source.shape[0]):
        highmiddlesource[i] = sourcehigh[i-r]
        lowmiddlesource[i] = sourcelow[i-r]
        if (np.all(highmiddlesource[i] >= sourcehigh[i-(l+r):i-(r)]) and np.all(highmiddlesource[i] > sourcehigh[i-(r-1):i+1])) and pivothigh[i-1] == 0:
            pivothigh[i] = 1  
            lastpivothighprice[i] = highmiddlesource[i]  
        else:
            pivothigh[i] = 0 
            lastpivothighprice[i] = lastpivothighprice[i-1] 
        if (np.all(lowmiddlesource[i] <= sourcelow[i-(l+r):i-(r)]) and np.all(lowmiddlesource[i] < sourcelow[i-(r-1):i+1])) and pivotlow[i-1] == 0:    
            pivotlow[i] = 1  
            lastpivotlowprice[i] = lowmiddlesource[i] 
        else:
            pivotlow[i] = 0 
            lastpivotlowprice[i] = lastpivotlowprice[i-1] 
        if pivothigh[i-1] == 1 and pivotlow[i-1] == 1:
            prices[i] = candles[:,4][i-3] 
            indices[i] = (i-3) 
        if pivothigh[i] == 1 and pivotlow[i] == 0:
            prices[i] = candles[:,3][i-2] 
            indices[i] = (i-2) 
        if pivotlow[i] == 1 and pivothigh[i] == 0:
            prices[i] = candles[:,4][i-2] 
            indices[i] = (i-2)
        if pivothigh[i] == 1 and pivotlow[i] == 1:
            prices[i] = candles[:,3][i-2] 
            indices[i] = (i-2) 
        if pivothigh[i] == 0 and pivotlow[i] == 0 and (pivothigh[i-1] == 0 and pivotlow[i-1] == 0):
            prices[i] = np.nan
            indices[i] = np.nan 
    prices =  prices[~np.isnan(prices)]
    prices = np.concatenate((np.full((source.shape[0] - prices.shape[0]), np.nan), prices)) if prices.shape[0] < source.shape[0] else prices
    indices = indices[~np.isnan(indices)]
    indices = np.concatenate((np.full((source.shape[0] - indices.shape[0]), np.nan), indices)) if indices.shape[0] < source.shape[0] else indices
    if np.isnan((prices[-(length+1)])):
        prices[:source.shape[0]-(length+1)] = candles[:source.shape[0]-(length+1),1]
    return indices, prices
                 
            
        
     
	
	
