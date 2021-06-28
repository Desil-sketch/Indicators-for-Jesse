from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple


'''
https://www.tradingview.com/script/4wdh1Y1s-NET-MyRSI/#chart-view-comment-form
ehler Noise Elimination Technology with regular rsi, ehler rsi, and by itself
''' 
  
def ehlernet(candles: np.ndarray, period: int= 14, source_type: str = "close",ind_type: str = "net", sequential: bool = False ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    rsi = fast_rsi(source,period)
    ehlerrsi = ehlerrsi_fast(source,period)
    net = ehlernet_fast(source,candles, source, period)
    regrsi = ehlernet_fast(source,candles,rsi,period)
    myrsi = ehlernet_fast(source,candles,ehlerrsi,period)
    if ind_type == 'net':
        res = net 
    elif ind_type == 'regrsi':
        res = regrsi 
    elif ind_type == 'ehlerrsi':
        res = myrsi 
    else:
        res = np.nan 
        
    if sequential:
        return res
    else:
        return res[-1]
@njit
def ehlernet_fast(source,candles,src,period):
    numfinal = np.full_like(source,0)
    for i in range(source.shape[0]):
        demon = period*(period-1)/2
        x = 0.0
        num = 0.0
        for j in range(1,period):
            for k in range(0,j):
                num = num - np.sign(src[i-j] - src[i-k])
        numfinal[i] = num/demon
    return numfinal

@njit
def ehlerrsi_fast(source,period):

    myrsi = np.full_like(source,0)
    for i in range(source.shape[0]):
        cu = 0.0
        cd = 0.0
        for j in range(period):
            if (source[i-j] - source[i-(j+1)]) > 0:
                cu = cu + source[i-j] - source[i-(j+1)]
            if (source[i-j] - source[i-(j+1)]) < 0:
                cd = cd + source[i-(j+1)] - source[i-j] 
        if (cu + cd) != 0:
            myrsi[i] = (cu - cd)/(cu + cd)
    return myrsi             
    
@njit
def fast_rsi(source,length):
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
    
    
