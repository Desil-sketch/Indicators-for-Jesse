from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

TMO = namedtuple('TMO',['mainLine', 'signalLine', 'barcolor'])

'''
https://www.tradingview.com/script/HibAeQPA-TMO-with-TTM-Squeeze/
''' 
  
def tmo(candles: np.ndarray, tmolength:int=21,calcLength:int=5,smoothLength:int=3, source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential) 
    source = get_candle_source(candles, source_type=source_type)
    mainLine,signalLine,barcolor = fast_tmo(source,candles,tmolength,calcLength,smoothLength)
    if sequential:
        return TMO(mainLine,signalLine,barcolor)
    else:
        return TMO(mainLine[-1],signalLine[-1],barcolor[-1])
    
@njit
def fast_tmo(source,candles,tmolength,calcLength,smoothLength):
    EMA5 = np.full_like(source,0)
    data1 = np.full_like(source,0)
    Main = np.full_like(source,0)
    Signal = np.full_like(source,0)
    mainLine = np.full_like(source,0)
    signalLine = np.full_like(source,0)
    barcolor = np.full_like(source,0)
    for i in range(tmolength,source.shape[0]):
        data = 0.0
        for j in range(1,tmolength):
            if candles[:,2][i] > candles[:,1][i-j]:
                data = data + 1 
            elif candles[:,2][i] < candles[:,1][i-j]:
                data = data - 1 
        data1[i] = data 
        EMA5[i] = pine_ema(source,data1,calcLength)[i]
        Main[i] = pine_ema(source,EMA5,smoothLength)[i]
        Signal[i] = pine_ema(source,Main,smoothLength)[i]
        mainLine[i] = 100*Main[i]/tmolength
        signalLine[i] = 100*Signal[i]/tmolength
        barcolor[i] = 1 if mainLine[i] > signalLine[i] else 0 
    return mainLine,signalLine,barcolor
        
@njit 
def pine_ema(source1, source2, length):
    alpha = 2 / (length + 1)
    sum1 = np.full_like(source1,0)
    for i in range(source1.shape[0]):
        sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1] 
    return sum1 
    
    
