from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple

'''
https://www.tradingview.com/script/RLnFKoAx-VAMA-Volume-Adjusted-Moving-Average/
Cumulative Total Volume Indicator, may work, can't evalute compared to tradingview 
''' 
  
def vama(candles: np.ndarray, useMethod:str='Current', avgVolInp:float=0.0, nDataBars:int=100,vamaLength:int=55,fctVolume:float=0.67, ignoreMax:bool=True, source_type: str = "close", sequential: bool = True ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential) if sequential else candles[-800:]
    source = get_candle_source(candles, source_type=source_type)
    vama_ma = mainfunction(source,candles, avgVolInp, nDataBars, vamaLength, fctVolume, ignoreMax, useMethod)
    if sequential:
        return vama_ma[-1]
    else:
        return vama_ma[-1] 
    
# @njit
# def sumFrom(source,source2,length,_idx,maxLength):
    # sumreturn = np.full_like(source,0)
    # for i in range(0,source.shape[0]):
        # sum1 = 0.0 
        # for j in range(_idx,(length-1+_idx)):
            # sum1 = sum1 + source2[i-j]
        # sumreturn[i] = sum1 
    # return sumreturn

@njit            
def mainfunction(source,candles,avgVolInp,nDataBars,vamaLength,fctVolume,ignoreMax,useMethod):
    sumreturn = np.full_like(source,0)
    tot_nBars = np.full_like(source,0)
    totVolume = np.full_like(source,0)
    avgVolInp = np.full_like(source,0)
    avgVolNow = np.full_like(source,0)
    avgVolPrv = np.full_like(source,0)
    avgVolSub = np.full_like(source,0)
    avgVolume = np.full_like(source,0)
    incVolume = np.full_like(source,0)
    rtoVol2VI = np.full_like(source,0)
    wtdPrices = np.full_like(source,0)
    vama = np.full_like(source,0)
    sum3i = np.full_like(source,0)
    sum4i = np.full_like(source,0)
    bars1 = np.full_like(source,0)
    for i in range(nDataBars,source.shape[0]):
        sum1 = 0.0
        for j in range(0,nDataBars):
            sum1 = sum1 + candles[:,5][i-j] 
        sumreturn[i] = sum1
        tot_nBars[i] = tot_nBars[i-1] + 1
        totVolume[i] = np.cumsum(candles[:,5])[i]
        avgVolNow[i] = totVolume[i] / tot_nBars[i]
        avgVolPrv[i] = avgVolNow[i-1] 
        avgVolSub[i] = sumreturn[i]/nDataBars
        if useMethod == "Current":
            avgVolume[i] = avgVolNow[i] 
        elif useMethod == "input":
            avgVolume[i] = avgVolInp[i]
        else:
            avgVolume[i] = avgVolSub[i] 
        incVolume[i] = avgVolume[i] * fctVolume
        rtoVol2VI[i] = candles[:,5][i]/ incVolume[i]
        wtdPrices[i] = source[i] * rtoVol2VI[i] 
        bars = 1 
        sum2 = 0.0 
        for j in range(1,((vamaLength*5)+1)):
            insidesum = 0.0
            strict = False if ignoreMax == True else j == vamaLength
            for k in range(0,j+1):
                insidesum = insidesum + rtoVol2VI[i-k]
            if insidesum >= vamaLength or not ignoreMax:
                break
            bars = bars + 1 
        bars1 = bars 
        sum3 = 0.0
        sum4 = 0.0
        for j in range(0,bars):
            sum3 = sum3 + wtdPrices[i-j]
            sum4 = sum4 + rtoVol2VI[i-j]
        sum3i[i] = sum3
        sum4i[i] = sum4 
        vama[i] = (sum3i[i] - (sum4i[i] - vamaLength) * source[i-bars1]) / vamaLength 
    return vama
    
    

    
