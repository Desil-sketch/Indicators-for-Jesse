from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

BBB = namedtuple('BBB',['basis', 'upper', 'lower','colorcenter','bandcolor'])

'''
https://www.tradingview.com/script/E02jMupd-Better-Bollinger-Bands-now-open-source/
Better Bollinger Bands
sequential must remain True, 
Only default settings tested to be working 
''' 

def bbb(candles: np.ndarray, mult: float=2.0, length: int = 20, timewindow: str = "Simple", use_linear_regression: bool = False, ignore_volume: bool = False, centerscheme: str = "Hybrid", bandscheme: str = "Centerline",source_type: str = "close", sequential: bool = True ) -> BBB:#Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    multiple = 8
    rsi_length = length
    hysteresislength = 7
    t = 1.2 
    r = 1.1
    hybrid_parameter = 1.0
    hybrid_trend = 4.5
    n = len(source)
    lsrc = np.log(source)
    basis1 = linear_regression(lsrc,length,n,timewindow) if use_linear_regression else mean(candles,lsrc,length,timewindow)
    dev = mult *(lsq_stdev(candles,lsrc,length,n,timewindow) if use_linear_regression else sdev(candles,lsrc,length,timewindow))
    upper1 = basis1 + dev
    lower1 = basis1 - dev 
    shorttrend = linreg_slope(candles,lsrc,length/2,n,timewindow)*(length/4)
    mediumtrend = volume_ema(candles,lsrc,50*length/100) - volume_ema(candles,lsrc,length)
    longtrend = talib.EMA(lsrc,length) - talib.EMA(lsrc,multiple*length)
    nocleartrend = 1 if np.any(np.abs(longtrend[-1]) <= (2*talib.SMA(dev,multiple*length))) else 0 
    rn = rsinorm(source,rsi_length,timewindow)
    overbought = np.full_like(source,0)
    oversold = np.full_like(source,0)
    rsihit = np.full_like(source,0)
    rsima = np.full_like(source,0)
    for i in range(source.shape[0]):
        overbought[i] = 1 if (overbought[i-1] == 1 or (rn[i] > t)) and not rn[i-1] > t and not rn[i] < r else 0 
        oversold[i] = 1 if (oversold[i-1] == 1 or (rn[i] < -t)) and not rn[i-1] > -r else 0 
        rsihit[i] = 1 if overbought[i] == 1 or oversold[i] == 1 else 0 
        rsima[i] = -rn[i] if rsihit[i] == 1 else 0 
    rsi_hysteresis = talib.EMA(rsima,hysteresislength) + talib.SMA(rsima,hysteresislength)
    hybrid_oscillator = mediumtrend + hybrid_parameter*rsi_hysteresis*(talib.EMA(np.abs(mediumtrend),5*length)) + hybrid_trend*talib.SMA(rsi_hysteresis,2*hysteresislength)*shorttrend 
    centerline_delta = basis1 - talib.EMA(basis1,2)
    volatility_delta = dev - talib.EMA(dev,2)
    momentumcolor = 1 if mediumtrend[-1] > 0 else 0 
    if np.abs(rsi_hysteresis[-1]) > 0.1:
        rsicolor = 1 if rsi_hysteresis[-1] > 0 else -1 
    hybridcolor = 1 if hybrid_oscillator[-1] > 0 else -1 
    deltacolor = 1 if centerline_delta[-1:] > 0 else -1 
    if centerscheme == 'Centerline change':
        colorcenter = deltacolor 
    elif centerscheme == 'Momentum':
        colorcenter = momentumcolor 
    elif centerscheme == 'RSI hysteresis':
        colorcenter = rsicolor
    else:
        colorcenter = hybridcolor 
    basis = np.exp(basis1) 
    upper = np.exp(upper1)
    lower = np.exp(lower1)
    if nocleartrend == 1:
        markettypecolor = 0 
    elif longtrend[-1] > 0:
        markettypecolor = 1 
    else:
        markettypecolor = -1 
    volatility_color = 1 if volatility_delta[-1] > 0 else 0 
    if bandscheme == 'Centerline':
        bandcolor = colorcenter
    elif bandscheme == 'Volatility Change':
        bandcolor = volatility_color
    elif bandscheme == 'Market type':
        bandcolor = markettypecolor
    else:
        bandcolor = 0
        
    if sequential:
        return BBB(basis[-1],upper[-1],lower[-1],colorcenter,bandcolor)
    else:
        return BBB(basis[-1], upper[-1], lower[-1], colorcenter,bandcolor)
     
def volume_sma(candles,series,period):
    res = talib.SMA(series*(candles[:,5] + 1) ,period)/talib.SMA((candles[:,5] + 1) ,period)
    return res 

def volume_ema(candles,series,period):
    res = talib.EMA(series*(candles[:,5] + 1) ,period)/talib.EMA((candles[:,5] + 1) ,period)
    return res 
    
def volume_wma(candles,series,period):
    res = talib.WMA(series*(candles[:,5] + 1) ,period)/talib.WMA((candles[:,5] + 1) ,period)
    return res 
    
def basselcorrection_sma(period):
    res = (period + 1)/period 
    return res 
    
def basselcorrection_ema(period):
    res = (period + 1)/(2 * (period-1))
    return res 
    
def basselcorrection_wma(period):
    res = (3 * period*(period+1)/(3*period*period+period-1))
    return res 
    
def mean(candles,series,length,timewindow):
    if timewindow == 'Exponential':
        res = volume_ema(candles,series,length)
    elif timewindow == 'Sawtooth':
        res = volume_wma(candles,series,length)
    else:
        res = volume_sma(candles,series,length)
    return res 
 
def basselcorrection(length, timewindow):
    if timewindow == 'Exponential':
        res = basselcorrection_ema(length)
    elif timewindow == 'Sawtooth':
        res = basselcorrection_wma(length)
    else:
        res = basselcorrection_sma(length)
    return res 
    
def cov(candles,x,y,length,timewindow):
    res = (mean(candles,x*y,length,timewindow) - mean(candles,x,length,timewindow)*mean(candles,y,length,timewindow))*basselcorrection(length,timewindow)
    return res 
    
def var(candles,series,length,timewindow):
    res = (mean(candles,series*series,length,timewindow) - mean(candles,series,length,timewindow)*mean(candles,series,length,timewindow))*basselcorrection(length,timewindow)
    return res 
    
def sdev(candles,series,length,timewindow):
    res = (np.sqrt(var(candles,series,length,timewindow)))
    return res 
   
def linreg_slope(candles,series,length,n,timewindow):
    res = cov(candles,n,series,length,timewindow)/var(candles,n,length,timewindow)
    return res 
    
def linear_regression(candles,series,length,n,timewindow):
    m = linreg_slope(candles,series,length,timewindow)
    res = mean(candles,series,length,timewindow) + m*(n - mean(candles,n,length,timewindow))
    return res 
    
def lsq_variance(candles,series,length,n,timewindow):
    m = linreg_slope(candles,series,length,timewindow)
    res = var(candles,series,length,timewindow) + m*m*var(candles,n,length,timewindow) - 2*m*cov(candles,series,n,length,timewindow)
    return res 
    
def lsq_stdev(candles,series,length,n,timewindow):
    res = np.sqrt(lsq_variance(candles,series,length,n,timewindow))
    return res 
    
def rsinorm(series,length,timewindow):
    rsi = fast_rsi(series,length)
    res = (rsi - talib.EMA(rsi,4*length))/15 
    return res 
    
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
    
