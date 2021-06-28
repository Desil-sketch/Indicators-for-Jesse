from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length 
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

AKC = namedtuple('AKC',['ma', 'hbandtop', 'hbandlmid','lbandbottom','lbandmid','macolor','barcolor'])

'''
https://www.tradingview.com/script/F6BVjYMH-Index-Adaptive-Keltner-Channels-DW/
AMA and VMA are exponential in nature so slight deviation in accuracy relative to Tradingview is expected. 
''' 

def akc(candles: np.ndarray, per: int= 9, amode:str='AMA',smode:str='KER',scomod:str='Standard',gatethreshold:float=0.5,comp_knee:float=0.5,comp_intentsity:float=80.0,fastper:int=3,slowper:int=30,natr:float=3.0,sbp:float=61.8, source_type: str = "close", sequential: bool = False ) -> AKC:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    if smode == 'KER':
        erdata = er(source,source,per)
        oer = erdata 
    elif smode == 'TCER':
        tccdata = er(source,tcc(source,source,per),per)
        oer = tccdata 
    elif smode == 'CCIER':
        rollwin = talib.SMA(source,per)
        ccidata = er(source,pine_cci(source,per,rollwin),per)
        oer = ccidata 
    elif smode == 'RSIER':
        rsidata = er(source,pine_rsi(source,per),per)
        oer = rsidata 
    elif smode == 'KVOER':
        kvodata = er(source,kvo(source,candles,source),per)
        oer = kvodata 
    elif smode == 'MFIER':
        mfidata = er(source,mfi(source,source,per,candles),per)
        oer = mfidata
    elif smode == 'FTER':
        highval = same_length(candles,max_rolling1(source,per))
        lowval = same_length(candles,min_rolling1(source,per))
        fishdata = er(source,fish(source,source,per,highval,lowval),per)
        oer = fishdata
    else:
        rollwin = rolling_window(source,per)
        avg = talib.SMA(source,per)
        stdevdata = er(source,std2(source,rollwin,avg,per),per)
        oer = stdevdata 
    if scomod == 'Standard':
        sco = np.abs(oer)
    elif scomod == 'Inverse Fisher Normalize':
        ifish2 = ifish(source,oer,per,candles)
        xwinmax = same_length(candles,max_rolling1(ifish2,per))
        xwinmin = same_length(candles,min_rolling1(ifish2,per))
        sco = fs_norm(source,ifish2,per,xwinmax,xwinmin)  
    elif scomod == 'Max Min Normalize':
        xwinmax2 = same_length(candles,max_rolling1(oer,per))
        xwinmin2 = same_length(candles,min_rolling1(oer,per))
        sco = fs_norm(source,oer,per,xwinmax2,xwinmin2)
    elif scomod == 'Gate':
        sco = gate(source,oer,per,gatethreshold)
    else:
        sco = comp(source,oer,comp_knee,comp_intentsity)
    
    AMA = ama(source,source,per,fastper,slowper,sco)
    VMA = vma(source,source,per,sco)
    ma = AMA if amode=='AMA' else VMA
    tr = talib.TRANGE(candles[:,3],candles[:,4],candles[:,2])
    ATR = ama(source,tr,per,fastper,slowper,sco) if amode == 'AMA' else vma(source,tr,per,sco)
    hbandlg = ma + ATR*natr 
    hbandsm = ma + ATR*natr*(sbp/100)
    lbandsm = ma - ATR*natr*(sbp/100)
    lbandlg = ma - ATR*natr 
    macolor,barcolor = barcolor1(source,ma,hbandlg,hbandsm,lbandsm,lbandlg)
    if sequential:
        return AKC(ma,hbandlg,hbandsm,lbandlg,lbandsm,macolor,barcolor)
    else:
        return AKC(ma[-1], hbandlg[-1], hbandsm[-1], lbandlg[-1], lbandsm[-1], macolor[-1], barcolor[-1])

@njit 
def gate(source,oer,per,gatethreshold):
    sco = np.full_like(source,0)
    for i in range(source.shape[0]):
        sco[i] = 0 if np.abs(oer[i]) < gatethreshold else np.abs(oer[i])
    return sco 
    
@njit
def std2(source, rollwin,avg,per):
    std1 = np.full_like(source,0)
    for i in range(source.shape[0]):
        sum1 = 0.0
        sum2 = 0.0
        for j in range(per):
            sum1 = (source[i-j] + -avg[i])
            sum2 = sum2 + sum1 * sum1 
        std1[i] = np.sqrt(sum2 / per)
    return std1 
    
@njit
def er(source,x,t):	
    dist = np.full_like(source,0)
    signal = np.full_like(source,0)
    noise = np.full_like(source,0)
    er = np.full_like(source,0)
    for i in range(source.shape[0]):
        dist[i] = np.abs(x[i] - x[i-1])
        signal[i] = np.abs(x[i] - x[i-t])
        noise[i] = np.sum(dist[i-t+1:])
        er[i] = signal[i]/noise[i] if noise[i] != 0 else 1 
    return er 
    
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return as_strided(a, shape=shape, strides=strides)    

@njit
def tcc(source,series1,period):
    period = np.maximum(2,period)
    denominator = np.full_like(source,0)
    denominator1 = np.full_like(source,0)
    bars = np.full_like(source,0)
    for i in range(source.shape[0]):
        bars[i] = bars[i-1] + 1 
        x = 0.0
        y = 0.0
        ex = 0.0 
        ex2 = 0.0 
        exy = 0.0 
        ey2 = 0.0
        ey = 0.0
        for j in range(period): 
            x = source[i-j]
            y = bars[i-j]
            ex = ex + x
            ex2 = ex2 + x * x 
            exy = exy + x * y 
            ey2 = ey2 + y * y 
            ey = ey + y 
        denominator1[i] = (period * ex2 - ex * ex) * (period * ey2 - ey * ey)
        denominator[i] = 0 if denominator1[i] == 0 else (period * exy - ex * ey)/np.sqrt(denominator1[i])
    return denominator 
    
@njit
def kvo(source,candles,x):
    sv = np.full_like(source,0)
    kvo = np.full_like(source,0)
    zeros = np.full_like(source,0)
    for i in range(source.shape[0]):
        if (x[i] > x[i-1]):
            sv[i] = candles[:,5][i] 
        elif (x[i] < x[i-1]):
            sv[i] = -(candles[:,5][i])
        else:
            sv[i] = zeros[i] 
        kvo = (pine_ema(source,sv,34) - pine_ema(source,sv,55))
    return kvo 

@jit(nopython=True, error_model="numpy")
def mfi(source,x,t,candles):
    mf = np.full_like(source,0)
    premf = np.full_like(source,0)
    pmf = np.full_like(source,0)
    nmf = np.full_like(source,0)
    mfi = np.full_like(source,0)
    prenf = np.full_like(source,0)
    for i in range(0,source.shape[0]):
        mf[i] = x[i]*candles[:,5][i] 
        premf[i] = mf[i] if (x[i] > x[i-1]) else 0 
        pmf[i] = np.sum(premf[i-t+1:])
        prenf[i] = mf[i] if (x[i] < x[i-1]) else 0 
        nmf[i] = np.sum(prenf[i-t+1:])
        mfi[i] = 100*(pmf[i]/(pmf[i] + nmf[i]))
    return mfi
  
@jit(nopython=True, error_model="numpy")
def fish(source,x,t,highval,lowval):
    val1 = np.full_like(source,0)
    val2 = np.full_like(source,0)
    fish = np.full_like(source,0)
    for i in range(t,source.shape[0]):
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
def ifish(source,x,t,candles):
    val = np.full_like(source,0)
    wval1 = np.full_like(source,0.0)
    wval2 = np.full_like(source,0.0)
    ifish = np.full_like(source,0)
    for i in range(source.shape[0]):
        val[i] = 0.1*x[i]
        wval1 = pine_wma(source,val,t)
        wval2 = np.concatenate((np.full((candles.shape[0] - wval1.shape[0]), np.nan), wval1))
        ifish[i] = (np.exp(2*wval2[i]) - 1)/(np.exp(2*wval2[i])+1)
    return ifish 

@njit    
def fs_norm(source,x,t,xwinmax,xwinmin):
    res = np.full_like(source,0)
    for i in range(source.shape[0]):
        res[i] = (x[i] - xwinmin[i])/(xwinmax[i] - xwinmin[i]) if xwinmax[i] != xwinmin[i] else 0 
    return res 
    
@njit    
def comp(source,x,knee,intensity):
    res = np.full_like(source,0)
    for i in range(source.shape[0]):
        res[i] = x[i] - intensity*(x[i]-knee)/100 if x[i] > knee else x[i] 
    return res 

@njit   
def ama(source,x,t,f,s,o):
    fastalpha = 2/(f+1)
    slowalpha = 2/(s+1)
    sc = np.full_like(source,0)
    ama = np.full_like(source,0)
    for i in range(source.shape[0]):
        sc[i] = np.power(o[i] *(fastalpha - slowalpha) + slowalpha,2)
        ama[i] = x[i] if np.isnan(ama[i-1]) else ama[i-1] + sc[i]*(x[i] - ama[i-1])
    return ama 

@njit     
def vma(source,x,t,o):
    vma = np.full_like(source,0)
    for i in range(source.shape[0]):    
        vma[i] = x[i] if np.isnan(vma[i-1]) else (1 - (1/t)*o[i])*vma[i-1]+(1/t)*o[i]*x[i]
    return vma 

@njit
def barcolor1(source,ma,hbandlg,hbandsm,lbandsm,lbandlg):
    barcolor = np.full_like(source,0)
    macolor = np.full_like(source,0)
    for i in range(source.shape[0]):
        if (source[i] > ma[i]) and (source[i] > source[i-1]) and (source[i] < hbandsm[i]):
            barcolor[i] = 1.5
        elif (source[i] > ma[i]) and (source[i] > source[i-1]) and (source[i] >= hbandsm[i]):
            barcolor[i] = 2 
        elif (source[i] > ma[i]) and (source[i] < source[i-1]) and (source[i] < hbandsm[i]):
            barcolor[i] = 0.5 
        elif (source[i] > ma[i]) and (source[i] < source[i-1]) and (source[i] >= hbandsm[i]):
            barcolor[i] = 1
        elif (source[i] < ma[i]) and (source[i] < source[i-1]) and (source[i] > lbandsm[i]):
            barcolor[i] = -1.5 
        elif (source[i] < ma[i]) and (source[i] < source[i-1]) and (source[i] <= lbandsm[i]):
            barcolor[i] = -2
        elif (source[i] < ma[i]) and (source[i] > source[i-1]) and (source[i] > lbandsm[i]):
            barcolor[i] = -0.5 
        elif (source[i] < ma[i]) and (source[i] > source[i-1]) and (source[i] <= lbandsm[i]):
            barcolor[i] = -1
        else:
            barcolor[i] = 0 
        if ma[i] > ma[i-1]:
            macolor[i] = 1 
        elif ma[i] < ma[i-1]:
            macolor[i] = -1
        else:
            macolor[i] = 0 
            
    return macolor, barcolor 
    
 
@njit
def pine_ema(source1, source2, length):
    alpha = 2 / (length + 1)
    sum1 = np.full_like(source1,0)
    for i in range(source1.shape[0]):
        sum1[i] = np.mean(source2[-length:]) if np.isnan(sum1[i-1]) else alpha * source2[i] + (1 - alpha) * sum1[i-1]
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
def pine_cci(source,per,rollwin):
    mamean = np.full_like(source,0)
    cci = np.full_like(source,0)
    dev = np.full_like(source,0)
    for i in range(source.shape[0]):
        mamean = (rollwin)
        sum1 = 0.0
        val = 0.0
        for j in range(per):
            val = source[i-j]
            sum1 = sum1 + np.abs(val - mamean[i])
        dev[i] = sum1/per 
        cci[i] = (source[i] - mamean[i]) / (0.015 * dev[i])
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
    
    
    
    
