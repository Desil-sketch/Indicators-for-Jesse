from jesse.helpers import get_candle_source, slice_candles 
import talib		
import numpy as np 
from typing import Union
import math 
from numba import njit 

def ef(candles: np.ndarray, lp_per: int = 10, hp_per: int = 30, f_type: str = "Ehlers", normalize: bool = False, source_type: str = "close", sequential: bool = False) ->  Union[
    float, np.ndarray]:
	# added to definition : use_comp: bool = False, comp_intensity: float = 90.0,
    """
    https://www.tradingview.com/script/kPe86Nbc-Roofing-Filter-DW/
    compression function not working 
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    if f_type == "Ehlers":
        roof = erf( source, hp_per, lp_per)
    elif f_type == "Gaussian":
        roof = grf( source, hp_per, lp_per)
    elif f_type == "Butterworth":
        roof = brf( source, hp_per, lp_per)
        
    rms = RMS(source, roof, np.round((hp_per + lp_per)/2))
    if roof[-1] > 0:
        norm_roof =  roof/rms
    elif roof[-1] < 0: 
        norm_roof = -np.abs(roof)/rms
    else: 
        norm_roof = 0
    if normalize: 
        filt = norm_roof 
    else:
        filt = roof
        
    if sequential: 
        return filt
    else:    
        return filt[-1]
#jesse backtest  '2021-01-03' '2021-03-02'

@njit
def grf(x,t_hp, t_lp): 
    beta1 = (1 - np.cos(4*np.arcsin(1)/t_hp))/(np.sqrt(2)-1)
    alpha1 = -beta1 + np.sqrt(np.power(beta1,2) + 2*beta1)
    beta2 = (1 - np.cos(4*np.arcsin(1)/t_lp))/(np.sqrt(2)-1)
    alpha2 = -(beta2) + np.sqrt(np.power(beta2,2) + 2*beta2)
    ghp = np.zeros_like(x)
    grf = np.zeros_like(x)
    for i in range(0,x.shape[0]):  
        ghp[i] = (1 - np.power(alpha1, 2))*x[i] + 2*(1-alpha1) * (ghp[i-1] - x[i-1]) + np.power(1 - alpha1, 2) * (x[i-2] - ghp[i-2])
        grf[i] = np.power(alpha2, 2)*ghp[i] + 2*(1 - alpha2)*grf[i-1] - np.power(1 - alpha2, 2)* grf[i-2]
    return grf

@njit
def erf( x, t_hp, t_lp): 
    omega1 = 4*np.arcsin(1)/t_hp
    omega2 = 4*np.arcsin(1)/t_lp
    alpha = (np.cos((np.sqrt(2)/2)*omega1) + np.sin((np.sqrt(2)/2)*omega1) - 1)/np.cos((np.sqrt(2)/2)*omega1)
    hp = np.zeros_like(x)
    erf = np.zeros_like(x)
    for i in range(0,x.shape[0]):
        hp[i] = np.power(1 - alpha/2, 2)*(x[i] - 2*x[i-1] + x[i-2]) + 2*(1 - alpha) * (hp[i-1]) - np.power(1 - alpha,2) * (hp[i-2])
    a1 = np.exp(-np.sqrt(2)*2*np.arcsin(1)/t_lp)
    b1 = 2*a1*np.cos((np.sqrt(2)/2)*omega2)
    c2 = b1 
    c3 = -np.power(a1,2)
    c1 = 1 - c2 - c3 
    for i in range(x.shape[0]): 
        erf[i] = c1*hp[i]+c2*erf[i-1] + c3*erf[i-2]
    return erf

@njit
def brf(x, t_hp, t_lp):
    a1   = np.exp(-np.sqrt(2)*2*np.arcsin(1)/t_hp)
    b1   = 2*a1*np.cos(np.sqrt(2)*2*np.arcsin(1)/t_hp)
    c1   = np.power(a1, 2)
    d1   = ((1 - b1 + c1)/4)
    a2   = np.exp(-np.sqrt(2)*2*np.arcsin(1)/t_lp)
    b2   = 2*a2*np.cos(np.sqrt(2)*2*np.arcsin(1)/t_lp)
    c2   = np.power(a2, 2)
    d2   = ((1 - b2 + c2)/4)
    bhp = np.zeros_like(x)
    brf = np.zeros_like(x)
    for i in range(x.shape[0]): 
        bhp[i] = b1 * bhp[i-1] - c1*bhp[i-2] + (1 - d1)*x[i] - (b1 + 2*d1) * x[i-1] + (c1 - d1)*x[i-2]
        brf[i] = b2 * brf[i-1] - c2*brf[i-2] + d2*(bhp[i] + 2*bhp[i-1] + bhp[i-2])
    return brf 
    
def RMS(source,x,t):
    rms = np.full_like(source, 0)
    rms = np.sqrt(talib.SMA(np.power(x,2),t))
    return rms
    

    
    
