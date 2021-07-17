from jesse.helpers import get_candle_source, slice_candles, np_shift,same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from numpy.lib.stride_tricks import as_strided

REB = namedtuple('REB',['mband', 'hband', 'lband', 'barcolor','m_color','bg_color'])

'''
https://www.tradingview.com/script/pK0AIDt2-Resampling-Reverse-Engineering-Bands-DW/
Price Action function only accurate with twice the amount of preloaded candles: 240 -> 480. 
Only RRSI works with the Price Action function and not RCCI or Rstoch yet. 
StochRSI smoothing not working 
''' 

def reb(candles: np.ndarray, period: int= 14, source_type: str = "close",band_type:str='RStoch',rng_qty:float=1,rng_scale:str='Average Change',rng_per:int=14,smooth_range:bool=False,smooth_per:int=27,rsi_per:int=14,rsi_ht:int=70,rsi_lt:int=30,stoch_sper:int=1,stoch_per:int=14,stoch_ht:int=80,stoch_lt:int=20,cci_per:int=20,cci_ht:int=100,cci_lt:int=-100, s_off:int=0, sequential: bool = False ) -> REB:    
    candles = slice_candles(candles, sequential) if sequential else candles[-480:]
    source = get_candle_source(candles, source_type=source_type)
    ones = np.full_like(source,1)
    tr = talib.TRANGE(candles[:,3],candles[:,4],candles[:,2])
    new_sample3_1 = new_sample_p(source,candles,source,rng_scale,rng_qty,rng_per,smooth_range,smooth_per,s_off,tr)
    new_sample = ones if rng_qty == 0 else new_sample3_1
    if band_type == 'RRSI':
        rrsi_h = Cond_RRSI(source,candles,source,new_sample,rsi_per,rsi_ht)
        rrsi_l = Cond_RRSI(source,candles,source,new_sample,rsi_per,rsi_lt)
        rrsi_m = Cond_RRSI(source,candles,source,new_sample,rsi_per,50) 
        hband = rrsi_h 
        lband = rrsi_l
        mband = rrsi_m 
    elif band_type == 'RStoch':
        # high = candles[:,3]
        # low = candles[:,4]
        # H = same_length(candles,max_rolling1(high,stoch_per)) 
        # L = same_length(candles,min_rolling1(low,stoch_per))
        rstoch_h = RStoch(source,candles,source,stoch_sper,stoch_ht,new_sample,stoch_per)
        rstoch_l = RStoch(source,candles,source,stoch_sper,stoch_lt,new_sample,stoch_per) 
        rstoch_m = RStoch(source,candles,source,stoch_sper,50,new_sample,stoch_per) 
        hband = rstoch_h
        lband = rstoch_l 
        mband = rstoch_m
    else:
        valmean = same_length(candles,talib.SMA(source,cci_per))
        rcci_h = cond_rcci(source,candles,source,cci_per,cci_ht,new_sample,valmean) 
        rcci_l = cond_rcci(source,candles,source,cci_per,cci_lt,new_sample,valmean) 
        rcci_m = cond_rcci(source,candles,source,cci_per,0,new_sample,valmean) 
        hband = rcci_h 
        lband = rcci_l 
        mband = rcci_m 
    barcolor,m_color,bg_color = Values(source,candles,mband,hband,lband,new_sample)
    if sequential:
        return REB(mband,hband,lband,barcolor,m_color,bg_color)
    else:
        return REB(mband[-1],hband[-1],lband[-1],barcolor[-1],m_color[-1],bg_color[-1])
    
@njit  
def Values(source,candles,mband,hband,lband,new_sample):
    ds_src = np.full_like(source,0)
    barcolor = np.full_like(source,0)
    m_color = np.full_like(source,0)
    bg_color = np.full_like(source,0)
    for i in range(source.shape[0]):
        if new_sample[i] == 1:
            ds_src[i] = source[i] 
        else:
            ds_src[i] = ds_src[i-1] 
        if (source[i] > mband[i]) and (source[i] >= ds_src[i]):
            barcolor[i] = 2 if (source[i] > source[i-1]) else 1 
        elif (source[i] < mband[i]) and (source[i] <= ds_src[i]):
            barcolor[i] = -2 if source[i] < source[i-1] else -1 
        else:
            barcolor[i] = 0 
        if ds_src[i] > mband[i]:
            m_color[i] = 1 
        elif ds_src[i] < mband[i]:
            m_color[i] = -1
        else:
            m_color[i] = 0 
        if source[i] > hband[i]:
            bg_color[i] = 1 
        elif source[i] < lband[i]:
            bg_color[i] = -1 
        else:
            bg_color[i] = 0 
    return barcolor,m_color,bg_color    

@jit(nopython=True, error_model="numpy")    
def Cond_RRSI(source,candles,x,cond,n,v):
    vals = np.full_like(source,0)
    gain = np.full_like(source,0)
    loss = np.full_like(source,0)
    av_gain = np.full_like(source,0)
    av_loss = np.full_like(source,0)
    RS = np.full_like(source,0)
    RSI = np.full_like(source,0)
    gain_proj = np.full_like(source,0)
    loss_proj = np.full_like(source,0)
    dist = np.full_like(source,0)
    RRSI = np.full_like(source,0)
    for i in range(source.shape[0]):
        if cond[i] == 1:
            vals[i] = x[i] 
            gain[i] = np.abs(vals[i] - vals[i-1]) if vals[i] > vals[i-1] else 0
            loss[i] = np.abs(vals[i] - vals[i-1]) if vals[i] < vals[i-1] else 0 
            av_gain = pine_rma(source,gain,(n),cond) 
            av_loss = pine_rma(source,loss,(n),cond) 
            RS[i] = av_gain[i]/av_loss[i]
            RSI[i] = 100*(1 - (1/(1 + RS[i])))
            gain_proj[i] = ((v*(n-1)*av_loss[i])/(100 - v)) - (n - 1)*av_gain[i] if RSI[i] < v else 0
            loss_proj[i] = (((100 - v)*(n-1)*av_gain[i])/v) - (n - 1)*av_loss[i] if RSI[i] > v else 0 
            dist[i] = gain_proj[i] - loss_proj[i] 
            RRSI[i] = vals[i] + dist[i] 
        else:
            vals[i] = vals[i-1] 
            gain[i] = gain[i-1]
            loss[i] = loss[i-1]
            av_gain[i] = av_gain[i-1]
            av_loss[i] = av_loss[i-1]
            RS[i] = RS[i-1]
            RSI[i] = RSI[i-1]
            gain_proj[i] = gain_proj[i-1]
            loss_proj[i] = loss_proj[i-1]
            dist[i] = dist[i-1] 
            RRSI[i] = RRSI[i-1] 
    return RRSI    


@jit(nopython=True, error_model="numpy")  
def RStoch(source,candles,x,ns,v,cond,t):
    RStoch = np.full_like(source,0)
    H = np.full_like(source,0)
    L = np.full_like(source,0)
    for i in range(t,source.shape[0]):
        if cond[i] == 1:
            H[i] = np.amax(candles[i-(t-1):i+1,3])
            L[i] = np.amin(candles[i-(t-1):i+1,4])
            RStoch[i] = ((v*ns)*(H[i]-L[i])/100)+L[i] 
        else:
            H[i] = H[i-1] 
            L[i] = L[i-1] 
            RStoch[i] = RStoch[i-1] 
    return RStoch
    
@njit 
def cond_rcci(source,candles,x,n,v,cond,valmean):
    c = 0.015 
    m = np.full_like(source,0)
    vals = np.full_like(source,0)
    RCCI = np.full_like(source,0)
    for i in range(source.shape[0]):  
        if cond[i] == 1:
            vals[i] = x[i] 
            dvals = 0.0
            for j in range(n):
                dvals = dvals + np.abs(vals[i-j] - valmean[i])
            RCCI[i] = c*v*(dvals/n) + valmean[i]
        else:
            vals[i] = vals[i-1]
            RCCI[i] = RCCI[i-1] 
    return RCCI
      
@njit    
def new_sample_p(source,candles,x,scale,qty,t,smooth,st,o,tr):
    r1 = np.full_like(source,0)
    r2 = np.full_like(source,0)
    r = np.full_like(source,0)
    pa_line = np.full_like(source,0)
    new_sample_p1 = np.full_like(source,0)
    new_sample_p2 = np.full_like(source,0)
    closediff = np.full_like(source,0)
    ach_ = np.full_like(source,0)
    ach1 = np.full_like(source,0)
    rng_size = np.full_like(source,0)
    atr_ = np.full_like(source,0)
    ones = np.full_like(source,1)
    for i in range(source.shape[0]):    
        atr_[i] = qty*(pine_ema(source,tr,t,ones)[i])
        closediff[i] = np.abs(candles[:,2][i] - candles[:,2][i-1])
        ach1[i] = qty*(pine_ema(source,closediff,t,ones)[i])
        ach_[i] = ach1[i] 
        if scale=='% of Price':
            rng_size[i] = candles[:,2][i]*qty/100 
        elif scale == 'ATR':
            rng_size[i] = atr_[i]
        elif scale == 'Average Change':
            rng_size[i] = ach_[i]
        else:
            rng_size[i] = qty
        r1[i] = rng_size[i]
        r2[i] = pine_ema(source,r1,st,ones)[i]
        r[i] = r2[i] if smooth else r1[i] 
        pa_line[i] = pa_line[i-1]
        if np.abs(x[i] - pa_line[i]) >= r[i]:
            pa_line[i] = x[i]
        new_sample_p1[i] = 1 if pa_line[i] != pa_line[i-1] else 0 
        new_sample_p2[i] = 1 if new_sample_p1[i-o] == 1 else 0 
    return new_sample_p2 
 
@njit 
def pine_rma(source1, source2, length,cond):
    alpha = 1/length
    sum1 = np.full_like(source1,0)
    for i in range(source1.shape[0]):
        if cond[i] == 1 :
            sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1]
        else:
            sum1[i] = sum1[i-1] 
    return sum1 
      
@njit 
def pine_ema(source1, source2, length,cond):
    alpha = 2 / (length + 1)
    sum1 = np.full_like(source1,0)
    for i in range(source1.shape[0]):
        if cond[i] == 1 :
            sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1] 
        else:
            sum1[i] = sum1[i-1] 
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
        
