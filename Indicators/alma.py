from typing import Union
import numpy as np
from jesse.helpers import get_config
from jesse.helpers import get_candle_source

def strided_axis0(a, L):
    # Store the shape and strides info
    shp = a.shape
    s  = a.strides

    # Compute length of output array along the first axis
    nd0 = shp[0]-L+1

    # Setup shape and strides for use with np.lib.stride_tricks.as_strided
    # and get (n+1) dim output array
    shp_in = (nd0,L)+shp[1:]
    strd_in = (s[0],) + s
    return np.lib.stride_tricks.as_strided(a, shape=shp_in, strides=strd_in)
    
def alma(candles: np.ndarray, period: int = 9, offset: float = 0.85, sigma: int = 6, source_type: str = "close", sequential: bool = False) -> Union[
    float, np.ndarray]:    
    warmup_candles_num = get_config('env.data.warmup_candles_num', 240)
    if not sequential and len(candles) > warmup_candles_num:
        candles = candles[-warmup_candles_num:]
        
    source = get_candle_source(candles, source_type = source_type)
    alma = np.zeros(source.shape)
    wtd_sum = np.zeros(source.shape)
    asize = period - 1
    m = offset * asize
    s = period / sigma
    dss = 2 * s * s
    wtds = np.exp(-(np.arange(period) - m)**2/dss)
    source3D = strided_axis0(source,len(wtds))
    alma = np.zeros(alma.shape)
    alma[period-1:] = np.tensordot(source3D,wtds,axes=((1),(0)))[:]
    alma /= wtds.sum()

    if sequential: 
        return alma
    else: 
        return alma[-3:-1] 

