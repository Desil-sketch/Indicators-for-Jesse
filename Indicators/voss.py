from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length, get_config
import numpy as np
from collections import namedtuple
from numba import njit
import talib 
from typing import Union

VossFilter = namedtuple('VossFilter', ['voss', 'bpf'])

def vpf(candles: np.ndarray, period: int = 20, bandwidth:float = 0.25, bars_prediction:int=3, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    voss,bpf = bpf_filt(source,source,period,bandwidth, bars_prediction)
    if sequential: 
        return VossFilter(voss,bpf)
    else:    
        return VossFilter(voss[-1],bpf[-1])
     
@njit
def bpf_filt(source,series,period,bandwidth, bars_prediction):
    PIx2 = 4.0 * np.arcsin(1.0)
    alpha = PIx2 / period
    gamma = np.cos(alpha*bandwidth)
    delta = 1.0/gamma - np.sqrt(1.0 / np.power(gamma, 2.0) -1.0)
    bandPass = np.full_like(source,0)
    order = np.int(3.0 * bars_prediction)
    voss = np.full_like(source,0)
    for i in range(source.shape[0]):
        bandPass[i] = (1.0 - delta) * 0.5 * (source[i] - source[i-2]) + np.cos(alpha) * (1.0 + delta) * bandPass[i-1] - delta * bandPass[i-2]
        E = 0
        for k in range(order):
            E = E + ((1 + k)/(order)) * voss[i - (order - k)] 
        voss[i] = (0.5 * (3.0 + order)) * bandPass[i] - E
    return voss,bandPass



