from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config


def jma(candles: np.ndarray, period: int= 14,phase: int = 50, power: float= 2,  source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    jma = jma_fast(source, period, phase, power)
    if sequential:
        return jma
    else:
        return jma[-2:]


@njit
def jma_fast(source, period, phase, power):
    e0 =   np.full_like(source, 0)
    e1 =  np.full_like(source, 0)
    e2 =   np.full_like(source, 0)
    jma =   np.full_like(source, 0)
    phaseRatio = 0.0
    beta = 0.45 * (period - 1) / (0.45 * (period -1) + 2) 
    alpha = np.power(beta,power)
    if phase < -100: 
        phaseRatio = 0.5
    elif phase > 100: 
        phaseRatio = 2.5
    else: 
        phaseRatio = phase / 100 + 1.5
    for i in range(source.shape[0]):
        e0[i] = (1 - alpha) * source[i] + alpha * e0[i-1]
        e1[i] = (source[i] - e0[i]) * (1 - beta) + beta * e1[i-1] 
        e2[i] = (e0[i] + phaseRatio * e1[i] - jma[i-1]) * np.power(1 - alpha, 2) + np.power(alpha, 2) * e2[i-1] 
        jma[i] = e2[i] + jma[i-1]
    return jma
