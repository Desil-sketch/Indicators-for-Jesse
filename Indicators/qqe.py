from jesse.helpers import get_candle_source, slice_candles 
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

QQE = namedtuple("qqe", ["QQF", "QQS","QUP","QDN"]) 
"""
https://www.tradingview.com/script/0vn4HZ7O-Quantitative-Qualitative-Estimation-QQE/
"""

def qqe(candles: np.ndarray, length: int = 14, SSF: int = 5, source_type: str = "close", sequential: bool = False) -> QQE:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type) 
    wwalpha = 1/length
    TR = np.full_like(source, np.nan)
    RSII = np.empty_like(source)
    WWMA = np.empty_like(source)
    ATRRSI = np.empty_like(source)
    QUP = np.empty_like(source)
    QDN = np.empty_like(source)
    QQES = np.empty_like(source)
    QQF = np.empty_like(source)
    QQS = np.empty_like(source)
    QQEF = talib.EMA((fast_rsi(source,  length)),SSF)
    for i in range(source.shape[0]):
        RSII = talib.EMA((fast_rsi(source,length)),SSF)
        TR[i] = np.abs(RSII[i] - RSII[i-1])
        WWMA[i] = wwalpha*np.nan_to_num(TR[i]) + (1-wwalpha)*WWMA[i-1]
        ATRRSI[i] = wwalpha*WWMA[i] + (1-wwalpha)*(ATRRSI[i-1])
        QUP[i] = QQEF[i]+ATRRSI[i]*4.236
        QDN[i] = QQEF[i]-ATRRSI[i]*4.236
        if QUP[i] < QQES[i-1]:
            QQES[i] = QUP[i]
        elif QQEF[i] > QQES[i-1] and QQEF[i-1] < QQES[i-1]:
            QQES[i] = QDN[i]
        elif QDN[i] > QQES[i-1]:
            QQES[i] = QDN[i]
        elif QQEF[i] < QQES[i-1] and QQEF[i-1] > QQES[i-1]:
            QQES[i] = QUP[i]
        else:
            QQES[i] = QQES[i-1]
        QQS[i] = QQES[i] - 50
        QQF[i] = QQEF[i] - 50
    
    if sequential: 
        return QQE(QQF,QQS, QUP,QDN)
    else:    
        return QQE(QQF[-1],QQS[-1],QUP[-1],QDN[-1])     
        
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
        rs[i] = (sumation1[i])/(sumation2[i])
        res[i] = 100 - 100 / ( 1 + rs[i])
    return res
    
