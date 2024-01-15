
import pickle
from collections import deque
import numpy as np

def Load(file):
    print("loading",file)
    f = open(file, "rb")
    obj = pickle.load(f)
    f.close()
    return obj


def save(_object, file):
    filehandler = open(file, 'wb') 
    pickle.dump(_object, filehandler)




def make_price_relative(price, nullprice, scale):
    return (price - nullprice) / max(scale, 0.25)

def scale_timeofday(t):
    return (t[3]*60+t[4])#+(t[5])*60*24

def get_inputs_from_ret(ret, x):
    
    info = ret[0]
    midnight, avg_15m_candle_range, last, last_time = info
    
    input_close_scaled = make_price_relative(last, midnight, avg_15m_candle_range)
    input_time_scaled = scale_timeofday(last_time)
    day = last_time[5]

    candles_m15 = ret[1][0]
    candles_scaled_m15 = []
    for i in candles_m15:
        o = make_price_relative(i.o, midnight, avg_15m_candle_range)
        h = make_price_relative(i.h, midnight, avg_15m_candle_range)
        l = make_price_relative(i.l, midnight, avg_15m_candle_range)
        c = make_price_relative(i.c, midnight, avg_15m_candle_range)
        t = scale_timeofday(i.t)
        candles_scaled_m15.append([o,h,l,c,t])

    candles_m5 = ret[1][1]
    candles_scaled_m5 = []
    for i in candles_m5:
        o = make_price_relative(i.o, midnight, avg_15m_candle_range)
        h = make_price_relative(i.h, midnight, avg_15m_candle_range)
        l = make_price_relative(i.l, midnight, avg_15m_candle_range)
        c = make_price_relative(i.c, midnight, avg_15m_candle_range)
        t = scale_timeofday(i.t)
        candles_scaled_m5.append([o,h,l,c,t])

    candles_m1 = ret[1][2]
    candles_scaled_m1 = []
    for i in candles_m1:
        o = make_price_relative(i.o, midnight, avg_15m_candle_range)
        h = make_price_relative(i.h, midnight, avg_15m_candle_range)
        l = make_price_relative(i.l, midnight, avg_15m_candle_range)
        c = make_price_relative(i.c, midnight, avg_15m_candle_range)
        t = scale_timeofday(i.t)
        candles_scaled_m1.append([o,h,l,c,t])
        
    
    return [
        input_close_scaled,
        input_time_scaled,
        day,
        np.array(candles_scaled_m1),
        np.array(candles_scaled_m5),
        np.array(candles_scaled_m15)
    ]