
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


#import tensorflow as tf
class PositionalEncoding_Layer():
    def __init__(self):
        pass
    def encode(self, inputs, max_len, d_model):
        pe = np.zeros([max_len, d_model], dtype=np.float32)
        position = np.expand_dims(inputs, axis=1)
        div_term = np.exp(np.arange(0.0, float(d_model), 2.0, dtype=np.float32) * -(np.log(1000.0) / float(d_model)))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = np.expand_dims(pe, axis=0)
        return pe
    # batch_size, seq_max_len, dim
    def __call__(self, inputs, max_len, inputs_dim):
        pe = self.encode(inputs, max_len, inputs_dim)
        return pe


def get_inputs_from_ret(ret, x):
    pe_layer = PositionalEncoding_Layer()
    info = ret[0]
    midnight, avg_15m_candle_range, last, last_time = info
    
    input_close_scaled = make_price_relative(last, midnight, avg_15m_candle_range)
    input_time_scaled = scale_timeofday(last_time)
    day = last_time[5]

    candles_m15 = ret[1][0]
    candles_scaled_m15 = []
    m15_sin_array = []
    for i in candles_m15:
        o = make_price_relative(i.o, midnight, avg_15m_candle_range)
        h = make_price_relative(i.h, midnight, avg_15m_candle_range)
        l = make_price_relative(i.l, midnight, avg_15m_candle_range)
        c = make_price_relative(i.c, midnight, avg_15m_candle_range)
        t = scale_timeofday(i.t)
        candles_scaled_m15.append([o,h,l,c,t])
        
        m15_sin_array.append([o,t])
        m15_sin_array.append([h,t])
        m15_sin_array.append([l,t])
        m15_sin_array.append([c,t])
        
    m15_sin_outputs = pe_layer([x[0] for x in m15_sin_array], len(m15_sin_array), 32)[0]
    t_info = []
    for i in range(int(len(m15_sin_outputs) / 4)):
        t = m15_sin_outputs[i][1]
        t_info.append([1,0,0,0,t])
        t_info.append([0,1,0,0,t])
        t_info.append([0,0,1,0,t])
        t_info.append([0,0,0,1,t])
    candles_scaled_m15 = np.concatenate([m15_sin_outputs, t_info],axis=1)

    candles_m5 = ret[1][1]
    candles_scaled_m5 = []
    
    m5_sin_array = []
    for i in candles_m5:
        o = make_price_relative(i.o, midnight, avg_15m_candle_range)
        h = make_price_relative(i.h, midnight, avg_15m_candle_range)
        l = make_price_relative(i.l, midnight, avg_15m_candle_range)
        c = make_price_relative(i.c, midnight, avg_15m_candle_range)
        t = scale_timeofday(i.t)
        candles_scaled_m5.append([o,h,l,c,t])
        
        m5_sin_array.append([o,t])
        m5_sin_array.append([h,t])
        m5_sin_array.append([l,t])
        m5_sin_array.append([c,t])
        
    m5_sin_outputs = pe_layer([x[0] for x in m5_sin_array], len(m5_sin_array), 32)[0]
    t_info = []
    for i in range(int(len(m5_sin_outputs) / 4)):
        t = m5_sin_outputs[i][1]
        t_info.append([1,0,0,0,t])
        t_info.append([0,1,0,0,t])
        t_info.append([0,0,1,0,t])
        t_info.append([0,0,0,1,t])
    candles_scaled_m5 = np.concatenate([m5_sin_outputs, t_info],axis=1)

    candles_m1 = ret[1][2]
    candles_scaled_m1 = []
    
    m1_sin_array = []
    for i in candles_m1:
        o = make_price_relative(i.o, midnight, avg_15m_candle_range)
        h = make_price_relative(i.h, midnight, avg_15m_candle_range)
        l = make_price_relative(i.l, midnight, avg_15m_candle_range)
        c = make_price_relative(i.c, midnight, avg_15m_candle_range)
        t = scale_timeofday(i.t)
        candles_scaled_m1.append([o,h,l,c,t])
        
        m1_sin_array.append([o,t])
        m1_sin_array.append([h,t])
        m1_sin_array.append([l,t])
        m1_sin_array.append([c,t])
        
    m1_sin_outputs = pe_layer([x[0] for x in m1_sin_array], len(m1_sin_array), 32)[0]
    t_info = []
    for i in range(int(len(m1_sin_outputs) / 4)):
        t = m1_sin_outputs[i][1]
        t_info.append([1,0,0,0,t])
        t_info.append([0,1,0,0,t])
        t_info.append([0,0,1,0,t])
        t_info.append([0,0,0,1,t])
    candles_scaled_m1 = np.concatenate([m1_sin_outputs, t_info],axis=1)
    
    return [
        input_close_scaled,
        input_time_scaled,
        day,
        np.array(candles_scaled_m1),
        np.array(candles_scaled_m5),
        np.array(candles_scaled_m15)
    ]
    
