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
    return (t[3]*60+t[4])+(t[5])*60*24

def get_inputs_from_ret(ret, x):
    
    info = ret[0]
    midnight, avg_15m_candle_range, last, last_time = info
    
    input_close_scaled = make_price_relative(last, midnight, avg_15m_candle_range)
    input_time_scaled = scale_timeofday(last_time)
    
    
    
    
    input_close_scaled, input_time_scaled
    
    active_pd_arrays_m1 = deque(maxlen = 256)
    active_pd_arrays_m5 = deque(maxlen = 256)
    active_pd_arrays_m15 = deque(maxlen = 256)
    active_pd_arrays_m60 = deque(maxlen = 512)
    active_pd_arrays_d1 = deque(maxlen = 512)
    
    actions_m1 = deque(maxlen = 256)
    for _ in range(256):
        actions_m1.append([0, 0, 0, 0, 0])

    actions_m5 = deque(maxlen = 256)
    for _ in range(256):
        actions_m5.append([0, 0, 0, 0, 0])
        
    actions_m15 = deque(maxlen = 256)
    for _ in range(256):
        actions_m15.append([0, 0, 0, 0, 0])
    
    for i in ret[1]:
        tf = i.tf
        t = scale_timeofday(i.time)
        pd_type = i.pda.type
        
        if pd_type == "BUYSIDE":
            pd_type = 0
            pd_price = i.pda.price
            if tf == "m1":
                active_pd_arrays_m1.append([t,pd_type,make_price_relative(pd_price, midnight, avg_15m_candle_range)])
            elif tf == "m5":
                active_pd_arrays_m5.append([t,pd_type,make_price_relative(pd_price, midnight, avg_15m_candle_range)])
            elif tf == "m15":
                active_pd_arrays_m15.append([t,pd_type,make_price_relative(pd_price, midnight, avg_15m_candle_range)])
            elif tf == "m60":
                active_pd_arrays_m60.append([t,pd_type,make_price_relative(pd_price, midnight, avg_15m_candle_range)])
            elif tf == "d1":
                active_pd_arrays_d1.append([t,pd_type,make_price_relative(pd_price, midnight, avg_15m_candle_range)])
            else:
                print("error in tf")
    
        if pd_type == "SELLSIDE":
            pd_type = 1
            pd_price = i.pda.price
            if tf == "m1":
                active_pd_arrays_m1.append([t,pd_type,make_price_relative(pd_price, midnight, avg_15m_candle_range)])
            elif tf == "m5":
                active_pd_arrays_m5.append([t,pd_type,make_price_relative(pd_price, midnight, avg_15m_candle_range)])
            elif tf == "m15":
                active_pd_arrays_m15.append([t,pd_type,make_price_relative(pd_price, midnight, avg_15m_candle_range)])
            elif tf == "m60":
                active_pd_arrays_m60.append([t,pd_type,make_price_relative(pd_price, midnight, avg_15m_candle_range)])
            elif tf == "d1":
                active_pd_arrays_d1.append([t,pd_type,make_price_relative(pd_price, midnight, avg_15m_candle_range)])
            else:
                print("error in tf")
    
        if pd_type == "BISI":
            pd_type1 = 3
            pd_type2 = 4
            pd_price1 = i.pda.fvg_high
            pd_price2 = i.pda.fvg_low
            if tf == "m1":
                active_pd_arrays_m1.append([t,pd_type1,make_price_relative(pd_price1, midnight, avg_15m_candle_range)])
                active_pd_arrays_m1.append([t,pd_type2,make_price_relative(pd_price2, midnight, avg_15m_candle_range)])
            elif tf == "m5":
                active_pd_arrays_m5.append([t,pd_type1,make_price_relative(pd_price1, midnight, avg_15m_candle_range)])
                active_pd_arrays_m5.append([t,pd_type2,make_price_relative(pd_price2, midnight, avg_15m_candle_range)])
            elif tf == "m15":
                active_pd_arrays_m15.append([t,pd_type1,make_price_relative(pd_price1, midnight, avg_15m_candle_range)])
                active_pd_arrays_m15.append([t,pd_type2,make_price_relative(pd_price2, midnight, avg_15m_candle_range)])
            elif tf == "m60":
                active_pd_arrays_m60.append([t,pd_type1,make_price_relative(pd_price1, midnight, avg_15m_candle_range)])
                active_pd_arrays_m60.append([t,pd_type2,make_price_relative(pd_price2, midnight, avg_15m_candle_range)])
            elif tf == "d1":
                active_pd_arrays_d1.append([t,pd_type1,make_price_relative(pd_price1, midnight, avg_15m_candle_range)])
                active_pd_arrays_d1.append([t,pd_type2,make_price_relative(pd_price2, midnight, avg_15m_candle_range)])
            else:
                print("error in tf")
    
        if pd_type == "SIBI":
            pd_type1 = 5
            pd_type2 = 6
            pd_price1 = i.pda.fvg_high
            pd_price2 = i.pda.fvg_low
            if tf == "m1":
                active_pd_arrays_m1.append([t,pd_type1,make_price_relative(pd_price1, midnight, avg_15m_candle_range)])
                active_pd_arrays_m1.append([t,pd_type2,make_price_relative(pd_price2, midnight, avg_15m_candle_range)])
            elif tf == "m5":
                active_pd_arrays_m5.append([t,pd_type1,make_price_relative(pd_price1, midnight, avg_15m_candle_range)])
                active_pd_arrays_m5.append([t,pd_type2,make_price_relative(pd_price2, midnight, avg_15m_candle_range)])
            elif tf == "m15":
                active_pd_arrays_m15.append([t,pd_type1,make_price_relative(pd_price1, midnight, avg_15m_candle_range)])
                active_pd_arrays_m15.append([t,pd_type2,make_price_relative(pd_price2, midnight, avg_15m_candle_range)])
            elif tf == "m60":
                active_pd_arrays_m60.append([t,pd_type1,make_price_relative(pd_price1, midnight, avg_15m_candle_range)])
                active_pd_arrays_m60.append([t,pd_type2,make_price_relative(pd_price2, midnight, avg_15m_candle_range)])
            elif tf == "d1":
                active_pd_arrays_d1.append([t,pd_type1,make_price_relative(pd_price1, midnight, avg_15m_candle_range)])
                active_pd_arrays_d1.append([t,pd_type2,make_price_relative(pd_price2, midnight, avg_15m_candle_range)])
            else:
                print("error in tf")
    
    
    for i in ret[2][0]:
        tp = i.type
        if tp == "BUYSIDE_TAKEN":
            tp = 0
        elif tp == "SELLSIDE_TAKEN":
            tp = 1
            
        elif tp == "BISI_HIGH_TOUCHED_FROM_INSIDE":
            tp = 2
        elif tp == "BISI_HIGH_TOUCHED_FROM_OUTSIDE":
            tp = 3
        elif tp == "BISI_HIGH_BROKEN_UPSIDE":
            tp = 4
        elif tp == "BISI_LOW_TOUCHED_FROM_INSIDE":
            tp = 5
        elif tp == "BISI_LOW_TOUCHED_FROM_OUTSIDE":
            tp = 6
        elif tp == "BISI_LOW_BROKEN_DOWNSIDE":
            tp = 7
            
        elif tp == "SIBI_HIGH_TOUCHED_FROM_INSIDE":
            tp = 8
        elif tp == "SIBI_HIGH_TOUCHED_FROM_OUTSIDE":
            tp = 9
        elif tp == "SIBI_HIGH_BROKEN_UPSIDE":
            tp = 10
        elif tp == "SIBI_LOW_TOUCHED_FROM_INSIDE":
            tp = 11
        elif tp == "SIBI_LOW_TOUCHED_FROM_OUTSIDE":
            tp = 12
        elif tp == "SIBI_LOW_BROKEN_DOWNSIDE":
            tp = 13

        elif tp == "SWING_HIGH_FORMED":
            tp = 14
        elif tp == "SWING_LOW_FORMED":
            tp = 15
        elif tp == "BISI_FORMED":
            tp = 16
        elif tp == "SIBI_FORMED":
            tp = 17
            
        else:print("error in action type")
    
        pd_formed_scaled = scale_timeofday(i.pd_array_formed_time)
        pd_array_tf = i.pd_array_tf
        if pd_array_tf == "m1":
            pd_array_tf = 0
        elif pd_array_tf == "m5":
            pd_array_tf = 1
        elif pd_array_tf == "m15":
            pd_array_tf = 2
        elif pd_array_tf == "m60":
            pd_array_tf = 3
        elif pd_array_tf == "d1":
            pd_array_tf = 4
        hit_price = make_price_relative(i.price, midnight, avg_15m_candle_range)
        hit_time = scale_timeofday(i.time)
    
        actions_m1.append([tp, pd_formed_scaled, pd_array_tf, hit_price, hit_time, 0])
        

    for i in ret[2][1]:
        tp = i.type
        if tp == "BUYSIDE_TAKEN":
            tp = 0
        elif tp == "SELLSIDE_TAKEN":
            tp = 1
            
        elif tp == "BISI_HIGH_TOUCHED_FROM_INSIDE":
            tp = 2
        elif tp == "BISI_HIGH_TOUCHED_FROM_OUTSIDE":
            tp = 3
        elif tp == "BISI_HIGH_BROKEN_UPSIDE":
            tp = 4
        elif tp == "BISI_LOW_TOUCHED_FROM_INSIDE":
            tp = 5
        elif tp == "BISI_LOW_TOUCHED_FROM_OUTSIDE":
            tp = 6
        elif tp == "BISI_LOW_BROKEN_DOWNSIDE":
            tp = 7
            
        elif tp == "SIBI_HIGH_TOUCHED_FROM_INSIDE":
            tp = 8
        elif tp == "SIBI_HIGH_TOUCHED_FROM_OUTSIDE":
            tp = 9
        elif tp == "SIBI_HIGH_BROKEN_UPSIDE":
            tp = 10
        elif tp == "SIBI_LOW_TOUCHED_FROM_INSIDE":
            tp = 11
        elif tp == "SIBI_LOW_TOUCHED_FROM_OUTSIDE":
            tp = 12
        elif tp == "SIBI_LOW_BROKEN_DOWNSIDE":
            tp = 13

        elif tp == "SWING_HIGH_FORMED":
            tp = 14
        elif tp == "SWING_LOW_FORMED":
            tp = 15
        elif tp == "BISI_FORMED":
            tp = 16
        elif tp == "SIBI_FORMED":
            tp = 17
            
        else:print("error in action type")
    
        pd_formed_scaled = scale_timeofday(i.pd_array_formed_time)
        pd_array_tf = i.pd_array_tf
        if pd_array_tf == "m1":
            pd_array_tf = 0
            print("error in action_m5 tf")
        elif pd_array_tf == "m5":
            pd_array_tf = 1
        elif pd_array_tf == "m15":
            pd_array_tf = 2
        elif pd_array_tf == "m60":
            pd_array_tf = 3
        elif pd_array_tf == "d1":
            pd_array_tf = 4
        hit_price = make_price_relative(i.price, midnight, avg_15m_candle_range)
        hit_time = scale_timeofday(i.time)
    
        actions_m5.append([tp, pd_formed_scaled, pd_array_tf, hit_price, hit_time, 1])
        
    for i in ret[2][2]:
        tp = i.type
        if tp == "BUYSIDE_TAKEN":
            tp = 0
        elif tp == "SELLSIDE_TAKEN":
            tp = 1
            
        elif tp == "BISI_HIGH_TOUCHED_FROM_INSIDE":
            tp = 2
        elif tp == "BISI_HIGH_TOUCHED_FROM_OUTSIDE":
            tp = 3
        elif tp == "BISI_HIGH_BROKEN_UPSIDE":
            tp = 4
        elif tp == "BISI_LOW_TOUCHED_FROM_INSIDE":
            tp = 5
        elif tp == "BISI_LOW_TOUCHED_FROM_OUTSIDE":
            tp = 6
        elif tp == "BISI_LOW_BROKEN_DOWNSIDE":
            tp = 7
            
        elif tp == "SIBI_HIGH_TOUCHED_FROM_INSIDE":
            tp = 8
        elif tp == "SIBI_HIGH_TOUCHED_FROM_OUTSIDE":
            tp = 9
        elif tp == "SIBI_HIGH_BROKEN_UPSIDE":
            tp = 10
        elif tp == "SIBI_LOW_TOUCHED_FROM_INSIDE":
            tp = 11
        elif tp == "SIBI_LOW_TOUCHED_FROM_OUTSIDE":
            tp = 12
        elif tp == "SIBI_LOW_BROKEN_DOWNSIDE":
            tp = 13

        elif tp == "SWING_HIGH_FORMED":
            tp = 14
        elif tp == "SWING_LOW_FORMED":
            tp = 15
        elif tp == "BISI_FORMED":
            tp = 16
        elif tp == "SIBI_FORMED":
            tp = 17
            
        else:print("error in action type")
    
        pd_formed_scaled = scale_timeofday(i.pd_array_formed_time)
        pd_array_tf = i.pd_array_tf
        if pd_array_tf == "m1":
            pd_array_tf = 0
            print("error in action_m15 tf 1")
        elif pd_array_tf == "m5":
            pd_array_tf = 1
            print("error in action_m15 tf 5")
        elif pd_array_tf == "m15":
            pd_array_tf = 2
        elif pd_array_tf == "m60":
            pd_array_tf = 3
        elif pd_array_tf == "d1":
            pd_array_tf = 4
        hit_price = make_price_relative(i.price, midnight, avg_15m_candle_range)
        hit_time = scale_timeofday(i.time)
    
        actions_m15.append([tp, pd_formed_scaled, pd_array_tf, hit_price, hit_time, 2])
        
           
    
    def get_closest_pda(last_close, pd_arrays):
            
            pda_below = []
            pda_above = []
            for i in pd_arrays:
                d = i[2] - last_close
                if d > 0:
                    pda_above.append(i)
                if d < 0:
                    pda_below.append(i)
            
            pda_below = sorted(pda_below, key=lambda x:x[2], reverse = True)[0:5]
            pda_above = sorted(pda_above, key=lambda x:x[2])[0:5]
            while True:
                if len(pda_below) == 5:
                    break
                #print(pda_below)
                pda_below.append([0,0,0])
                #print("?")
                #print(pda_below)
            while True:
                if len(pda_above) == 5:
                    break
                #print(pda_above)
                pda_above.append([0,0,0])
                #print("#")
                #print(pda_above)
            return np.array(pda_above), np.array(pda_below)

        
    
    return [
        input_close_scaled,
        input_time_scaled,
        get_closest_pda(input_close_scaled,active_pd_arrays_m1),
        get_closest_pda(input_close_scaled,active_pd_arrays_m5), 
        get_closest_pda(input_close_scaled,active_pd_arrays_m15), 
        get_closest_pda(input_close_scaled,active_pd_arrays_m60), 
        get_closest_pda(input_close_scaled,active_pd_arrays_d1), 
        np.array(actions_m1),
        np.array(actions_m5),
        np.array(actions_m15)
    ]