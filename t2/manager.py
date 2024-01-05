import numpy as np
import copy
from collections import deque
from datetime import datetime

class candle_class:
    def __init__(self, o=0,h=0,l=0,c=0,t=0):
        self.o=o
        self.h=h
        self.l=l
        self.c=c
        self.t=t

class fvg:
    def __init__(self, bisi_sibi, fvg1, fvg2):
        self.fvg_high = max(fvg1,fvg2)
        self.fvg_low = min(fvg1,fvg2)
        self.ce = (self.fvg_high + self.fvg_low)/2
        self.q1 = (self.fvg_high - self.fvg_low)/4+self.fvg_low
        self.q3 = (self.fvg_high - self.fvg_low)/4*3+self.fvg_low
        self.type = bisi_sibi.upper()

class liquidity:
    def __init__(self, bs_ss, price):
        self.type = bs_ss.upper()
        self.price = price
    

class pd_array:
    def __init__(self, timeframe, pda, pd_time):
        self.pda = pda
        self.time = pd_time
        self.tf = timeframe

class action:
    def __init__(self, action_type, price, action_time, pd_array_tf, pd_array_formed_time):
        self.type = action_type
        self.price = price
        self.time = action_time
        self.pd_array_tf = pd_array_tf
        self.pd_array_formed_time = pd_array_formed_time

def MHDMoY_to_minutes(m,h,d,mo,y):
        
    # Create a datetime object
    dt = datetime(y, mo, d, h, m)
    
    # Create a reference datetime object (here, the start of the year 2000)
    ref_dt = datetime(2000, 1, 1)
    
    # Calculate the difference between the two datetime objects
    delta = dt - ref_dt
    
    # Convert the difference into minutes
    total_minutes = delta.total_seconds() // 60

    return total_minutes

class manager:
    maxlen = 100
    def __init__(self):
        self.m1_candles = deque(maxlen = self.maxlen)
        self.m5_candles = deque(maxlen = self.maxlen)
        self.m15_candles = deque(maxlen = self.maxlen)
        self.m60_candles = deque(maxlen = self.maxlen)
        self.d1_candles = deque(maxlen = self.maxlen)

        self.action_history_m1 = deque(maxlen = 256)
        self.action_history_m5 = deque(maxlen = 256)
        self.action_history_m15 = deque(maxlen = 256)
        
        self.nymidnight_price = 0
        pass

    
    def push_m1_candle(self, __candle, scan= True):
        candle = copy.deepcopy(__candle)
        candle.t = [int(i) for i in  candle.t.split(":")]
        candle_minute = candle.t[4]
        candle_hour = candle.t[3]
        if candle_minute == 0 and candle_hour == 0 or self.nymidnight_price == 0:
            self.nymidnight_price = candle.o

        update_action_m1 = True
        update_action_m5 = False
        update_action_m15 = False
        
        self.m1_candles.append(candle)
        self.detect_pd_arrays("m1")
        
        

        # m5 candles
        if len(self.m5_candles) > 0:
            last_minute = self.m5_candles[-1].t[4]
            last_hour = self.m5_candles[-1].t[3]
            minute_rounded = int(candle_minute/5) * 5
            if minute_rounded != last_minute or candle_hour != last_hour:
                self.detect_pd_arrays("m5")
                update_action_m5 = True
                c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.m5_candles.append(c)
            else:
                self.m5_candles[-1].c = candle.c
                self.m5_candles[-1].h = max(candle.h, self.m5_candles[-1].h)
                self.m5_candles[-1].l = min(candle.l, self.m5_candles[-1].l)
        else:
            self.detect_pd_arrays("m5")
            update_action_m5 = True
            c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.m5_candles.append(c)
            

       # m15 candles
        if len(self.m15_candles) > 0:
            last_minute = self.m15_candles[-1].t[4]
            last_hour = self.m15_candles[-1].t[3]
            minute_rounded = int(candle_minute/15) * 15
            if minute_rounded != last_minute or candle_hour != last_hour:
                self.detect_pd_arrays("m15")
                update_action_m15 = True
                c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.m15_candles.append(c)
            else:
                self.m15_candles[-1].c = candle.c
                self.m15_candles[-1].h = max(candle.h, self.m15_candles[-1].h)
                self.m15_candles[-1].l = min(candle.l, self.m15_candles[-1].l)
        else:
            self.detect_pd_arrays("m15")
            update_action_m15 = True
            c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.m15_candles.append(c)

        # m60 candles
        if len(self.m60_candles) > 0:
            last_minute = self.m60_candles[-1].t[4]
            last_hour = self.m60_candles[-1].t[3]
            
            if candle_hour != last_hour:
                self.detect_pd_arrays("m60")
                c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.m60_candles.append(c)
            else:
                self.m60_candles[-1].c = candle.c
                self.m60_candles[-1].h = max(candle.h, self.m60_candles[-1].h)
                self.m60_candles[-1].l = min(candle.l, self.m60_candles[-1].l)
        else:
            self.detect_pd_arrays("m60")
            c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.m60_candles.append(c)


       # d1 candles
        if len(self.d1_candles) > 0:
            last_candle_hour = self.m1_candles[-2].t[3]
            #print(candle_hour)
            if candle_hour != last_candle_hour and candle_hour == 18:
                self.detect_pd_arrays("d1")
                c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.d1_candles.append(c)
            else:
                self.d1_candles[-1].c = candle.c
                self.d1_candles[-1].h = max(candle.h, self.d1_candles[-1].h)
                self.d1_candles[-1].l = min(candle.l, self.d1_candles[-1].l)
        else:
            self.detect_pd_arrays("d1")
            c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.d1_candles.append(c)



        if not scan:return

        self.pd_arrays = self.m1_pda+self.m5_pda+self.m15_pda+self.m60_pda+self.d1_pda
        self.pd_arrays_expired_indicies = []
        
        if update_action_m1: # always True
            self.compute_action_history("m1")
        if update_action_m5:
            self.compute_action_history("m5")
        if update_action_m15:
            self.compute_action_history("m15")
            
        
        expired_set = set(self.pd_arrays_expired_indicies)
        active_pd_arrays = [ i for index, i in enumerate(self.pd_arrays) if not index in expired_set]
        
        avg_candle_range = np.mean([x.h-x.l for x in self.m15_candles])
        info = [self.nymidnight_price, avg_candle_range, self.m1_candles[-1].c, self.m1_candles[-1].t]
        
        return info, active_pd_arrays, [self.action_history_m1, self.action_history_m5, self.action_history_m15]

    def detect_pd_arrays(self, tf):
        if tf == "m1":
            self.m1_pda = deque(maxlen = self.maxlen)
            if len(self.m1_candles) > 5:
                for i in range(1,len(self.m1_candles)-1):
                    if self.m1_candles[i].h > self.m1_candles[i-1].h and self.m1_candles[i].h > self.m1_candles[i+1].h:
                        x = liquidity("BUYSIDE", self.m1_candles[i].h)
                        pda = pd_array("m1", x, self.m1_candles[i].t)
                        self.m1_pda.append(pda)
                    if self.m1_candles[i].l < self.m1_candles[i-1].l and self.m1_candles[i].l < self.m1_candles[i+1].l:
                        x = liquidity("SELLSIDE", self.m1_candles[i].l)
                        pda = pd_array("m1", x, self.m1_candles[i].t)
                        self.m1_pda.append(pda)
    
                    if self.m1_candles[i-1].h < self.m1_candles[i+1].l:
                        x = fvg("BISI", self.m1_candles[i-1].h, self.m1_candles[i+1].l)
                        pda = pd_array("m1", x, self.m1_candles[i].t)
                        self.m1_pda.append(pda)
    
                    if self.m1_candles[i-1].l > self.m1_candles[i+1].h:
                        x = fvg("SIBI", self.m1_candles[i-1].l, self.m1_candles[i+1].h)
                        pda = pd_array("m1", x, self.m1_candles[i].t)
                        self.m1_pda.append(pda)
        if tf =="m5":
            self.m5_pda = deque(maxlen = self.maxlen)
            if len(self.m5_candles) > 5:
                for i in range(1,len(self.m5_candles)-1):
                    if self.m5_candles[i].h > self.m5_candles[i-1].h and self.m5_candles[i].h > self.m5_candles[i+1].h:
                        x = liquidity("BUYSIDE", self.m5_candles[i].h)
                        pda = pd_array("m5", x, self.m5_candles[i].t)
                        self.m5_pda.append(pda)
                    if self.m5_candles[i].l < self.m5_candles[i-1].l and self.m5_candles[i].l < self.m5_candles[i+1].l:
                        x = liquidity("SELLSIDE", self.m5_candles[i].l)
                        pda = pd_array("m5", x, self.m5_candles[i].t)
                        self.m5_pda.append(pda)
    
                    if self.m5_candles[i-1].h < self.m5_candles[i+1].l:
                        x = fvg("BISI", self.m5_candles[i-1].h, self.m5_candles[i+1].l)
                        pda = pd_array("m5", x, self.m5_candles[i].t)
                        self.m5_pda.append(pda)
    
                    if self.m5_candles[i-1].l > self.m5_candles[i+1].h:
                        x = fvg("SIBI", self.m5_candles[i-1].l, self.m5_candles[i+1].h)
                        pda = pd_array("m5", x, self.m5_candles[i].t)
                        self.m5_pda.append(pda)
        if tf == "m15":
            self.m15_pda = deque(maxlen = self.maxlen)
            if len(self.m15_candles) > 5:
                for i in range(1,len(self.m15_candles)-1):
                    if self.m15_candles[i].h > self.m15_candles[i-1].h and self.m15_candles[i].h > self.m15_candles[i+1].h:
                        x = liquidity("BUYSIDE", self.m15_candles[i].h)
                        pda = pd_array("m15", x, self.m15_candles[i].t)
                        self.m15_pda.append(pda)
                    if self.m15_candles[i].l < self.m15_candles[i-1].l and self.m15_candles[i].l < self.m15_candles[i+1].l:
                        x = liquidity("SELLSIDE", self.m15_candles[i].l)
                        pda = pd_array("m15", x, self.m15_candles[i].t)
                        self.m15_pda.append(pda)
    
                    if self.m15_candles[i-1].h < self.m15_candles[i+1].l:
                        x = fvg("BISI", self.m15_candles[i-1].h, self.m15_candles[i+1].l)
                        pda = pd_array("m15", x, self.m15_candles[i].t)
                        self.m15_pda.append(pda)
    
                    if self.m15_candles[i-1].l > self.m15_candles[i+1].h:
                        x = fvg("SIBI", self.m15_candles[i-1].l, self.m15_candles[i+1].h)
                        pda = pd_array("m15", x, self.m15_candles[i].t)
                        self.m15_pda.append(pda)
        if tf == "m60":
            self.m60_pda = deque(maxlen = self.maxlen)
            if len(self.m60_candles) > 5:
                for i in range(1,len(self.m60_candles)-1):
                    if self.m60_candles[i].h > self.m60_candles[i-1].h and self.m60_candles[i].h > self.m60_candles[i+1].h:
                        x = liquidity("BUYSIDE", self.m60_candles[i].h)
                        pda = pd_array("m60", x, self.m60_candles[i].t)
                        self.m60_pda.append(pda)
                    if self.m60_candles[i].l < self.m60_candles[i-1].l and self.m60_candles[i].l < self.m60_candles[i+1].l:
                        x = liquidity("SELLSIDE", self.m60_candles[i].l)
                        pda = pd_array("m60", x, self.m60_candles[i].t)
                        self.m60_pda.append(pda)
    
                    if self.m60_candles[i-1].h < self.m60_candles[i+1].l:
                        x = fvg("BISI", self.m60_candles[i-1].h, self.m60_candles[i+1].l)
                        pda = pd_array("m60", x, self.m60_candles[i].t)
                        self.m60_pda.append(pda)
    
                    if self.m60_candles[i-1].l > self.m60_candles[i+1].h:
                        x = fvg("SIBI", self.m60_candles[i-1].l, self.m60_candles[i+1].h)
                        pda = pd_array("m60", x, self.m60_candles[i].t)
                        self.m60_pda.append(pda)
        if tf == "d1":
            self.d1_pda = deque(maxlen = self.maxlen)
            if len(self.d1_candles) > 5:
                for i in range(1,len(self.d1_candles)-1):
                    if self.d1_candles[i].h > self.d1_candles[i-1].h and self.d1_candles[i].h > self.d1_candles[i+1].h:
                        x = liquidity("BUYSIDE", self.d1_candles[i].h)
                        pda = pd_array("d1", x, self.d1_candles[i].t)
                        self.d1_pda.append(pda)
                    if self.d1_candles[i].l < self.d1_candles[i-1].l and self.d1_candles[i].l < self.d1_candles[i+1].l:
                        x = liquidity("SELLSIDE", self.d1_candles[i].l)
                        pda = pd_array("d1", x, self.d1_candles[i].t)
                        self.d1_pda.append(pda)
    
                    if self.d1_candles[i-1].h < self.d1_candles[i+1].l:
                        x = fvg("BISI", self.d1_candles[i-1].h, self.d1_candles[i+1].l)
                        pda = pd_array("d1", x, self.d1_candles[i].t)
                        self.d1_pda.append(pda)
    
                    if self.d1_candles[i-1].l > self.d1_candles[i+1].h:
                        x = fvg("SIBI", self.d1_candles[i-1].l, self.d1_candles[i+1].h)
                        pda = pd_array("d1", x, self.d1_candles[i].t)
                        self.d1_pda.append(pda)
        
        

            
    
    def compute_action_history(self, tf):
        self.pd_copy = copy.deepcopy(self.pd_arrays)
        for i in range(len(self.pd_copy)):
                self.pd_copy[i].invalid = False
                self.pd_copy[i].active = False
                self.pd_copy[i].livetime = 0       
                if type(self.pd_copy[i].pda) == fvg:
                    self.pd_copy[i].filled = False

        
        if tf == "m5":
            
            self.action_history_m5 = deque(maxlen = 256)
            for i in range(len(self.m5_candles)):
                for pd_index, o in enumerate(self.pd_copy):
    
    
                    if o.tf == "m1":
                            continue
                        
                    if o.active:
                        
                        if o.tf == "m5":
                            o.livetime+=1
                        
                        if type(o.pda) == fvg:
                            if o.livetime > 20 and o.filled:
                                o.invalid = True
                                self.pd_arrays_expired_indicies.append(pd_index)
                    
                    
                    if o.invalid:continue
                    
                    candle_time = [int(p) for p in self.m5_candles[i].t]
                    pd_array_formed = [int(p) for p in o.time]
                    candle_time_abs = MHDMoY_to_minutes(candle_time[4], candle_time[3], candle_time[0], candle_time[1], candle_time[2])
                    pd_array_formed_abs = MHDMoY_to_minutes(pd_array_formed[4],pd_array_formed[3],pd_array_formed[0],pd_array_formed[1],pd_array_formed[2])
                    
                    if pd_array_formed_abs < candle_time_abs:o.active = True
    
                    if pd_array_formed_abs == candle_time_abs:
                        if o.tf == "m5":
                            if o.pda.type == "BUYSIDE":
                                a = action("SWING_HIGH_FORMED", o.pda.price, self.m5_candles[i].t, o.tf,o.time)
                                self.action_history_m5.append(a)
                            if o.pda.type == "SELLSIDE":
                                a = action("SWING_LOW_FORMED", o.pda.price, self.m5_candles[i].t, o.tf,o.time)
                                self.action_history_m5.append(a)
                            if o.pda.type == "BISI":
                                a = action("BISI_FORMED", o.pda.ce, self.m5_candles[i].t, o.tf,o.time)
                                self.action_history_m5.append(a)
                            if o.pda.type == "SIBI":
                                a = action("SIBI_FORMED", o.pda.ce, self.m5_candles[i].t, o.tf,o.time)
                                self.action_history_m5.append(a)
                    
                                
                            
            
                    if not o.active:continue
                        
                    if o.pda.type == "BUYSIDE":
                        if self.m5_candles[i].h > o.pda.price:
                            o.invalid = True
                            self.pd_arrays_expired_indicies.append(pd_index)
                            a = action("BUYSIDE_TAKEN", o.pda.price, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
            
                    if o.pda.type == "SELLSIDE":
                        if self.m5_candles[i].l < o.pda.price:
                            o.invalid = True
                            self.pd_arrays_expired_indicies.append(pd_index)
                            a = action("SELLSIDE_TAKEN", o.pda.price, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
            
    
                    
                    if o.pda.type == "BISI":
                        if self.m5_candles[i].h >= o.pda.fvg_high and self.m5_candles[i].o <= o.pda.fvg_high and self.m5_candles[i].c <= o.pda.fvg_high :
                            a = action("BISI_HIGH_TOUCHED_FROM_INSIDE", o.pda.fvg_high, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
                        if self.m5_candles[i].l <= o.pda.fvg_high and self.m5_candles[i].o >= o.pda.fvg_high:
                            a = action("BISI_HIGH_TOUCHED_FROM_OUTSIDE", o.pda.fvg_high, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
                        if self.m5_candles[i].o < o.pda.fvg_high and self.m5_candles[i].c > o.pda.fvg_high :
                            a = action("BISI_HIGH_BROKEN_UPSIDE", o.pda.fvg_high, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
            
                        if self.m5_candles[i].l <= o.pda.fvg_low and self.m5_candles[i].o >= o.pda.fvg_low and self.m5_candles[i].c >= o.pda.fvg_low :
                            a = action("BISI_LOW_TOUCHED_FROM_INSIDE", o.pda.fvg_low, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
                            o.filled=True
                        if self.m5_candles[i].h >= o.pda.fvg_low and self.m5_candles[i].o <= o.pda.fvg_low:
                            a = action("BISI_LOW_TOUCHED_FROM_OUTSIDE", o.pda.fvg_low, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
                        if self.m5_candles[i].o > o.pda.fvg_low and self.m5_candles[i].c < o.pda.fvg_low :
                            a = action("BISI_LOW_BROKEN_DOWNSIDE", o.pda.fvg_low, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
                            o.filled=True
                        
            
                    if o.pda.type == "SIBI":
                        if self.m5_candles[i].h >= o.pda.fvg_high and self.m5_candles[i].o <= o.pda.fvg_high and self.m5_candles[i].c <= o.pda.fvg_high :
                            a = action("SIBI_HIGH_TOUCHED_FROM_INSIDE", o.pda.fvg_high, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
                            o.filled=True
                        if self.m5_candles[i].l <= o.pda.fvg_high and self.m5_candles[i].o >= o.pda.fvg_high:
                            a = action("SIBI_HIGH_TOUCHED_FROM_OUTSIDE", o.pda.fvg_high, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
                        if self.m5_candles[i].o < o.pda.fvg_high and self.m5_candles[i].c > o.pda.fvg_high :
                            a = action("SIBI_HIGH_BROKEN_UPSIDE", o.pda.fvg_high, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
                            o.filled=True
            
                        if self.m5_candles[i].l <= o.pda.fvg_low and self.m5_candles[i].o >= o.pda.fvg_low and self.m5_candles[i].c >= o.pda.fvg_low :
                            a = action("SIBI_LOW_TOUCHED_FROM_INSIDE", o.pda.fvg_low, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
                        if self.m5_candles[i].h >= o.pda.fvg_low and self.m5_candles[i].o <= o.pda.fvg_low:
                            a = action("SIBI_LOW_TOUCHED_FROM_OUTSIDE", o.pda.fvg_low, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
                        if self.m5_candles[i].o > o.pda.fvg_low and self.m5_candles[i].c < o.pda.fvg_low :
                            a = action("SIBI_LOW_BROKEN_DOWNSIDE", o.pda.fvg_low, self.m5_candles[i].t, o.tf,o.time)
                            self.action_history_m5.append(a)
                        
                        
        
        if tf == "m15":
            
            self.action_history_m15 = deque(maxlen = 256)
            for i in range(len(self.m15_candles)):
                for pd_index, o in enumerate(self.pd_copy):
    
                    if o.tf == "m1":
                            continue
                    if o.tf == "m5":
                            continue
                        
                    if o.active:
                        
                        if o.tf == "m15":
                            o.livetime+=1
                            
                        if type(o.pda) == fvg:
                            if o.livetime > 20 and o.filled:
                                o.invalid = True
                                self.pd_arrays_expired_indicies.append(pd_index)
                    
                    
                    if o.invalid:continue
                    
                    candle_time = [int(p) for p in self.m15_candles[i].t]
                    pd_array_formed = [int(p) for p in o.time]
                    candle_time_abs = MHDMoY_to_minutes(candle_time[4], candle_time[3], candle_time[0], candle_time[1], candle_time[2])
                    pd_array_formed_abs = MHDMoY_to_minutes(pd_array_formed[4],pd_array_formed[3],pd_array_formed[0],pd_array_formed[1],pd_array_formed[2])
                    
                    if pd_array_formed_abs < candle_time_abs:o.active = True
    
                    if pd_array_formed_abs == candle_time_abs:
                        if o.tf == "m15":
                            if o.pda.type == "BUYSIDE":
                                a = action("SWING_HIGH_FORMED", o.pda.price, self.m15_candles[i].t, o.tf,o.time)
                                self.action_history_m15.append(a)
                            if o.pda.type == "SELLSIDE":
                                a = action("SWING_LOW_FORMED", o.pda.price, self.m15_candles[i].t, o.tf,o.time)
                                self.action_history_m15.append(a)
                            if o.pda.type == "BISI":
                                a = action("BISI_FORMED", o.pda.ce, self.m15_candles[i].t, o.tf,o.time)
                                self.action_history_m15.append(a)
                            if o.pda.type == "SIBI":
                                a = action("SIBI_FORMED", o.pda.ce, self.m15_candles[i].t, o.tf,o.time)
                                self.action_history_m15.append(a)            
                    
                    if not o.active:continue
                        
                    if o.pda.type == "BUYSIDE":
                        if self.m15_candles[i].h > o.pda.price:
                            o.invalid = True
                            self.pd_arrays_expired_indicies.append(pd_index)
                            a = action("BUYSIDE_TAKEN", o.pda.price, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
            
                    if o.pda.type == "SELLSIDE":
                        if self.m15_candles[i].l < o.pda.price:
                            o.invalid = True
                            self.pd_arrays_expired_indicies.append(pd_index)
                            a = action("SELLSIDE_TAKEN", o.pda.price, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
            
    
                    
                    if o.pda.type == "BISI":
                        if self.m15_candles[i].h >= o.pda.fvg_high and self.m15_candles[i].o <= o.pda.fvg_high and self.m15_candles[i].c <= o.pda.fvg_high :
                            a = action("BISI_HIGH_TOUCHED_FROM_INSIDE", o.pda.fvg_high, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
                        if self.m15_candles[i].l <= o.pda.fvg_high and self.m15_candles[i].o >= o.pda.fvg_high:
                            a = action("BISI_HIGH_TOUCHED_FROM_OUTSIDE", o.pda.fvg_high, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
                        if self.m15_candles[i].o < o.pda.fvg_high and self.m15_candles[i].c > o.pda.fvg_high :
                            a = action("BISI_HIGH_BROKEN_UPSIDE", o.pda.fvg_high, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
            
                        if self.m15_candles[i].l <= o.pda.fvg_low and self.m15_candles[i].o >= o.pda.fvg_low and self.m15_candles[i].c >= o.pda.fvg_low :
                            a = action("BISI_LOW_TOUCHED_FROM_INSIDE", o.pda.fvg_low, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
                            o.filled=True
                        if self.m15_candles[i].h >= o.pda.fvg_low and self.m15_candles[i].o <= o.pda.fvg_low:
                            a = action("BISI_LOW_TOUCHED_FROM_OUTSIDE", o.pda.fvg_low, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
                        if self.m15_candles[i].o > o.pda.fvg_low and self.m15_candles[i].c < o.pda.fvg_low :
                            a = action("BISI_LOW_BROKEN_DOWNSIDE", o.pda.fvg_low, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
                            o.filled=True
                        
            
                    if o.pda.type == "SIBI":
                        if self.m15_candles[i].h >= o.pda.fvg_high and self.m15_candles[i].o <= o.pda.fvg_high and self.m15_candles[i].c <= o.pda.fvg_high :
                            a = action("SIBI_HIGH_TOUCHED_FROM_INSIDE", o.pda.fvg_high, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
                            o.filled=True
                        if self.m15_candles[i].l <= o.pda.fvg_high and self.m15_candles[i].o >= o.pda.fvg_high:
                            a = action("SIBI_HIGH_TOUCHED_FROM_OUTSIDE", o.pda.fvg_high, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
                        if self.m15_candles[i].o < o.pda.fvg_high and self.m15_candles[i].c > o.pda.fvg_high :
                            a = action("SIBI_HIGH_BROKEN_UPSIDE", o.pda.fvg_high, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
                            o.filled=True
            
                        if self.m15_candles[i].l <= o.pda.fvg_low and self.m15_candles[i].o >= o.pda.fvg_low and self.m15_candles[i].c >= o.pda.fvg_low :
                            a = action("SIBI_LOW_TOUCHED_FROM_INSIDE", o.pda.fvg_low, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
                        if self.m15_candles[i].h >= o.pda.fvg_low and self.m15_candles[i].o <= o.pda.fvg_low:
                            a = action("SIBI_LOW_TOUCHED_FROM_OUTSIDE", o.pda.fvg_low, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
                        if self.m15_candles[i].o > o.pda.fvg_low and self.m15_candles[i].c < o.pda.fvg_low :
                            a = action("SIBI_LOW_BROKEN_DOWNSIDE", o.pda.fvg_low, self.m15_candles[i].t, o.tf,o.time)
                            self.action_history_m15.append(a)
                        
                                         
                
    
    


        if tf == "m1":
            
            self.action_history_m1 = deque(maxlen = 256)
            for i in range(len(self.m1_candles)):
                for pd_index,o in enumerate(self.pd_copy):
                    
                    if o.active:
                        
                        if o.tf == "m1":
                            o.livetime+=1
                            
                        if type(o.pda) == fvg:
                            if o.livetime > 20 and o.filled:
                                o.invalid = True
                                self.pd_arrays_expired_indicies.append(pd_index)
                    
                    
                    if o.invalid:continue
                    
                    candle_time = [int(p) for p in self.m1_candles[i].t]
                    pd_array_formed = [int(p) for p in o.time]
                    candle_time_abs = MHDMoY_to_minutes(candle_time[4], candle_time[3], candle_time[0], candle_time[1], candle_time[2])
                    pd_array_formed_abs = MHDMoY_to_minutes(pd_array_formed[4],pd_array_formed[3],pd_array_formed[0],pd_array_formed[1],pd_array_formed[2])
                    
                    if pd_array_formed_abs < candle_time_abs:o.active = True
    
                    if pd_array_formed_abs == candle_time_abs:
                        if o.tf == "m1":
                            if o.pda.type == "BUYSIDE":
                                a = action("SWING_HIGH_FORMED", o.pda.price, self.m1_candles[i].t, o.tf,o.time)
                                self.action_history_m1.append(a)
                            if o.pda.type == "SELLSIDE":
                                a = action("SWING_LOW_FORMED", o.pda.price, self.m1_candles[i].t, o.tf,o.time)
                                self.action_history_m1.append(a)
                            if o.pda.type == "BISI":
                                a = action("BISI_FORMED", o.pda.ce, self.m1_candles[i].t, o.tf,o.time)
                                self.action_history_m1.append(a)
                            if o.pda.type == "SIBI":
                                a = action("SIBI_FORMED", o.pda.ce, self.m1_candles[i].t, o.tf,o.time)
                                self.action_history_m1.append(a)            
                    
                    if not o.active:continue
                        
                    if o.pda.type == "BUYSIDE":
                        if self.m1_candles[i].h > o.pda.price:
                            o.invalid = True
                            self.pd_arrays_expired_indicies.append(pd_index)
                            a = action("BUYSIDE_TAKEN", o.pda.price, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
            
                    if o.pda.type == "SELLSIDE":
                        if self.m1_candles[i].l < o.pda.price:
                            o.invalid = True
                            self.pd_arrays_expired_indicies.append(pd_index)
                            a = action("SELLSIDE_TAKEN", o.pda.price, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
            
    
                    
                    if o.pda.type == "BISI":
                        if self.m1_candles[i].h >= o.pda.fvg_high and self.m1_candles[i].o <= o.pda.fvg_high and self.m1_candles[i].c <= o.pda.fvg_high :
                            a = action("BISI_HIGH_TOUCHED_FROM_INSIDE", o.pda.fvg_high, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
                        if self.m1_candles[i].l <= o.pda.fvg_high and self.m1_candles[i].o >= o.pda.fvg_high:
                            a = action("BISI_HIGH_TOUCHED_FROM_OUTSIDE", o.pda.fvg_high, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
                        if self.m1_candles[i].o < o.pda.fvg_high and self.m1_candles[i].c > o.pda.fvg_high :
                            a = action("BISI_HIGH_BROKEN_UPSIDE", o.pda.fvg_high, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
            
                        if self.m1_candles[i].l <= o.pda.fvg_low and self.m1_candles[i].o >= o.pda.fvg_low and self.m1_candles[i].c >= o.pda.fvg_low :
                            a = action("BISI_LOW_TOUCHED_FROM_INSIDE", o.pda.fvg_low, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
                            o.filled=True
                        if self.m1_candles[i].h >= o.pda.fvg_low and self.m1_candles[i].o <= o.pda.fvg_low:
                            a = action("BISI_LOW_TOUCHED_FROM_OUTSIDE", o.pda.fvg_low, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
                        if self.m1_candles[i].o > o.pda.fvg_low and self.m1_candles[i].c < o.pda.fvg_low :
                            a = action("BISI_LOW_BROKEN_DOWNSIDE", o.pda.fvg_low, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
                            o.filled=True
                        
            
                    if o.pda.type == "SIBI":
                        if self.m1_candles[i].h >= o.pda.fvg_high and self.m1_candles[i].o <= o.pda.fvg_high and self.m1_candles[i].c <= o.pda.fvg_high :
                            a = action("SIBI_HIGH_TOUCHED_FROM_INSIDE", o.pda.fvg_high, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
                            o.filled=True
                        if self.m1_candles[i].l <= o.pda.fvg_high and self.m1_candles[i].o >= o.pda.fvg_high:
                            a = action("SIBI_HIGH_TOUCHED_FROM_OUTSIDE", o.pda.fvg_high, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
                        if self.m1_candles[i].o < o.pda.fvg_high and self.m1_candles[i].c > o.pda.fvg_high :
                            a = action("SIBI_HIGH_BROKEN_UPSIDE", o.pda.fvg_high, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
                            o.filled=True
            
                        if self.m1_candles[i].l <= o.pda.fvg_low and self.m1_candles[i].o >= o.pda.fvg_low and self.m1_candles[i].c >= o.pda.fvg_low :
                            a = action("SIBI_LOW_TOUCHED_FROM_INSIDE", o.pda.fvg_low, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
                        if self.m1_candles[i].h >= o.pda.fvg_low and self.m1_candles[i].o <= o.pda.fvg_low:
                            a = action("SIBI_LOW_TOUCHED_FROM_OUTSIDE", o.pda.fvg_low, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
                        if self.m1_candles[i].o > o.pda.fvg_low and self.m1_candles[i].c < o.pda.fvg_low :
                            a = action("SIBI_LOW_BROKEN_DOWNSIDE", o.pda.fvg_low, self.m1_candles[i].t, o.tf,o.time)
                            self.action_history_m1.append(a)
           
        
        if tf == "m60": # has no action history (yet...?)
            for i in range(len(self.m60_candles)):
                for pd_index, o in enumerate(self.pd_copy):
    
                    if o.tf == "m1":
                            continue
                    if o.tf == "m5":
                            continue
                    if o.tf == "m15":
                            continue
                        
                    if o.active:
                        if o.tf == "m60":
                            o.livetime+=1
                        if type(o.pda) == fvg:
                            if o.livetime > 20 and o.filled:
                                o.invalid = True
                                self.pd_arrays_expired_indicies.append(pd_index)
                    
                    
                    if o.invalid:continue
                    
                    candle_time = [int(p) for p in self.m60_candles[i].t]
                    pd_array_formed = [int(p) for p in o.time]
                    candle_time_abs = MHDMoY_to_minutes(candle_time[4], candle_time[3], candle_time[0], candle_time[1], candle_time[2])
                    pd_array_formed_abs = MHDMoY_to_minutes(pd_array_formed[4],pd_array_formed[3],pd_array_formed[0],pd_array_formed[1],pd_array_formed[2])
                    
                    if pd_array_formed_abs < candle_time_abs:o.active = True
                                
                            
            
                    if not o.active:continue
                        
                    if o.pda.type == "BUYSIDE":
                        if self.m60_candles[i].h > o.pda.price:
                            o.invalid = True
                            self.pd_arrays_expired_indicies.append(pd_index)
            
                    if o.pda.type == "SELLSIDE":
                        if self.m60_candles[i].l < o.pda.price:
                            o.invalid = True
                            self.pd_arrays_expired_indicies.append(pd_index)
            
    
                    
                    if o.pda.type == "BISI":
                        if self.m60_candles[i].l <= o.pda.fvg_low:
                            o.filled=True
                        
            
                    if o.pda.type == "SIBI":
                        if self.m60_candles[i].h >= o.pda.fvg_high:
                            o.filled=True

        if tf == "d1": # has no action history (yet...?)
            for i in range(len(self.d1_candles)):
                for pd_index, o in enumerate(self.pd_copy):
    
                    if o.tf == "m1":
                            continue
                    if o.tf == "m5":
                            continue
                    if o.tf == "m15":
                            continue
                    if o.tf == "m60":
                            continue
                        
                    if o.active:
                        if o.tf == "d1":
                            o.livetime+=1
                        if type(o.pda) == fvg:
                            if o.livetime > 20 and o.filled:
                                o.invalid = True
                                self.pd_arrays_expired_indicies.append(pd_index)
                    
                    
                    if o.invalid:continue
                    
                    candle_time = [int(p) for p in self.d1_candles[i].t]
                    pd_array_formed = [int(p) for p in o.time]
                    candle_time_abs = MHDMoY_to_minutes(candle_time[4], candle_time[3], candle_time[0], candle_time[1], candle_time[2])
                    pd_array_formed_abs = MHDMoY_to_minutes(pd_array_formed[4],pd_array_formed[3],pd_array_formed[0],pd_array_formed[1],pd_array_formed[2])
                    
                    if pd_array_formed_abs < candle_time_abs:o.active = True
                                
                            
            
                    if not o.active:continue
                        
                    if o.pda.type == "BUYSIDE":
                        if self.d1_candles[i].h > o.pda.price:
                            o.invalid = True
                            self.pd_arrays_expired_indicies.append(pd_index)
            
                    if o.pda.type == "SELLSIDE":
                        if self.d1_candles[i].l < o.pda.price:
                            o.invalid = True
                            self.pd_arrays_expired_indicies.append(pd_index)
            
    
                    
                    if o.pda.type == "BISI":
                        if self.d1_candles[i].l <= o.pda.fvg_low:
                            o.filled=True
                        
            
                    if o.pda.type == "SIBI":
                        if self.d1_candles[i].h >= o.pda.fvg_high:
                            o.filled=True
                        
                        
    def plot_candles(self, name, pda=False):
        import cv2
        if name == "m1":candles = self.m1_candles
        if name == "m5":candles = self.m5_candles
        if name == "m15":candles = self.m15_candles
        if name == "m60":candles = self.m60_candles
        if name == "d1":candles = self.d1_candles
    
        def scale_p(p):
            return (p - max_l) / hlrange * h
            
        w = 400
        h = 300
        canvas = np.zeros((h,w,3), np.uint8) 
        l = len(candles)
        single_candle_w = w / l * 0.95
        max_h = 0
        max_l = 1000000
        for i in candles:
            if i.h > max_h:
                max_h = i.h
            if i.l < max_l:
                max_l = i.l
        hlrange = max(max_h - max_l,1)

        for i in range(len(candles)):  
            
            color = (0,100,0) if candles[i].o<candles[i].c else (0,0,100) if candles[i].o>candles[i].c else (100,100,100)
            cv2.rectangle(canvas, (int(i*single_candle_w),int(scale_p(candles[i].o))), (int((i+1)*single_candle_w),int(scale_p(candles[i].c))), color, -1)
            cv2.line(canvas, (int((i+0.5)*single_candle_w),int(scale_p(candles[i].h))), (int((i+0.5)*single_candle_w),int(scale_p(candles[i].l))), color)

            if pda:
                for o in self.pd_arrays:
                    if o.time == candles[i].t and o.tf == name:
                        if o.pda.type == "BUYSIDE":
                            price = o.pda.price
                            cv2.line(canvas, (int((i+0.5)*single_candle_w),int(scale_p(price))), (int((i+0.5+3)*single_candle_w),int(scale_p(price))), (0,0,200))
                        if o.pda.type == "SELLSIDE":
                            price = o.pda.price
                            cv2.line(canvas, (int((i+0.5)*single_candle_w),int(scale_p(price))), (int((i+0.5+3)*single_candle_w),int(scale_p(price))), (0,200,0))
    
                        if o.pda.type == "BISI":
                            price1 = o.pda.fvg_high
                            price2 = o.pda.fvg_low
                            ce = o.pda.ce
                            
                            cv2.line(canvas, (int((i+0.5)*single_candle_w),int(scale_p(price1))), (int((i+0.5+20)*single_candle_w),int(scale_p(price1))), (0,200,0), thickness=2)
                            cv2.line(canvas, (int((i+0.5)*single_candle_w),int(scale_p(price2))), (int((i+0.5+20)*single_candle_w),int(scale_p(price2))), (0,200,0), thickness=2)
    
    
                        if o.pda.type == "SIBI":
                            price1 = o.pda.fvg_high
                            price2 = o.pda.fvg_low
                            ce = o.pda.ce
                            
                            cv2.line(canvas, (int((i+0.5)*single_candle_w),int(scale_p(price1))), (int((i+0.5+20)*single_candle_w),int(scale_p(price1))), (0,0,200), thickness=2)
                            cv2.line(canvas, (int((i+0.5)*single_candle_w),int(scale_p(price2))), (int((i+0.5+20)*single_candle_w),int(scale_p(price2))), (0,0,200), thickness=2)
    
            
        canvas = canvas[::-1]

        cv2.imshow(name, canvas)
        cv2.waitKey(1)

