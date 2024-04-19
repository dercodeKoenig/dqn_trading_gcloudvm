
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
        self.comp_minutes()
    def comp_minutes(self):
        candle_time = [int(p) for p in self.t]                    
        self.candle_time_minutes = MHDMoY_to_minutes(candle_time[4], candle_time[3], candle_time[0], candle_time[1], candle_time[2])


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
    
    def __init__(self):
        l = 120
        self.m1_candles = deque(maxlen = l)
        self.m5_candles = deque(maxlen = l)
        self.m15_candles = deque(maxlen = l)
        self.m60_candles = deque(maxlen = 1000)  
        self.d1_candles = deque(maxlen = 1000)   

      
        
        self.nymidnight_price = 0
        pass

    
    def push_m1_candle(self, __candle):
        candle = copy.deepcopy(__candle)
        candle.t = [int(i) for i in  candle.t.split(":")]
        candle_minute = candle.t[4]
        candle_hour = candle.t[3]
        if candle_minute == 0 and candle_hour == 0 or self.nymidnight_price == 0:
            self.nymidnight_price = candle.o

   
        candle.comp_minutes()
        self.m1_candles.append(candle)
    
        

        # m5 candles
        if len(self.m5_candles) > 0:
            last_minute = self.m5_candles[-1].t[4]
            last_hour = self.m5_candles[-1].t[3]
            minute_rounded = int(candle_minute/5) * 5
            if minute_rounded != last_minute or candle_hour != last_hour:
                
                c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.m5_candles.append(c)
            else:
                self.m5_candles[-1].c = candle.c
                self.m5_candles[-1].h = max(candle.h, self.m5_candles[-1].h)
                self.m5_candles[-1].l = min(candle.l, self.m5_candles[-1].l)
        else:

            c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.m5_candles.append(c)
            

       # m15 candles
        if len(self.m15_candles) > 0:
            last_minute = self.m15_candles[-1].t[4]
            last_hour = self.m15_candles[-1].t[3]
            minute_rounded = int(candle_minute/15) * 15
            if minute_rounded != last_minute or candle_hour != last_hour:
               
                c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.m15_candles.append(c)
            else:
                self.m15_candles[-1].c = candle.c
                self.m15_candles[-1].h = max(candle.h, self.m15_candles[-1].h)
                self.m15_candles[-1].l = min(candle.l, self.m15_candles[-1].l)
        else:
            
            c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.m15_candles.append(c)

        # m60 candles
        if len(self.m60_candles) > 0:
            last_minute = self.m60_candles[-1].t[4]
            last_hour = self.m60_candles[-1].t[3]
            
            if candle_hour != last_hour:
                
                c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.m60_candles.append(c)
            else:
                self.m60_candles[-1].c = candle.c
                self.m60_candles[-1].h = max(candle.h, self.m60_candles[-1].h)
                self.m60_candles[-1].l = min(candle.l, self.m60_candles[-1].l)
        else:
           
            c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.m60_candles.append(c)


       # d1 candles
        if len(self.d1_candles) > 0:
            last_candle_hour = self.m1_candles[-2].t[3]
            #print(candle_hour)
            if candle_hour != last_candle_hour and candle_hour == 18:
                
                c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.d1_candles.append(c)
            else:
                self.d1_candles[-1].c = candle.c
                self.d1_candles[-1].h = max(candle.h, self.d1_candles[-1].h)
                self.d1_candles[-1].l = min(candle.l, self.d1_candles[-1].l)
        else:
            
            c = candle_class(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.d1_candles.append(c)



        avg_candle_range = np.mean([x.h-x.l for x in self.m15_candles])
        return [self.nymidnight_price, max(0.25, avg_candle_range), self.m1_candles[-1].c, self.m1_candles[-1].t], [self.m15_candles,self.m5_candles,self.m1_candles]

      