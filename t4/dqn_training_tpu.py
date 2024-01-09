#########################################
# this version is including no position #
#########################################


from manager import *
from utils import *
from make_model import *

import tensorflow as tf
from multiprocessing import Queue, Process
import time
from manager import candle_class
import random
import numpy as np
from collections import deque
import time
import os

num_model_inputs = 3+5+3+1
n_actions = 3
m1 = np.eye(n_actions, dtype="float32")
batch_q_size = 512
data_q_maxlen = 128
target_model_sync = 10000

save_freq = 50 # save after x epochs


verb = False
import sys
argv = sys.argv
if len(argv) == 2:
    if argv[1] == "v":
        verb = True
       
       
def threaded_data_generation(q,num):
    first_run = True
    while True:
        data_dir = "data/"
        parts = [("parts_EP_1.o/", 0.4), ("parts_EP_1_inverted.o/", 0.4), ("parts_ENQ_1.o/", 2.4), ("parts_ENQ_1_inverted.o/", 2.4), ("parts_YM_1.o/", 2.4), ("parts_YM_1_inverted.o/", 2.4)]
        c = random.choice(parts)
        path = data_dir+c[0]
        cm = c[1]
        name = path.split("/")[-2]
        print(num,"-","path:", path)
        print(num,"-","cm:", cm)
    
        subfiles = sorted([x for x in os.listdir(path) if "part" in x])

        
        start_ofs = 50000

        start = 0
        if first_run:
            start = random.randint(0,(len(subfiles)-1)*100000)
            first_run = False

        
        print(num,"-","start at", name, "-", start)
        
        candle_counter = -1
        last_state = 0

        x = manager()
        for part in subfiles:
            if candle_counter + 100000 < start:
                candle_counter+=100000
                continue
            candles = Load(path+part)
            for i in range(0, len(candles)):
                candle_counter+=1
                while q.qsize() > data_q_maxlen:
                    #print(num,"-","Data Queue full - waiting...")
                    time.sleep(random.randint(10,50)/10)
                
                if candle_counter>=start+start_ofs:
                    ret = x.push_m1_candle(candles[i])
                    
                    #disable relative scaling
                    #ret[0][1] = 1
                    #########
                    
                    inp = get_inputs_from_ret(ret, x)
                    current_close = (candles[i].c - candles[i].o) / ret[0][1]
            
                    if last_state != 0:
                        inp_long = [1]+last_state
                        inp_short = [-1]+last_state
                        inp_neutral = [0]+last_state
            
                        
                        
                        #state, action, reward, next state (terminus)
                        scaled_cm = make_price_relative(cm, 0, ret[0][1])
                    #prev long holding:
                        # new long holding    
                        new_state = [1]+inp
                        pair = (inp_long, 1, current_close*1,0,new_state)
                        q.put(pair)
                        
                        # new short holding    
                        new_state = [-1]+inp
                        pair = (inp_long, 0, current_close*-1 - scaled_cm,0,new_state)
                        q.put(pair)
                        
                        # new neutral holding    
                        new_state = [0]+inp
                        pair = (inp_long, 2, 0 - scaled_cm / 2,0,new_state)
                        q.put(pair)
                        
                    #prev short holding:
                        # new long holding    
                        new_state = [1]+inp
                        pair = (inp_short, 1, current_close*1 - scaled_cm,0,new_state)
                        q.put(pair)
                        
                        # new short holding    
                        new_state = [-1]+inp
                        pair = (inp_short, 0, current_close*-1,0,new_state)
                        q.put(pair)
                        
                        # new neutral holding    
                        new_state = [0]+inp
                        pair = (inp_short, 2, 0 - scaled_cm / 2,0,new_state)
                        q.put(pair)
                        
                        
            
                    #prev neutral holding:
                        # new long holding    
                        new_state = [1]+inp
                        pair = (inp_neutral, 1, current_close*1 - scaled_cm / 2,0,new_state)
                        q.put(pair)
                        
                        # new short holding    
                        new_state = [-1]+inp
                        pair = (inp_neutral, 0, current_close*-1 - scaled_cm / 2,0,new_state)
                        q.put(pair)
                        
                        # new neutral holding    
                        new_state = [0]+inp
                        pair = (inp_neutral, 2, 0, 0, new_state)
                        q.put(pair)
                        
                        
                    last_state = inp
                else:
                    ret = x.push_m1_candle(candles[i], scan = False)
                


def data_get_func(data_qs, batch_q):
    ssrtm_memory = []
    print("data_get_func online")
    while True:

        nn = 0
        
        while True:
            has_items = False
            if len(ssrtm_memory) < memory_size:
                for data_q in data_qs:
                    if data_q.qsize() == 0:continue
                    has_items = True
                    nn+=1
                    #print("get_item",len(ssrtm_memory), nn)
                    if nn > batch_size * 1:
                        has_items = False
                        #break

                    p = data_q.get()

                    p[0][4] = np.concatenate((p[0][4][0],p[0][4][1]))
                    p[0][5] = np.concatenate((p[0][5][0],p[0][5][1]))
                    p[0][6] = np.concatenate((p[0][6][0],p[0][6][1]))
                    p[0][7] = np.concatenate((p[0][7][0],p[0][7][1]))
                    p[0][8] = np.concatenate((p[0][8][0],p[0][8][1]))

                    p[4][4] = np.concatenate((p[4][4][0],p[4][4][1]))
                    p[4][5] = np.concatenate((p[4][5][0],p[4][5][1]))
                    p[4][6] = np.concatenate((p[4][6][0],p[4][6][1]))
                    p[4][7] = np.concatenate((p[4][7][0],p[4][7][1]))
                    p[4][8] = np.concatenate((p[4][8][0],p[4][8][1]))

                    ssrtm_memory.append(p)
            if has_items == False:
                    #print("break")
                    break
            
            
        if batch_q.qsize() > batch_q_size:
            #print("batch Queue full - waiting")
            #print("memory len:", len(ssrtm_memory))
            time.sleep(0.05)
            continue


        if len(ssrtm_memory) < batch_size:
            time.sleep(1)
            continue
        else:
            sarts_batch = []
            for i in range(batch_size):
                l = len(ssrtm_memory)-1
                random_index = random.randint(0,l)
                sarts_batch.append(ssrtm_memory[random_index])
                del ssrtm_memory[random_index]
            
            states = [x[0] for x in sarts_batch]
            states_array = []
            for i in range(num_model_inputs):
                    states_array.append(np.array([x[i] for x in states], dtype = "float32"))
            
                        
            actions = [x[1] for x in sarts_batch]
            rewards = np.array([x[2] for x in sarts_batch], dtype="float32")
            terminals = np.array([x[3] for x in sarts_batch], dtype="float32")
            
            next_states = [x[4] for x in sarts_batch]
            next_states_array = []
            for i in range(num_model_inputs):
                    next_states_array.append(np.array([x[i] for x in next_states], dtype = "float32"))
                    
            
            #print(actions)
            masks = m1[actions]
            batch_q.put( [states_array, next_states_array, rewards, terminals, masks] )
    

################
    
def filesave(filename, value):
    f = open(filename, "a")
    f.write(str(value)+"\n")
    f.close()
    
def main():


    
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    
    
    batch_q = Queue()
    
    ## start batch generation threads
    for i in range(batch_generation_threads):
        data_qs = []
        
        ## start data generation threads
        for i in range(num_data_generation_threads):
            data_q = Queue()
            data_qs.append(data_q)
            p = Process(target = threaded_data_generation, args = (data_q, i), daemon = True)
            p.start()
            time.sleep(0.05)

    
        p = Process(target = data_get_func, args = (data_qs, batch_q), daemon = True)
        p.start()
        time.sleep(0.05)


    with strategy.scope():
        model = make_model()
        target_model = make_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    print("loading model weights...")
    try:
        model.load_weights("dqn_weights.h5")
        target_model.load_weights("dqn_weights.h5")
    except Exception as e:
        print(e)
        

    @tf.function()
    def get_target_q(next_states, rewards, terminals):
            estimated_q_values_next = target_model(next_states)
            q_batch = tf.math.reduce_max(estimated_q_values_next, axis=1)
            target_q_values = q_batch * gamma * (1-terminals) + rewards
            return target_q_values
    
            
    @tf.function()
    def tstep(data):
            states, next_states, rewards, terminals, masks = data
            target_q_values = get_target_q(next_states, rewards, terminals)
            
            with tf.GradientTape() as t:
                model_return = model(states, training=True) 
                mask_return = model_return * masks
                estimated_q_values = tf.math.reduce_sum(mask_return, axis=1)
                #print(estimated_q_values, mask_return, model_return, masks)
                loss_e = tf.math.square(target_q_values - estimated_q_values)
                loss = tf.reduce_mean(loss_e)
            
            
            gradient = t.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            
            return loss, tf.reduce_mean(estimated_q_values)

    def tpu_data_get_func(_n):
        return batch_q.get()
    
    counter = 0
    counter_total = 0
    loss_mean = []
    q_mean = []
    while True:
        counter += 1
        t0 = time.time()
        if verb:
            progbar = tf.keras.utils.Progbar(ep_len)
        losses = []
        qs = []
        if verb:
            bq_pv = batch_q.qsize() / batch_q_size * 100
            print("num ready batches:", bq_pv, "%", "(good)" if bq_pv >= 100 else "")
        for i in range(ep_len):
            
            if batch_q.qsize() >= 8:
                counter_total += 1
                
                distributed_values = (strategy.experimental_distribute_values_from_function(tpu_data_get_func))
                loss, qv = strategy.reduce(tf.distribute.ReduceOp.MEAN, strategy.run(tstep, args = (distributed_values,)), axis = None)
                
                losses.append(loss.numpy())
                qs.append(qv.numpy())
                if verb:
                    progbar.update(i+1, values = [("loss", loss), ("qv", qv)])
                    
                if counter_total % target_model_sync == 0:   
                    print("\ncopy model weights...\n")
                    target_model.set_weights(model.get_weights())        
                
            
            else:
                time.sleep(1)
                print("waiting for batch generation...")

            
        
        loss_mean.append(np.mean(losses))
        q_mean.append(np.mean(qs))
        
        if counter % save_freq == 0:
                    print("\nsaving...\n")
                    for l in loss_mean:
                        filesave("loss.txt", l)        
                        loss_mean = []
                    for q in q_mean:
                        filesave("qv.txt", q)
                        q_mean = []
                    model.save_weights("dqn_weights.h5")
        
         
        if not verb:
            print("loss:", np.mean(losses), "- expected Q values:", np.mean(qs), "- time:", time.time() - t0)
        
            

if __name__ == "__main__":
    main()