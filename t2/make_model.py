
## tx + gru large

import tensorflow as tf

#config
batch_size = 64
gamma = 0.995
learning_rate=0.00002
num_data_generation_threads = 12
batch_generation_threads = 8
memory_size = 200_000
ep_len = 100


num_model_inputs = 3+5+3+1
n_actions = 3
batch_q_size = 512
data_q_maxlen = 128
target_model_sync = 1000

save_freq = 20 # save after x epochs


def make_model():

  embed_times = tf.keras.layers.Embedding(60*24,8)

  input_m15 = tf.keras.layers.Input(shape = (500,5))
  input_m5 = tf.keras.layers.Input(shape = (500,5))
  input_m1 = tf.keras.layers.Input(shape = (500,5))
    
    
  
  m15_o = tf.keras.layers.Reshape((-1,1))(input_m15[::,::,0])
  m15_h = tf.keras.layers.Reshape((-1,1))(input_m15[::,::,1])
  m15_l = tf.keras.layers.Reshape((-1,1))(input_m15[::,::,2])
  m15_c = tf.keras.layers.Reshape((-1,1))(input_m15[::,::,3])
  m15_times = input_m15[::,::,4]
  m15_times = embed_times(m15_times)
  chart_m15 = tf.keras.layers.Concatenate(axis=-1)([m15_o,m15_h,m15_l,m15_c, m15_times])
    
  
  m5_o = tf.keras.layers.Reshape((-1,1))(input_m5[::,::,0])
  m5_h = tf.keras.layers.Reshape((-1,1))(input_m5[::,::,1])
  m5_l = tf.keras.layers.Reshape((-1,1))(input_m5[::,::,2])
  m5_c = tf.keras.layers.Reshape((-1,1))(input_m5[::,::,3])
  m5_times = input_m5[::,::,4]
  m5_times = embed_times(m5_times)
  chart_m5 = tf.keras.layers.Concatenate(axis=-1)([m5_o,m5_h,m5_l,m5_c, m5_times])
    
  
  m1_o = tf.keras.layers.Reshape((-1,1))(input_m1[::,::,0])
  m1_h = tf.keras.layers.Reshape((-1,1))(input_m1[::,::,1])
  m1_l = tf.keras.layers.Reshape((-1,1))(input_m1[::,::,2])
  m1_c = tf.keras.layers.Reshape((-1,1))(input_m1[::,::,3])
  m1_times = input_m1[::,::,4]
  m1_times = embed_times(m1_times)
  chart_m1 = tf.keras.layers.Concatenate(axis=-1)([m1_o,m1_h,m1_l,m1_c, m1_times])
  
  


  input_closing_prices = tf.keras.layers.Input(shape=(1,))
  input_current_pos = tf.keras.layers.Input(shape=(1,))
  input_current_day_in = tf.keras.layers.Input(shape=(1,))
  input_current_day = tf.keras.layers.Embedding(7,3)(input_current_day_in)
  input_current_day = tf.keras.layers.Flatten()(input_current_day)
  input_closing_times_in = tf.keras.layers.Input(shape=(1,))
  input_closing_times = embed_times(input_closing_times_in)
  input_closing_times = tf.keras.layers.Flatten()(input_closing_times)
  

  gru_units = 256

  def embed_information(input_state):
      input_state = tf.keras.layers.Dense(256, activation = "relu")(input_state)
      input_state = tf.keras.layers.Dense(256, activation = "relu")(input_state)
      input_state_tx = tf.keras.layers.Dense(12, activation = "relu")(input_state)
      input_state_tx = tf.keras.layers.Reshape((1,12))(input_state_tx)
      return input_state_tx
  
  def process_chart(chart, additional_info):
    input_state_tx = embed_information(additional_info)
    chart = tf.keras.layers.Concatenate(axis=1)([input_state_tx, chart])
   
    x = tf.keras.layers.GRU(gru_units, return_sequences=True)(chart)
    x = tf.keras.layers.GRU(gru_units, return_sequences=True)(x)
    x = tf.keras.layers.GRU(gru_units, return_sequences=True)(x)
    x = tf.keras.layers.GRU(gru_units, return_sequences=True)(x)
    x = tf.keras.layers.GRU(gru_units, return_sequences=True)(x)
    x = tf.keras.layers.GRU(gru_units, return_sequences=True)(x)
    x = tf.keras.layers.GRU(gru_units, return_sequences=True)(x)
    x = tf.keras.layers.GRU(gru_units, return_sequences=True)(x)
    x = tf.keras.layers.GRU(gru_units, return_sequences=True)(x)
    
    x = tf.keras.layers.GRU(1024, return_sequences=False)(x)
    
    
    return x

  input_state = tf.keras.layers.Concatenate()([input_current_pos, input_closing_prices, input_closing_times, input_current_day])
  output_m15 = process_chart(chart_m15, input_state)
  input_state = tf.keras.layers.Concatenate()([input_current_pos, input_closing_prices, input_closing_times, input_current_day, output_m15])
  output_m5 = process_chart(chart_m5, input_state)
  input_state = tf.keras.layers.Concatenate()([input_current_pos, input_closing_prices, input_closing_times, input_current_day, output_m15, output_m5])
  output_m1 = process_chart(chart_m1, input_state)
  
  dense_input = tf.keras.layers.Concatenate()([output_m15, output_m5, output_m1])
  

  x = tf.keras.layers.Dense(1024)(dense_input)
  x = tf.keras.layers.LeakyReLU()(x)
    
  x = tf.keras.layers.Dense(1024)(x)
  x = tf.keras.layers.LeakyReLU()(x)
    
  x = tf.keras.layers.Dense(1024)(x)
  x = tf.keras.layers.LeakyReLU()(x)
    

  x = tf.keras.layers.Dense(3, activation = "linear")(x)

  model = tf.keras.Model(inputs = [input_current_pos, input_closing_prices, input_closing_times_in, input_current_day_in, input_m1, input_m5, input_m15], outputs=x)
  return model

make_model()