import tensorflow as tf

def make_model():

  embed_times = tf.keras.layers.Embedding(60*24*7,8)
  embed_pda_types = tf.keras.layers.Embedding(7,6)

  pd_arrays_m1_input = tf.keras.layers.Input(shape = (10,3))
  pda_m1_times = pd_arrays_m1_input[::,::,0]
  pda_m1_types = pd_arrays_m1_input[::,::,1]
  pda_m1_prices = tf.keras.layers.Reshape((-1,1))(pd_arrays_m1_input[::,::,2])
  pda_m1_times = embed_times(pda_m1_times)
  pda_m1_types = embed_pda_types(pda_m1_types)
  pda_list_m1 = tf.keras.layers.Concatenate(axis=-1)([pda_m1_prices, pda_m1_times, pda_m1_types])
  pda_list_m1 = tf.keras.layers.Flatten()(pda_list_m1)
  

  pd_arrays_m5_input = tf.keras.layers.Input(shape = (10,3))
  pda_m5_times = pd_arrays_m5_input[::,::,0]
  pda_m5_types = pd_arrays_m5_input[::,::,1]
  pda_m5_prices = tf.keras.layers.Reshape((-1,1))(pd_arrays_m5_input[::,::,2])
  pda_m5_times = embed_times(pda_m5_times)
  pda_m5_types = embed_pda_types(pda_m5_types)
  pda_list_m5 = tf.keras.layers.Concatenate(axis=-1)([pda_m5_prices, pda_m5_times, pda_m5_types])
  pda_list_m5 = tf.keras.layers.Flatten()(pda_list_m5)
  

  pd_arrays_m15_input = tf.keras.layers.Input(shape = (10,3))
  pda_m15_times = pd_arrays_m15_input[::,::,0]
  pda_m15_types = pd_arrays_m15_input[::,::,1]
  pda_m15_prices = tf.keras.layers.Reshape((-1,1))(pd_arrays_m15_input[::,::,2])
  pda_m15_times = embed_times(pda_m15_times)
  pda_m15_types = embed_pda_types(pda_m15_types)
  pda_list_m15 = tf.keras.layers.Concatenate(axis=-1)([pda_m15_prices, pda_m15_times, pda_m15_types])
  pda_list_m15 = tf.keras.layers.Flatten()(pda_list_m15)
  

  pd_arrays_m60_input = tf.keras.layers.Input(shape = (10,3))
  pda_m60_times = pd_arrays_m60_input[::,::,0]
  pda_m60_types = pd_arrays_m60_input[::,::,1]
  pda_m60_prices = tf.keras.layers.Reshape((-1,1))(pd_arrays_m60_input[::,::,2])
  pda_m60_times = embed_times(pda_m60_times)
  pda_m60_types = embed_pda_types(pda_m60_types)
  pda_list_m60 = tf.keras.layers.Concatenate(axis=-1)([pda_m60_prices, pda_m60_times, pda_m60_types])
  pda_list_m60 = tf.keras.layers.Flatten()(pda_list_m60)
  

  pd_arrays_d1_input = tf.keras.layers.Input(shape = (10,3))
  pda_d1_times = pd_arrays_d1_input[::,::,0]
  pda_d1_types = pd_arrays_d1_input[::,::,1]
  pda_d1_prices = tf.keras.layers.Reshape((-1,1))(pd_arrays_d1_input[::,::,2])
  pda_d1_times = embed_times(pda_d1_times)
  pda_d1_types = embed_pda_types(pda_d1_types)
  pda_list_d1 = tf.keras.layers.Concatenate(axis=-1)([pda_d1_prices, pda_d1_times, pda_d1_types])
  pda_list_d1 = tf.keras.layers.Flatten()(pda_list_d1)
  



  action_m1_inputs = tf.keras.layers.Input(shape=(256,5))
  action_m5_inputs = tf.keras.layers.Input(shape=(256,5))
  action_m15_inputs = tf.keras.layers.Input(shape=(256,5))

  embed_action_type = tf.keras.layers.Embedding(18,6)

  t1 = action_m1_inputs[::,::,1]
  t2 = action_m1_inputs[::,::,4]
  at = action_m1_inputs[::,::,0]
  v1 = tf.keras.layers.Reshape((-1,1))(action_m1_inputs[::,::,2])
  v3 = tf.keras.layers.Reshape((-1,1))(action_m1_inputs[::,::,3])
  t1 = embed_times(t1)
  t2 = embed_times(t2)
  at = embed_times(at)
  actions_m1 = tf.keras.layers.Concatenate(axis=-1)([v1,at,v3,t1,t2])
  

  t1 = action_m5_inputs[::,::,1]
  t2 = action_m5_inputs[::,::,4]
  at = action_m5_inputs[::,::,0]
  v1 = tf.keras.layers.Reshape((-1,1))(action_m5_inputs[::,::,2])
  v3 = tf.keras.layers.Reshape((-1,1))(action_m5_inputs[::,::,3])
  t1 = embed_times(t1)
  t2 = embed_times(t2)
  at = embed_times(at)
  actions_m5 = tf.keras.layers.Concatenate(axis=-1)([v1,at,v3,t1,t2])
  

  t1 = action_m15_inputs[::,::,1]
  t2 = action_m15_inputs[::,::,4]
  at = action_m15_inputs[::,::,0]
  v1 = tf.keras.layers.Reshape((-1,1))(action_m15_inputs[::,::,2])
  v3 = tf.keras.layers.Reshape((-1,1))(action_m15_inputs[::,::,3])
  t1 = embed_times(t1)
  t2 = embed_times(t2)
  at = embed_times(at)
  actions_m15 = tf.keras.layers.Concatenate(axis=-1)([v1,at,v3,t1,t2])
  



  input_closing_prices = tf.keras.layers.Input(shape=(1))
  input_current_pos = tf.keras.layers.Input(shape=(1))
  input_closing_times_in = tf.keras.layers.Input(shape=(1))
  input_closing_times = embed_times(input_closing_times_in)
  input_closing_times = tf.keras.layers.Flatten()(input_closing_times)
  


  actions_m1 = tf.keras.layers.Dense(32)(actions_m1)
  actions_m1 = tf.keras.layers.LeakyReLU()(actions_m1)
  actions_m1 = tf.keras.layers.Dense(64)(actions_m1)
  actions_m1 = tf.keras.layers.LeakyReLU()(actions_m1)
  actions_m1 = tf.keras.layers.GRU(256, return_sequences=True)(actions_m1)
  actions_m1 = tf.keras.layers.GRU(256)(actions_m1)

  actions_m5 = tf.keras.layers.Dense(32)(actions_m5)
  actions_m5 = tf.keras.layers.LeakyReLU()(actions_m5)
  actions_m5 = tf.keras.layers.Dense(64)(actions_m5)
  actions_m5 = tf.keras.layers.LeakyReLU()(actions_m5)
  actions_m5 = tf.keras.layers.GRU(256, return_sequences=True)(actions_m5)
  actions_m5 = tf.keras.layers.GRU(256)(actions_m5)

  actions_m15 = tf.keras.layers.Dense(32)(actions_m15)
  actions_m15 = tf.keras.layers.LeakyReLU()(actions_m15)
  actions_m15 = tf.keras.layers.Dense(64)(actions_m15)
  actions_m15 = tf.keras.layers.LeakyReLU()(actions_m15)
  actions_m15 = tf.keras.layers.GRU(256, return_sequences=True)(actions_m15)
  actions_m15 = tf.keras.layers.GRU(256)(actions_m15)

  dense_input = tf.keras.layers.Concatenate()([input_current_pos, input_closing_prices, input_closing_times, pda_list_m60, pda_list_d1, pda_list_m15, pda_list_m5, pda_list_m1, actions_m1, actions_m5, actions_m15])
  

  x = tf.keras.layers.Dense(1024)(dense_input)
  x = tf.keras.layers.LeakyReLU()(x)
    
  x = tf.keras.layers.Dense(1024)(x)
  x = tf.keras.layers.LeakyReLU()(x)
    
  x = tf.keras.layers.Dense(1024)(x)
  x = tf.keras.layers.LeakyReLU()(x)
    

  x = tf.keras.layers.Dense(2, activation = "linear")(x)

  model = tf.keras.Model(inputs = [input_current_pos, input_closing_prices, input_closing_times_in, pd_arrays_m1_input, pd_arrays_m5_input, pd_arrays_m15_input, pd_arrays_m60_input, pd_arrays_d1_input, action_m1_inputs, action_m5_inputs, action_m15_inputs], outputs=x)
  return model
