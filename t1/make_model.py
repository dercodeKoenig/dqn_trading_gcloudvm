## tx + gru large

import tensorflow as tf

#config
batch_size = 24
gamma = 0.995
learning_rate=0.000002
num_data_generation_threads = 12
batch_generation_threads = 8
memory_size = 300_000
ep_len = 100



from tensorflow.keras import layers
from tensorflow import keras

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super().__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

def make_model():
    
  

  embed_times = tf.keras.layers.Embedding(60*24,8)
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
  at = embed_action_type(at)
  actions_m1 = tf.keras.layers.Concatenate(axis=-1)([v1,at,v3,t1,t2])
  

  t1 = action_m5_inputs[::,::,1]
  t2 = action_m5_inputs[::,::,4]
  at = action_m5_inputs[::,::,0]
  v1 = tf.keras.layers.Reshape((-1,1))(action_m5_inputs[::,::,2])
  v3 = tf.keras.layers.Reshape((-1,1))(action_m5_inputs[::,::,3])
  t1 = embed_times(t1)
  t2 = embed_times(t2)
  at = embed_action_type(at)
  actions_m5 = tf.keras.layers.Concatenate(axis=-1)([v1,at,v3,t1,t2])
  

  t1 = action_m15_inputs[::,::,1]
  t2 = action_m15_inputs[::,::,4]
  at = action_m15_inputs[::,::,0]
  v1 = tf.keras.layers.Reshape((-1,1))(action_m15_inputs[::,::,2])
  v3 = tf.keras.layers.Reshape((-1,1))(action_m15_inputs[::,::,3])
  t1 = embed_times(t1)
  t2 = embed_times(t2)
  at = embed_action_type(at)
  actions_m15 = tf.keras.layers.Concatenate(axis=-1)([v1,at,v3,t1,t2])
  



  input_closing_prices = tf.keras.layers.Input(shape=(1))
  input_current_pos = tf.keras.layers.Input(shape=(1))
  input_current_day_in = tf.keras.layers.Input(shape=(1))
  input_current_day = tf.keras.layers.Embedding(7,3)(input_current_day_in)
  input_current_day = tf.keras.layers.Flatten()(input_current_day)
  input_closing_times_in = tf.keras.layers.Input(shape=(1))
  input_closing_times = embed_times(input_closing_times_in)
  input_closing_times = tf.keras.layers.Flatten()(input_closing_times)
  


  gru_units = 32
  tx_embed_len = 8
  tx_embed_units = 32
  
  pos_embedding = PositionEmbedding(256, tx_embed_units)
  
  def embed_information(input_state):
      input_state = tf.keras.layers.Dense(512, activation = "relu")(input_state)
      input_state = tf.keras.layers.Dense(512, activation = "relu")(input_state)
      input_state_rnn = tf.keras.layers.Dense(gru_units, activation = "relu")(input_state)
      input_state_tx = tf.keras.layers.Dense(tx_embed_units*tx_embed_len, activation = "relu")(input_state)
      input_state_tx = tf.keras.layers.Reshape((-1,tx_embed_units))(input_state_tx)
      return input_state_tx, input_state_rnn
  
  def process_actions(actions, additional_info):
    actions = tf.keras.layers.Dense(tx_embed_units)(actions)
    actions = tf.keras.layers.LeakyReLU()(actions)
    input_state_tx, input_state_rnn = embed_information(additional_info)
    actions = tf.keras.layers.Concatenate(axis=1)([input_state_tx, actions])
    actions = pos_embedding(actions)
    actions = TransformerBlock(tx_embed_units, 4, 256)(actions)
    actions = tf.keras.layers.GRU(gru_units)(actions, initial_state = input_state_rnn)
    return actions
  
  input_state = tf.keras.layers.Concatenate()([input_current_pos, input_closing_prices, input_closing_times, input_current_day, pda_list_m60, pda_list_d1])
  actions_m15 = process_actions(actions_m15, input_state)
  input_state = tf.keras.layers.Concatenate()([input_current_pos, input_closing_prices, input_closing_times, input_current_day, pda_list_m60, pda_list_d1, pda_list_m15, actions_m15])
  actions_m5 = process_actions(actions_m5, input_state)
  input_state = tf.keras.layers.Concatenate()([input_current_pos, input_closing_prices, input_closing_times, input_current_day, pda_list_m60, pda_list_d1, pda_list_m15, pda_list_m5, actions_m15, actions_m5])
  actions_m1 = process_actions(actions_m1, input_state)
  
  dense_input = tf.keras.layers.Concatenate()([input_current_pos, input_closing_prices, input_closing_times, input_current_day, pda_list_m60, pda_list_d1, pda_list_m15, pda_list_m5, pda_list_m1, actions_m1, actions_m5, actions_m15])
  

  x = tf.keras.layers.Dense(1024)(dense_input)
  x = tf.keras.layers.LeakyReLU()(x)
    
  x = tf.keras.layers.Dense(1024)(x)
  x = tf.keras.layers.LeakyReLU()(x)
    
  x = tf.keras.layers.Dense(1024)(x)
  x = tf.keras.layers.LeakyReLU()(x)
    

  x = tf.keras.layers.Dense(2, activation = "linear")(x)

  model = tf.keras.Model(inputs = [input_current_pos, input_closing_prices, input_closing_times_in, input_current_day_in, pd_arrays_m1_input, pd_arrays_m5_input, pd_arrays_m15_input, pd_arrays_m60_input, pd_arrays_d1_input, action_m1_inputs, action_m5_inputs, action_m15_inputs], outputs=x)
  return model
